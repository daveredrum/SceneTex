import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, dtype=torch.float32):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias, dtype=dtype))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x
    
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        outputs = self.net(coords)
        return outputs
    

class HashGrid(nn.Module):
    def __init__(self, in_channels,
        otype, n_levels, n_features_per_level, log2_hashmap_size, base_resolution, # the same as in tinycudann
        max_resolution, # NOTE need to compute per_level_scale ,
        dtype=torch.float32 # half precision might lead to NaN
    ):
        
        super().__init__()

        self.otype = otype
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.per_level_scale = self.get_per_level_scale()

        self.config = {
            "otype": self.otype,
            "n_levels": self.n_levels,
            "n_features_per_level": self.n_features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.per_level_scale
        }
        self.hashgrid = tcnn.Encoding(in_channels, self.config, dtype=dtype)

    def get_per_level_scale(self):
        return np.power(self.max_resolution / self.base_resolution, 1 / self.n_levels)
    
    def forward(self, inputs):
        return self.hashgrid(inputs)
    
class FasterMLP(nn.Module):
    def __init__(self, in_channels, out_channels,
        n_neurons, n_hidden_layers,
        dtype=torch.float32 # half precision might lead to NaN
    ):
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers

        self.config = {
            "otype": "CutlassMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": n_neurons,
            "n_hidden_layers": n_hidden_layers
        }
        self.net = tcnn.Network(in_channels, out_channels, self.config)
    
    def forward(self, inputs):
        return self.net(inputs)
    
class HashGridMLP(nn.Module):
    def __init__(self, in_channels,
        hashgrid_config, mlp_config
    ):
        
        super().__init__()

        self.hashgrid_config = {
            "otype": hashgrid_config.otype,
            "n_levels": hashgrid_config.n_levels,
            "n_features_per_level": hashgrid_config.n_features_per_level,
            "log2_hashmap_size": hashgrid_config.log2_hashmap_size,
            "base_resolution": hashgrid_config.base_resolution,
            "per_level_scale": self.get_per_level_scale(
                hashgrid_config.max_resolution,
                hashgrid_config.base_resolution,
                hashgrid_config.n_levels
            )
        }
        self.MLP_config = {
            "otype": mlp_config.otype,
            "activation": mlp_config.activation,
            "output_activation": mlp_config.output_activation,
            "n_neurons": mlp_config.n_neurons,
            "n_hidden_layers": mlp_config.n_hidden_layers
        }

        self.net = tcnn.NetworkWithInputEncoding(in_channels, mlp_config.out_channels, self.hashgrid_config, self.MLP_config)

    def get_per_level_scale(self, max_resolution, base_resolution, n_levels):
        return np.power(max_resolution / base_resolution, 1 / n_levels)
    
    def forward(self, inputs):
        return self.net(inputs)