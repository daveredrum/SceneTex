import os
import json

import torch
import torch.nn as nn

import pytorch_lightning as pl

from pathlib import Path

# pytorch3d
# from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer import TexturesUV

import sys
sys.path.append("./lib")
from lib.mesh_helper import init_multiple_meshes_xatlas, init_multiple_meshes_as_scene, init_background
from lib.build_instance_map import build_instance_map

sys.path.append("./models")
from models.modules.modules import MLP, Siren, HashGrid, HashGridMLP

"""
    Textured Mesh model - Defining the texture and background for optimization
"""
class TextureMesh(nn.Module):
    def __init__(self, 
        config,
        device
    ): 
        
        super().__init__()
        
        self.config = config
        self.device = device

        self.num_instances = 0

        self._init_mesh()

    def apply_texture_to_mesh(self, mesh, faces, aux, texture_tensor, sampling_mode="bilinear"):
        new_mesh = mesh.clone() # in-place operation - DANGER!!!
        new_mesh.textures = TexturesUV(
            maps=texture_tensor, # B, H, W, C
            faces_uvs=faces.textures_idx[None, ...],
            verts_uvs=aux.verts_uvs[None, ...],
            sampling_mode=sampling_mode,
            # align_corners=False
        )

        return new_mesh
    
    def repeat_meshes_as_batch(self, mesh, batch_size):
        return join_meshes_as_batch(
            [mesh for _ in range(batch_size)],
            include_textures=True
        )

    def _init_mesh(self):
        cache_dir = self.config.log_dir

        self.mesh_dict = init_multiple_meshes_as_scene(
            json.load(open(self.config.scene_config_path)), 
            str(cache_dir), 
            self.device, 
            subdivide_factor=self.config.subdivide_factor,
            return_dict=True
        )

        self.mesh, self.texture = self._init_texture(self.mesh_dict)

        if self.config.use_background:
            self.background_mesh_dict = init_background(
                self.config.background,
                self.mesh.get_bounding_boxes().cpu().numpy()[0],
                str(cache_dir),
                self.device,
                return_dict=True
            )

            self.background_mesh, self.background_texture = self._init_texture(self.background_mesh_dict)

    def _init_texture(self, mesh_dict):
        texture = torch.randn((
                1, 
                self.config.latent_texture_size, 
                self.config.latent_texture_size, 
                self.config.latent_channels
            ), requires_grad=True, device=self.device)

        mesh = self.apply_texture_to_mesh(
            mesh_dict["mesh"],
            mesh_dict["faces"],
            mesh_dict["aux"],
            texture
        )

        if self.config.texture_type == "hashgrid":
            texture = HashGrid(
                2,
                self.config.hashgrid_config.otype,
                self.config.hashgrid_config.n_levels,
                self.config.hashgrid_config.n_features_per_level,
                self.config.hashgrid_config.log2_hashmap_size,
                self.config.hashgrid_config.base_resolution,
                self.config.hashgrid_config.max_resolution,
                torch.float16 if self.config.hashgrid_config.dtype == "half" else torch.float32 # full precision to avoid NaN
            )
        
        elif self.config.texture_type == "hashgrid_mlp":
            texture = HashGridMLP(
                2,
                self.config.hashgrid_config,
                self.config.mlp_config
            )

        else:
            texture = torch.randn((
                1, 
                self.config.latent_texture_size, 
                self.config.latent_texture_size, 
                self.config.latent_channels
            ), requires_grad=True, device=self.device)

            mesh = self.apply_texture_to_mesh(
                mesh_dict["mesh"],
                mesh_dict["faces"],
                mesh_dict["aux"],
                texture
            )

        return mesh, texture
    
    def sort_rand_gpu(self, pop_size, num_samples):
        """Generate a random torch.Tensor (GPU) and sort it to generate indices."""
        return torch.argsort(torch.rand(pop_size, device=self.device))[:num_samples]

    def build_instance_map(self, studio, cache_dir):
        # build instance masks
        instance_map = build_instance_map(studio, 
            cache_dir, cache_dir,
            self.config.dummy_texture_path, 
            self.device, self.config.texture_size, self.config.render_size, 500).to(self.device)

        assert len(instance_map.shape) == 2, "instance map should be in shape (W, H)"

        # replace the dummy texture with the instance map
        self.mesh = self.apply_texture_to_mesh(
            self.mesh_dict["mesh"],
            self.mesh_dict["faces"],
            self.mesh_dict["aux"],
            instance_map[None, :, :, None].repeat(1, 1, 1, 3),
            "nearest"
        )
            
        self.instance_map = instance_map
    
    def sample_instance_anchors(self, cache_dir):
        cache_path = Path(cache_dir) / "anchors.pth"

        if cache_path.exists():
            print("=> loading instance anchors from {}...".format(str(cache_path)))
            self.instance_anchors = torch.load(str(cache_path))
            self.num_instances = self.instance_anchors.shape[0]
        else:
            print("=> sampling instance anchors...")
            instance_labels = torch.unique(self.instance_map)
            assert instance_labels.shape[0] > 1
            instance_labels = instance_labels[instance_labels != 0]

            instance_anchors = []
            for instance_id in instance_labels:
                instance_mask = self.instance_map == instance_id
                uv_coords = torch.nonzero(instance_mask) # NumInsTex, 2
                sampled_ids = self.sort_rand_gpu(uv_coords.shape[0], self.config.num_anchors)
                sampled_uv_coords = uv_coords[sampled_ids, :]
                instance_anchors.append(sampled_uv_coords)

            instance_anchors = torch.stack(instance_anchors) # M, NumAnchor, 2
            instance_anchors = instance_anchors.float() / self.config.texture_size

            assert instance_anchors.min() >= 0 and instance_anchors.max() <= 1

            print("=> saving anchors to {}".format(str(cache_path)))
            torch.save(instance_anchors, str(cache_path))

            self.instance_anchors = instance_anchors
            self.num_instances = self.instance_anchors.shape[0]

    