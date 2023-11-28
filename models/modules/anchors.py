import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

sys.path.append("./models")
from models.modules.modules import MLP
from models.modules.multihead_attention import MHA, FlashCrossAttention

class AnchorTransformer(nn.Module):
    def __init__(self, 
        config,
        device,
        anchor_dim,
        num_instances # this must be specified on init
    ): 
        
        super().__init__()

        self.config = config
        self.device = device

        self.anchor_dim = anchor_dim
        self.num_instances = num_instances

        self.hidden_size = config.anchor_config.hidden_size
        self.num_heads = config.anchor_config.num_heads
        self.num_mapping_layers = config.anchor_config.num_mapping_layers

        if self.config.anchor_config.anchor_type == "self-attention":
            if self.num_mapping_layers == 0 and self.num_heads == 1:
                self.hidden_size = anchor_dim
                self.map_key = nn.Identity()
                self.map_query = nn.Identity()
                self.map_value = nn.Identity()

                self.attention = nn.MultiheadAttention(
                    anchor_dim,
                    1,
                    batch_first=True # (batch, seq, feature)
                )
                self.map_outputs = nn.Identity()
            else:
                self.map_key = MLP(anchor_dim, self.hidden_size * self.num_heads, self.hidden_size, self.num_mapping_layers)
                self.map_query = MLP(anchor_dim, self.hidden_size * self.num_heads, self.hidden_size, self.num_mapping_layers)
                self.map_value = MLP(anchor_dim, self.hidden_size * self.num_heads, self.hidden_size, self.num_mapping_layers)

                self.attention = nn.MultiheadAttention(
                    self.hidden_size * self.num_heads,
                    self.num_heads,
                    batch_first=True # (batch, seq, feature)
                )
                self.map_outputs = MLP(self.hidden_size * self.num_heads, anchor_dim, self.hidden_size, self.num_mapping_layers)

        elif self.config.anchor_config.anchor_type == "cross-attention":
            if self.num_mapping_layers == 0 and self.num_heads == 1:
                self.hidden_size = anchor_dim
                self.map_key = nn.Identity()
                self.map_query = nn.Identity()
                self.map_value = nn.Identity()

                self.attention = nn.MultiheadAttention(
                    anchor_dim,
                    1,
                    batch_first=True # (batch, seq, feature)
                )
                self.map_outputs = nn.Identity()

            else:
                self.map_key = MLP(anchor_dim, self.hidden_size * self.num_heads, self.hidden_size, self.num_mapping_layers)
                self.map_query = MLP(anchor_dim, self.hidden_size * self.num_heads, self.hidden_size, self.num_mapping_layers)
                self.map_value = MLP(anchor_dim, self.hidden_size * self.num_heads, self.hidden_size, self.num_mapping_layers)

                self.attention = nn.MultiheadAttention(
                    self.hidden_size * self.num_heads,
                    self.num_heads,
                    batch_first=True # (batch, seq, feature)
                )
                self.map_outputs = MLP(self.hidden_size * self.num_heads, anchor_dim, self.hidden_size, self.num_mapping_layers)

        elif self.config.anchor_config.anchor_type == "flash-attention":
            if self.num_mapping_layers == 0 and self.num_heads == 1:
                self.hidden_size = anchor_dim
                self.map_key = nn.Identity()
                self.map_query = nn.Identity()
                self.map_value = nn.Identity()
                self.map_outputs = nn.Identity()

            else:
                self.map_key = MLP(anchor_dim, self.hidden_size * self.num_heads, self.hidden_size, self.num_mapping_layers)
                self.map_query = MLP(anchor_dim, self.hidden_size * self.num_heads, self.hidden_size, self.num_mapping_layers)
                self.map_value = MLP(anchor_dim, self.hidden_size * self.num_heads, self.hidden_size, self.num_mapping_layers)
                self.map_outputs = MLP(self.hidden_size * self.num_heads, anchor_dim, self.hidden_size, self.num_mapping_layers)

            self.attention = FlashCrossAttention() # NOTE input must be half precision

    def _map_inputs(self, query, key, value):
        if self.config.anchor_config.output_type == "token":
            # inputs = torch.cat([inputs, self.embedding.unsqueeze(1)], dim=1)
            avg_token = query.mean(1, keepdim=True)
            query = torch.cat([query, avg_token], dim=1)

        key = self.map_key(key)
        query = self.map_query(query)
        value = self.map_value(value)
        
        return query, key, value
    
    def _map_outputs(self, inputs):
        outputs = self.map_outputs(inputs) # (M, L, C)

        if self.config.anchor_config.output_type == "token":
            outputs = outputs[:, -1, :]
        elif self.config.anchor_config.output_type == "mean":
            outputs = outputs.mean(1)
        else:
            pass

        return outputs
    
    def _map_features(self, features, anchors, instances_in_view, pad_seq=False):
        B, H, W, C = features.shape
        features = features.reshape(-1, C)
        instances_in_view = instances_in_view.reshape(-1)
        labels = torch.unique(instances_in_view).long()

        # outputs
        seq_features, seq_anchors, seq_labels, seqlens, seqlens_k = [], [], [], [0], [0]
        cu_seqlens, max_seqlen, cu_seqlens_k, max_seqlen_k = None, None, None, None
        map_flag = False

        for label in labels:
            if label == 0: continue
            instance_mask = instances_in_view == label
            instance_feature = features[instance_mask]
            instace_labels = instances_in_view[instance_mask]
            seq_features.append(instance_feature)
            seq_anchors.append(anchors[label-1])
            seqlen = instance_feature.shape[0]
            seqlens.append(seqlen)
            seqlens_k.append(anchors.shape[1])
            seq_labels.append(instace_labels)

        if len(seq_features) > 0:

            map_flag = True

            if pad_seq:
                seq_features = pad_sequence(seq_features, batch_first=True)
                seq_labels = pad_sequence(seq_labels, batch_first=True)
                seq_anchors = torch.stack(seq_anchors)
            else:
                seq_features = torch.cat(seq_features, dim=0)
                seq_labels = torch.cat(seq_labels, dim=0)
                seq_anchors = torch.cat(seq_anchors, dim=0)
                cu_seqlens = torch.cumsum(torch.IntTensor(seqlens), dim=0).to(self.device).int()
                max_seqlen = max(seqlens)
                cu_seqlens_k = torch.cumsum(torch.IntTensor(seqlens_k), dim=0).to(self.device).int()
                max_seqlen_k = max(seqlens_k)

        return seq_features, seq_labels, seq_anchors, cu_seqlens, max_seqlen, cu_seqlens_k, max_seqlen_k, map_flag

    def _unmap_features(self, features, seq_labels, instances_in_view):
        *_, C = features.shape
        B, H, W = instances_in_view.shape
        unmapped = torch.zeros(B, H, W, C).to(self.device)

        if self.config.anchor_config.anchor_type == "flash-attention":
            unmapped = unmapped.reshape(-1, C)
            instances_in_view = instances_in_view.reshape(-1)
            assert unmapped.shape[0] == instances_in_view.shape[0]
            labels = torch.unique(instances_in_view)
            for label in labels:
                if label == 0: continue
                unmapped[instances_in_view == label] = features[seq_labels == label]

            unmapped = unmapped.reshape(B, H, W, C)

        elif self.config.anchor_config.anchor_type == "cross-attention":
            unmapped = unmapped.reshape(-1, C)
            instances_in_view = instances_in_view.reshape(-1)
            assert unmapped.shape[0] == instances_in_view.shape[0]
            for i in range(features.shape[0]): # feature indices indicate instances
                unmapped[instances_in_view == i+1] = features[seq_labels == i+1]

            unmapped = unmapped.reshape(B, H, W, C)

        return unmapped
    
    def _apply_outputs(self, features, anchors, instances_in_view):
        if self.config.anchor_config.anchor_type in ["self-attention", "mean"]:
            B, H, W = instances_in_view.shape # NOTE instance_in_view must in shape (B, H, W)
            instances_in_view = instances_in_view.reshape(-1) - 1 # instances are indexed from 0, -1 is the background
            background_mask = instances_in_view == -1
            anchor_features = anchors[instances_in_view.long(), :]
            anchor_features[background_mask] = 0
            anchor_features = anchor_features.reshape(B, H, W, -1)
        else:
            anchor_features = anchors

        # output
        features = features + anchor_features

        return features
    
    def _prepare_flash_attention_inputs(self, query, key, value):
        query = query.reshape(-1, self.num_heads, self.hidden_size)
        key = key.reshape(-1, self.num_heads, self.hidden_size)
        value = value.reshape(-1, self.num_heads, self.hidden_size)
        key_value = torch.stack([key, value], dim=1)

        return query, key_value

    def forward(self, anchors, features, instances_in_view):
        assert len(anchors.shape) == 3, "anchors should be in shape (M, L, C)"
        assert len(features.shape) == 4, "features should be in shape (B, H, W, C)"

        if self.config.anchor_config.anchor_type == "self-attention":
            query, key, value = self._map_inputs(anchors, anchors, anchors)
            anchors, _ = self.attention(query, key, value)
            anchors = self._map_outputs(anchors)
        elif self.config.anchor_config.anchor_type == "cross-attention":
            seq_features, seq_labels, seq_anchors, cu_seqlens, max_seqlen, cu_seqlens_k, max_seqlen_k, map_flag = self._map_features(features, anchors, instances_in_view, True)
            if map_flag:
                seq_features, seq_anchors, seq_anchors = self._map_inputs(seq_features, seq_anchors, seq_anchors)
                seq_features, _ = self.attention(
                    seq_features,
                    seq_anchors,
                    seq_anchors
                )
                seq_features = self._map_outputs(seq_features)
                seq_features = self._unmap_features(seq_features, seq_labels, instances_in_view)
                anchors = seq_features
            else:
                anchors = features
        elif self.config.anchor_config.anchor_type == "flash-attention":
            seq_features, seq_labels, seq_anchors, cu_seqlens, max_seqlen, cu_seqlens_k, max_seqlen_k, map_flag = self._map_features(features, anchors, instances_in_view)
            if map_flag:
                seq_features, seq_anchors, seq_anchors = self._map_inputs(seq_features, seq_anchors, seq_anchors)
                seq_query, seq_key_value = self._prepare_flash_attention_inputs(seq_features, seq_anchors, seq_anchors)
                seq_features = self.attention(
                    seq_query.half(), # (Sq, H, C)
                    seq_key_value.half(), # (Sk, 2, H_k, C)
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                    cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k
                ).to(torch.float32) # (Sq, H, C)
                seq_features = self._map_outputs(seq_features.reshape(seq_features.shape[0], -1)) # (Sq, C)
                seq_features = self._unmap_features(seq_features, seq_labels, instances_in_view)
                anchors = seq_features
            else:
                anchors = features
        else:
            anchors = anchors.mean(1)

        # output
        features = self._apply_outputs(features, anchors, instances_in_view)

        return features


        
    