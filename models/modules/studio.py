import json
import random

import torch
import torch.nn as nn

import numpy as np

from omegaconf import OmegaConf

from pytorch3d.ops import interpolate_face_attributes

# customized
import sys
sys.path.append("./lib")
from lib.camera_helper import init_trajectory, init_blender_trajectory, init_blenderproc_trajectory, init_camera_R_T
from lib.render_helper import init_renderer
from lib.shading_helper import init_flat_texel_shader

sys.path.append("./models")
from models.modules.modules import MLP
from models.modules.anchors import AnchorTransformer

class Studio(nn.Module):
    def __init__(self, 
        config,
        device
    ): 
        
        super().__init__()
        
        self.config = config
        self.device = device

        # render function
        self.render_func = self._init_render_func()

        self._init_camera_settings()

    def _init_camera_settings(self):

        self.Rs, self.Ts, self.fovs = [], [], []
        self.inference_Rs, self.inference_Ts, self.inference_fovs = [], [], []

        if self.config.use_sphere_cameras:
            sphere_cameras = OmegaConf.load(self.config.sphere_cameras)

            dist_linspace = np.linspace(
                sphere_cameras.dist.min,
                sphere_cameras.dist.max,
                1 if sphere_cameras.dist.min == sphere_cameras.dist.max else sphere_cameras.dist.num_linspace,
            )
            elev_linspace = np.linspace(
                sphere_cameras.elev.min,
                sphere_cameras.elev.max,
                1 if sphere_cameras.elev.min == sphere_cameras.elev.max else sphere_cameras.elev.num_linspace,
            )
            azim_linspace = np.linspace(
                sphere_cameras.azim.min,
                sphere_cameras.azim.max,
                1 if sphere_cameras.azim.min == sphere_cameras.azim.max else sphere_cameras.azim.num_linspace,
            )
            fov_linspace = np.linspace(
                sphere_cameras.fov.min,
                sphere_cameras.fov.max,
                1 if sphere_cameras.fov.min == sphere_cameras.fov.max else sphere_cameras.fov.num_linspace,
            )
            at = np.array(sphere_cameras.at)

            combinations = np.array(np.meshgrid(dist_linspace, elev_linspace, azim_linspace, fov_linspace)).T.reshape(-1, 4)
            dist_list = combinations[:, 0].tolist()
            elev_list = combinations[:, 1].tolist()
            azim_list = combinations[:, 2].tolist()

            Rs, Ts = init_trajectory(dist_list, elev_list, azim_list, at)
            fovs = combinations[:, 3].tolist()

            self.Rs += Rs
            self.Ts += Ts
            self.fovs += fovs

            print("=> using {} spherical cameras for training".format(len(Rs)))

            # inference cameras
            if len(self.inference_Rs) == 0 and len(self.inference_Ts) == 0 and len(self.inference_fovs) == 0:
                dist_linspace = [sphere_cameras.dist.min] # always take the min dist from spherical cameras
                elev_linspace = [self.config.elev]
                azim_linspace = np.linspace(
                    self.config.azim[0],
                    self.config.azim[1],
                    self.config.log_latents_views,
                )
                fov_linspace = [self.config.fov]
                at = np.array(sphere_cameras.at) # always take the cameras center from spherical cameras

                combinations = np.array(np.meshgrid(dist_linspace, elev_linspace, azim_linspace, fov_linspace)).T.reshape(-1, 4)
                inference_dist_list = combinations[:, 0].tolist()
                inference_elev_list = combinations[:, 1].tolist()
                inference_azim_list = combinations[:, 2].tolist()
                inference_fov_list = combinations[:, 3].tolist()
                inference_at = at

                self.inference_Rs, self.inference_Ts = init_trajectory(inference_dist_list, inference_elev_list, inference_azim_list, inference_at)
                self.infernece_fovs = inference_fov_list

        if self.config.use_blenderproc_cameras:
            poses = json.load(open(self.config.blenderproc_cameras))
            Rs, Ts = init_blenderproc_trajectory(poses, self.device)
            fovs = [self.config.fov] * len(Rs)

            self.Rs += Rs
            self.Ts += Ts
            self.fovs += fovs

            print("=> using {} blenderproc cameras for training".format(len(Rs)))

            # inference cameras
            if len(self.inference_Rs) == 0 and len(self.inference_Ts) == 0 and len(self.inference_fovs) == 0:
                interval = len(self.Rs) // self.config.log_latents_views
                self.inference_Rs = [r for i, r in enumerate(self.Rs) if i % interval == 0]
                self.inference_Ts = [t for i, t in enumerate(self.Ts) if i % interval == 0]
                self.infernece_fovs = [self.config.fov for _ in range(self.config.log_latents_views)]

        if self.config.use_blender_cameras:
            poses = json.load(open(self.config.blender_cameras))
            Rs, Ts = init_blender_trajectory(poses, self.device)
            fovs = [self.config.fov] * len(Rs)

            self.Rs += Rs
            self.Ts += Ts
            self.fovs += fovs

            print("=> using {} blender cameras for training".format(len(Rs)))

            # inference cameras
            if len(self.inference_Rs) == 0 and len(self.inference_Ts) == 0 and len(self.inference_fovs) == 0:
                interval = len(self.Rs) // self.config.log_latents_views
                self.inference_Rs = [r for i, r in enumerate(self.Rs) if i % interval == 0]
                self.inference_Ts = [t for i, t in enumerate(self.Ts) if i % interval == 0]
                self.infernece_fovs = [self.config.fov for _ in range(self.config.log_latents_views)]

        self.num_cameras = len(self.Rs)
        self.num_inference_cameras = len(self.inference_Rs)
        assert self.num_cameras > 0 and self.num_inference_cameras > 0, "no camera defined!"

        print("=> using {} cameras for training, {} cameras for inference.".format(self.num_cameras, self.num_inference_cameras))


    def _init_render_func(self):
        if self.config.render_func_type == "mlp":
            if self.config.texture_type == "hashgrid":
                in_channels = self.config.hashgrid_config.n_levels * self.config.hashgrid_config.n_features_per_level
            elif self.config.texture_type == "hashgrid_mlp":
                in_channels = self.config.mlp_config.out_channels
            else:
                in_channels = self.config.latent_channels

            render_func = MLP(
                in_channels,
                self.config.render_channels,
                self.config.view_embedding_hidden_dim,
                self.config.num_view_embedding_layers,
                dtype=torch.float32
            ).to(self.device)
        
        elif self.config.render_func_type == "none":
            render_func = nn.Identity()

        else:
            raise NotImplementedError("not supported render function type: {}".format(self.config.render_func_type))

        return render_func
    
    def init_anchor_func(self, num_instances):
        if self.config.texture_type == "hashgrid":
            anchor_dim = self.config.hashgrid_config.n_levels * self.config.hashgrid_config.n_features_per_level
        elif self.config.texture_type == "hashgrid_mlp":
            anchor_dim = self.config.mlp_config.out_channels
        else:
            anchor_dim = self.config.latent_channels

        anchor_func = AnchorTransformer(self.config, self.device, anchor_dim=anchor_dim, num_instances=num_instances).to(self.device)

        self.anchor_func = anchor_func

    def set_cameras(self, R, T, fov, image_size):
        return init_camera_R_T(R, T, image_size, self.device, fov)
    
    def set_renderer(self, camera, image_size):
        return init_renderer(camera,
            shader=init_flat_texel_shader(
                camera=camera,
                device=self.device
            ),
            image_size=image_size, 
            faces_per_pixel=self.config.faces_per_pixel
        )

    def _sample_one_camera(self, step, random_cameras=False, inference=False):
        R, T, fov, idx = None, None, None, None
        if inference:
            idx = step % self.num_inference_cameras
            R, T, fov = self.inference_Rs[idx], self.inference_Ts[idx], self.infernece_fovs[idx]
        else:

            if random_cameras:
                idx = random.choice(range(self.num_cameras))
            else:
                idx = step % self.num_cameras

            R, T, fov = self.Rs[idx], self.Ts[idx], self.fovs[idx]

        return R, T, fov, idx
    
    def sample_cameras(self, step, num_samples, random_cameras=False, inference=False):
        if num_samples == 1:
            return self._sample_one_camera(step, random_cameras, inference)
        else:
            Rs, Ts, fovs, ids = [], [], [], []
            cur_step = step % self.num_cameras
    
            if random_cameras:
                pool = [e for e in range(self.num_cameras) if e != cur_step]
                next_steps = random.sample(pool, k=num_samples-1)
            else:
                next_steps = [(cur_step+s+1) % self.num_cameras for s in range(num_samples-1)]

            steps = [cur_step] + next_steps
            for s in steps:
                R, T, fov, idx = self._sample_one_camera(s)
                Rs.append(R)
                Ts.append(T)
                fovs.append(fov)
                ids.append(idx)

            Rs = torch.cat(Rs, dim=0)
            Ts = torch.cat(Ts, dim=0)

            return Rs, Ts, fovs, ids

    def get_uv_coordinates(self, mesh, fragments):
        xyzs = mesh.verts_padded() # (N, V, 3)
        faces = mesh.faces_padded() # (N, F, 3)

        faces_uvs = mesh.textures.faces_uvs_padded()
        verts_uvs = mesh.textures.verts_uvs_padded()

        # NOTE Meshes are replicated in batch. Taking the first one is enough.
        batch_size, _, _ = xyzs.shape
        xyzs, faces, faces_uvs, verts_uvs = xyzs[0], faces[0], faces_uvs[0], verts_uvs[0]
        faces_coords = verts_uvs[faces_uvs] # (F, 3, 2)

        # replicate the coordinates as batch
        faces_coords = faces_coords.repeat(batch_size, 1, 1)

        invalid_mask = fragments.pix_to_face == -1
        target_coords = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_coords
        ) # (N, H, W, 1, 3)
        _, H, W, K, _ = target_coords.shape
        target_coords[invalid_mask] = 0
        assert K == 1 # pixel_per_faces should be 1
        target_coords = target_coords.squeeze(3) # (N, H, W, 2)

        return target_coords

    def get_relative_depth_map(self, zbuf, pad_value=10):
        absolute_depth = zbuf[..., 0] # B, H, W
        no_depth = -1

        depth_min, depth_max = absolute_depth[absolute_depth != no_depth].min(), absolute_depth[absolute_depth != no_depth].max()
        target_min, target_max = 50, 255

        depth_value = absolute_depth[absolute_depth != no_depth]
        depth_value = depth_max - depth_value # reverse values

        depth_value /= (depth_max - depth_min)
        depth_value = depth_value * (target_max - target_min) + target_min

        relative_depth = absolute_depth.clone()
        relative_depth[absolute_depth != no_depth] = depth_value
        relative_depth[absolute_depth == no_depth] = pad_value # not completely black

        return absolute_depth, relative_depth

    def query_texture(self, coords, texture, encode=True):
        assert "hashgrid" in self.config.texture_type

        if encode:
            B, H, W, C = coords.shape
            inputs = coords.reshape(-1, C)
            outputs = texture(inputs)
            outputs = outputs.reshape(B, H, W, -1)
        else:
            outputs = coords

        return outputs.to(torch.float32)
    
    def query_anchor_features(self, anchors, texture, features, instances_in_view, is_background=False):
        if is_background:
            anchor_features = features
        else:
            # with torch.no_grad():
            #     anchors = self.query_texture(anchors.unsqueeze(2), texture).squeeze(2) # M, NumAnchor, C
            #     if self.config.detach_anchors:
            #         anchors = anchors.detach() # the original UV features won't be updated

            anchors = self.query_texture(anchors.unsqueeze(2), texture).squeeze(2) # M, NumAnchor, C
            if self.config.detach_anchors:
                anchors = anchors.detach() # the original UV features won't be updated
            
            anchor_features = self.anchor_func(anchors, features, instances_in_view) # M, C

        return anchor_features

    def render_features(self, renderer, mesh, texture, is_direct=False, is_background=False, anchors=None):
        # if enable_anchor_embedding is True
        # latents will be the rendered instance map
        latents, fragments = renderer(mesh) # image: (N, H, W, C)

        if is_direct:
            features = latents
        else:
            uv_coords = self.get_uv_coordinates(mesh, fragments)
            features = self.query_texture(uv_coords, texture)

            if self.config.enable_anchor_embedding:
                features = self.query_anchor_features(anchors, texture, features, latents[..., 0], is_background)

        features = self.render_func(features)

        absolute_depth, relative_depth = self.get_relative_depth_map(fragments.zbuf)

        return features, fragments, absolute_depth, relative_depth # (N, H, W, C)
    
    def render(self, renderer, mesh, texture, background=None, background_texture=None, anchors=None, is_direct=False):
        features, fragments, absolute_depth, relative_depth = self.render_features(renderer, mesh, texture, is_direct=is_direct, is_background=False, anchors=anchors)

        # blend background
        # NOTE there's no need to render background if no views see the background
        if background is not None and -1 in fragments.zbuf:
            background_features, background_fragments, _, _ = self.render_features(renderer, background, background_texture, is_direct=is_direct, is_background=True, anchors=None)

            # blend rendering
            background_mask = fragments.zbuf == -1
            background_mask = background_mask.repeat(1, 1, 1, background_features.shape[-1])
            features[background_mask] = background_features[background_mask]

            # blend depth
            background_mask = fragments.zbuf == -1
            blend_zbuf = fragments.zbuf
            blend_zbuf[background_mask] = background_fragments.zbuf[background_mask]
            absolute_depth, relative_depth = self.get_relative_depth_map(blend_zbuf)

        return features, absolute_depth, relative_depth
