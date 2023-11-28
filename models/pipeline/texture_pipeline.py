import random
import wandb
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np

import pytorch_lightning as pl

# mat
import matplotlib.pyplot as plt

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR
from omegaconf import OmegaConf

from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image
from copy import deepcopy
from pathlib import Path

from pytorch3d.io import (
    load_obj,
    load_objs_as_meshes
)
from pytorch3d.renderer import TexturesUV
from pytorch3d.ops import interpolate_face_attributes

# customized
import sys
sys.path.append("./lib")
from models.modules import TextureMesh, Studio, Guidance

class TexturePipeline(nn.Module):
    def __init__(self, 
        config,
        stamp,
        device
    ): 
        
        super().__init__()

        self.config = config
        self.stamp = stamp

        self.prompt = config.prompt + ", " + config.a_prompt if config.a_prompt else config.prompt
        self.n_prompt = config.n_prompt
        
        self.device = device
        self.weights_dtype = torch.float16 if self.config.enable_half_precision else torch.float32
        print("=> Use precision: {}".format(self.weights_dtype))

        pl.seed_everything(self.config.seed)


    """call this after to(device)"""
    def configure(self, inference_mode=False):
        if not inference_mode:
            self.log_name = "_".join(self.config.prompt.split(' '))
            self.log_stamp = self.stamp
            self.log_dir = os.path.join(self.config.log_dir, self.log_name, self.config.loss_type, self.log_stamp)

            # override config
            self.config.log_name = self.log_name
            self.config.log_stamp = self.log_stamp
            self.config.log_dir = self.log_dir

        # 3D assets
        self._init_mesh()

        # studio
        self._init_studio()

        # instances
        self._init_anchors()

        if not inference_mode:
            # diffusion
            self._init_guidance()

            # optimization
            self._configure_optimizers()
            self._init_logger()

        if self.config.enable_clip_benchmark:
            import open_clip
            self.clip, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

    def _init_studio(self):
        self.studio = Studio(self.config, self.device)

    def _init_mesh(self):
        self.texture_mesh = TextureMesh(self.config, self.device)

    def _init_guidance(self):
        self.guidance = Guidance(self.config, self.device)

    def _init_anchors(self):
        if self.config.enable_anchor_embedding:
            self.texture_mesh.build_instance_map(self.studio, self.config.log_dir)
            self.texture_mesh.sample_instance_anchors(self.config.log_dir)
            self.studio.init_anchor_func(self.texture_mesh.num_instances)

    def _init_logger(self):
        os.makedirs(self.log_dir, exist_ok=True)

        model_type = "controlnet" if "controlnet" in self.config.diffusion_type else "SD"

        wandb.login()
        wandb.init(
            project="SceneTex",
            name=self.log_name+"_"+self.log_stamp+"_"+model_type,
            dir=self.log_dir
        )

        with open(os.path.join(self.log_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config=self.config, f=f)

        self.avg_loss_vsd = []
        self.avg_loss_phi = []

    def _get_texture_parameters(self):
        if "hashgrid" not in self.config.texture_type:
            texture_params = [self.texture_mesh.texture]

            if self.config.use_background: 
                texture_params += [self.texture_mesh.background_texture]

        else:
            texture_params = [p for p in self.texture_mesh.texture.parameters() if p.requires_grad]

            if self.config.use_background: 
                texture_params += [p for p in self.texture_mesh.background_texture.parameters() if p.requires_grad]

        # render function
        texture_params += [p for p in self.studio.render_func.parameters() if p.requires_grad]
        
        # anchor function
        if self.config.enable_anchor_embedding: 
            texture_params += [p for p in self.studio.anchor_func.parameters() if p.requires_grad]

        return texture_params

    def _get_guidance_parameters(self):
        return [p for p in self.guidance.unet_phi_layers.parameters() if p.requires_grad]

    def _configure_optimizers(self):
        texture_params = self._get_texture_parameters()

        print("=> Total number of trainable parameters for texture: {}".format(
            sum(p.numel() for p in texture_params if p.requires_grad)))

        self.texture_optimizer = AdamW(texture_params, lr=self.config.latent_lr)

        if self.config.loss_type == "vsd":
            guidance_params = self._get_guidance_parameters()

            print("=> Number of trainable parameters of phi model: {}".format(
                sum(p.numel() for p in guidance_params if p.requires_grad)))

            self.phi_optimizer = AdamW(guidance_params, lr=self.config.phi_lr)

    def _downsample(self, inputs, in_size, out_size, mode="direct", type_="interpolate"):
        if mode == "iterative":
            assert type_ != "encoder"
            down_size_list = []
            down_size = in_size
            num_max_down = in_size // out_size
            for _ in range(num_max_down):
                down_size = down_size // 2
                if down_size <= out_size:
                    down_size = out_size

                down_size_list.append(down_size)

                if down_size == out_size: break

            outputs = inputs
            for down_size in down_size_list:
                if type_ == "interpolate":
                    outputs = F.interpolate(outputs, size=(down_size, down_size), mode="bilinear")
                elif type_ == "avg_pool":
                    outputs = F.adaptive_avg_pool2d(outputs, (down_size, down_size))
                else:
                    raise ValueError("invalid downsampling type.")

        elif mode == "direct":
                if type_ == "interpolate":
                    outputs = F.interpolate(inputs, size=(out_size, out_size), mode="bilinear")
                elif type_ == "avg_pool":
                    outputs = F.adaptive_avg_pool2d(inputs, (out_size, out_size))
                elif type_ == "encoder":
                    # outputs = self.latent_encoder(inputs)
                    raise NotImplementedError
                else:
                    raise ValueError("invalid downsampling type.")
        else:
            raise ValueError("invalid downsampling mode.")

        return outputs

    def _rgb_to_rgba(self, input_tensor):
        B, C, H, W = input_tensor.shape
        assert C == 3
        alpha = torch.ones(B, 1, H, W).to(self.device)
        output_tensor = torch.cat([input_tensor, alpha], dim=1)

        return output_tensor
    
    def _rgba_to_rgb(self, input_tensor):
        B, C, H, W = input_tensor.shape
        assert C == 4
        output_tensor = input_tensor[:, :3]

        return output_tensor
    
    def load_checkpoint(self, checkpoint_dir, checkpoint_step):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(checkpoint_step))
        checkpoint = torch.load(checkpoint_path)

        # load weights
        self.texture_mesh.texture.load_state_dict(checkpoint["texture"])
        self.studio.render_func.load_state_dict(checkpoint["render_func"])

        if self.config.enable_anchor_embedding: 
            self.studio.anchor_func.state_dict(checkpoint["anchor_func"])

    @torch.no_grad()
    def inference(self, checkpoint_dir, checkpoint_step, texture_size):
        u, v = torch.arange(texture_size).to(self.device), torch.arange(texture_size).to(self.device)
        u, v = torch.meshgrid(u, v, indexing='ij')
        inputs = torch.stack([u, v]).permute(1, 2, 0).unsqueeze(0) / (texture_size - 1)

        texture = torch.zeros(texture_size, texture_size, 3)
        
        if self.config.enable_anchor_embedding: 
            instance_map = self.texture_mesh.instance_map[None, None, :, :]
            instance_map = F.interpolate(instance_map, (texture_size, texture_size), mode="nearest")
            instance_map = instance_map.permute(0, 1, 3, 2)
            instance_map = torch.flip(instance_map, dims=[3])

        for i in tqdm(range(texture_size)):
            r_inputs = inputs[:, i].float()
            r_inputs = self.studio.query_texture(r_inputs.unsqueeze(1), self.texture_mesh.texture)
            
            if self.config.enable_anchor_embedding: 
                r_inputs = self.studio.query_anchor_features(
                    self.texture_mesh.instance_anchors, 
                    self.texture_mesh.texture, 
                    r_inputs, 
                    instance_map[:, :, i]
                )

            r = self.studio.render_func(r_inputs)[0, 0]
            texture[i] = r.detach().cpu()
        
        texture = (texture / 2 + 0.5).clamp(0, 1)
        texture = texture.cpu()
        texture = texture.permute(1, 0, 2)
        texture = torch.flip(texture, dims=[0])

        assert texture.min() >= 0 and texture.max() <= 1

        texture = torchvision.transforms.ToPILImage()(texture.permute(2, 0, 1)).convert("RGB")
        texture.save(os.path.join(checkpoint_dir, "texture_{}.png".format(checkpoint_step)))

    def _prepare_mesh(self, inference):
        mesh = self.texture_mesh.mesh
        background_mesh = self.texture_mesh.background_mesh if self.config.use_background and not inference else None

        if self.config.batch_size > 1 and not inference:
            mesh = self.texture_mesh.repeat_meshes_as_batch(self.texture_mesh.mesh, self.config.batch_size)
            if background_mesh is not None: 
                background_mesh = self.texture_mesh.repeat_meshes_as_batch(self.texture_mesh.background_mesh, self.config.batch_size)
        
        # textures
        texture = self.texture_mesh.texture
        background_texture = self.texture_mesh.background_texture if self.config.use_background and not inference else None

        return mesh, texture, background_mesh, background_texture

    def forward(self, camera, inference=False, downsample=True, normalize_depth=True, is_direct=False):
        renderer = self.studio.set_renderer(camera, self.config.render_size)

        mesh, texture, background_mesh, background_texture = self._prepare_mesh(inference)

        anchors = self.texture_mesh.instance_anchors if self.config.enable_anchor_embedding else None

        # for VSD -> 512x512
        # this is to get more texels involved
        latents, abs_depth, rel_depth = self.studio.render(renderer, mesh, texture, background_mesh, background_texture, anchors, is_direct)
        latents = latents.permute(0, 3, 1, 2)

        if downsample:
            if self.config.downsample == "vae":
                latents = self.guidance.encode_latent_texture(latents, self.config.deterministic_vae_encoding)
            elif self.config.downsample == "interpolate":
                latents = F.interpolate(latents, (self.config.latent_size, self.config.latent_size), mode="bilinear", align_corners=False)
            else:
                raise ValueError("invalid downsampling mode.")

        rel_depth_normalized = rel_depth.unsqueeze(1)
        if normalize_depth:
            rel_depth_normalized = self.guidance.prepare_depth_map(rel_depth_normalized) # -1 - 1

        return latents, abs_depth, rel_depth, rel_depth_normalized

    @torch.no_grad()
    def _benchmark_step(self, image, text):
        if self.config.enable_clip_benchmark:
            image = self.clip_preprocess(image).unsqueeze(0)
            text = self.clip_tokenizer([text])

            image_features = self.clip.encode_image(image)
            text_features = self.clip.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            return F.cosine_similarity(image_features, text_features).item()
        else:
            return 0

    def fit(self):
        pbar = tqdm(self.guidance.chosen_ts)

        self.guidance.init_text_embeddings(self.config.batch_size)

        for step, chosen_t in enumerate(pbar):

            Rs, Ts, fovs, ids = self.studio.sample_cameras(step, self.config.batch_size, self.config.use_random_cameras)
            cameras = self.studio.set_cameras(Rs, Ts, fovs, self.config.render_size)
            latents, _, _, rel_depth_normalized = self.forward(cameras, is_direct=("hashgrid" not in self.config.texture_type))
            t, noise, noisy_latents, _ = self.guidance.prepare_latents(latents, chosen_t, self.config.batch_size)

            # compute loss
            if self.config.loss_type == "sds":
                
                self.texture_optimizer.zero_grad()

                sds_loss = self.guidance.compute_sds_loss(
                    latents, noisy_latents, noise, t.to(latents.dtype), 
                    control=rel_depth_normalized if "d2i" in self.config.diffusion_type else None
                )

                sds_loss.backward()
                self.texture_optimizer.step()

                vsd_loss = sds_loss
                vsd_phi_loss = torch.zeros_like(vsd_loss)

            elif self.config.loss_type == "vsd":

                # VSD
                self.texture_optimizer.zero_grad()

                vsd_loss_pixel, vsd_loss = self.guidance.compute_vsd_loss(
                    latents, noisy_latents, noise, t.to(latents.dtype), 
                    cross_attention_kwargs={'scale': self.config.phi_scale},
                    control=rel_depth_normalized if "d2i" in self.config.diffusion_type else None
                )

                vsd_loss.backward()
                self.texture_optimizer.step()


                # phi
                torch.cuda.empty_cache()

                self.phi_optimizer.zero_grad()

                if self.config.use_different_t:
                    t_phi = random.choice(range(self.guidance.num_train_timesteps))
                else:
                    t_phi = chosen_t

                t_phi, noise_phi, noisy_latents_phi, clean_latents_phi = self.guidance.prepare_latents(latents, t_phi, self.config.batch_size)
                
                vsd_phi_loss_pixel, vsd_phi_loss = self.guidance.compute_vsd_phi_loss(
                    noisy_latents_phi.detach(), clean_latents_phi, noise_phi, t_phi.to(latents.dtype), 
                    cross_attention_kwargs={'scale': self.config.phi_scale},
                    control=rel_depth_normalized if "d2i" in self.config.diffusion_type else None
                )

                vsd_phi_loss.backward()
                self.phi_optimizer.step()

            elif self.config.loss_type == "l2": # only for debugging

                raise NotImplementedError

            else:
                raise ValueError("invalid loss type")
            
            wandb.log({
                "train/vsd_loss": vsd_loss.item(),
                "train/vsd_lora_loss": vsd_phi_loss.item()
            })
            
            self.avg_loss_vsd.append(vsd_loss.item())
            self.avg_loss_phi.append(vsd_phi_loss.item())
            
            max_memory_allocated = torch.cuda.max_memory_allocated()
            pbar.set_description(f'Loss: {vsd_loss.item():.6f}, sampled t : {t.item()}, GPU: {max_memory_allocated / 1024**3:.2f} GB')
            
            if step % self.config.log_steps == 0:

                # save texture field
                checkpoint = {
                    "render_func": self.studio.render_func.state_dict()
                }

                if "hashgrid" in self.config.texture_type:
                    checkpoint["texture"] = self.texture_mesh.texture.state_dict()
                else:
                    checkpoint["texture"] = self.texture_mesh.texture

                if self.config.enable_anchor_embedding: 
                    checkpoint["anchor_func"] = self.studio.anchor_func.state_dict()

                torch.save(
                    checkpoint,
                    os.path.join(self.log_dir, "checkpoint_{}.pth".format(step))
                )

                # visualize
                wandb_images = []
                wandb_images_depths = []
                clip_scores = []

                if self.config.show_original_texture:
                    if self.config.texture_type == "latent":
                        decoded_texture = self.guidance.decode_latent_texture(self.texture_mesh.texture.permute(0, 3, 1, 2))
                        decoded_texture = torchvision.transforms.ToPILImage()(decoded_texture[0]).convert("RGB")
                        decoded_texture.save(os.path.join(self.config.log_dir, "texture_{}.png".format(step)))
                    elif self.config.texture_type == "rgb":
                        decoded_texture = (self.texture_mesh.texture / 2 + 0.5).clamp(0, 1)
                        decoded_texture = torchvision.transforms.ToPILImage()(decoded_texture[0].permute(2, 0, 1)).convert("RGB")
                        decoded_texture.save(os.path.join(self.config.log_dir, "texture_{}.png".format(step)))
                    else:
                        self.inference(self.config.log_dir, step, self.config.texture_size)

                if self.config.show_decoded_latents:
                    wandb_renderings, wandb_depths = [], []
                    for view_id in range(self.config.log_latents_views):
                        Rs, Ts, fovs, _ = self.studio.sample_cameras(view_id, 1, inference=True)
                        cameras = self.studio.set_cameras(Rs, Ts, fovs, self.config.render_size)

                        if self.config.texture_type == "latent":
                            with torch.no_grad():
                                latents, _, rel_depth, _ = self.forward(cameras, False, True, is_direct=("hashgrid" not in self.config.texture_type))
                                latents = self.guidance.decode_latent_texture(latents)
                        else:
                            with torch.no_grad():
                                latents, _, rel_depth, _ = self.forward(cameras, False, False, is_direct=("hashgrid" not in self.config.texture_type))
                                latents = (latents / 2 + 0.5).clamp(0, 1)
                        
                        latents_image = torchvision.transforms.ToPILImage()(latents[0]).convert("RGB").resize((self.config.decode_size, self.config.decode_size))

                        clip_score = self._benchmark_step(latents_image, self.config.prompt)
                        clip_scores.append(clip_score)

                        wandb_renderings.append(wandb.Image(latents_image))

                        # depth
                        depth_image = Image.fromarray(rel_depth[0].cpu().numpy().astype(np.uint8)).convert("L").resize((self.config.decode_size, self.config.decode_size))
                        wandb_depths.append(wandb.Image(depth_image))

                    wandb_images += wandb_renderings
                    wandb_images_depths += wandb_depths

                wandb.log({
                    "images": wandb_images,
                    "depths": wandb_images_depths,
                    "train/avg_loss": np.mean(self.avg_loss_vsd),
                    "train/avg_loss_lora": np.mean(self.avg_loss_phi),
                    "train/clip_score": np.mean(clip_scores)
                })
        

