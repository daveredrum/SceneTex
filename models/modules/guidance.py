import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import numpy as np

import torchvision

from PIL import Image
from diffusers import DDIMScheduler, ControlNetModel

# customized
import sys
sys.path.append("./models")
from models.utils.lora import extract_lora_diffusers

class Guidance(nn.Module):
    def __init__(self, 
        config,
        device
    ): 
        
        super().__init__()
        
        self.config = config
        self.device = device

        self.prompt = config.prompt + ", " + config.a_prompt if config.a_prompt else config.prompt
        self.n_prompt = config.n_prompt
        
        self.weights_dtype = torch.float16 if self.config.enable_half_precision else torch.float32

        self._init_guidance()

    def _init_guidance(self):
        self._init_backbone()
        self._init_t_schedule()

    def _init_backbone(self):
        if self.config.diffusion_type == "t2i":
            from diffusers import StableDiffusionPipeline as DiffusionPipeline
            checkpoint_name = "stabilityai/stable-diffusion-2-1-base"
            # diffusion_model = DiffusionPipeline.from_pretrained(checkpoint_name).to(self.device)
            # checkpoint_name = "runwayml/stable-diffusion-v1-5"
            diffusion_model = DiffusionPipeline.from_pretrained(checkpoint_name).to(self.device)
        elif self.config.diffusion_type == "d2i":
            from diffusers import StableDiffusionDepth2ImgPipeline as DiffusionPipeline
            checkpoint_name = "stabilityai/stable-diffusion-2-depth"
            diffusion_model = DiffusionPipeline.from_pretrained(checkpoint_name).to(self.device)
        elif self.config.diffusion_type == "d2i_controlnet":
            from diffusers import StableDiffusionControlNetPipeline as DiffusionPipeline
            controlnet_name = "lllyasviel/control_v11f1p_sd15_depth"
            controlnet = ControlNetModel.from_pretrained(controlnet_name)
            checkpoint_name = "runwayml/stable-diffusion-v1-5"
            diffusion_model = DiffusionPipeline.from_pretrained(checkpoint_name, controlnet=controlnet).to(self.device)

            # freeze controlnet
            self.controlnet = diffusion_model.controlnet.to(self.weights_dtype)
            self.controlnet.requires_grad_(False)
        else:
            raise ValueError("invalid diffusion type.")

        if self.config.enable_memory_efficient_attention:
            print("=> Enable memory efficient attention.")
            diffusion_model.enable_xformers_memory_efficient_attention()

        # pretrained diffusion model
        self.tokenizer = diffusion_model.tokenizer
        self.text_encoder = diffusion_model.text_encoder
        self.vae = diffusion_model.vae
        self.unet = diffusion_model.unet.to(self.weights_dtype)

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        # use DDIMScheduler by default
        self.scheduler = DDIMScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")
        self.scheduler.betas = self.scheduler.betas.to(self.device)
        self.scheduler.alphas = self.scheduler.alphas.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        self.num_train_timesteps = len(self.scheduler.betas)

        if self.config.generation_mode == "t2i":
            self.scheduler.set_timesteps(self.config.num_steps)
            raise NotImplementedError
        else:
            self.scheduler.set_timesteps(self.num_train_timesteps)

        # phi
        # unet_phi is the same instance as unet that has been modified in-place
        # unet_phi not grad -> only train unet_phi_layers
        if self.config.loss_type == "vsd":
            self.unet_phi, self.unet_phi_layers = extract_lora_diffusers(self.unet, self.device)

            # load pretrained lora
            if len(self.config.load_lora_weights) > 0 and os.path.exists(self.config.load_lora_weights):
                print("=> loading pretrained LoRA weights from: {}".format(self.config.load_lora_weights))
                self.unet_phi.load_attn_procs(self.config.load_lora_weights)

        # loss weights
        self.loss_weights = self._init_loss_weights(self.scheduler.betas)

        self.avg_loss_vsd = []
        self.avg_loss_phi = []
        self.avg_loss_rgb = []

        if self.config.loss_type == "l2": 
            self.label = torchvision.io.read_image(self.config.label_path).float().to(self.device) / 255.
            self.label = self.label * 2 - 1 # -1 to 1
            self.label = self.label.unsqueeze(0)

        max_memory_allocated = torch.cuda.max_memory_allocated()
        print(f"=> Maximum GPU memory allocated by PyTorch: {max_memory_allocated / 1024**3:.2f} GB")

    def _init_loss_weights(self, betas):    
        num_train_timesteps = len(betas)
        betas = torch.tensor(betas).to(torch.float32) if not torch.is_tensor(betas) else betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        sqrt_1m_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
            
        weights = []
        for i in range(num_train_timesteps):
            weights.append(sqrt_1m_alphas_cumprod[i]**2)
            
        return weights
    
    def _init_t_schedule(self, t_start=0.02, t_end=0.98):
        # Create a list of time steps from 0 to num_train_timesteps
        ts = list(range(self.num_train_timesteps))
        # set ts to U[0.02,0.98] as least
        t_start = int(t_start * self.num_train_timesteps)
        t_end = int(t_end * self.num_train_timesteps)
        ts = ts[t_start:t_end]

        # If the scheduling strategy is "random", choose args.num_steps random time steps without replacement
        if self.config.t_schedule == "random":
            chosen_ts = np.random.choice(ts, self.config.num_steps, replace=True)

        # If the scheduling strategy is "t_stages", the total number of time steps are divided into several stages.
        # In each stage, a decreasing portion of the total time steps is considered for selection.
        # For each stage, time steps are randomly selected with replacement from the respective portion.
        # The final list of chosen time steps is a concatenation of the time steps selected in all stages.
        # Note: The total number of time steps should be evenly divisible by the number of stages.
        elif "t_stages" in self.config.t_schedule:
            # Parse the number of stages from the scheduling strategy string
            num_stages = int(self.config.t_schedule[8:]) if len(self.config.t_schedule[8:]) > 0 else 2
            chosen_ts = []
            for i in range(num_stages):
                # Define the portion of ts to be considered in this stage
                portion = ts[:int((num_stages-i)*len(ts)//num_stages)]
                selected_ts = np.random.choice(portion, self.config.num_steps//num_stages, replace=True).tolist()
                chosen_ts += selected_ts
        
        elif "anneal" in self.config.t_schedule:
            print("=> time step annealing after {} steps".format(self.config.num_anneal_steps))

            ts_before_anneal = np.random.choice(ts, self.config.num_anneal_steps, replace=True).tolist()
            ts_after_anneal = np.random.choice(ts[:len(ts)//2], self.config.num_steps-self.config.num_anneal_steps, replace=True).tolist()
            chosen_ts = ts_before_anneal + ts_after_anneal
        
        else:
            raise ValueError(f"Unknown scheduling strategy: {self.config.t_schedule}")

        # Return the list of chosen time steps
        self.chosen_ts = chosen_ts

    def init_text_embeddings(self, batch_size):
        ### get text embedding
        text_input = self.tokenizer(
            [self.prompt], 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(self.device)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input)[0].repeat(batch_size, 1, 1)

        max_length = text_input.shape[-1]
        uncond_input = self.tokenizer(
            [self.n_prompt], 
            padding="max_length", 
            max_length=max_length, 
            return_tensors="pt"
        ).input_ids.to(self.device)

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input)[0].repeat(batch_size, 1, 1)

        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    def prepare_depth_map(self, depth_map):
        assert len(depth_map.shape) == 4
        if "controlnet" in self.config.diffusion_type:
            depth_map = depth_map.repeat(1, 3, 1, 1).float()
            depth_map = F.interpolate(depth_map, (self.config.render_size, self.config.render_size), mode="bilinear", align_corners=False)
        
            # expected range [0,1]
            depth_map /= 255.0
        else:
            # down-sample and normalize
            depth_map = F.interpolate(depth_map, (self.config.latent_size, self.config.latent_size), mode="bilinear", align_corners=False)

            # expected range [-1,1]
            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
            # depth_map /= 255.0
            # depth_map = 2.0 * depth_map - 1.0

        depth_map = depth_map.to(torch.float32)

        return depth_map
    
    @torch.no_grad()
    def decode_latent_texture(self, inputs, use_patches=False):
        outputs = 1 / self.vae.config.scaling_factor * inputs

        if use_patches:
            assert self.config.latent_texture_size % self.config.decode_texture_size == 0
            batch_size = inputs.shape[0]
            num_iter_x = self.config.latent_texture_size // self.config.decode_texture_size
            num_iter_y = self.config.latent_texture_size // self.config.decode_texture_size
            patch_stride = self.config.decode_texture_size
            decoded_stride = self.config.decode_texture_size * 8
            decoded_size = self.config.latent_texture_size * 8
            decoded_texture = torch.zeros(batch_size, 3, decoded_size, decoded_size).to(self.device)

            for x in range(num_iter_x):
                for y in range(num_iter_y):
                    patch = outputs[:, :, x*patch_stride:(x+1)*patch_stride, y*patch_stride:(y+1)*patch_stride]
                    patch = self.vae.decode(patch.contiguous()).sample # B, 3, H, W

                    decoded_texture[:, :, x*decoded_stride:(x+1)*decoded_stride, y*decoded_stride:(y+1)*decoded_stride] = patch
        
            outputs = (decoded_texture / 2 + 0.5).clamp(0, 1)

        else:
            outputs = self.vae.decode(outputs.contiguous()).sample # B, 3, H, W
            outputs = (outputs / 2 + 0.5).clamp(0, 1)

        return outputs
    
    def encode_latent_texture(self, inputs, deterministic=False):
        inputs = inputs.clamp(-1, 1)
        
        h = self.vae.encoder(inputs)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        std = torch.zeros_like(mean) if deterministic else torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn_like(mean)
        
        return self.vae.config.scaling_factor * sample

    def normalize_latent_texture(self, inputs):
        outputs = (inputs / 2 + 0.5).clamp(0, 1)

        return outputs
    
    def prepare_one_latent(self, latents, t):
        noise = torch.randn_like(latents).to(self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
        clean_latents = self.scheduler.step(noise, t, noisy_latents).pred_original_sample

        return noise, noisy_latents, clean_latents

    def prepare_latents(self, latents, t, batch_size):
        t = torch.tensor([t]).to(self.device)
        noise, noisy_latents, clean_latents = self.prepare_one_latent(latents, t)

        return t, noise, noisy_latents, clean_latents
    
    def predict_noise(self, unet, noisy_latents, t, cross_attention_kwargs, guidance_scale, control=None):
        down_block_res_samples, mid_block_res_sample = None, None

        if guidance_scale == 1:
            latent_model_input = noisy_latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            text_embeddings, _ = self.text_embeddings.chunk(2)

            if control is not None: 
                if "controlnet" in self.config.diffusion_type:
                    with torch.no_grad():
                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            latent_model_input.to(self.weights_dtype),
                            t,
                            encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                            controlnet_cond=control.to(self.weights_dtype),
                            conditioning_scale=1.0,
                            guess_mode=False,
                            return_dict=False,
                        )

                        down_block_res_samples = [e.to(self.weights_dtype) for e in down_block_res_samples]
                        mid_block_res_sample = mid_block_res_sample.to(self.weights_dtype)
                else:
                    latent_model_input = torch.cat([latent_model_input, control], dim=1)

            # if self.config.verbose_mode: start = time.time()
            noise_pred = unet(
                latent_model_input.to(self.weights_dtype), 
                t, 
                encoder_hidden_states=text_embeddings.to(self.weights_dtype), 
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            ).sample.to(torch.float32)
            # if self.config.verbose_mode: print("=> UNet forward: {}s".format(time.time() - start))
        else:
            latent_model_input = torch.cat([noisy_latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            if control is not None: 
                if "controlnet" in self.config.diffusion_type:
                    with torch.no_grad():
                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            latent_model_input.to(self.weights_dtype),
                            t,
                            encoder_hidden_states=self.text_embeddings.to(self.weights_dtype),
                            controlnet_cond=torch.cat([control]*2).to(self.weights_dtype),
                            conditioning_scale=1.0,
                            guess_mode=False,
                            return_dict=False,
                        )

                        down_block_res_samples = [e.to(self.weights_dtype) for e in down_block_res_samples]
                        mid_block_res_sample = mid_block_res_sample.to(self.weights_dtype)
                else:
                    latent_model_input = torch.cat([latent_model_input, torch.cat([control]*2)], dim=1)

            # if self.config.verbose_mode: start = time.time()
            noise_pred = unet(
                latent_model_input.to(self.weights_dtype), 
                t, 
                encoder_hidden_states=self.text_embeddings.to(self.weights_dtype), 
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            ).sample.to(torch.float32)
            # if self.config.verbose_mode: print("=> UNet forward: {}s".format(time.time() - start))

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

    def compute_sds_loss(self, latents, noisy_latents, noise, t, control=None):
        with torch.no_grad():
            noise_pred = self.predict_noise(
                self.unet, 
                noisy_latents, 
                t, 
                cross_attention_kwargs={},
                guidance_scale=self.config.guidance_scale,
                control=control
            )

        grad = self.config.grad_scale * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        grad *= self.loss_weights[int(t)]
        
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="mean")

        return loss
    
    def compute_vsd_loss(self, latents, noisy_latents, noise, t, cross_attention_kwargs, control=None):    
        with torch.no_grad():
            # predict the noise residual with unet
            # set cross_attention_kwargs={"scale": 0} to use the pre-trained model
            if self.config.verbose_mode: start = time.time()
            noise_pred = self.predict_noise(
                self.unet, 
                noisy_latents, 
                t, 
                cross_attention_kwargs={"scale": 0},
                guidance_scale=self.config.guidance_scale,
                control=control
            )
            if self.config.verbose_mode: print("=> VSD pretrained forward: {}s".format(time.time() - start))

            if self.config.verbose_mode: start = time.time()
            noise_pred_phi = self.predict_noise(
                self.unet_phi, 
                noisy_latents, 
                t, 
                cross_attention_kwargs=cross_attention_kwargs,
                guidance_scale=self.config.guidance_scale_phi,
                control=control
            )
            if self.config.verbose_mode: print("=> VSD lora forward: {}s".format(time.time() - start))

        grad = self.config.grad_scale * (noise_pred - noise_pred_phi.detach())
        grad = torch.nan_to_num(grad)

        grad *= self.loss_weights[int(t)]
        
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="none")

        return loss, loss.mean()
    
    def compute_vsd_phi_loss(self, noisy_latents, clean_latents, noise, t, cross_attention_kwargs, control=None):
        if self.config.verbose_mode: start = time.time()
        noise_pred_phi = self.predict_noise(
            self.unet_phi, 
            noisy_latents, 
            t, 
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_scale=self.config.guidance_scale_phi,
            control=control
        )

        if self.config.verbose_mode: print("=> phi lora forward: {}s".format(time.time() - start))

        target = noise

        loss = self.config.grad_scale * F.mse_loss(noise_pred_phi, target, reduction="none")

        return loss, loss.mean()
