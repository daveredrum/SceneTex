from typing import NamedTuple, Sequence

from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer import (
    AmbientLights,
    SoftPhongShader
)
from pytorch3d.ops import interpolate_face_attributes

import torch
import torch.nn.functional as F

class BlendParams(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Sequence = (1, 1, 1)


class FlatTexelShader(ShaderBase):

    def __init__(self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__(device, cameras, lights, materials, blend_params)

    # override to enable half precision
    def _sample_textures(self, texture_maps, fragments, faces_verts_uvs):
        """
        Interpolate a 2D texture map using uv vertex texture coordinates for each
        face in the mesh. First interpolate the vertex uvs using barycentric coordinates
        for each pixel in the rasterized output. Then interpolate the texture map
        using the uv coordinate for each pixel.

        Args:
            fragments:
                The outputs of rasterization. From this we use

                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
                the barycentric coordinates of each pixel
                relative to the faces (in the packed
                representation) which overlap the pixel.

        Returns:
            texels: tensor of shape (N, H, W, K, C) giving the interpolated
            texture for each pixel in the rasterized image.
        """


        # pixel_uvs: (N, H, W, K, 2)
        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
        )

        N, H_out, W_out, K = fragments.pix_to_face.shape
        N, H_in, W_in, C = texture_maps.shape  # 3 for RGB

        # pixel_uvs: (N, H, W, K, 2) -> (N, K, H, W, 2) -> (NK, H, W, 2)
        pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 2)

        # textures.map:
        #   (N, H, W, C) -> (N, C, H, W) -> (1, N, C, H, W)
        #   -> expand (K, N, C, H, W) -> reshape (N*K, C, H, W)
        texture_maps = (
            texture_maps.permute(0, 3, 1, 2)[None, ...]
            .expand(K, -1, -1, -1, -1)
            .transpose(0, 1)
            .reshape(N * K, C, H_in, W_in)
        )

        # Textures: (N*K, C, H, W), pixel_uvs: (N*K, H, W, 2)
        # Now need to format the pixel uvs and the texture map correctly!
        # From pytorch docs, grid_sample takes `grid` and `input`:
        #   grid specifies the sampling pixel locations normalized by
        #   the input spatial dimensions It should have most
        #   values in the range of [-1, 1]. Values x = -1, y = -1
        #   is the left-top pixel of input, and values x = 1, y = 1 is the
        #   right-bottom pixel of input.

        pixel_uvs = pixel_uvs * 2.0 - 1.0

        texture_maps = torch.flip(texture_maps, [2])  # flip y axis of the texture map
        if texture_maps.device != pixel_uvs.device:
            texture_maps = texture_maps.to(pixel_uvs.device)

        pixel_uvs = pixel_uvs.to(texture_maps.dtype)

        texels = F.grid_sample(
            texture_maps,
            pixel_uvs
        )
        # texels now has shape (NK, C, H_out, W_out)
        texels = texels.reshape(N, K, C, H_out, W_out).permute(0, 3, 4, 1, 2)

        return texels

    def forward(self, fragments, meshes, **_kwargs):
        texels = meshes.sample_textures(fragments)
        texels[(fragments.pix_to_face == -1), :] = 0
        texels = texels.squeeze(-2) 

        # blend background
        B, H, W, C = texels.shape
        if C == 3:
            background_color = torch.FloatTensor(self.blend_params.background_color).to(texels.device)
            background_mask = fragments.zbuf == -1
            background_mask = background_mask.repeat(1, 1, 1, 3)
            background_color = background_color.reshape(1, 1, 1, 3).repeat(B, H, W, 1)
            texels[background_mask] = background_color[background_mask]

        return texels


def init_soft_phong_shader(camera, device, blend_params=BlendParams()):
    lights = AmbientLights(device=device)
    shader = SoftPhongShader(
        cameras=camera,
        lights=lights,
        device=device,
        blend_params=blend_params
    )

    return shader


def init_flat_texel_shader(camera, device, blend_params=BlendParams()):
    shader=FlatTexelShader(
        cameras=camera,
        device=device,
        blend_params=blend_params
    )
    
    return shader