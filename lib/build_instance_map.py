import os
import sys
import torch
import torchvision

from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path

from pytorch3d.io import (
    load_obj,
    load_objs_as_meshes
)
from pytorch3d.renderer import TexturesUV
from pytorch3d.ops import interpolate_face_attributes

# # customized
# sys.path.append(".")
# from models.modules import Studio

TEXTURE = "./samples/textures/white.png"
WORK_DIR = "./cache/debug/"
NUM_VIEWS = 500
UV_SIZE = 4096
IMAGE_SIZE = 512

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()

# def init_studio(work_dir=WORK_DIR, device=DEVICE):
#     studio = Studio(OmegaConf.load(os.path.join(work_dir, "config.yaml")), "run", device)

#     return studio

def init_meshes(work_dir=WORK_DIR, texture=TEXTURE, device=DEVICE):
    meshes = {}

    assets_dir = Path(work_dir) / "meshes"
    for category in ["things", "stuff"]:
        for obj_path in (assets_dir / category).glob("*.obj"):
            verts, faces, aux = load_obj(str(obj_path), device=device)
            mesh = load_objs_as_meshes([str(obj_path)], device=device)

            texture_tensor = Image.open(texture)
            texture_tensor = torchvision.transforms.ToTensor()(texture_tensor).to(device)
            texture_tensor = texture_tensor.unsqueeze(0).permute(0, 2, 3, 1)
            mesh.textures = TexturesUV(
                maps=texture_tensor, # B, H, W, 3
                faces_uvs=faces.textures_idx[None, ...],
                verts_uvs=aux.verts_uvs[None, ...]
            )

            meshes[str(obj_path.stem)] = {
                "verts": verts,
                "faces": faces,
                "aux": aux,
                "mesh": mesh,
                "path": str(obj_path)
            }

    return meshes

def get_all_4_locations(values_y, values_x):
    y_0 = torch.floor(values_y)
    y_1 = torch.ceil(values_y)
    x_0 = torch.floor(values_x)
    x_1 = torch.ceil(values_x)

    return torch.cat([y_0, y_0, y_1, y_1], 0).long(), torch.cat([x_0, x_1, x_0, x_1], 0).long()

def get_coordinates(mesh, fragments):
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

    target_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_coords
    ) # (N, H, W, 1, 3)
    _, H, W, K, _ = target_coords.shape
    assert K == 1 # pixel_per_faces should be 1
    target_coords = target_coords.squeeze(3) # (N, H, W, 3)

    return target_coords

def build_instance_map(studio, 
    mesh_dir=WORK_DIR, cache_dir=WORK_DIR, texture=TEXTURE, 
    device=DEVICE, uv_size=UV_SIZE, image_size=IMAGE_SIZE, num_views=NUM_VIEWS, 
    threshold=512,
    show_progress=True):

    meshes = init_meshes(mesh_dir, texture, device)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "instances.pth"

    if cache_path.exists():
        print("=> loading instance map...")
        global_instance_map = torch.load(str(cache_path)).to(device)
    else:
        print("=> building instance map...")
        global_instance_map = torch.zeros(uv_size, uv_size).to(device)

        instance_id = 1 # indexed from 1
        for mesh_name, mesh_data in meshes.items():
            if show_progress: print("=> processing instance mask for {}".format(mesh_name))
            instance_map = torch.zeros(uv_size, uv_size).to(device)
            iterator = tqdm(range(num_views)) if show_progress else range(num_views)
            for view_id in iterator:
                Rs, Ts, fovs, ids = studio.sample_cameras(view_id, 1, True)
                cameras = studio.set_cameras(Rs, Ts, fovs, image_size)
                renderer = studio.set_renderer(cameras, image_size)

                _, fragments = renderer(mesh_data["mesh"])
                uvs = get_coordinates(mesh_data["mesh"], fragments).reshape(-1, 2)

                texture_locations_y, texture_locations_x = get_all_4_locations(
                    (1 - uvs[:, 1]).reshape(-1) * (uv_size - 1), 
                    uvs[:, 0].reshape(-1) * (uv_size - 1)
                )
                instance_map[texture_locations_y, texture_locations_x] = 1
                global_instance_map[texture_locations_y, texture_locations_x] = instance_id

            save_path = Path(mesh_data["path"])
            save_path = save_path.parent / "{}.png".format(str(save_path.stem))

            num_texels = int(instance_map.sum())
            if show_progress: print("=> processed {} texels for {}".format(num_texels, mesh_name))

            if num_texels < threshold:
                # neglect this instance
                if show_progress: print("=> neglect {}".format(mesh_name))
                global_instance_map[global_instance_map == instance_id] = 0
            else:
                instance_id += 1
                instance_map = torchvision.transforms.ToPILImage()(instance_map)
                instance_map.save(str(save_path))

        torch.save(global_instance_map, str(cache_path))

    return global_instance_map

# if __name__ == "__main__":
#     build_semantic_masks()
