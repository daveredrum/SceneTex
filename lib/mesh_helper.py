import os
import torch
import trimesh
import xatlas
import json
import subprocess

import numpy as np

from pathlib import Path
from sklearn.decomposition import PCA
from PIL import Image

from torchvision import transforms

from tqdm import tqdm

from pytorch3d.structures import (
    Meshes,
    join_meshes_as_scene
)
from pytorch3d.io import (
    load_obj,
    load_objs_as_meshes
)
from pytorch3d.renderer import TexturesAtlas

import sys
sys.path.append(".")

from lib.colors import PALETTE


def compute_principle_directions(model_path, num_points=20000):
    mesh = trimesh.load_mesh(model_path, force="mesh")
    pc, _ = trimesh.sample.sample_surface_even(mesh, num_points)

    pc -= np.mean(pc, axis=0, keepdims=True)

    principle_directions = PCA(n_components=3).fit(pc).components_
    
    return principle_directions


def rotate_mesh(mesh, verts, axis, theta, device):
    if axis == "x":
        R = torch.tensor([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]).float().to(device)
    elif axis == "y":
        R = torch.tensor([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ]).float().to(device)
    else:
        R = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ]).float().to(device)

    new_verts = R @ verts.T
    offsets = new_verts.T - verts
    new_mesh = mesh.offset_verts(offsets)

    return new_mesh


def apply_offsets_to_mesh(mesh, offsets):
    new_mesh = mesh.offset_verts(offsets)

    return new_mesh

def apply_scale_to_mesh(mesh, scale):
    new_mesh = mesh.scale_verts(scale)

    return new_mesh


def adjust_uv_map(faces, aux, init_texture, uv_size):
    """
        adjust UV map to be compatiable with multiple textures.
        UVs for different materials will be decomposed and placed horizontally

        +-----+-----+-----+--
        |  1  |  2  |  3  |
        +-----+-----+-----+--

    """

    textures_ids = faces.textures_idx
    materials_idx = faces.materials_idx
    verts_uvs = aux.verts_uvs

    num_materials = torch.unique(materials_idx).shape[0]

    new_verts_uvs = verts_uvs.clone()
    for material_id in range(num_materials):
        # apply offsets to horizontal axis
        faces_ids = textures_ids[materials_idx == material_id].unique()
        new_verts_uvs[faces_ids, 0] += material_id

    new_verts_uvs[:, 0] /= num_materials

    init_texture_tensor = transforms.ToTensor()(init_texture)
    init_texture_tensor = torch.cat([init_texture_tensor for _ in range(num_materials)], dim=-1)
    init_texture = transforms.ToPILImage()(init_texture_tensor).resize((uv_size, uv_size))

    return new_verts_uvs, init_texture

def adjust_uv_map_blender(faces, aux):
    """
        adjust UV map to be compatiable with multiple textures.
        UVs for different materials will be decomposed and placed horizontally

        +-----+-----+-----+--
        |  1  |  2  |  3  |
        +-----+-----+-----+--

    """

    textures_ids = faces.textures_idx
    materials_idx = faces.materials_idx
    verts_uvs = aux.verts_uvs

    num_materials = torch.unique(materials_idx).shape[0]

    new_verts_uvs = verts_uvs.clone()
    for material_id in range(num_materials):
        # apply offsets to horizontal axis
        faces_ids = textures_ids[materials_idx == material_id].unique()
        new_verts_uvs[faces_ids, 0] += material_id

    new_verts_uvs[:, 0] /= num_materials

    return new_verts_uvs


@torch.no_grad()
def update_face_angles(mesh, cameras, fragments):
    def get_angle(x, y):
        x = torch.nn.functional.normalize(x)
        y = torch.nn.functional.normalize(y)
        inner_product = (x * y).sum(dim=1)
        x_norm = x.pow(2).sum(dim=1).pow(0.5)
        y_norm = y.pow(2).sum(dim=1).pow(0.5)
        cos = inner_product / (x_norm * y_norm)
        angle = torch.acos(cos)
        angle = angle * 180 / 3.14159

        return angle

    # face normals
    face_normals = mesh.faces_normals_padded()[0]

    # view vector (object center -> camera center)
    camera_center = cameras.get_camera_center()

    face_angles = get_angle(
        face_normals, 
        camera_center.repeat(face_normals.shape[0], 1)
    ) # (F)

    face_angles_rev = get_angle(
        face_normals, 
        -camera_center.repeat(face_normals.shape[0], 1)
    ) # (F)

    face_angles = torch.minimum(face_angles, face_angles_rev)

    # Indices of unique visible faces
    visible_map = fragments.pix_to_face.unique()  # (num_visible_faces)
    invisible_mask = torch.ones_like(face_angles)
    invisible_mask[visible_map] = 0
    face_angles[invisible_mask == 1] = 10000.  # angles of invisible faces are ignored

    return face_angles


def create_face_semantics(faces, semantic_list, offset_list, device):
    """
        create semantic map for mesh faces
        faces should be a tensor in shape (F, 3)
        semantic_list contains all semantics 0-N for each vertex
        offset_list contains vertices offsets for each instance

        e.g. semantic_list = [0, 1, 2]
               offset_list = [3, 2, 1]
                           |---> vertices_semantics = [0, 0, 0, 1, 1, 2]
                                 note that vertices_semantics should be in shape (V, 1)

        return: 
            vertices_semantics in shape (V, 1)
            faces_semantics in shape (F, 3, 1)
    """
    
    assert len(semantic_list) == len(offset_list)
    vertices_semantics = []
    for semantic, offset in zip(semantic_list, offset_list):
        for _ in range(offset):
            vertices_semantics.append(semantic)

    vertices_semantics = torch.tensor(vertices_semantics).to(device).unsqueeze(1).float()
    faces_semantics = vertices_semantics[faces]

    return vertices_semantics, faces_semantics

def propagate_texture_by_example(texture_image, prior_mask, exist_mask, instance_mask, 
    propagate_dir, view_idx, device, mode="random"):
    texture_tensor = transforms.ToTensor()(texture_image).to(device)
    _, H, W = texture_tensor.shape
    texture_tensor_flatten = texture_tensor.reshape(3, -1).permute(1, 0) # H x W, 3
    propagated_texture = None

    if mode == "ebsynth":
        pallete = torch.FloatTensor(PALETTE).to(device) # 41, 3
        assert torch.unique(instance_mask).shape[0] <= pallete.shape[0], \
            "number of instance is beyond capacity."

        source_image = texture_image

        source_guide_tensor = instance_mask * exist_mask # H, W
        source_guide_tensor = pallete[source_guide_tensor.long()] # H, W, 3

        # target_guide_tensor = instance_mask * (1 - exist_mask) # H, W
        target_guide_tensor = instance_mask # H, W
        target_guide_tensor = pallete[target_guide_tensor.long()] # H, W, 3

        source_guide = Image.fromarray(source_guide_tensor.cpu().numpy().astype(np.uint8))
        target_guide = Image.fromarray(target_guide_tensor.cpu().numpy().astype(np.uint8))

        source_path = os.path.join(propagate_dir, "source_{}.png".format(view_idx))
        target_path = os.path.join(propagate_dir, "target_{}.png".format(view_idx))
        source_guide_path = os.path.join(propagate_dir, "source_guide_{}.png".format(view_idx))
        target_guide_path = os.path.join(propagate_dir, "target_guide_{}.png".format(view_idx))

        source_image.save(source_path)
        source_guide.save(source_guide_path)
        target_guide.save(target_guide_path)

        # synthesis via ebsynth
        cmd = [
            "ebsynth", 
            "-patchsize", "3", 
            # "-uniformity", "1000", 
            "-style", source_path, 
            "-guide", source_guide_path, target_guide_path,
            "-output", target_path
        ]
        _ = subprocess.call(cmd)

        # output
        prior_mask[(1 - exist_mask) == 1] = 1
        target_image = Image.open(target_path)

        # combine
        target_image_tensor = transforms.ToTensor()(target_image).to(device).permute(1, 2, 0) # H, W, 3
        source_image_tensor = transforms.ToTensor()(source_image).to(device).permute(1, 2, 0) # H, W, 3

        exist_mask = exist_mask.unsqueeze(-1)
        propagated_texture_tensor = source_image_tensor * exist_mask + target_image_tensor * (1 - exist_mask)
        propagated_texture = transforms.ToPILImage()(propagated_texture_tensor.permute(2, 0, 1))

    elif mode == "reformer":
        source_image = texture_image

        source_guide_tensor = instance_mask * exist_mask
        source_guide_tensor = source_guide_tensor.unsqueeze(-1).repeat(1, 1, 3)
        target_guide_tensor = instance_mask * (1 - exist_mask)
        target_guide_tensor = target_guide_tensor.unsqueeze(-1).repeat(1, 1, 3)

        source_guide = Image.fromarray(source_guide_tensor.cpu().numpy().astype(np.uint8))
        target_guide = Image.fromarray(target_guide_tensor.cpu().numpy().astype(np.uint8))

        source_path = os.path.join(propagate_dir, "source_{}.png".format(view_idx))
        target_path = os.path.join(propagate_dir, "target_{}.png".format(view_idx))
        source_guide_path = os.path.join(propagate_dir, "source_guide_{}.png".format(view_idx))
        target_guide_path = os.path.join(propagate_dir, "target_guide_{}.png".format(view_idx))

        source_image.save(source_path)
        source_guide.save(source_guide_path)
        target_guide.save(target_guide_path)

        # synthesis via ebsynth
        cmd = [
            "python", "Texture-Reformer/transfer.py", 
            "-style", source_path, 
            "-style_sem", source_guide_path,
            "-content_sem", target_guide_path,
            "-outf", target_path,
            "-compress"
        ]
        _ = subprocess.call(cmd)

        # output
        prior_mask[(1 - exist_mask) == 1] = 1
        propagated_texture = Image.open(target_path)

    else:
        # for each instance
        instance_ids = torch.unique(instance_mask)
        for instance_id in instance_ids:
            if instance_id == 0: continue
            instance_mask_single = (instance_mask == instance_id).float()

            instance_exist_mask = (instance_mask_single * exist_mask).reshape(-1) # H x W
            instance_nonexist_mask = (instance_mask_single * (1 - exist_mask)).reshape(-1) # H x W

            # mark the propagated texels as 1
            prior_mask[instance_nonexist_mask.reshape(H, W) == 1] = 1

            if mode == "random":
                
                exist_texture_values = texture_tensor_flatten[instance_exist_mask == 1]
                num_nonexist_texels = (instance_nonexist_mask == 1).sum()

                sample_weights = torch.ones(exist_texture_values.shape[0])
                sample_weights /= sample_weights.sum()
                sample_ids = torch.multinomial(sample_weights, num_nonexist_texels, replacement=True)
                sampled_texture_values = exist_texture_values[sample_ids]

                texture_tensor_flatten[instance_nonexist_mask == 1] = sampled_texture_values

                propagated_texture_tensor = texture_tensor_flatten.permute(1, 0).reshape(3, H, W)
                propagated_texture = transforms.ToPILImage()(propagated_texture_tensor)

            elif mode == "majority": 
                exist_texture_values = texture_tensor_flatten[instance_exist_mask == 1]
                unique_values = torch.unique(exist_texture_values, dim=0)
                major_value = unique_values[-1]
                num_nonexist_texels = (instance_nonexist_mask == 1).sum()

                texture_tensor_flatten[instance_nonexist_mask == 1] = major_value.unsqueeze(0).repeat(num_nonexist_texels, 1)

                propagated_texture_tensor = texture_tensor_flatten.permute(1, 0).reshape(3, H, W)
                propagated_texture = transforms.ToPILImage()(propagated_texture_tensor)

            elif mode == "average":
                exist_texture_values = texture_tensor_flatten[instance_exist_mask == 1]
                average_values = torch.mean(exist_texture_values, dim=0)
                num_nonexist_texels = (instance_nonexist_mask == 1).sum()

                texture_tensor_flatten[instance_nonexist_mask == 1] = average_values.unsqueeze(0).repeat(num_nonexist_texels, 1)

                propagated_texture_tensor = texture_tensor_flatten.permute(1, 0).reshape(3, H, W)
                propagated_texture = transforms.ToPILImage()(propagated_texture_tensor)

            else:
                raise ValueError("invalid propagation mode.")
        
    # exit()

    return propagated_texture, prior_mask


######################################################################
#
#                       initialization helpers                     
#
#######################################################################


def init_mesh(model_path, device):
    print("=> loading target mesh...")

    # principle_directions = compute_principle_directions(model_path)
    principle_directions = None
    
    _, faces, aux = load_obj(model_path, device=device)
    mesh = load_objs_as_meshes([model_path], device=device)

    num_verts = mesh.verts_packed().shape[0]

    # make sure mesh center is at origin
    bbox = mesh.get_bounding_boxes()
    mesh_center = bbox.mean(dim=2).repeat(num_verts, 1)
    mesh = apply_offsets_to_mesh(mesh, -mesh_center)

    # make sure mesh size is normalized
    box_size = bbox[..., 1] - bbox[..., 0]
    box_max = box_size.max(dim=1, keepdim=True)[0].repeat(num_verts, 3)
    mesh = apply_scale_to_mesh(mesh, 1 / box_max)

    return mesh, mesh.verts_packed(), faces, aux, principle_directions, mesh_center, box_max


def init_mesh_blender(model_path, device):
    verts, faces, aux = load_obj(model_path, device=device)
    mesh = load_objs_as_meshes([model_path], device=device)

    return mesh, verts, faces, aux


def init_mesh_atlas(model_path, atlas_size, device):
    verts, faces, aux = load_obj(
        model_path, 
        device=device,
        load_textures=True,
        create_texture_atlas=True,
        texture_atlas_size=atlas_size,
        texture_wrap=None
    )

    textures = aux.texture_atlas
    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
        textures=TexturesAtlas(atlas=[textures]),
    )

    return mesh, verts, faces, aux


def init_multiple_meshes_as_scene(scene_config, output_dir, device, join_mesh=True, subdivide_factor=0, is_force=False, return_mesh=True, return_dict=False):
    """
        Load a list of meshes and reparameterize the UVs
        NOTE: all UVs are packed into one texture space
        Those input meshes will be combined as one mesh for the scene.

        scene_config: {
            "0": {              # instance ID
                "name": ...,    # instance name
                "type": ...,    # instance type ("things" or "stuff")
                "path": ...,    # path to obj file
                "prompt": ...   # description for the instance (left empty for "stuff")
            }
        }

    """

    output_dir = Path(output_dir) / "meshes"

    # parameterize raw inputs
    if not output_dir.exists() or is_force:

        things_dir = output_dir / "things"
        stuff_dir = output_dir / "stuff"
        os.makedirs(things_dir, exist_ok=True)
        os.makedirs(stuff_dir, exist_ok=True)

        if join_mesh:
            atlas = xatlas.Atlas()

            # load scene and init atlas
            verts_list, faces_list = [], []
            for _, mesh_info in scene_config.items():
                mesh = trimesh.load_mesh(mesh_info["path"], force='mesh')

                if subdivide_factor == 0:
                    vertices, faces = mesh.vertices, mesh.faces
                else:
                    vertices, faces = trimesh.remesh.subdivide_loop(mesh.vertices, mesh.faces, iterations=subdivide_factor)

                # mesh_list.append(mesh)

                verts_list.append(vertices)
                faces_list.append(faces)

                atlas.add_mesh(vertices, faces)

            # pack UVs
            atlas.generate()

            # combine meshes
            vertices_list, indices_list, uvs_list = [], [], []
            offset = 0
            for i in range(atlas.mesh_count):
                vmapping, indices, uvs = atlas[i]

                vertices = verts_list[i][vmapping]
                indices += offset

                vertices_list.append(vertices)
                indices_list.append(indices)
                uvs_list.append(uvs)

                num_vertices = vertices.shape[0]
                offset += num_vertices

                # HACK store the number of vertices for each mesh
                # this will be used for indexing the instance vertices
                scene_config[list(scene_config.keys())[i]]["num_vertices"] = num_vertices

            # export parameterized individual meshes
            for mesh_id, mesh_info in scene_config.items():
                mesh_id = int(mesh_id)
                output_subdir = things_dir if mesh_info["type"] == "things" else stuff_dir
                output_path = output_subdir / "{}_{}.obj".format(mesh_id, mesh_info["name"])

                vmapping, indices, uvs = atlas[int(mesh_id)]

                xatlas.export(str(output_path), verts_list[mesh_id][vmapping], indices, uvs)

            # export combined scene as one mesh
            scene_mesh_vertices = np.concatenate(vertices_list)
            scene_mesh_indices = np.concatenate(indices_list)
            scene_mesh_uvs = np.concatenate(uvs_list)

            scene_mesh_path = output_dir / "scene.obj"
            xatlas.export(str(scene_mesh_path), scene_mesh_vertices, scene_mesh_indices, scene_mesh_uvs)

        else:
            for mesh_id, mesh_info in scene_config.items():
                mesh = trimesh.load_mesh(mesh_info["path"], force='mesh')

                if subdivide_factor == 0:
                    vertices, faces = mesh.vertices, mesh.faces
                else:
                    vertices, faces = trimesh.remesh.subdivide_loop(mesh.vertices, mesh.faces, iterations=subdivide_factor)

                vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

                output_subdir = things_dir if mesh_info["type"] == "things" else stuff_dir
                output_path = output_subdir / "{}_{}.obj".format(mesh_id, mesh_info["name"])

                xatlas.export(str(output_path), vertices[vmapping], indices, uvs)

        # export scene configuration
        scene_config_path = output_dir / "scene_config.json"
        with open(scene_config_path, "w") as f:
            json.dump(scene_config, f, indent=4)

    if not return_mesh:
        return
    
    # load the parameterized assets
    scene_config_path = output_dir / "scene_config.json"
    with open(scene_config_path) as f:
        scene_config = json.load(f)

    # load the combined scene
    # NOTE only the combined scene mesh is needed here
    scene_mesh_path = output_dir / "scene.obj"

    verts, faces, aux = load_obj(scene_mesh_path, device=device)
    mesh = load_objs_as_meshes([scene_mesh_path], device=device)

    # create semantics
    semantic_list, offset_list = [], []
    for mesh_id, mesh_info in scene_config.items():
        semantic_list.append(int(mesh_id))
        offset_list.append(int(mesh_info["num_vertices"])) 

    vertices_semantics, faces_semantics = create_face_semantics(faces.verts_idx, semantic_list, offset_list, device)

    return {
        "mesh": mesh,
        "verts": verts,
        "faces": faces,
        "aux": aux,
        "scene_config": scene_config,
        "vertices_semantics": vertices_semantics,
        "faces_semantics": faces_semantics,
    }
    
def init_multiple_meshes_xatlas(scene_config, output_dir, device, is_force=False, subdivide_factor=0, return_mesh=True, return_dict=False):
    """
        Load a list of meshes and reparameterize the UVs
        NOTE: all UVs are packed into one texture space
        Those input meshes will be combined as one mesh for the scene.

        scene_config: {
            "0": {              # instance ID
                "name": ...,    # instance name
                "type": ...,    # instance type ("things" or "stuff")
                "path": ...,    # path to obj file
                "prompt": ...   # description for the instance (left empty for "stuff")
            }
        }

    """

    output_dir = Path(output_dir) / "meshes"

    # parameterize raw inputs
    if not output_dir.exists() or is_force:

        things_dir = output_dir / "things"
        stuff_dir = output_dir / "stuff"
        os.makedirs(things_dir, exist_ok=True)
        os.makedirs(stuff_dir, exist_ok=True)

        for mesh_id, mesh_info in scene_config.items():
            mesh = trimesh.load_mesh(mesh_info["path"], force='mesh')

            if subdivide_factor == 0:
                vertices, faces = mesh.vertices, mesh.faces
            else:
                vertices, faces = trimesh.remesh.subdivide_loop(mesh.vertices, mesh.faces, iterations=subdivide_factor)

            vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

            output_subdir = things_dir if mesh_info["type"] == "things" else stuff_dir
            output_path = output_subdir / "{}_{}.obj".format(mesh_id, mesh_info["name"])

            xatlas.export(str(output_path), vertices[vmapping], indices, uvs)

        # export scene configuration
        scene_config_path = output_dir / "scene_config.json"
        with open(scene_config_path, "w") as f:
            json.dump(scene_config, f, indent=4)

    if not return_mesh:
        return
    
    things_dir = output_dir / "things"
    stuff_dir = output_dir / "stuff"

    mesh_path_list, mesh_name_list = [], []

    for path in stuff_dir.glob("*.obj"):
        mesh_path_list.append(str(path))
        mesh_name_list.append(path.stem)
    
    for path in things_dir.glob("*.obj"):
        mesh_path_list.append(str(path))
        mesh_name_list.append(path.stem)

    # load the parameterized assets
    scene_config_path = output_dir / "scene_config.json"
    with open(scene_config_path) as f:
        scene_config = json.load(f)

    mesh_list, verts_list, faces_list, aux_list = [], [], [], []
    for mesh_path in mesh_path_list:
        verts, faces, aux = load_obj(mesh_path, device=device)
        mesh = load_objs_as_meshes([mesh_path], device=device)

        mesh_list.append(mesh)
        verts_list.append(verts)
        faces_list.append(faces)
        aux_list.append(aux)

    return {
        "mesh": mesh_list,
        "mesh_path": mesh_path_list,
        "mesh_name": mesh_name_list,
        "verts": verts_list,
        "faces": faces_list,
        "aux": aux_list,
        "scene_config": scene_config,
    }

def init_mesh_xatlas(model_path, output_dir, device, is_force=False, return_dict=False):
    output_dir = Path(output_dir) / "meshes"
    
    # parameterize the mesh
    mesh = trimesh.load_mesh(model_path, force='mesh')
    vertices, faces = mesh.vertices, mesh.faces

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    output_path = output_dir / "object.obj"
    xatlas.export(str(output_path), vertices[vmapping], indices, uvs)

    # load mesh
    verts, faces, aux = load_obj(str(output_path), device=device)
    mesh = load_objs_as_meshes([str(output_path)], device=device)
    
    return {
        "mesh": mesh,
        "verts": verts,
        "faces": faces,
        "aux": aux
    } 

def init_background(model_path, bounding_box, output_dir, device, is_force=False, return_dict=False):
    output_dir = Path(output_dir) / "meshes"
    
    # parameterize the mesh
    mesh = trimesh.load_mesh(model_path, force='mesh')
    vertices, faces = mesh.vertices, mesh.faces
    
    offsets = bounding_box.mean(1, keepdims=True) # 3, 1
    bounding_box -= offsets
    scale = bounding_box.max() / vertices.max()
    vertices *= scale
    vertices += offsets.T

    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    output_path = output_dir / "background.obj"
    xatlas.export(str(output_path), vertices[vmapping], indices, uvs)

    # load mesh
    verts, faces, aux = load_obj(str(output_path), device=device)
    mesh = load_objs_as_meshes([str(output_path)], device=device)
    
    return {
        "mesh": mesh,
        "verts": verts,
        "faces": faces,
        "aux": aux
    }