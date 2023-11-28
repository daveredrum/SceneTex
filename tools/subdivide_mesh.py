import os
import argparse
import torch

import numpy as np

from pathlib import Path
from PIL import Image
from pytorch3d.io import (
    load_objs_as_meshes,
    save_obj
)
from pytorch3d.ops import SubdivideMeshes


def init_args():
    print("=> initializing input arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_path", type=str, required=True)
    parser.add_argument("--num_iter", type=int, default=2)

    args = parser.parse_args()

    return args

def init_mesh(args):
    print("=> loading target mesh...")
    
    meshes = load_objs_as_meshes([args.obj_path])

    return meshes

def subdivide(args, meshes):
    new_meshes = meshes.clone()
    num_iter = args.num_iter
    assert num_iter > 0

    for _ in range(num_iter):
        new_meshes = SubdivideMeshes()(new_meshes)

    return new_meshes

if __name__ == "__main__":
    args = init_args()

    meshes = init_mesh(args)
    new_meshes = subdivide(args, meshes)

    input_path = Path(args.obj_path)
    output_path = input_path.parent / "{}_subdivided.obj".format(input_path.stem)

    save_obj(
        str(output_path),
        verts=new_meshes.verts_packed(),
        faces=new_meshes.faces_packed(),
        decimal_places=5
    )

