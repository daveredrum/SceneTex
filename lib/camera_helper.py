import torch

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from pytorch3d.renderer import (
    PerspectiveCameras,
    FoVPerspectiveCameras,
    look_at_view_transform,
    camera_position_from_spherical_angles,
    look_at_rotation,
)

# customized
import sys
sys.path.append(".")

from lib.constants import VIEWPOINTS

# ---------------- UTILS ----------------------

def degree_to_radian(d):
    return d * np.pi / 180

def radian_to_degree(r):
    return 180 * r / np.pi

def xyz_to_polar(xyz):
    """ assume y-axis is the up axis """
    
    x, y, z = xyz
    
    theta = 180 * np.arccos(z) / np.pi
    phi = 180 * np.arccos(y) / np.pi

    return theta, phi

def polar_to_xyz(theta, phi, dist):
    """ assume y-axis is the up axis """

    theta = degree_to_radian(theta)
    phi = degree_to_radian(phi)

    x = np.sin(phi) * np.sin(theta) * dist
    y = np.cos(phi) * dist
    z = np.sin(phi) * np.cos(theta) * dist

    return [x, y, z]


# ---------------- VIEWPOINTS ----------------------


def filter_viewpoints(pre_viewpoints: dict, viewpoints: dict):
    """ return the binary mask of viewpoints to be filtered """

    filter_mask = [0 for _ in viewpoints.keys()]
    for i, v in viewpoints.items():
        x_v, y_v, z_v = polar_to_xyz(v["azim"], 90 - v["elev"], v["dist"])

        for _, pv in pre_viewpoints.items():
            x_pv, y_pv, z_pv = polar_to_xyz(pv["azim"], 90 - pv["elev"], pv["dist"])
            sim = cosine_similarity(
                np.array([[x_v, y_v, z_v]]),
                np.array([[x_pv, y_pv, z_pv]])
            )[0, 0]

            if sim > 0.9:
                filter_mask[i] = 1

    return filter_mask


def init_viewpoints(mode, sample_space, init_dist, init_elev, principle_directions, 
    use_principle=True, use_shapenet=False, use_objaverse=False):

    if mode == "predefined":

        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list
        ) = init_predefined_viewpoints(sample_space, init_dist, init_elev)

    elif mode == "hemisphere":

        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list
        ) = init_hemisphere_viewpoints(sample_space, init_dist)

    else:
        raise NotImplementedError()

    # punishments for views -> in case always selecting the same view
    view_punishments = [1 for _ in range(len(dist_list))]

    if use_principle:

        (
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list,
            view_punishments
        ) = init_principle_viewpoints(
            principle_directions, 
            dist_list, 
            elev_list, 
            azim_list, 
            sector_list,
            view_punishments,
            use_shapenet,
            use_objaverse
        )

    return dist_list, elev_list, azim_list, sector_list, view_punishments


def init_principle_viewpoints(
    principle_directions, 
    dist_list, 
    elev_list, 
    azim_list, 
    sector_list,
    view_punishments,
    use_shapenet=False,
    use_objaverse=False
):

    if use_shapenet:
        key = "shapenet"

        pre_elev_list = [v for v in VIEWPOINTS[key]["elev"]]
        pre_azim_list = [v for v in VIEWPOINTS[key]["azim"]]
        pre_sector_list = [v for v in VIEWPOINTS[key]["sector"]]

        num_principle = 10
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]

    elif use_objaverse:
        key = "objaverse"

        pre_elev_list = [v for v in VIEWPOINTS[key]["elev"]]
        pre_azim_list = [v for v in VIEWPOINTS[key]["azim"]]
        pre_sector_list = [v for v in VIEWPOINTS[key]["sector"]]

        num_principle = 10
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]
    else:
        num_principle = 6
        pre_elev_list = [v for v in VIEWPOINTS[num_principle]["elev"]]
        pre_azim_list = [v for v in VIEWPOINTS[num_principle]["azim"]]
        pre_sector_list = [v for v in VIEWPOINTS[num_principle]["sector"]]
        pre_dist_list = [dist_list[0] for _ in range(num_principle)]
        pre_view_punishments = [0 for _ in range(num_principle)]

    dist_list = pre_dist_list + dist_list
    elev_list = pre_elev_list + elev_list
    azim_list = pre_azim_list + azim_list
    sector_list = pre_sector_list + sector_list
    view_punishments = pre_view_punishments + view_punishments

    return dist_list, elev_list, azim_list, sector_list, view_punishments


def init_predefined_viewpoints(sample_space, init_dist, init_elev):
    
    viewpoints = VIEWPOINTS[sample_space]

    assert sample_space == len(viewpoints["sector"])

    dist_list = [init_dist for _ in range(sample_space)] # always the same dist
    elev_list = [viewpoints["elev"][i] for i in range(sample_space)]
    azim_list = [viewpoints["azim"][i] for i in range(sample_space)]
    sector_list = [viewpoints["sector"][i] for i in range(sample_space)]

    return dist_list, elev_list, azim_list, sector_list


def init_hemisphere_viewpoints(sample_space, init_dist):
    """
        y is up-axis
    """

    num_points = 2 * sample_space
    ga = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    flags = []
    elev_list = [] # degree
    azim_list = [] # degree

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1

        # only take the north hemisphere
        if y >= 0: 
            flags.append(True)
        else:
            flags.append(False)

        theta = ga * i  # golden angle increment

        elev_list.append(radian_to_degree(np.arcsin(y)))
        azim_list.append(radian_to_degree(theta))

        radius = np.sqrt(1 - y * y)  # radius at y
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

    elev_list = [elev_list[i] for i in range(len(elev_list)) if flags[i]]
    azim_list = [azim_list[i] for i in range(len(azim_list)) if flags[i]]

    dist_list = [init_dist for _ in elev_list]
    sector_list = ["good" for _ in elev_list] # HACK don't define sector names for now

    return dist_list, elev_list, azim_list, sector_list

def init_trajectory(dist_list, elev_list, azim_list, at):
    Rs, Ts = [], []
    for dist, elev, azim in zip(dist_list, elev_list, azim_list):
        R, T = look_at_view_transform(dist, elev, azim, at=at)

        Rs.append(R) # 1, 3, 3
        Ts.append(T) # 1, 3

    return Rs, Ts

def init_meshlab_trajectory(trajectory, device):
    Rs, Ts = [], []
    for _, viewpoint in trajectory.items():
        R = torch.FloatTensor(viewpoint["rotation"]).to(device)
        T = torch.FloatTensor(viewpoint["translation"]).to(device)

        # convert to PyTorch3D's camera convention
        # flipping z-axis
        R[:, 2] *= -1
        T *= torch.FloatTensor([0, 0, -1]).to(device)

        Rs.append(R.unsqueeze(0))
        Ts.append(T.unsqueeze(0))

    return Rs, Ts

def init_blender_trajectory(trajectory, device):
    Rs, Ts = [], []
    for _, viewpoint in trajectory.items():
        c2w = torch.FloatTensor(viewpoint["matrix"]).to(device)

        calibrate_axis = torch.FloatTensor([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ]).to(device)
        rot_z = torch.FloatTensor([
            [np.cos(np.pi), -np.sin(np.pi), 0],
            [np.sin(np.pi), np.cos(np.pi), 0],
            [0, 0, 1]
        ]).to(device)
        rot_y = torch.FloatTensor([
            [np.cos(np.pi), 0, -np.sin(np.pi), 0],
            [0, 1, 0, 0],
            [np.sin(np.pi), 0, np.cos(np.pi), 0],
            [0, 0, 0, 1]
        ]).to(device)

        c2w = calibrate_axis @ c2w
        c2w = rot_y @ c2w

        t = c2w[:3,-1]  # Extract translation of the camera
        r = c2w[:3, :3] @ rot_z # Extract rotation matrix of the camera

        # horizontally flip the image
        flip_x = torch.FloatTensor([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).to(device)
        r = r @ flip_x

        t = t @ r # Make rotation local

        Rs.append(r.unsqueeze(0))
        Ts.append(t.unsqueeze(0))

    return Rs, Ts

def init_blenderproc_trajectory(trajectory, device):
    """
        This function only applies for Blenderproc cameras and original mesh data
    """
    Rs, Ts = [], []
    for _, viewpoint in trajectory.items():
        c2w = torch.FloatTensor(viewpoint["matrix"]).to(device)

        calibrate_axis = torch.FloatTensor([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]).to(device)
        rot_z = torch.FloatTensor([
            [np.cos(np.pi), -np.sin(np.pi), 0],
            [np.sin(np.pi), np.cos(np.pi), 0],
            [0, 0, 1]
        ]).to(device)
        rot_x = torch.FloatTensor([
            [1, 0, 0, 0],
            [0, np.cos(np.pi/2), -np.sin(np.pi/2), 0],
            [0, np.sin(np.pi/2), np.cos(np.pi/2), 0],
            [0, 0, 0, 1]
        ]).to(device)

        c2w = calibrate_axis @ c2w
        c2w = rot_x @ c2w

        t = c2w[:3,-1]  # Extract translation of the camera
        r = c2w[:3, :3] @ rot_z # Extract rotation matrix of the camera

        # horizontally flip the image
        flip_x = torch.FloatTensor([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).to(device)
        r = r @ flip_x

        t = t @ r # Make rotation local

        Rs.append(r.unsqueeze(0))
        Ts.append(t.unsqueeze(0))

    return Rs, Ts

def init_3dfront_trajectory_from_blenderproc(trajectory, device):
    """
        This function only applies for 3D-FRONT scenes and cameras processed by Blenderproc
    """
    Rs, Ts = [], []
    for _, viewpoint in trajectory.items():
        c2w = torch.FloatTensor(viewpoint["matrix"]).to(device)

        calibrate_axis = torch.FloatTensor([
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]).to(device)
        rot_z = torch.FloatTensor([
            [np.cos(np.pi), -np.sin(np.pi), 0],
            [np.sin(np.pi), np.cos(np.pi), 0],
            [0, 0, 1]
        ]).to(device)
        rot_x = torch.FloatTensor([
            [1, 0, 0, 0],
            [0, np.cos(np.pi/2), -np.sin(np.pi/2), 0],
            [0, np.sin(np.pi/2), np.cos(np.pi/2), 0],
            [0, 0, 0, 1]
        ]).to(device)

        c2w = calibrate_axis @ c2w
        c2w = rot_x @ c2w

        t = c2w[:3,-1]  # Extract translation of the camera
        r = c2w[:3, :3] @ rot_z # Extract rotation matrix of the camera

        # horizontally flip the image
        flip_x = torch.FloatTensor([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).to(device)
        r = r @ flip_x

        t = t @ r # Make rotation local

        Rs.append(r.unsqueeze(0))
        Ts.append(t.unsqueeze(0))

    return Rs, Ts

# ---------------- CAMERAS ----------------------

def init_camera(camera_params, image_size, device):
    if "dist" in camera_params:
        cameras = init_camera_lookat(
            camera_params["dist"],
            camera_params["elev"],
            camera_params["azim"],
            image_size,
            device
        )
    elif "rotation" in camera_params:
        cameras = init_camera_R_T(
            camera_params["rotation"],
            camera_params["translation"],
            image_size,
            device
        )
    else:
        raise ValueError("invalid camera parameters.")
    
    return cameras

def init_camera_lookat(dist, elev, azim, image_size, device, fov=60, at=torch.FloatTensor([[0, 0, 0]])):
    R, T = look_at_view_transform(dist, elev, azim, at=at)
    image_size = torch.tensor([image_size, image_size]).unsqueeze(0)
    # cameras = PerspectiveCameras(R=R, T=T, device=device, image_size=image_size)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device, fov=fov)

    return cameras

def init_camera_R_T(R, T, image_size, device, fov=60):
    """init camera using R and T matrics

    Args:
        R (torch.FloatTensor): Rotation matrix, (N, 3, 3)
        T (torch.FloatTensor): Translation matrix, (N, 3)
        image_size (int): rendering size
        device (torch.device): CPU or GPU

    Returns:
        camera: PyTorch3D camera instance
    """

    if isinstance(image_size, int):
        image_size = torch.tensor([image_size, image_size]).unsqueeze(0)
    elif isinstance(image_size, tuple):
        image_size = torch.tensor(image_size).unsqueeze(0)
    else:
        raise TypeError("invalid image size.")

    # cameras = PerspectiveCameras(R=R, T=T, device=device, image_size=image_size)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device, fov=fov)

    return cameras