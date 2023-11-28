import numpy as np

from torch.utils.data import Dataset


class ViewpointsDataset(Dataset):
    def __init__(self, config):
        
        self.config = config

        self._init_camera_settings()

    def _init_camera_settings(self):
        if self.config.use_center_cameras or self.config.use_grid_cameras: # use random cameras
            # self.camera_config = OmegaConf.load(self.config.trajectory_path)
            self.camera_config = self.config.camera_space
            
            dist_linspace = np.linspace(
                self.camera_config.dist.min,
                self.camera_config.dist.max,
                1 if self.camera_config.dist.min == self.camera_config.dist.max else self.camera_config.dist.num_linspace,
            )
            elev_linspace = np.linspace(
                self.camera_config.elev.min,
                self.camera_config.elev.max,
                1 if self.camera_config.elev.min == self.camera_config.elev.max else self.camera_config.elev.num_linspace,
            )
            azim_linspace = np.linspace(
                self.camera_config.azim.min,
                self.camera_config.azim.max,
                1 if self.camera_config.azim.min == self.camera_config.azim.max else self.camera_config.azim.num_linspace,
            )
            fov_linspace = np.linspace(
                self.camera_config.fov.min,
                self.camera_config.fov.max,
                1 if self.camera_config.fov.min == self.camera_config.fov.max else self.camera_config.azim.num_linspace,
            )

            if self.config.use_center_cameras:
                at = np.array(self.camera_config.at)
            else:
                at = np.load(self.config.trajectory_path)

            combinations = np.array(np.meshgrid(dist_linspace, elev_linspace, azim_linspace, fov_linspace)).T.reshape(-1, 4)
            self.dist_list = combinations[:, 0].tolist()
            self.elev_list = combinations[:, 1].tolist()
            self.azim_list = combinations[:, 2].tolist()
            self.fov_list = combinations[:, 3].tolist()
            self.at = at

            self.num_cameras = len(self.dist_list)

            # for inference
            dist_linspace = [self.config.dist]
            elev_linspace = [self.config.elev]
            azim_linspace = np.linspace(
                self.camera_config.azim.min,
                self.camera_config.azim.max,
                self.config.log_latents_views,
            )
            fov_linspace = [self.config.fov]
            at = np.array(self.config.at)

            combinations = np.array(np.meshgrid(dist_linspace, elev_linspace, azim_linspace, fov_linspace)).T.reshape(-1, 4)
            self.inference_dist_list = combinations[:, 0].tolist()
            self.inference_elev_list = combinations[:, 1].tolist()
            self.inference_azim_list = combinations[:, 2].tolist()
            self.inference_fov_list = combinations[:, 3].tolist()
            self.inference_at = at

            self.num_inference_cameras = len(self.inference_dist_list)

        else: # use fixed cameras
            raise NotImplementedError

        print("=> using {} cameras for training, {} cameras for inference.".format(self.num_cameras, self.num_inference_cameras))

    def __len__(self):
        return self.num_cameras

    def __getitem__(self, index):
        dist = self.dist_list[index]
        elev = self.elev_list[index]
        azim = self.azim_list[index]
        fov = self.fov_list[index]
        at = self.at

        return {
            "dist": dist,
            "elev": elev,
            "azim": azim,
            "fov": fov,
            "at": at
        }

