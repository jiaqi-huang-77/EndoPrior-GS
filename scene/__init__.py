#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import os

import numpy as np

import arguments
import datasets.endonerf as endonerf_dataset
import datasets.scared as scared_dataset
import datasets.stereomis as stereomis_dataset
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import BasicPointCloud, getWorld2View2


def _initialisation_arguments(args):
    prior_policy = arguments.prior_component_policy(args)
    return {
        "use_prior_initialisation": prior_policy.initialisation,
        "initial_point_budget": args.initial_point_budget,
        "initialise_from_all_frames": args.initialise_from_all_frames,
        "prior_uniform_mix": args.prior_uniform_mix,
        "prior_erosion_kernel": args.prior_erosion_kernel,
        "prior_brightness_threshold": args.prior_brightness_threshold,
        "prior_brightness_percentile": args.prior_brightness_percentile,
        "prior_saturation_threshold": args.prior_saturation_threshold,
        "prior_gradient_percentile": args.prior_gradient_percentile,
        "initialisation_seed": args.initialisation_seed,
    }


def _load_dataset(args):
    readers = {
        "endonerf": endonerf_dataset.EndoNeRFDataset,
        "scared": scared_dataset.SCAREDDataset,
        "stereomis": stereomis_dataset.StereoMISDataset,
    }
    return readers[args.dataset_type.lower()](
        args.source_path, **_initialisation_arguments(args)
    )


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}

class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=(1.0,),
        coarse_only=False,
    ):
        """Load a configured dataset and either initialise or restore its Gaussians."""
        del shuffle, resolution_scales
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration is not None:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}.")

        print(f"Data source: {os.path.abspath(args.source_path)}")
        dataset = _load_dataset(args)
        self.train_camera = dataset.format_infos(split="train")
        self.test_camera = dataset.format_infos(split="test")
        self.video_camera = dataset.format_infos(split="video")
        self.cameras_extent = getNerfppNorm(self.train_camera)["radius"]
        self.maxtime = dataset.get_maxtime()

        if self.loaded_iter is not None:
            prefix = "coarse_iteration" if coarse_only else "iteration"
            iteration_directory = os.path.join(
                self.model_path,
                "point_cloud",
                f"{prefix}_{self.loaded_iter}",
            )
            self.gaussians.load_ply(
                os.path.join(iteration_directory, "point_cloud.ply")
            )
            self.gaussians.load_model(iteration_directory)
            points = self.gaussians.get_xyz.detach().cpu().numpy()
            if points.shape[0] == 0:
                raise RuntimeError("The loaded Gaussian point cloud is empty.")
        else:
            points, colours, normals = dataset.get_initial_points()
            point_cloud = BasicPointCloud(points=points, colors=colours, normals=normals)
            if points.shape[0] == 0:
                raise RuntimeError("The initial point cloud is empty.")
            xyz_max = points.max(axis=0)
            xyz_min = points.min(axis=0)
            self.gaussians._deformation.deformation_net.grid.set_aabb(
                xyz_max,
                xyz_min,
            )
            extent = (
                args.camera_extent
                if str(args.dataset_type).lower() in {"endonerf", "stereomis"}
                else self.cameras_extent
            )
            if extent is None:
                raise ValueError("camera_extent must be set for image/depth datasets.")
            self.gaussians.create_from_pcd(
                point_cloud,
                extent,
            )

    def save(self, iteration, stage):
        prefix = "coarse_iteration" if stage == "coarse" else "iteration"
        point_cloud_path = os.path.join(
            self.model_path,
            "point_cloud",
            f"{prefix}_{iteration}",
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        del scale
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        del scale
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        del scale
        return self.video_camera
