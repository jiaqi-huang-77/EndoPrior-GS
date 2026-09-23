"""Shared camera construction, initial point sampling and prepared-image loading."""

import glob
import os

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from scene.cameras import Camera
from utils.graphics_utils import focal2fov
from utils.texture_prior import sampling_probability_np


class DataLoader:
    """Dataset readers provide load_meta(), camera_data() and point geometry."""

    image_size = (640, 512)

    def __init__(self, datadir, downsample=1.0, test_every=8, *,
                 use_prior_initialisation=True, initial_point_budget=20_000,
                 initialise_from_all_frames=True, initial_points_per_frame_min=160,
                 initial_points_per_frame_max=330, initial_points_ess_ratio=0.20,
                 prior_uniform_mix=0.45, prior_erosion_kernel=9,
                 prior_brightness_threshold=0.85, prior_brightness_percentile=97.0,
                 prior_saturation_threshold=0.35, prior_gradient_percentile=99.0,
                 initialisation_seed=0):
        self.root_dir = datadir
        self.downsample = downsample
        self.img_wh = tuple(int(size / downsample) for size in self.image_size)
        self.transform = T.ToTensor()
        self.white_bg = False
        self.use_prior_initialisation = use_prior_initialisation
        self.initial_point_budget = initial_point_budget
        self.initialise_from_all_frames = initialise_from_all_frames
        self.initial_points_per_frame_min = initial_points_per_frame_min
        self.initial_points_per_frame_max = initial_points_per_frame_max
        self.initial_points_ess_ratio = initial_points_ess_ratio
        self.prior_uniform_mix = prior_uniform_mix
        self.prior_erosion_kernel = prior_erosion_kernel
        self.prior_brightness_threshold = prior_brightness_threshold
        self.prior_brightness_percentile = prior_brightness_percentile
        self.prior_saturation_threshold = prior_saturation_threshold
        self.prior_gradient_percentile = prior_gradient_percentile
        self.initialisation_seed = initialisation_seed
        count = self.load_meta()
        self.train_idxs = [i for i in range(count) if (i - 1) % test_every != 0]
        self.test_idxs = [i for i in range(count) if (i - 1) % test_every == 0]
        self.video_idxs = list(range(count))
        self.maxtime = 1.0
        print(f"Loaded metadata for {count} images.")

    def format_infos(self, split):
        indices = {"train": self.train_idxs, "test": self.test_idxs,
                   "video": self.video_idxs}[split]
        cameras = []
        for index in tqdm(indices, desc=f"Loading {split} cameras"):
            data = self.camera_data(index)
            focal_x, focal_y = data.pop("focal")
            data["image"] = self.transform(data.pop("colour"))
            data["depth"] = torch.from_numpy(data["depth"])
            data["mask"] = self.transform(data["mask"]).bool()
            data["motion_mask"] = self.transform(data["motion_mask"]).bool()
            cameras.append(Camera(
                FoVx=focal2fov(focal_x, self.img_wh[0]),
                FoVy=focal2fov(focal_y, self.img_wh[1]),
                image_name=str(index), uid=index, **data,
            ))
        return cameras

    def _initial_frame_indices(self):
        if self.initialise_from_all_frames:
            return list(self.train_idxs)

        maximum_frames = 10
        step = max(1, len(self.train_idxs) // maximum_frames)
        return list(self.train_idxs[::step][:maximum_frames])

    def _sampling_distribution(self, colour, mask, flattened_mask):
        number_of_points = int(np.asarray(flattened_mask, dtype=bool).sum())
        if number_of_points == 0:
            return np.empty(0, dtype=np.float64)

        if not self.use_prior_initialisation:
            return np.full(number_of_points, 1.0 / number_of_points, dtype=np.float64)

        probability_map, _ = sampling_probability_np(
            colour=colour,
            mask=mask,
            mode="texture_prior",
            uniform_mix=self.prior_uniform_mix,
            erode_kernel_size=self.prior_erosion_kernel,
            specular_min_v=self.prior_brightness_threshold,
            specular_brightness_percentile=self.prior_brightness_percentile,
            specular_s_threshold=self.prior_saturation_threshold,
            grad_percentile=self.prior_gradient_percentile,
        )
        flattened_mask = np.asarray(flattened_mask, dtype=bool)
        probability = probability_map.reshape(-1)[flattened_mask].astype(np.float64)

        if probability.sum() <= 0.0:
            probability.fill(1.0 / number_of_points)

        probability = np.maximum(probability, 0.0)
        probability /= probability.sum()
        return probability

    def _automatic_frame_budget(self, probability):
        effective_sample_size = 1.0 / (np.square(probability).sum() + 1e-12)
        budget = int(np.ceil(self.initial_points_ess_ratio * effective_sample_size))
        return int(
            np.clip(
                budget,
                self.initial_points_per_frame_min,
                self.initial_points_per_frame_max,
            )
        )

    def _sample_initial_points_from_frame(self, index, frame_budget, rng):
        colour, mask, flattened_mask, camera_points, colours, pose = (
            self._frame_data_for_initialisation(index)
        )
        if camera_points.shape[0] == 0:
            return None, None

        probability = self._sampling_distribution(colour, mask, flattened_mask)
        if self.initial_point_budget == 0:
            frame_budget = self._automatic_frame_budget(probability)

        sample_count = min(
            frame_budget,
            camera_points.shape[0],
            int((probability > 0).sum()),
        )
        if sample_count <= 0:
            return None, None

        selected = rng.choice(
            camera_points.shape[0],
            size=sample_count,
            replace=False,
            p=probability,
        )
        world_points = self.get_world_points(
            camera_points[selected], pose
        )
        return world_points, colours[selected]

    def get_initial_points(self):
        frame_indices = self._initial_frame_indices()
        if not frame_indices:
            raise RuntimeError("No training frames are available for initialisation.")

        automatic_budget = self.initial_point_budget == 0
        if automatic_budget:
            base_budget = remainder = 0
            budget_description = (
                "automatic ESS budget "
                f"[{self.initial_points_per_frame_min}, "
                f"{self.initial_points_per_frame_max}]"
            )
        else:
            base_budget, remainder = divmod(
                self.initial_point_budget, len(frame_indices)
            )
            budget_description = f"global budget {self.initial_point_budget}"

        print(
            f"Initialising Gaussians from {len(frame_indices)} training frames "
            f"with {budget_description}."
        )
        rng = np.random.default_rng(self.initialisation_seed)
        all_points = []
        all_colours = []
        for frame_number, index in enumerate(frame_indices):
            frame_budget = (
                0
                if automatic_budget
                else base_budget + int(frame_number < remainder)
            )
            points, colours = self._sample_initial_points_from_frame(
                index, frame_budget, rng
            )
            if points is not None:
                all_points.append(points)
                all_colours.append(colours)

        if not all_points:
            raise RuntimeError("No points were sampled for initialisation.")

        points = np.concatenate(all_points, axis=0)
        colours = np.concatenate(all_colours, axis=0)
        if not automatic_budget and points.shape[0] > self.initial_point_budget:
            selected = rng.choice(
                points.shape[0], size=self.initial_point_budget, replace=False
            )
            points = points[selected]
            colours = colours[selected]

        points, colours, normals = self._consolidate_initial_points(points, colours)
        print(f"Initialised {points.shape[0]} points.")
        return points, colours, normals

    @staticmethod
    def _consolidate_initial_points(points, colours):
        points = points.astype(np.float32)
        colours = colours.astype(np.float32)
        return points, colours, np.zeros(points.shape, dtype=np.float32)

    def get_maxtime(self):
        return self.maxtime


class ImageDepthDataset(DataLoader):
    """Shared prepared PNG/poses_bounds format for EndoNeRF and StereoMIS."""

    def __init__(self, datadir, downsample=1.0, test_every=8, *,
                 initial_point_budget=30_000, initialise_from_all_frames=False,
                 **initialisation):
        super().__init__(datadir, downsample, test_every,
                         initial_point_budget=initial_point_budget,
                         initialise_from_all_frames=initialise_from_all_frames,
                         **initialisation)

    @staticmethod
    def _clip_depth_outliers(depth, depth_path):
        valid_depth = depth != 0
        if not valid_depth.any():
            raise ValueError(f"Depth map contains no valid pixels: {depth_path}")
        near = np.percentile(depth[valid_depth], 3.0)
        far = np.percentile(depth[valid_depth], 99.8)
        return np.clip(depth, near, far)

    def load_meta(self):
        poses_path = os.path.join(self.root_dir, "poses_bounds.npy")
        poses_array = np.load(poses_path)
        poses = poses_array[:, :-2].reshape([-1, 3, 5])
        _, _, focal = poses[0, :, -1]
        focal /= self.downsample
        self.focal = (focal, focal)

        poses = np.concatenate(
            [poses[..., :1], poses[..., 1:2], poses[..., 2:3], poses[..., 3:4]],
            axis=-1,
        )
        self.image_poses = []
        self.image_times = []
        for index, pose in enumerate(poses):
            camera_to_world = np.concatenate(
                (pose, np.array([[0, 0, 0, 1]])), axis=0
            )
            world_to_camera = np.linalg.inv(camera_to_world)
            rotation = np.transpose(world_to_camera[:3, :3])
            translation = world_to_camera[:3, -1]
            self.image_poses.append((rotation, translation))
            self.image_times.append(index / poses.shape[0])

        def sorted_pngs(directory):
            return sorted(glob.glob(os.path.join(self.root_dir, directory, "*.png")))

        self.image_paths = sorted_pngs("images")
        self.depth_paths = sorted_pngs("depth")
        self.mask_paths = sorted_pngs("masks")
        ground_truth_masks = sorted_pngs("gt_masks")
        self.motion_masks_by_name = {
            os.path.basename(path): path for path in ground_truth_masks
        }

        expected = poses.shape[0]
        if len(self.image_paths) != expected:
            raise ValueError("The number of images must match the poses.")
        if len(self.depth_paths) != expected:
            raise ValueError("The number of depth maps must match the poses.")
        if len(self.mask_paths) != expected:
            raise ValueError("The number of masks must match the poses.")

        return len(self.image_paths)

    def camera_data(self, index):
        colour, depth, mask = self.get_colour_depth_mask(index)
        motion_mask = mask
        gt_mask = self.motion_masks_by_name.get(os.path.basename(self.mask_paths[index]))
        if gt_mask is not None:
            motion_mask = (1.0 - np.asarray(Image.open(gt_mask)) / 255.0) * mask
        rotation, translation = self.image_poses[index]
        return dict(colour=colour, depth=depth, mask=mask, motion_mask=motion_mask,
                    R=rotation, T=translation, focal=self.focal,
                    time=self.image_times[index], Znear=None, Zfar=None)

    def _frame_data_for_initialisation(self, index):
        colour, depth, mask = self.get_colour_depth_mask(index)
        points, colours, flat_mask = self.get_camera_points(depth, mask, colour)
        return colour, mask, flat_mask, points, colours, self.image_poses[index]

    @staticmethod
    def _consolidate_initial_points(points, colours):
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        colours = colours[finite]
        if points.shape[0] == 0:
            raise RuntimeError("Initialisation produced no finite points.")

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colours)

        maximum_extent = float(
            np.max(point_cloud.get_axis_aligned_bounding_box().get_extent())
        )
        if maximum_extent > 0.0:
            point_cloud = point_cloud.voxel_down_sample(maximum_extent / 400.0)
        if len(point_cloud.points) > 20:
            point_cloud, _ = point_cloud.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=1.5,
            )

        final_points = np.asarray(point_cloud.points).astype(np.float32)
        final_colours = np.asarray(point_cloud.colors).astype(np.float32)
        normals = np.zeros((final_points.shape[0], 3), dtype=np.float32)
        return final_points, final_colours, normals

    @staticmethod
    def get_world_points(points, pose):
        rotation, translation = pose
        world_to_camera = np.concatenate(
            (np.transpose(rotation), translation[..., None]), axis=-1
        )
        world_to_camera = np.concatenate(
            (world_to_camera, np.array([[0, 0, 0, 1]])), axis=0
        )
        camera_to_world = np.linalg.inv(world_to_camera)
        homogeneous = np.concatenate(
            (points, np.ones((points.shape[0], 1))), axis=-1
        )
        return (camera_to_world @ homogeneous.T).T[:, :3]

    def get_colour_depth_mask(self, index):
        depth_path = self.depth_paths[index]
        depth = self._clip_depth_outliers(
            np.asarray(Image.open(depth_path)), depth_path
        )
        mask = 1.0 - np.asarray(Image.open(self.mask_paths[index])) / 255.0
        colour = np.asarray(Image.open(self.image_paths[index])) / 255.0
        return colour, depth, mask

    def get_camera_points(self, depth, mask, colour):
        width, height = self.img_wh
        horizontal, vertical = np.meshgrid(
            np.linspace(0, width - 1, width),
            np.linspace(0, height - 1, height),
        )
        x_over_z = (horizontal - width / 2) / self.focal[0]
        y_over_z = (vertical - height / 2) / self.focal[1]
        camera_points = np.stack(
            (x_over_z * depth, y_over_z * depth, depth), axis=-1
        ).reshape(-1, 3)
        colours = colour.reshape(-1, 3)
        flattened_mask = mask.reshape(-1).astype(bool)
        return (
            camera_points[flattened_mask],
            colours[flattened_mask],
            flattened_mask,
        )
