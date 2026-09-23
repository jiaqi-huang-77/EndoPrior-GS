"""SCARED: read processed frames, build cameras, then sample initial points.

Raw stereo preprocessing and validation are in the lower part of this file.
Run them with ``python -m datasets.scared KEYFRAME [options]``.
"""

import argparse
import json
import os
import re
import shutil
import tarfile
import tempfile
import warnings
from pathlib import Path, PurePosixPath

import cv2
import imageio.v2 as iio
import numpy as np
import tifffile
from tqdm import trange

from scene.data_loader import DataLoader


SUPPORTED_SCARED_GEOMETRY_VERSIONS = frozenset({1, 2})

SCARED_FRAME_STRIDES = {
    "dataset_1": 2,
    "dataset_2": 1,
    "dataset_3": 4,
    "dataset_6": 8,
    "dataset_7": 8,
}


def _depth_from_rectified_disparity(disparity, reprojection):
    """Reproject a disparity image through the complete OpenCV Q matrix."""

    disparity = np.asarray(disparity, dtype=np.float64)
    reprojection = np.asarray(reprojection, dtype=np.float64)
    if disparity.ndim != 2 or reprojection.shape != (4, 4):
        raise ValueError("SCARED disparity and reprojection shapes are invalid.")

    height, width = disparity.shape
    horizontal, vertical = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.stack(
        (horizontal, vertical, disparity, np.ones_like(disparity)),
        axis=-1,
    )
    homogeneous = pixels @ reprojection.T
    denominator = homogeneous[..., 3]
    valid = (
        (disparity > 0)
        & np.all(np.isfinite(homogeneous), axis=-1)
        & (np.abs(denominator) > 1e-12)
    )
    depth = np.zeros(disparity.shape, dtype=np.float32)
    depth[valid] = (homogeneous[..., 2][valid] / denominator[valid]).astype(
        np.float32
    )
    depth[~np.isfinite(depth) | (depth <= 0)] = 0
    return depth

def _scared_frame_stride(datadir):
    """Return the published frame stride for a supported SCARED sequence."""
    path_components = set(Path(datadir).parts)
    matches = [name for name in SCARED_FRAME_STRIDES if name in path_components]
    if len(matches) != 1:
        supported = ", ".join(SCARED_FRAME_STRIDES)
        raise ValueError(
            "The SCARED path must contain exactly one supported dataset component "
            f"({supported}); received: {datadir}"
        )
    return SCARED_FRAME_STRIDES[matches[0]]

def _geometry_version(payload):
    version = payload.get("schema_version", 1)
    if type(version) is not int:
        raise ValueError("Invalid geometry schema version")
    if version not in SUPPORTED_SCARED_GEOMETRY_VERSIONS:
        raise ValueError(f"Unsupported SCARED geometry schema v{version}")
    return version


def _camera_geometry(calibration, geometry, image_size):
    """Decode camera geometry once for both loading and offline validation."""
    version = _geometry_version(geometry)
    world_to_camera = np.asarray(calibration["camera-pose"], dtype=np.float64)
    if version == 2:
        rotation = np.asarray(geometry["left_rectification_rotation"], dtype=np.float64)
        left = np.asarray(geometry["left_projection_matrix"], dtype=np.float64)
        right = np.asarray(geometry["right_projection_matrix"], dtype=np.float64)
        reprojection = np.asarray(geometry["reprojection_matrix"], dtype=np.float64)
        matrices = [(rotation, (3, 3)), (left, (3, 4)), (right, (3, 4))]
        if tuple(geometry["image_size"]) != image_size or any(
            a.shape != shape or not np.isfinite(a).all() for a, shape in matrices
        ):
            raise ValueError("Invalid rectified camera geometry")
        intrinsics = left[:, :3]
        rectification = np.eye(4)
        rectification[:3, :3] = rotation
        world_to_camera = rectification @ world_to_camera
    else:
        intrinsics = np.asarray(calibration["camera-calibration"]["KL"], dtype=np.float64)
        reprojection = np.asarray(geometry["reprojection-matrix"], dtype=np.float64)
    matrices = [(intrinsics, (3, 3)), (world_to_camera, (4, 4)), (reprojection, (4, 4))]
    if any(a.shape != shape or not np.isfinite(a).all() for a, shape in matrices):
        raise ValueError("Invalid camera matrix or reprojection matrix")
    if (intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or abs(np.linalg.det(intrinsics)) <= 1e-12
            or abs(np.linalg.det(world_to_camera)) <= 1e-12):
        raise ValueError("Invalid camera focal length or singular pose")
    return intrinsics, world_to_camera, reprojection, version


def _single_geometry_version(versions):
    if len(versions) != 1:
        raise ValueError("Frames mix incompatible geometry schema versions")
    version = next(iter(versions))
    if version == 1:
        warnings.warn("Using SCARED geometry v1 compatibility mode (legacy geometry).",
                      RuntimeWarning, stacklevel=2)
    return version


class SCAREDDataset(DataLoader):
    """Read prepared SCARED frames; share cameras and initial sampling."""

    image_size = (1280, 1024)

    def __init__(self, datadir, downsample=1.0, test_every=8, **initialisation):
        self.skip_every = _scared_frame_stride(datadir)
        self.depth_far_threshold = 300.0
        self.depth_near_threshold = 0.03
        super().__init__(datadir, downsample, test_every, **initialisation)

    def load_meta(self):
        data_directory = os.path.join(self.root_dir, "data")
        calibration_directory = os.path.join(data_directory, "frame_data")
        rgb_directory = os.path.join(data_directory, "left_finalpass")
        disparity_directory = os.path.join(data_directory, "disparity")
        reprojection_directory = os.path.join(data_directory, "reprojection_data")

        frame_ids = sorted(
            Path(filename).stem
            for filename in os.listdir(calibration_directory)
            if filename.endswith(".json")
        )[:: self.skip_every]
        if not frame_ids:
            raise ValueError(f"No SCARED calibration frames found in {calibration_directory}.")

        rgbs = []
        bounds = []
        masks = []
        valid_depth_masks = []
        depths = []
        poses = []
        camera_matrices = []

        reference_pose = None
        geometry_versions = set()
        image_size = None
        for frame_number in trange(len(frame_ids), desc="Loading frames"):
            frame_id = frame_ids[frame_number]
            calibration_path = os.path.join(calibration_directory, f"{frame_id}.json")
            with open(calibration_path, "r", encoding="utf-8") as calibration_file:
                calibration = json.load(calibration_file)

            rgb_path = os.path.join(rgb_directory, f"{frame_id}.png")
            rgb = iio.imread(rgb_path)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"Invalid SCARED RGB image shape: {rgb_path}")

            disparity_path = os.path.join(disparity_directory, f"{frame_id}.tiff")
            disparity = iio.imread(disparity_path).astype(np.float32)
            if disparity.ndim != 2:
                raise ValueError(f"Invalid SCARED disparity shape: {disparity_path}")
            height, width = disparity.shape
            if rgb.shape[:2] != (height, width):
                raise ValueError(
                    f"SCARED RGB and disparity dimensions differ for {frame_id}."
                )
            frame_image_size = (width, height)
            if image_size is None:
                image_size = frame_image_size
            elif image_size != frame_image_size:
                raise ValueError("SCARED frames have inconsistent image dimensions.")
            rgbs.append(rgb)
            reprojection_path = os.path.join(
                reprojection_directory, f"{frame_id}.json"
            )
            with open(reprojection_path, "r", encoding="utf-8") as reprojection_file:
                geometry = json.load(reprojection_file)

            camera_intrinsics, world_to_camera, reprojection, version = _camera_geometry(
                calibration, geometry, frame_image_size
            )
            geometry_versions.add(version)

            camera_matrix = np.eye(4)
            camera_matrix[:3, :3] = camera_intrinsics
            camera_matrices.append(camera_matrix)

            camera_to_world = np.linalg.inv(world_to_camera)
            if reference_pose is None:
                reference_pose = camera_to_world
            poses.append(np.linalg.inv(reference_pose) @ camera_to_world)

            depth = _depth_from_rectified_disparity(disparity, reprojection)
            depth[depth > self.depth_far_threshold] = 0
            depth[depth < self.depth_near_threshold] = 0
            valid_depth = depth != 0
            if not valid_depth.any():
                raise ValueError(f"SCARED depth map contains no valid pixels: {disparity_path}")
            depths.append(depth)
            valid_depth_masks.append(valid_depth.astype(np.float32))

            kernel_size = max(1, int(width / 128))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            masks.append(
                cv2.morphologyEx(
                    valid_depth.astype(np.float32), cv2.MORPH_CLOSE, kernel
                )
            )
            bounds.append(np.array([depth[valid_depth].min(), depth[valid_depth].max()]))

        geometry_version = _single_geometry_version(geometry_versions)

        self.rgbs = np.stack(rgbs).astype(np.float32) / 255.0
        self.img_wh = image_size
        self.pose_mat = np.stack(poses).astype(np.float32)
        self.camera_mat = np.stack(camera_matrices).astype(np.float32)
        self.depths = np.stack(depths).astype(np.float32)
        self.masks = np.stack(masks).astype(np.float32)
        self.valid_depth_masks = np.stack(valid_depth_masks).astype(np.float32)
        self.bds = np.stack(bounds).astype(np.float32)
        self.times = np.linspace(0, 1, num=len(rgbs)).astype(np.float32)
        self.frame_ids = frame_ids
        self.geometry_version = geometry_version

        return len(self.rgbs)

    def camera_data(self, index):
        world_to_camera = np.linalg.inv(self.pose_mat[index])
        matrix = self.camera_mat[index]
        return dict(
            colour=self.rgbs[index], depth=self.depths[index], mask=self.masks[index],
            motion_mask=self.valid_depth_masks[index],
            R=np.transpose(world_to_camera[:3, :3]), T=world_to_camera[:3, -1],
            focal=(matrix[0, 0], matrix[1, 1]), time=self.times[index],
            Znear=self.depth_near_threshold, Zfar=self.depth_far_threshold,
            principal_x=float(matrix[0, 2]), principal_y=float(matrix[1, 2]),
        )

    def _frame_data_for_initialisation(self, index):
        colour = self.rgbs[index]
        depth = self.depths[index]
        mask = (
            self.masks[index].astype(bool)
            & (depth > self.depth_near_threshold)
            & (depth < self.depth_far_threshold)
        )
        camera_points, colours, flattened_mask = self.get_camera_points(
            depth,
            mask,
            colour,
            self.camera_mat[index, :3, :3],
        )
        return colour, mask, flattened_mask, camera_points, colours, self.pose_mat[index]

    @staticmethod
    def get_world_points(points, camera_to_world):
        homogeneous = np.concatenate(
            (points, np.ones((points.shape[0], 1))), axis=-1
        )
        return (camera_to_world @ homogeneous.T).T[:, :3]

    def get_camera_points(self, depth, mask, colour, camera_intrinsics):
        width, height = self.img_wh
        horizontal, vertical = np.meshgrid(
            np.arange(width),
            np.arange(height),
        )
        pixels = np.stack(
            (horizontal, vertical, np.ones_like(horizontal)),
            axis=-1,
        )
        rays = pixels @ np.linalg.inv(camera_intrinsics).T
        camera_points = (rays * depth[..., None]).reshape(-1, 3)
        colours = colour.reshape(-1, 3)
        flattened_mask = mask.reshape(-1).astype(bool)
        return (
            camera_points[flattened_mask],
            colours[flattened_mask],
            flattened_mask,
        )


def detect_scared_geometry_version(source_directory):
    """Read version tags without duplicating camera geometry validation."""
    directory = Path(source_directory) / "data" / "reprojection_data"
    files = sorted(directory.glob("*.json"))
    if not files:
        raise ValueError(f"Missing SCARED reprojection metadata: {directory}")
    versions = {_geometry_version(json.loads(path.read_text())) for path in files}
    if len(versions) != 1:
        raise ValueError("Mixed SCARED geometry schema versions")
    return versions.pop()


# Offline preprocessing: raw stereo video and scene points to rectified frames.

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1280
FRAME_PATTERN = re.compile(r"^frame_data\d{6}$")
GEOMETRY_SCHEMA_VERSION = 2


def split_stereo_frame(frame, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH):
    """Split a vertically stacked SCARED frame into left and right images."""
    expected_shape = (image_height * 2, image_width, 3)
    if frame.shape != expected_shape:
        raise ValueError(
            f"expected stacked stereo frame with shape {expected_shape}, got {frame.shape}"
        )
    return frame[:image_height].copy(), frame[image_height:].copy()


def project_scene_points_to_disparity(
    scene_points,
    left_rectification_rotation,
    left_projection_matrix,
    right_projection_matrix,
):
    """Project raw left-camera points into the rectified stereo pair."""
    points = np.asarray(scene_points)
    rotation = np.asarray(left_rectification_rotation, dtype=np.float64)
    left_projection = np.asarray(left_projection_matrix, dtype=np.float64)
    right_projection = np.asarray(right_projection_matrix, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 scene-point array, got {points.shape}")
    if rotation.shape != (3, 3):
        raise ValueError("left rectification rotation must have shape (3, 3)")
    if left_projection.shape != (3, 4) or right_projection.shape != (3, 4):
        raise ValueError("stereo projection matrices must have shape (3, 4)")
    if not all(
        np.all(np.isfinite(matrix))
        for matrix in (rotation, left_projection, right_projection)
    ):
        raise ValueError("rectification metadata contains non-finite values")

    height, width = points.shape[:2]
    flattened = points.reshape(-1, 3).astype(np.float64, copy=False)
    valid = np.all(np.isfinite(flattened), axis=1) & (flattened[:, 2] > 0)
    candidates = flattened[valid] @ rotation.T

    output = np.zeros((height, width), dtype=np.float32)
    if candidates.size == 0:
        return output

    homogeneous = np.concatenate(
        (candidates, np.ones((candidates.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    left_homogeneous = homogeneous @ left_projection.T
    right_homogeneous = homogeneous @ right_projection.T
    left_z = left_homogeneous[:, 2]
    right_z = right_homogeneous[:, 2]
    nonzero_depth = (np.abs(left_z) > 1e-12) & (np.abs(right_z) > 1e-12)
    pixel_x = np.full(left_z.shape, np.nan, dtype=np.float64)
    pixel_y = np.full(left_z.shape, np.nan, dtype=np.float64)
    right_x = np.full(right_z.shape, np.nan, dtype=np.float64)
    pixel_x[nonzero_depth] = left_homogeneous[nonzero_depth, 0] / left_z[nonzero_depth]
    pixel_y[nonzero_depth] = left_homogeneous[nonzero_depth, 1] / left_z[nonzero_depth]
    right_x[nonzero_depth] = (
        right_homogeneous[nonzero_depth, 0] / right_z[nonzero_depth]
    )
    disparity = pixel_x - right_x
    rounded_x = np.rint(pixel_x)
    rounded_y = np.rint(pixel_y)
    projected = (
        np.isfinite(disparity)
        & np.isfinite(rounded_x)
        & np.isfinite(rounded_y)
        & (disparity > 0)
        & (rounded_x >= 0)
        & (rounded_x < width)
        & (rounded_y >= 0)
        & (rounded_y < height)
    )
    if not np.any(projected):
        return output

    x_index = rounded_x[projected].astype(np.intp)
    y_index = rounded_y[projected].astype(np.intp)
    flat_index = y_index * width + x_index
    flat_size = height * width
    disparity_sum = np.bincount(
        flat_index, weights=disparity[projected], minlength=flat_size
    )
    observation_count = np.bincount(flat_index, minlength=flat_size)
    observed = observation_count > 0
    flat_output = output.reshape(-1)
    flat_output[observed] = (
        disparity_sum[observed] / observation_count[observed]
    ).astype(np.float32)
    return output


def _validate_flat_tar_members(archive, expected_pattern):
    members = archive.getmembers()
    if not members:
        raise ValueError(f"{archive.name}: empty nested archive")

    validated = []
    for member in members:
        raw_name = member.name
        if "\\" in raw_name:
            raise ValueError(f"unsafe archive member: {raw_name}")
        path = PurePosixPath(raw_name)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
            raise ValueError(f"unsafe archive member: {raw_name}")
        if not member.isreg() or expected_pattern.fullmatch(path.name) is None:
            raise ValueError(f"unsafe archive member: {raw_name}")
        validated.append((member, path.name))
    return validated


def _extract_flat_tar(archive_path, output_directory, expected_pattern, overwrite=False):
    archive_path = Path(archive_path)
    output_directory = Path(output_directory)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    with tarfile.open(archive_path, "r:gz") as archive:
        members = _validate_flat_tar_members(archive, expected_pattern)
        if output_directory.is_symlink():
            raise ValueError(
                f"refusing symbolic-link output directory: {output_directory}"
            )
        output_directory.mkdir(parents=True, exist_ok=True)
        extracted = 0
        for member, basename in members:
            destination = output_directory / basename
            if destination.exists() and not overwrite:
                if destination.is_file() and destination.stat().st_size == member.size:
                    continue
                raise FileExistsError(
                    f"refusing to replace existing file with a different size: {destination}"
                )

            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"could not read archive member: {member.name}")
            temporary_name = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=output_directory, prefix=f".{basename}.", delete=False
                ) as temporary:
                    temporary_name = temporary.name
                    shutil.copyfileobj(source, temporary, length=1024 * 1024)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_name, destination)
                temporary_name = None
                extracted += 1
            finally:
                source.close()
                if temporary_name is not None:
                    Path(temporary_name).unlink(missing_ok=True)
    return extracted


def extract_nested_inputs(keyframe_directory, overwrite=False):
    """Safely extract SCARED metadata and scene-point nested archives."""
    keyframe = Path(keyframe_directory)
    data = keyframe / "data"
    extracted = _extract_flat_tar(
        data / "frame_data.tar.gz",
        data / "frame_data",
        re.compile(r"^frame_data\d{6}\.json$"),
        overwrite=overwrite,
    )
    extracted += _extract_flat_tar(
        data / "scene_points.tar.gz",
        data / "scene_points",
        re.compile(r"^scene_points\d{6}\.tiff$"),
        overwrite=overwrite,
    )
    return extracted


def _frame_identifiers(directory, extension):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"missing processed data directory: {directory}")
    identifiers = {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix == extension and FRAME_PATTERN.fullmatch(path.stem)
    }
    if not identifiers:
        raise ValueError(f"no processed frames found in: {directory}")
    return identifiers


def validate_processed_layout(keyframe_directory):
    """Validate the file-level contract consumed by ``SCAREDDataset``."""
    keyframe = Path(keyframe_directory)
    data = keyframe / "data"
    sets = {
        "frame_data": _frame_identifiers(data / "frame_data", ".json"),
        "left_finalpass": _frame_identifiers(data / "left_finalpass", ".png"),
        "disparity": _frame_identifiers(data / "disparity", ".tiff"),
        "reprojection_data": _frame_identifiers(data / "reprojection_data", ".json"),
    }
    reference = sets["frame_data"]
    if any(identifiers != reference for identifiers in sets.values()):
        summary = ", ".join(f"{name}={len(value)}" for name, value in sets.items())
        raise ValueError(f"processed frame identifiers do not match ({summary})")
    return sorted(reference)


def validate_processed_content(keyframe_directory, image_height=IMAGE_HEIGHT,
                               image_width=IMAGE_WIDTH, cv2_module=None, tiff_module=None):
    """Check prepared frames using the same geometry decoder as training."""
    cv2_module = cv2 if cv2_module is None else cv2_module
    tiff_module = tifffile if tiff_module is None else tiff_module
    keyframe = Path(keyframe_directory)
    frames = validate_processed_layout(keyframe)
    data = keyframe / "data"
    positive_pixels = total_pixels = 0
    minimum_positive, maximum_disparity = np.inf, 0.0
    versions = set()
    for frame in frames:
        image = cv2_module.imread(str(data / "left_finalpass" / f"{frame}.png"), cv2_module.IMREAD_COLOR)
        disparity = np.asarray(tiff_module.imread(str(data / "disparity" / f"{frame}.tiff")))
        if image is None or image.shape != (image_height, image_width, 3):
            raise ValueError(f"Invalid image shape: {frame}")
        if disparity.shape != (image_height, image_width) or disparity.dtype != np.float32:
            raise ValueError(f"Expected float32 disparity at the image resolution: {frame}")
        if not np.isfinite(disparity).all():
            raise ValueError(f"Non-finite disparity: {frame}")
        positive = disparity > 0
        if not positive.any():
            raise ValueError(f"no positive disparity values: {frame}")
        positive_pixels += int(positive.sum())
        total_pixels += disparity.size
        minimum_positive = min(minimum_positive, float(disparity[positive].min()))
        maximum_disparity = max(maximum_disparity, float(disparity.max()))
        calibration = json.loads((data / "frame_data" / f"{frame}.json").read_text())
        geometry = json.loads((data / "reprojection_data" / f"{frame}.json").read_text())
        _, _, _, version = _camera_geometry(calibration, geometry, (image_width, image_height))
        versions.add(version)
    return dict(frames=len(frames), positive_ratio=positive_pixels / total_pixels,
                disparity_min=minimum_positive, disparity_max=maximum_disparity,
                geometry_version=_single_geometry_version(versions))


def _load_stereo_calibration(metadata_path):
    with Path(metadata_path).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    calibration = metadata["camera-calibration"]
    left_camera = np.asarray(calibration["KL"], dtype=np.float64)
    right_camera = np.asarray(calibration["KR"], dtype=np.float64)
    left_distortion = np.asarray(calibration["DL"], dtype=np.float64)
    right_distortion = np.asarray(calibration["DR"], dtype=np.float64)
    rotation = np.asarray(calibration["R"], dtype=np.float64)
    translation = np.asarray(calibration["T"], dtype=np.float64).reshape(3, 1)
    return (
        left_camera,
        left_distortion,
        right_camera,
        right_distortion,
        rotation,
        translation,
    )


def _rectify_left_image(left_image, metadata_path, cv2_module):
    (
        left_camera,
        left_distortion,
        right_camera,
        right_distortion,
        rotation,
        translation,
    ) = _load_stereo_calibration(metadata_path)
    image_size = (left_image.shape[1], left_image.shape[0])
    rectification = cv2_module.stereoRectify(
        left_camera,
        left_distortion,
        right_camera,
        right_distortion,
        image_size,
        rotation,
        translation,
        flags=cv2_module.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    left_rotation, _, left_projection, right_projection, reprojection = (
        rectification[:5]
    )
    left_map_x, left_map_y = cv2_module.initUndistortRectifyMap(
        left_camera,
        left_distortion,
        left_rotation,
        left_projection,
        image_size,
        cv2_module.CV_32FC1,
    )
    left_rectified = cv2_module.remap(
        left_image, left_map_x, left_map_y, cv2_module.INTER_LINEAR
    )
    geometry = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "image_size": list(image_size),
        "left_rectification_rotation": np.asarray(
            left_rotation, dtype=np.float64
        ),
        "left_projection_matrix": np.asarray(left_projection, dtype=np.float64),
        "right_projection_matrix": np.asarray(
            right_projection, dtype=np.float64
        ),
        "reprojection_matrix": np.asarray(reprojection, dtype=np.float64),
    }
    return left_rectified, geometry


def _ordered_frame_inputs(data_directory):
    frame_data = Path(data_directory) / "frame_data"
    scene_points = Path(data_directory) / "scene_points"
    metadata_paths = sorted(frame_data.glob("frame_data*.json"))
    if not metadata_paths:
        raise ValueError(f"no SCARED frame metadata found in: {frame_data}")

    ordered = []
    for expected_index, metadata_path in enumerate(metadata_paths):
        frame_identifier = f"frame_data{expected_index:06d}"
        if metadata_path.stem != frame_identifier:
            raise ValueError(
                f"SCARED frame metadata must be contiguous from zero; expected "
                f"{frame_identifier}.json, got {metadata_path.name}"
            )
        scene_path = scene_points / f"scene_points{expected_index:06d}.tiff"
        if not scene_path.is_file():
            raise ValueError(f"missing scene-point TIFF for {frame_identifier}: {scene_path}")
        ordered.append((frame_identifier, metadata_path, scene_path))

    extra_scene_points = {
        path.name for path in scene_points.glob("scene_points*.tiff")
    } - {path.name for _, _, path in ordered}
    if extra_scene_points:
        raise ValueError(
            "scene-point frame identifiers do not match metadata: "
            + ", ".join(sorted(extra_scene_points)[:5])
        )
    return ordered


def _atomic_write(path, write):
    """Write each generated file completely before replacing its destination."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(descriptor)
    try:
        if write(temporary) is False:
            raise OSError(f"Could not write {path}")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def preprocess_keyframe(
    keyframe_directory,
    image_height=IMAGE_HEIGHT,
    image_width=IMAGE_WIDTH,
    overwrite=False,
    cv2_module=None,
    tiff_module=None,
):
    """Convert one extracted raw SCARED keyframe into the loader contract."""
    if cv2_module is None:
        cv2_module = cv2
    if tiff_module is None:
        tiff_module = tifffile

    keyframe = Path(keyframe_directory)
    data = keyframe / "data"
    nested_archives = (
        data / "frame_data.tar.gz",
        data / "scene_points.tar.gz",
    )
    if all(path.is_file() for path in nested_archives):
        extract_nested_inputs(keyframe, overwrite=overwrite)
    elif not (data / "frame_data").is_dir() or not (data / "scene_points").is_dir():
        missing = ", ".join(str(path) for path in nested_archives if not path.is_file())
        raise FileNotFoundError(f"missing nested SCARED archives: {missing}")
    inputs = _ordered_frame_inputs(data)
    video_path = data / "rgb.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    left_output = data / "left_finalpass"
    disparity_output = data / "disparity"
    reprojection_output = data / "reprojection_data"
    for directory in (
        left_output,
        disparity_output,
        reprojection_output,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    capture = cv2_module.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"OpenCV could not open SCARED video: {video_path}")

    processed_frames = 0
    valid_disparity_frames = 0
    try:
        for frame_identifier, metadata_path, scene_path in inputs:
            success, stacked_frame = capture.read()
            if not success:
                break

            left_raw, _ = split_stereo_frame(
                stacked_frame,
                image_height=image_height,
                image_width=image_width,
            )
            left_rectified, geometry = _rectify_left_image(
                left_raw, metadata_path, cv2_module
            )
            scene = np.asarray(tiff_module.imread(str(scene_path)))
            expected_scene_shape = (image_height * 2, image_width, 3)
            if scene.shape != expected_scene_shape:
                raise ValueError(
                    f"expected stacked scene points with shape {expected_scene_shape}, "
                    f"got {scene.shape} in {scene_path}"
                )
            left_scene_points = scene[:image_height]
            disparity = project_scene_points_to_disparity(
                left_scene_points,
                geometry["left_rectification_rotation"],
                geometry["left_projection_matrix"],
                geometry["right_projection_matrix"],
            )

            output_paths = (
                left_output / f"{frame_identifier}.png",
                disparity_output / f"{frame_identifier}.tiff",
                reprojection_output / f"{frame_identifier}.json",
            )
            existing = [path for path in output_paths if path.exists()]
            if existing and not overwrite:
                raise FileExistsError(
                    "refusing to mix existing and newly processed frames; use --overwrite: "
                    + ", ".join(str(path) for path in existing)
                )

            _atomic_write(output_paths[0], lambda path: cv2_module.imwrite(path, left_rectified))
            _atomic_write(output_paths[1], lambda path: tiff_module.imwrite(path, disparity.astype(np.float32)))
            payload = {key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in geometry.items()}
            _atomic_write(output_paths[2], lambda path: Path(path).write_text(json.dumps(payload, indent=2) + "\n"))
            processed_frames += 1
            valid_disparity_frames += int(np.any(disparity > 0))

        extra_success, _ = capture.read()
    finally:
        capture.release()

    if processed_frames != len(inputs):
        raise ValueError(
            f"video contains {processed_frames} frames but metadata contains {len(inputs)}"
        )
    if extra_success:
        raise ValueError(
            f"video contains more than {len(inputs)} frames but metadata contains {len(inputs)}"
        )
    validate_processed_layout(keyframe)
    return {
        "frames": processed_frames,
        "valid_disparity_frames": valid_disparity_frames,
    }


def _build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert extracted SCARED keyframes into the layout consumed by "
            "EndoPrior-GS."
        )
    )
    parser.add_argument(
        "keyframes",
        nargs="+",
        type=Path,
        help="one or more extracted dataset_#/keyframe_1 directories",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing extracted and processed frame files",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--extract-only",
        action="store_true",
        help="safely extract the nested metadata and scene-point archives only",
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an already processed keyframe without modifying it",
    )
    return parser


def main(arguments=None):
    options = _build_argument_parser().parse_args(arguments)
    for keyframe in options.keyframes:
        keyframe = keyframe.resolve()
        if options.validate_only:
            summary = validate_processed_content(keyframe)
            print(
                f"{keyframe}: validated {summary['frames']} frames; "
                f"positive disparity ratio={summary['positive_ratio']:.6f}, "
                f"range={summary['disparity_min']:.6f}.."
                f"{summary['disparity_max']:.6f}, "
                f"geometry schema=v{summary['geometry_version']}"
            )
        elif options.extract_only:
            extracted = extract_nested_inputs(keyframe, overwrite=options.overwrite)
            print(f"{keyframe}: extracted {extracted} nested archive members")
        else:
            summary = preprocess_keyframe(keyframe, overwrite=options.overwrite)
            print(
                f"{keyframe}: processed {summary['frames']} frames "
                f"({summary['valid_disparity_frames']} with valid disparity)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
