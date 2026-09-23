"""Read and prepare StereoMIS P2_7/P2_8 with shared camera/point loading.

Preserves the original 150-frame protocol, rectification, mask polarity,
pose units and Omnidata relative-depth conversion. No training runs here.
"""

import argparse
import configparser
import csv
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from scene.data_loader import ImageDepthDataset


SEQUENCES = {"P2_7": (7665, 37.0), "P2_8": (0, 39.0)}
IMAGE_SIZE = (640, 512)


class StereoMISDataset(ImageDepthDataset):
    """Read prepared StereoMIS frames using their recorded train/test split."""

    def __init__(self, datadir, downsample=1.0, test_every=8, **initialisation):
        super().__init__(datadir, downsample, test_every, **initialisation)
        split = json.loads((Path(datadir) / "split.json").read_text())
        self.train_idxs = split["train_indices"]
        self.test_idxs = split["test_indices"]
        indices = self.train_idxs + self.test_idxs
        if sorted(indices) != self.video_idxs:
            raise ValueError("StereoMIS split must cover every frame exactly once")


def frame_selection(sequence):
    """Select five seconds at 30 fps; hold out indices 1, 9, 17, ... ."""
    start, fps = SEQUENCES[sequence]
    frames = [start + math.floor(i * fps / 30.0 + 0.5) for i in range(150)]
    test = list(range(1, 150, 8))
    train = [i for i in range(150) if i not in test]
    return frames, {"sequence": sequence, "train_indices": train, "test_indices": test}


def rectification_maps(calibration):
    """Read StereoMIS INI calibration using the original resize/crop convention."""
    config = configparser.ConfigParser()
    if not config.read(calibration):
        raise FileNotFoundError(calibration)
    left, right = config["StereoLeft"], config["StereoRight"]
    scale = IMAGE_SIZE[0] / float(left["res_x"])
    crop = int((float(left["res_y"]) * scale - IMAGE_SIZE[1]) / 2)
    if crop < 0:
        raise ValueError("Calibration does not support the 640x512 vertical crop.")
    intrinsics, distortion = [], []
    for camera in (left, right):
        matrix = np.array([[float(camera["fc_x"]), 0, float(camera["cc_x"])],
                           [0, float(camera["fc_y"]), float(camera["cc_y"])],
                           [0, 0, 1]])
        matrix[:2] *= scale
        matrix[1, 2] -= crop
        intrinsics.append(matrix)
        distortion.append(np.array([float(camera[f"kc_{i}"]) for i in range(8)]))
    rotation = np.array([float(right[f"R_{i}"]) for i in range(9)]).reshape(3, 3)
    translation = np.array([float(right[f"T_{i}"]) for i in range(3)])
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        intrinsics[0], distortion[0], intrinsics[1], distortion[1],
        IMAGE_SIZE, rotation, translation, alpha=0,
    )
    maps = [cv2.initUndistortRectifyMap(k, d, r, p, IMAGE_SIZE, cv2.CV_32FC1)
            for k, d, r, p in zip(intrinsics, distortion, (r1, r2), (p1, p2))]
    return maps, p1[:3, :3]


def pose_bounds(rows, frames, focal):
    """Keep camera-to-world poses; convert metre translations to millimetres."""
    by_frame = {int(row[0]): row for row in np.atleast_2d(rows)}
    poses = []
    for frame in frames:
        row = by_frame[frame]
        quaternion = row[4:8]
        if np.linalg.norm(quaternion) == 0:
            raise ValueError(f"Zero quaternion at frame {frame}")
        x, y, z, w = quaternion / np.linalg.norm(quaternion)
        pose = np.zeros((3, 5))
        pose[:, :3] = [
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ]
        pose[:, 3] = row[1:4] * 1000.0
        pose[:, 4] = [IMAGE_SIZE[1], IMAGE_SIZE[0], focal]
        poses.append(np.concatenate((pose.reshape(-1), [0.0, 200.0])))
    return np.stack(poses)


def extract_frames(source, destination, sequence):
    """Rectify left RGB and masks and write the original pose/split metadata."""
    frames, split = frame_selection(sequence)
    maps, intrinsics = rectification_maps(source / "StereoCalibration.ini")
    videos = sorted(source.glob("*.mp4"))
    if len(videos) != 1:
        raise ValueError(f"Expected one vertically stacked stereo MP4 in {source}")
    for name in ("images", "masks"):
        (destination / name).mkdir()
    capture = cv2.VideoCapture(str(videos[0]))
    selected = {frame: i for i, frame in enumerate(frames)}
    count = 0
    try:
        for frame in tqdm(range(frames[-1] + 1), desc=f"Extract {sequence}"):
            ok, stacked = capture.read()
            if not ok:
                break
            if frame not in selected:
                continue
            left = cv2.resize(stacked[:stacked.shape[0] // 2], IMAGE_SIZE,
                              interpolation=cv2.INTER_AREA)
            left = cv2.remap(left, *maps[0], interpolation=cv2.INTER_CUBIC)
            mask_path = source / "masks" / f"{frame + 1:06d}l.png"
            tissue = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if tissue is None:
                raise FileNotFoundError(mask_path)
            tissue = cv2.resize(tissue, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
            tissue = cv2.remap(tissue, *maps[0], interpolation=cv2.INTER_NEAREST)
            tool_mask = 255 - np.where(tissue > 127, 255, 0).astype(np.uint8)
            name = f"{selected[frame]:06d}.png"
            for folder, array in (("images", left), ("masks", tool_mask)):
                if not cv2.imwrite(str(destination / folder / name), array):
                    raise OSError(f"Failed to write {folder}/{name}")
            count += 1
    finally:
        capture.release()
    if count != len(frames):
        raise ValueError(f"Extracted {count}/{len(frames)} frames")
    rows = np.loadtxt(source / "groundtruth.txt", dtype=np.float64)
    focal = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2)
    np.save(destination / "poses_bounds.npy", pose_bounds(rows, frames, focal))
    np.save(destination / "intrinsics.npy", intrinsics)
    (destination / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    with (destination / "frame_mapping.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("output_index", "source_frame", "split"))
        for i, frame in enumerate(frames):
            writer.writerow((i, frame, "test" if i in split["test_indices"] else "train"))


def generate_depth(destination, device, model_repository=None):
    """Reproduce Omnidata relative depth, not measured metric ground truth."""
    repository = str(model_repository) if model_repository else "alexsax/omnidata_models"
    source = "local" if model_repository else "github"
    model = torch.hub.load(repository, "depth_dpt_hybrid_384", source=source,
                           trust_repo=True).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize(384, interpolation=Image.Resampling.BILINEAR),
        transforms.CenterCrop(384), transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])
    (destination / "depth").mkdir()
    with torch.inference_mode():
        for path in tqdm(sorted((destination / "images").glob("*.png")), desc="Depth"):
            with Image.open(path) as image:
                tensor = transform(image.convert("RGB"))[:3].unsqueeze(0).to(device)
            prediction = model(tensor).clamp(0.0, 1.0)
            if prediction.ndim == 3:
                prediction = prediction.unsqueeze(1)
            prediction = F.interpolate(prediction, size=(512, 640), mode="bicubic",
                                       align_corners=False)[0, 0].cpu().numpy()
            if not np.isfinite(prediction).all():
                raise ValueError(f"Non-finite depth prediction: {path}")
            depth = np.rint(255 * (1 - np.clip(prediction, 0, 1))).astype(np.uint8)
            Image.fromarray(depth).save(destination / "depth" / path.name)


def link_prepared(source, destination):
    """Reuse existing preparation without regenerating images or depth."""
    mask_folder = "tool_masks" if (source / "tool_masks").is_dir() else "masks"
    for name in ("images", "depth", "masks", "poses_bounds.npy", "intrinsics.npy",
                 "split.json", "frame_mapping.csv"):
        target = source / (mask_folder if name == "masks" else name)
        if not target.exists():
            raise FileNotFoundError(target)
        (destination / name).symlink_to(os.path.relpath(target, destination),
                                       target_is_directory=target.is_dir())


def validate_sequence(directory, sequence):
    """Check the frozen frame selection, split, image layout and mask polarity."""
    frames, split = frame_selection(sequence)
    saved = json.loads((directory / "split.json").read_text())
    for key in split:
        if saved[key] != split[key]:
            raise ValueError(f"Unexpected StereoMIS {key}: {directory}")
    with (directory / "frame_mapping.csv").open() as stream:
        mapping = list(csv.DictReader(stream))
    expected = [(str(i), str(frame), "test" if i in split["test_indices"] else "train")
                for i, frame in enumerate(frames)]
    if [(r['output_index'], r['source_frame'], r['split']) for r in mapping] != expected:
        raise ValueError("Frame mapping differs from the original StereoMIS protocol")
    poses = np.load(directory / "poses_bounds.npy")
    if poses.shape != (150, 17) or not np.isfinite(poses).all():
        raise ValueError("Expected 150 finite camera poses")
    names = [f"{i:06d}.png" for i in range(150)]
    for folder in ("images", "depth", "masks"):
        if sorted(p.name for p in (directory / folder).glob('*.png')) != names:
            raise ValueError(f"Missing or extra frames in {folder}")
        for name in names:
            with Image.open(directory / folder / name) as image:
                if image.size != IMAGE_SIZE:
                    raise ValueError(f"Expected 640x512: {folder}/{name}")
                array = np.asarray(image)
            if folder == "masks" and (array.ndim != 2 or not np.isin(array, [0, 255]).all()):
                raise ValueError(f"Expected a binary tool mask: {name}")
            if folder == "depth" and (array.ndim != 2 or not (array > 0).any()):
                raise ValueError(f"Invalid depth: {name}")
    print(f"{sequence}: 150 frames, 131 train / 19 test; data ready at {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=SEQUENCES, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--raw", type=Path, help="Raw sequence folder containing MP4, calibration, poses and masks")
    inputs.add_argument("--prepared", type=Path, help="Existing common data or EndoPrior data view; reuse via symbolic links")
    inputs.add_argument("--validate", type=Path, help="Check an existing prepared sequence without writing")
    parser.add_argument("--output", type=Path, help="New or empty destination sequence folder")
    parser.add_argument("--device", default="cuda:0", help="Device for Omnidata preprocessing only")
    parser.add_argument("--omnidata-repository", type=Path, help="Optional local Omnidata Torch Hub checkout")
    args = parser.parse_args()
    if args.validate:
        validate_sequence(args.validate, args.sequence)
        return
    if args.output is None:
        parser.error("--output is required for data preparation")
    destination = args.output.resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Use a new or empty destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if args.prepared:
        link_prepared(args.prepared.resolve(), destination)
    else:
        extract_frames(args.raw.resolve(), destination, args.sequence)
        generate_depth(destination, args.device, args.omnidata_repository)
    validate_sequence(destination, args.sequence)


if __name__ == "__main__":
    main()
