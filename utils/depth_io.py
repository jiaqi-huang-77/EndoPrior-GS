"""Lossless depth-map I/O and masked depth evaluation utilities."""

from pathlib import Path
import warnings

import numpy as np
from PIL import Image


def _as_depth_array(depth):
    """Return a two-dimensional float32 depth array."""
    if hasattr(depth, "detach"):
        depth = depth.detach()
    if hasattr(depth, "cpu"):
        depth = depth.cpu()
    if hasattr(depth, "numpy"):
        depth = depth.numpy()

    array = np.asarray(depth)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(
            "A depth map must be two-dimensional after removing singleton "
            "batch or channel dimensions; received shape {}.".format(array.shape)
        )
    return np.asarray(array, dtype=np.float32)


def save_depth(path, depth):
    """Save a depth map as a lossless float32 NumPy array."""
    path = Path(path)
    if path.suffix.lower() != ".npy":
        raise ValueError("Depth maps must be saved with a .npy extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, _as_depth_array(depth), allow_pickle=False)


def load_depth_for_frame(depth_dir, frame_name):
    """Load depth by render-frame stem, preferring lossless NumPy files."""
    depth_dir = Path(depth_dir)
    stem = Path(frame_name).stem
    npy_path = depth_dir / (stem + ".npy")
    if npy_path.is_file():
        return _as_depth_array(np.load(npy_path, allow_pickle=False))

    legacy_path = depth_dir / (stem + ".png")
    if legacy_path.is_file():
        warnings.warn(
            "Lossless depth file {} is missing; using legacy PNG {}. "
            "Legacy PNG depth values may be quantised.".format(
                npy_path,
                legacy_path,
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        with Image.open(legacy_path) as image:
            return _as_depth_array(np.asarray(image).copy())

    raise FileNotFoundError(
        "No depth map found for frame {!r}; expected {} or {}.".format(
            frame_name,
            npy_path,
            legacy_path,
        )
    )


def masked_depth_rmse(prediction, ground_truth, tissue_mask):
    """Compute RMSE over finite, valid ground-truth tissue pixels only."""
    prediction = _as_depth_array(prediction)
    ground_truth = _as_depth_array(ground_truth)
    tissue_mask = _as_depth_array(tissue_mask)
    if prediction.shape != ground_truth.shape or prediction.shape != tissue_mask.shape:
        raise ValueError(
            "Prediction, ground truth and tissue mask must have identical shapes; "
            "received {}, {} and {}.".format(
                prediction.shape,
                ground_truth.shape,
                tissue_mask.shape,
            )
        )

    valid = (
        np.isfinite(prediction)
        & np.isfinite(ground_truth)
        & (ground_truth > 0.0)
        & (tissue_mask > 0.5)
    )
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return float("nan"), 0

    residual = prediction[valid].astype(np.float64) - ground_truth[valid].astype(
        np.float64
    )
    return float(np.sqrt(np.mean(np.square(residual)))), valid_count


def masked_adjacent_depth_instability(
    current_depth,
    previous_depth,
    current_tissue_mask,
    previous_tissue_mask,
):
    """Compute mean absolute rendered-depth change on shared valid tissue."""
    current_depth = _as_depth_array(current_depth)
    previous_depth = _as_depth_array(previous_depth)
    current_tissue_mask = _as_depth_array(current_tissue_mask)
    previous_tissue_mask = _as_depth_array(previous_tissue_mask)

    arrays = (
        current_depth,
        previous_depth,
        current_tissue_mask,
        previous_tissue_mask,
    )
    if any(array.shape != current_depth.shape for array in arrays[1:]):
        raise ValueError(
            "Current depth, previous depth and both tissue masks must have "
            "identical shapes; received {}, {}, {} and {}.".format(
                *(array.shape for array in arrays)
            )
        )

    valid = (
        np.isfinite(current_depth)
        & np.isfinite(previous_depth)
        & (current_depth > 0.0)
        & (previous_depth > 0.0)
        & (current_tissue_mask > 0.5)
        & (previous_tissue_mask > 0.5)
    )
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return float("nan"), 0

    difference = np.abs(
        current_depth[valid].astype(np.float64)
        - previous_depth[valid].astype(np.float64)
    )
    return float(np.mean(difference)), valid_count


def directory_depth_temporal_instability(depth_dir, masks_dir):
    """Evaluate adjacent rendered depths stored in chronological mask order."""
    depth_dir = Path(depth_dir)
    masks_dir = Path(masks_dir)
    mask_names = sorted(
        path.name
        for path in masks_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not mask_names:
        raise ValueError(
            "Depth temporal evaluation requires at least one tissue mask in {}."
            .format(masks_dir)
        )

    valid_pair_values = []
    per_view_values = {mask_names[0]: 0.0}
    previous_name = mask_names[0]
    previous_depth = load_depth_for_frame(depth_dir, previous_name)
    with Image.open(masks_dir / previous_name) as image:
        previous_mask = (
            np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        )

    for frame_name in mask_names[1:]:
        current_depth = load_depth_for_frame(depth_dir, frame_name)
        with Image.open(masks_dir / frame_name) as image:
            current_mask = (
                np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            )
        value, valid_count = masked_adjacent_depth_instability(
            current_depth,
            previous_depth,
            current_mask,
            previous_mask,
        )
        if valid_count > 0:
            valid_pair_values.append(value)
            per_view_values[frame_name] = value
        else:
            per_view_values[frame_name] = None
        previous_depth = current_depth
        previous_mask = current_mask

    aggregate = (
        float(np.mean(valid_pair_values)) if valid_pair_values else None
    )
    return aggregate, per_view_values
