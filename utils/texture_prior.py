"""Joint texture prior, persistent Gaussian scores and temporal weights."""


import cv2
import numpy as np
import torch
import torch.nn.functional as F


def previous_contiguous_training_frame_time(
    current_frame_id,
    training_frame_ids,
    video_frame_times,
):
    """Return the true preceding source-frame time without test-time leakage.

    A reference is valid only when the immediately preceding source frame is
    also in the training split.  This preserves real video adjacency while
    avoiding an optimisation constraint at a held-out timestamp.
    """

    current_frame_id = int(current_frame_id)
    training_frame_ids = {int(frame_id) for frame_id in training_frame_ids}
    video_frame_times = {
        int(frame_id): float(frame_time)
        for frame_id, frame_time in video_frame_times.items()
    }
    if current_frame_id not in video_frame_times:
        raise ValueError(
            f"Current frame {current_frame_id} is absent from the video timeline"
        )
    if current_frame_id not in training_frame_ids:
        raise ValueError(f"Current frame {current_frame_id} is not a training frame")

    previous_frame_id = current_frame_id - 1
    if previous_frame_id < 0 or previous_frame_id not in training_frame_ids:
        return None
    if previous_frame_id not in video_frame_times:
        raise ValueError(
            f"Previous frame {previous_frame_id} is absent from the video timeline"
        )
    return video_frame_times[previous_frame_id]


def _as_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(bool)


def gradient_confidence_np(colour, mask=None, percentile=99.0):
    """Return a robust [0, 1] Sobel confidence map for RGB images in [0, 1]."""
    try:
        gray = cv2.cvtColor(
            (np.clip(colour, 0.0, 1.0) * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
    except Exception:
        gray = np.mean(np.asarray(colour, dtype=np.float32), axis=-1)
        gy, gx = np.gradient(gray)
        grad = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)

    valid = _as_mask(mask) if mask is not None else np.ones_like(grad, dtype=bool)
    vals = grad[valid & np.isfinite(grad)]
    if vals.size == 0:
        return np.zeros_like(grad, dtype=np.float32)

    scale = np.percentile(vals, percentile)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = np.max(vals) + 1e-8
    return np.clip(grad / (scale + 1e-8), 0.0, 1.0).astype(np.float32)


def specular_confidence_np(
    colour,
    mask=None,
    min_v=0.85,
    s_threshold=0.35,
    brightness_percentile=97.0,
    v_softness=0.04,
    s_softness=0.08,
):
    """Soft specular likelihood. High values indicate specular highlights."""
    rgb = np.clip(np.asarray(colour, dtype=np.float32), 0.0, 1.0)
    vmax = rgb.max(axis=-1)
    vmin = rgb.min(axis=-1)
    sat = (vmax - vmin) / (vmax + 1e-6)

    valid = _as_mask(mask) if mask is not None else np.ones_like(vmax, dtype=bool)
    valid_brightness = vmax[valid & np.isfinite(vmax)]
    adaptive_threshold = (
        np.percentile(valid_brightness, float(brightness_percentile))
        if valid_brightness.size > 0
        else float(min_v)
    )
    brightness_threshold = max(float(min_v), float(adaptive_threshold))
    bright = 1.0 / (
        1.0
        + np.exp(
            -(vmax - brightness_threshold) / max(v_softness, 1e-6)
        )
    )
    low_sat = 1.0 / (1.0 + np.exp(-(s_threshold - sat) / max(s_softness, 1e-6)))
    white_like = bright * low_sat

    overexposed = (rgb > 0.92).sum(axis=-1) >= 2
    specular = np.maximum(white_like, overexposed.astype(np.float32))
    specular[~valid] = 0.0
    return np.clip(specular, 0.0, 1.0).astype(np.float32)


def erode_mask_np(mask, kernel_size=9):
    valid = _as_mask(mask)
    if kernel_size <= 1:
        return valid
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.erode(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        return valid


def sampling_probability_np(
    colour,
    mask,
    mode="texture_prior",
    uniform_mix=0.35,
    erode_kernel_size=9,
    specular_min_v=0.85,
    specular_brightness_percentile=97.0,
    specular_s_threshold=0.35,
    grad_percentile=99.0,
):
    """Build an init sampling distribution and diagnostic maps."""
    valid = _as_mask(mask)
    candidate = erode_mask_np(valid, erode_kernel_size)

    if not candidate.any():
        candidate = valid.copy()
    if not candidate.any():
        candidate = np.ones(valid.shape, dtype=bool)

    grad = gradient_confidence_np(
        colour,
        candidate,
        percentile=grad_percentile,
    )
    specular = specular_confidence_np(
        colour,
        candidate,
        min_v=specular_min_v,
        s_threshold=specular_s_threshold,
        brightness_percentile=specular_brightness_percentile,
    )
    non_specular = 1.0 - specular

    mode = (mode or "texture_prior").lower()
    if mode in ["random", "random_sparse", "uniform"]:
        structure = np.zeros_like(grad, dtype=np.float32)
        mix = 1.0
    elif mode in ["sobel", "gradient"]:
        structure = grad * candidate
        mix = 0.0
    elif mode == "texture_prior":
        structure = grad * non_specular * candidate
        mix = float(np.clip(uniform_mix, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown init sampling mode: {mode}")

    uniform = candidate.astype(np.float64)
    uniform /= uniform.sum() + 1e-12

    weighted = structure.astype(np.float64)
    weighted[~candidate] = 0.0
    if weighted.sum() > 0:
        weighted /= weighted.sum()
    else:
        weighted = uniform.copy()

    prob = mix * uniform + (1.0 - mix) * weighted
    prob[~candidate] = 0.0
    prob /= prob.sum() + 1e-12

    maps = {
        "candidate": candidate,
        "gradient": grad,
        "specular": specular,
        "non_specular": non_specular,
        "structure": structure.astype(np.float32),
    }
    return prob.astype(np.float32), maps


def compute_texture_prior_map(
    image,
    mask,
    specular_min_v=0.85,
    specular_s_threshold=0.35,
    grad_percentile=0.99,
):
    """Differentiability is not needed; this is for per-Gaussian statistics.

    The input mask is treated as a non-tool valid tissue mask: mask=1 keeps
    pixels, mask=0 filters surgical tool / invalid pixels.
    """

    rgb = image[:3].detach().clamp(0.0, 1.0)
    device = rgb.device
    dtype = rgb.dtype
    if mask.ndim == 3:
        valid = mask[0].detach().float() > 0.5
    else:
        valid = mask.detach().float() > 0.5

    gray = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).view(1, 1, *rgb.shape[-2:])
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)[0, 0]
    gy = F.conv2d(gray, ky, padding=1)[0, 0]
    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)

    vals = grad[valid & torch.isfinite(grad)]
    if vals.numel() > 16:
        scale = torch.quantile(vals, float(grad_percentile)).clamp_min(1e-6)
    elif vals.numel() > 0:
        scale = vals.max().clamp_min(1e-6)
    else:
        scale = grad.new_tensor(1.0)
    grad_conf = (grad / scale).clamp(0.0, 1.0)

    vmax = rgb.max(dim=0).values
    vmin = rgb.min(dim=0).values
    sat = (vmax - vmin) / (vmax + 1e-6)
    bright = torch.sigmoid((vmax - float(specular_min_v)) / 0.04)
    low_sat = torch.sigmoid((float(specular_s_threshold) - sat) / 0.08)
    white_like = bright * low_sat
    overexposed = (rgb > 0.92).float().sum(dim=0) >= 2
    specular = torch.maximum(white_like, overexposed.float())
    prior_map = grad_conf * (1.0 - specular) * valid.float()
    return prior_map.clamp(0.0, 1.0)


def effective_sample_size(weights):
    """Return the effective support represented by non-negative weights."""

    if weights is None or weights.numel() == 0:
        return 0.0
    detached = weights.detach()
    finite_weights = torch.where(
        torch.isfinite(detached),
        detached,
        torch.zeros_like(detached),
    ).clamp_min(0.0)
    denominator = torch.sum(finite_weights * finite_weights).clamp_min(1e-12)
    numerator = torch.sum(finite_weights) ** 2
    return float((numerator / denominator).item())


def adaptive_motion_prior_weights(
    prior_scores,
    initial_threshold,
    minimum_threshold,
    threshold_decay,
    initial_temperature,
    maximum_temperature,
    target_effective_support,
    maximum_evaluations=8,
    temperature_growth=1.5,
):

    maximum_evaluations = int(maximum_evaluations)
    if maximum_evaluations < 1:
        raise ValueError("maximum_evaluations must be at least one")
    if float(temperature_growth) < 1.0:
        raise ValueError("temperature_growth must be at least one")

    threshold_minimum = float(minimum_threshold)
    threshold = max(float(initial_threshold), threshold_minimum)
    threshold_decay = max(float(threshold_decay), 0.0)
    temperature = max(float(initial_temperature), 1e-6)
    temperature_maximum = max(float(maximum_temperature), temperature)
    target_support = min(
        max(float(target_effective_support), 0.0),
        float(prior_scores.numel()),
    )

    weights = torch.empty_like(prior_scores)
    for evaluation in range(maximum_evaluations):
        weights = torch.sigmoid((prior_scores - threshold) / temperature)
        support = effective_sample_size(weights)
        at_adaptation_limits = (
            temperature >= temperature_maximum
            and threshold <= threshold_minimum
        )
        if (
            support >= target_support
            or at_adaptation_limits
            or evaluation == maximum_evaluations - 1
        ):
            break
        temperature = min(
            temperature_maximum,
            temperature * float(temperature_growth),
        )
        threshold = max(threshold_minimum, threshold - threshold_decay)

    return weights, threshold, temperature


def raster_ndc_to_grid_coordinates(ndc_xy, image_width, image_height):
    """Match the rasteriser's half-pixel NDC convention in grid_sample."""


    width = int(image_width)
    height = int(image_height)
    grid_x = (
        ndc_xy[:, 0] * width / float(width - 1)
        if width > 1
        else torch.zeros_like(ndc_xy[:, 0])
    )
    grid_y = (
        ndc_xy[:, 1] * height / float(height - 1)
        if height > 1
        else torch.zeros_like(ndc_xy[:, 1])
    )
    return torch.stack((grid_x, grid_y), dim=-1)


def project_gaussians_to_ndc(means3d, camera):
    """Project Gaussian centres and mark finite, in-bounds image projections."""

    ones = torch.ones_like(means3d[:, :1])
    means_h = torch.cat([means3d, ones], dim=-1)
    proj_h = means_h @ camera.full_proj_transform.to(means3d.device)
    w = proj_h[:, 3:4]
    ndc = proj_h[:, :3] / (w + 1e-7)

    in_screen = (
        (w.squeeze(-1).abs() > 1e-7)
        & torch.isfinite(ndc).all(dim=-1)
        & (ndc[:, 0] > -1.0) & (ndc[:, 0] < 1.0)
        & (ndc[:, 1] > -1.0) & (ndc[:, 1] < 1.0)
        & (proj_h[:, 2] > 0)
    )
    return ndc, in_screen


def sample_scalar_map_at_gaussians(
    scalar_map,
    means3d,
    camera,
    visibility_filter=None,
):

    ndc, in_screen = project_gaussians_to_ndc(means3d, camera)
    if visibility_filter is not None:
        visibility = visibility_filter.detach().bool().view(-1)
        if visibility.numel() != in_screen.numel():
            raise ValueError("visibility_filter must match the Gaussian count")
        in_screen = in_screen & visibility

    scalar_img = scalar_map.detach().view(1, 1, *scalar_map.shape[-2:])
    height, width = scalar_img.shape[-2:]
    grid = raster_ndc_to_grid_coordinates(
        ndc[:, :2],
        image_width=width,
        image_height=height,
    ).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        scalar_img,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled.view(-1)
    sampled = torch.where(in_screen, sampled, torch.zeros_like(sampled))
    return sampled, in_screen


def sample_prior_at_gaussians(
    prior_map,
    valid_tissue_map,
    means3d,
    camera,
    visibility_filter=None,
    tissue_threshold=0.5,
):
    prior_values, projected = sample_scalar_map_at_gaussians(
        prior_map,
        means3d,
        camera,
        visibility_filter,
    )
    tissue_values, _ = sample_scalar_map_at_gaussians(
        valid_tissue_map.detach().float(),
        means3d,
        camera,
        visibility_filter,
    )
    valid_observation = projected & (tissue_values > float(tissue_threshold))
    prior_values = prior_values.masked_fill(~valid_observation, 0.0)
    return prior_values, valid_observation


def _empty_motion_selection_stats():
    return {
        "candidate_count": 0,
        "selected_count": 0,
        "raw_effective_support": 0.0,
        "effective_support": 0.0,
        "tau": 0.0,
        "temperature": 0.0,
        "weight_mean": 0.0,
        "weight_min": 0.0,
        "weight_max": 0.0,
    }


def _deterministic_weighted_subsample(point_indices, weights, max_points):

    max_points = int(max_points)
    if max_points <= 0 or point_indices.numel() <= max_points:
        return point_indices, weights

    weights = weights.detach()
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    weights = weights.clamp_min(0.0)
    total = torch.sum(weights)
    n_points = point_indices.numel()

    if total <= 0:
        sample_ids = torch.linspace(
            0, n_points - 1, steps=max_points, device=point_indices.device
        ).round().long()
        sample_ids = torch.unique(sample_ids, sorted=True)
    else:
        cdf = torch.cumsum(weights / total, dim=0)
        targets = (
            torch.arange(max_points, device=point_indices.device, dtype=weights.dtype) + 0.5
        ) / float(max_points)
        sample_ids = torch.searchsorted(cdf, targets).clamp(max=n_points - 1)
        sample_ids = torch.unique(sample_ids, sorted=True)

    if sample_ids.numel() < max_points:
        selected = torch.zeros(n_points, dtype=torch.bool, device=point_indices.device)
        selected[sample_ids] = True
        remaining = (~selected).nonzero(as_tuple=False).flatten()
        fill_count = min(max_points - sample_ids.numel(), remaining.numel())
        if fill_count > 0:
            fill_order = torch.argsort(weights[remaining], descending=True)[:fill_count]
            sample_ids = torch.cat([sample_ids, remaining[fill_order]], dim=0)

    sample_ids = sample_ids[:max_points]
    return point_indices[sample_ids], weights[sample_ids]


def select_unweighted_visible_points(
    visibility_filter,
    max_points,
    min_points,
):
    """Select visible Gaussians without using the joint-texture prior."""

    stats = _empty_motion_selection_stats()
    point_indices = visibility_filter.detach().bool().nonzero(as_tuple=False).flatten()
    stats["candidate_count"] = int(point_indices.numel())
    if point_indices.numel() < int(min_points):
        return None, None, stats

    max_points = int(max_points)
    if max_points > 0 and point_indices.numel() > max_points:
        sample_ids = torch.randperm(
            point_indices.numel(), device=point_indices.device
        )[:max_points]
        point_indices = point_indices[sample_ids]

    stats["selected_count"] = int(point_indices.numel())
    stats["raw_effective_support"] = float(point_indices.numel())
    stats["effective_support"] = float(point_indices.numel())
    stats["weight_mean"] = 1.0
    stats["weight_min"] = 1.0
    stats["weight_max"] = 1.0
    return point_indices, None, stats


def select_prior_weighted_visible_points(
    gaussians,
    visibility_filter,
    min_observations,
    threshold_initial,
    threshold_minimum,
    threshold_decay,
    temperature_initial,
    temperature_maximum,
    target_effective_support,
    max_points,
    min_points,
    maximum_evaluations,
    temperature_growth,
    minimum_weight,
):
    """Select raster-visible Gaussians and adapt their prior-derived weights."""
    stats = _empty_motion_selection_stats()
    if gaussians.prior_observation_count.numel() != gaussians.get_xyz.shape[0]:
        return None, None, stats

    prior_scores = gaussians.get_prior_scores().detach().squeeze()
    observed = gaussians.prior_observation_count.detach().squeeze() >= int(min_observations)
    candidate_mask = visibility_filter.detach().bool() & observed
    point_indices = candidate_mask.nonzero(as_tuple=False).flatten()
    stats["candidate_count"] = int(point_indices.numel())
    if point_indices.numel() < min_points:
        return None, None, stats

    temporal_weights, threshold, temperature = adaptive_motion_prior_weights(
        prior_scores[point_indices].clamp(0.0, 1.0),
        initial_threshold=threshold_initial,
        minimum_threshold=threshold_minimum,
        threshold_decay=threshold_decay,
        initial_temperature=temperature_initial,
        maximum_temperature=temperature_maximum,
        target_effective_support=target_effective_support,
        maximum_evaluations=maximum_evaluations,
        temperature_growth=temperature_growth,
    )
    stats["raw_effective_support"] = effective_sample_size(temporal_weights)
    stats["tau"] = float(threshold)
    stats["temperature"] = float(temperature)

    non_negligible = temporal_weights > minimum_weight
    point_indices = point_indices[non_negligible]
    temporal_weights = temporal_weights[non_negligible]
    if point_indices.numel() < min_points:
        return None, None, stats

    if max_points > 0 and point_indices.numel() > max_points:
        point_indices, temporal_weights = _deterministic_weighted_subsample(
            point_indices,
            temporal_weights,
            max_points,
        )

    stats["selected_count"] = int(point_indices.numel())
    stats["effective_support"] = effective_sample_size(temporal_weights)
    stats["weight_mean"] = float(temporal_weights.mean().item())
    stats["weight_min"] = float(temporal_weights.min().item())
    stats["weight_max"] = float(temporal_weights.max().item())
    return point_indices, temporal_weights, stats
