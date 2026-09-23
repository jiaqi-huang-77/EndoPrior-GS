"""Local soft-tissue motion regularisation for EndoPrior-GS."""


import torch
import torch.nn.functional as F


def compute_local_soft_tissue_motion_loss(
    means_current,
    means_t1,
    point_weights=None,
    k=8,
    sigma_scale=1.0,
    distance_weight=0.2,
    robust_delta=0.01,
    return_diagnostics=False,
):
    """Local soft-tissue motion regularisation on selected Gaussians.

    Optional point_weights are derived from the joint texture prior.
    Neighbour relations are built on the previous physical centres. At every
    update, the spatial affinity scale is the median finite K-nearest-neighbour
    distance in that reference set, multiplied by ``sigma_scale``. The default
    multiplier of 1.0 preserves the adaptive scale. The loss has two terms: neighbouring
    Gaussians should have similar low-frequency displacement, and their local
    pairwise distances should not change abruptly.
    """
    sigma_scale = float(sigma_scale)
    if sigma_scale <= 0.0:
        raise ValueError("sigma_scale must be positive")

    n_points = means_current.shape[0]
    if n_points <= 1:
        zero = means_current.sum() * 0.0
        if return_diagnostics:
            return zero, {
                "sigma_loc": 0.0,
                "displacement_loss": zero.detach(),
                "distance_loss": zero.detach(),
            }
        return zero

    k = max(1, min(int(k), n_points - 1))
    ref = means_t1.detach()

    with torch.no_grad():
        pair_dist = torch.cdist(ref, ref)
        pair_dist.fill_diagonal_(float("inf"))
        knn_dist, knn_idx = torch.topk(pair_dist, k=k, largest=False, dim=1)

        finite_dist = knn_dist[torch.isfinite(knn_dist)]
        local_scale = (
            float(finite_dist.median().item())
            if finite_dist.numel() > 0
            else 1.0
        )
        local_scale = max(local_scale * sigma_scale, 1e-6)
        edge_weight = torch.exp(
            -(knn_dist.clamp_max(1e6) ** 2)
            / (2.0 * local_scale * local_scale)
        )

        if point_weights is not None:
            weights = point_weights.detach().view(-1).clamp(0.0, 1.0)
            edge_weight = edge_weight * weights[:, None] * weights[knn_idx]
        edge_weight = edge_weight.clamp_min(1e-8)

    disp = means_current - ref
    disp_i = disp[:, None, :]
    disp_j = disp[knn_idx]
    disp_error = torch.sqrt(torch.sum((disp_i - disp_j) ** 2, dim=-1) + 1e-8)

    curr_i = means_current[:, None, :]
    curr_j = means_current[knn_idx]
    ref_i = ref[:, None, :]
    ref_j = ref[knn_idx]
    curr_dist = torch.sqrt(torch.sum((curr_i - curr_j) ** 2, dim=-1) + 1e-8)
    ref_dist = torch.sqrt(torch.sum((ref_i - ref_j) ** 2, dim=-1) + 1e-8)
    distance_error = F.smooth_l1_loss(
        curr_dist,
        ref_dist,
        beta=max(float(robust_delta), 1e-6),
        reduction="none",
    )

    denom = edge_weight.sum().clamp_min(1e-8)
    displacement_loss = (edge_weight * disp_error).sum() / denom
    distance_loss = (edge_weight * distance_error).sum() / denom
    total = displacement_loss + float(distance_weight) * distance_loss
    if return_diagnostics:
        return total, {
            "sigma_loc": local_scale,
            "displacement_loss": displacement_loss.detach(),
            "distance_loss": distance_loss.detach(),
        }
    return total
