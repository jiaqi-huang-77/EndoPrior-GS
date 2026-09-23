#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""Train EndoPrior-GS from one entry point.

Read main() for argument resolution, training() for stage orchestration and
scene_reconstruction() for the optimisation loop. Output and logging helpers
are kept below the loop. Ablations and hyperparameters live in arguments/.
"""

import csv
import os
import random
import sys
from random import randint
from time import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from arguments import (
    prior_component_policy,
    attach_scared_geometry_version, parse_training_args, save_run_config,
    validate_training_output,
)
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import TV_loss, l1_loss
from utils.regularisation import compute_local_soft_tissue_motion_loss
from utils.texture_prior import (
    _empty_motion_selection_stats, previous_contiguous_training_frame_time,
    sample_prior_at_gaussians, select_prior_weighted_visible_points,
    select_unweighted_visible_points, compute_texture_prior_map,
)


def main(argv=None):
    # Resolve the preset, ablations and hyperparameters.
    args, lp, op, pp, hp = parse_training_args(sys.argv[1:] if argv is None else argv)
    torch.cuda.empty_cache()

    # Initialise reproducible random states.
    safe_state(args.quiet)
    setup_seed(args.initialisation_seed)
    print("Starting training.")

    # Initialise the scene, then optimise its coarse and fine stages.
    training(
        args,
        lp.extract(args),
        hp.extract(args),
        op.extract(args),
        pp.extract(args),
    )

    print("\nTraining complete.")


def training(runtime_args, dataset, hyper, opt, pipe):
    """Train the coarse and fine stages, then save the final model once."""
    tb_writer = prepare_output_and_logger(runtime_args)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = runtime_args.model_path
    # Scene selects datasets/endonerf.py or datasets/scared.py via dataset_type.
    scene = Scene(dataset, gaussians, coarse_only=runtime_args.coarse_only)

    scene_reconstruction(
        dataset, opt, pipe, gaussians, scene, "coarse", tb_writer,
        opt.coarse_iterations,
    )
    if runtime_args.coarse_only:
        scene.save(opt.coarse_iterations, "coarse")
    else:
        scene_reconstruction(
            dataset, opt, pipe, gaussians, scene, "fine", tb_writer,
            opt.iterations,
        )
        scene.save(opt.iterations, "fine")
    tb_writer.close()


def scene_reconstruction(dataset, opt, pipe, gaussians, scene, stage,
                         tb_writer, train_iter):
    """Run one stage: render, compute losses, backpropagate and update."""
    gaussians.training_setup(opt)

    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    progress_bar = tqdm(range(train_iter), desc="Training progress")

    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
    training_frame_ids = {int(camera.uid) for camera in viewpoint_stack}
    video_frame_times = {
        int(camera.uid): float(camera.time) for camera in scene.getVideoCameras()
    }

    # CUDA event timers (measure GPU time, ms)
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # Collect compact training logs.
    gaussian_count_history = []
    motion_loss_history = []
    start_train_time = time()

    for iteration in range(1, train_iter + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # 1. Update learning rates, select a frame and render.
        if iteration % opt.sh_degree_interval == 0:
            gaussians.oneupSHdegree()
        if stage == 'coarse':
            idx = 0
        else:
            idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cams = [viewpoint_stack[idx]]

        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []

        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        prior_stats = []
        prior_policy = prior_component_policy(dataset)
        compute_prior_scores = (
            prior_policy.density_control
            or (
                prior_policy.temporal_regularisation
                and not dataset.disable_prior_temporal_weighting
            )
        )
        prior_interval = max(1, int(opt.prior_score_update_interval))
        continues_to_next_stage = stage == "coarse" and not dataset.coarse_only
        topology_updates_enabled = (
            iteration < opt.densify_until_iter
            and (iteration < train_iter or continues_to_next_stage)
        )
        densify_event = iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0
        prune_event = iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0
        collect_prior_stats = (
            compute_prior_scores
            and topology_updates_enabled
            and iteration % prior_interval == 0
            and (densify_event or prune_event)
        )

        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background,stage=stage,
                )
            image, depth, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg[
                    "visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            means3D_deform = render_pkg["means3D_deform"]
            scales_deform = render_pkg["scales_deform"]
            mask = viewpoint_cam.mask.cuda()
            tool_filtered_mask = getattr(viewpoint_cam, "motion_mask", mask).cuda()

            images.append(image.unsqueeze(0))
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            if collect_prior_stats:
                with torch.no_grad():
                    prior_map = compute_texture_prior_map(
                        gt_image,
                        tool_filtered_mask,
                        specular_min_v=dataset.prior_brightness_threshold,
                        specular_s_threshold=dataset.prior_saturation_threshold,
                        grad_percentile=dataset.prior_gradient_percentile / 100.0,
                    )
                    prior_values, valid_prior_observation = sample_prior_at_gaussians(
                        prior_map,
                        tool_filtered_mask,
                        means3D_deform.detach(),
                        viewpoint_cam,
                        visibility_filter,
                    )
                    prior_stats.append((prior_values, valid_prior_observation))

        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        rendered_images = torch.cat(images, 0)
        rendered_depths = torch.cat(depths, 0)
        gt_images = torch.cat(gt_images, 0)
        gt_depths = torch.cat(gt_depths, 0)
        masks = torch.cat(masks, 0)

        # Compute appearance, inverse-depth and total variation losses.
        Ll1 = l1_loss(rendered_images, gt_images, masks)

        if (gt_depths != 0).sum() < opt.minimum_valid_depth_pixels:
            depth_loss = torch.tensor(0.).cuda()
        else:
            rendered_depths[rendered_depths != 0] = 1 / rendered_depths[rendered_depths != 0]
            gt_depths[gt_depths != 0] = 1 / gt_depths[gt_depths != 0]
            depth_loss = l1_loss(rendered_depths, gt_depths, masks)

        depth_tvloss = TV_loss(rendered_depths)
        img_tvloss = TV_loss(rendered_images)
        tv_loss = opt.tv_loss_weight * (img_tvloss + depth_tvloss)
        loss_local_motion = torch.zeros((), device=rendered_images.device)
        loss_local_motion_raw = torch.zeros((), device=rendered_images.device)
        local_motion_diagnostics = None
        local_motion_points = 0
        local_motion_selection_stats = _empty_motion_selection_stats()
        reference_time = previous_contiguous_training_frame_time(
            viewpoint_cam.uid,
            training_frame_ids,
            video_frame_times,
        )
        # Apply temporal regularisation on eligible visible Gaussians.
        local_motion_event = (
            stage == "fine"
            and prior_policy.temporal_regularisation
            and iteration >= opt.local_motion_start_iteration
            and scales_deform is not None
            and reference_time is not None
        )

        if local_motion_event:
            if dataset.disable_prior_temporal_weighting:
                (
                    point_indices,
                    temporal_weights,
                    local_motion_selection_stats,
                ) = select_unweighted_visible_points(
                    visibility_filter, opt.local_motion_max_points, opt.local_motion_min_points
                )
            else:
                (
                    point_indices,
                    temporal_weights,
                    local_motion_selection_stats,
                ) = select_prior_weighted_visible_points(
                    gaussians,
                    visibility_filter,
                    min_observations=opt.motion_min_observations,
                    threshold_initial=opt.motion_prior_threshold_initial,
                    threshold_minimum=opt.motion_prior_threshold_minimum,
                    threshold_decay=opt.motion_prior_threshold_decay,
                    temperature_initial=opt.motion_prior_temperature_initial,
                    temperature_maximum=opt.motion_prior_temperature_maximum,
                    target_effective_support=opt.motion_target_effective_support,
                    max_points=opt.local_motion_max_points,
                    min_points=opt.local_motion_min_points,
                    maximum_evaluations=opt.motion_adaptation_evaluations,
                    temperature_growth=opt.motion_temperature_growth,
                    minimum_weight=opt.motion_minimum_weight,
                )
            if point_indices is not None:
                local_motion_points = int(point_indices.numel())
                means_previous = gaussians.get_deformed_centres_at_time(
                    reference_time,
                    point_indices,
                )
                loss_local_motion_raw, local_motion_diagnostics = compute_local_soft_tissue_motion_loss(
                    means3D_deform[point_indices],
                    means_previous,
                    point_weights=temporal_weights,
                    k=opt.local_motion_k,
                    sigma_scale=opt.local_motion_sigma_scale,
                    distance_weight=opt.local_motion_distance_weight,
                    robust_delta=opt.local_motion_robust_delta,
                    return_diagnostics=True,
                )
                loss_local_motion = opt.local_motion_lambda * loss_local_motion_raw

        loss = Ll1 + opt.depth_loss_weight * depth_loss + tv_loss + loss_local_motion
        psnr_ = psnr(rendered_images, gt_images, masks).mean().double()

        # Backpropagate once through the combined objective.
        loss.backward()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}",
                           "psnr": f"{psnr_:.{2}f}",
                           "point": f"{total_point}"}
                if loss_local_motion.item() != 0:
                    postfix["local"] = f"{loss_local_motion.item():.2e}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
                # Record population size.
                gaussian_count_history.append((iteration, total_point))
            if local_motion_event:
                motion_loss_history.append((
                    iteration,
                    stage,
                    float(loss_local_motion.detach().item()),
                    local_motion_points,
                    int(local_motion_selection_stats.get("candidate_count", 0)),
                    float(local_motion_selection_stats.get("raw_effective_support", 0.0)),
                    float(local_motion_selection_stats.get("effective_support", 0.0)),
                    float(local_motion_selection_stats.get("tau", 0.0)),
                    float(local_motion_selection_stats.get("temperature", 0.0)),
                    float(local_motion_selection_stats.get("weight_mean", 0.0)),
                    float(Ll1.detach().item()),
                    float(loss_local_motion_raw.detach().item()),
                    (
                        float(loss_local_motion.detach().item())
                        / max(float(Ll1.detach().item()), 1e-12)
                    ),
                    float(local_motion_diagnostics["sigma_loc"])
                    if local_motion_diagnostics is not None
                    else 0.0,
                    float(local_motion_diagnostics["displacement_loss"].item())
                    if local_motion_diagnostics is not None
                    else 0.0,
                    float(local_motion_diagnostics["distance_loss"].item())
                    if local_motion_diagnostics is not None
                    else 0.0,
                ))
            if iteration == train_iter:
                progress_bar.close()

            # Log training progress
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                loss_local_motion,
                psnr_,
                iter_start.elapsed_time(iter_end),
                stage,
            )
            # Update prior scores, densify and prune.
            if topology_updates_enabled:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                if collect_prior_stats:
                    for prior_values, valid_prior_observation in prior_stats:
                        gaussians.add_prior_stats(
                            prior_values, valid_prior_observation
                        )

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration * (
                                opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / (
                                            opt.densify_until_iter)
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration * (
                                opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / (
                                            opt.densify_until_iter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify(
                        densify_threshold, scene.cameras_extent,
                        prior_allocation=prior_policy.density_control,
                        prior_floor=opt.prior_densification_floor,
                        prior_weight=1.0 - opt.prior_densification_floor,
                    )

                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = opt.pruning_screen_size if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(
                        opacity_threshold, scene.cameras_extent, size_threshold,
                        prior_pruning=prior_policy.density_control,
                        prior_threshold=opt.prior_pruning_threshold,
                        prior_min_observations=opt.prior_min_observations,
                        prior_pruning_opacity=opt.prior_pruning_opacity_threshold,
                        prior_pruning_max_scale=opt.prior_pruning_max_scale
                        if prior_policy.density_control else 0.0,
                    )

                if iteration % opt.opacity_reset_interval == 0:
                    print("Resetting Gaussian opacity.")
                    gaussians.reset_opacity()

            # Apply the optimiser update.
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    save_training_logs(scene.model_path, stage, gaussian_count_history, motion_loss_history)
    total_time_min = (time() - start_train_time) / 60.0
    print(f"Training time: {total_time_min:.2f} minutes")


def prepare_output_and_logger(runtime_args):
    if not runtime_args.model_path:
        runtime_args.model_path = os.path.join("./output/", runtime_args.expname)
    runtime_args.model_path = os.path.abspath(runtime_args.model_path)
    runtime_args.source_path = os.path.abspath(runtime_args.source_path)
    scared_geometry_version = attach_scared_geometry_version(runtime_args)
    if scared_geometry_version is not None:
        print("Detected geometry schema v{}.".format(scared_geometry_version))
    validate_training_output(runtime_args.model_path)
    print("Output folder: {}".format(runtime_args.model_path))
    os.makedirs(runtime_args.model_path, exist_ok=True)
    save_run_config(runtime_args, runtime_args.model_path)
    return SummaryWriter(runtime_args.model_path)


def training_report(tb_writer, iteration, Ll1, loss, loss_local_motion,
                    psnr_val, elapsed, stage):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(
            f'{stage}/train_loss_patches/local_motion_weighted',
            loss_local_motion.item(),
            iteration,
        )
        tb_writer.add_scalar(f'{stage}/train_psnr', psnr_val.item() if torch.is_tensor(psnr_val) else psnr_val,
                             iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)


def save_training_logs(output_directory, stage, gaussian_count_history, motion_loss_history):
    """Save population counts and the local-motion loss/support statistics."""
    csv_path = os.path.join(output_directory, f"{stage}_gaussian_count.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'NumGaussians'])
        writer.writerows(gaussian_count_history)
    print(f"\nGaussian-count log saved to: {csv_path}")
    if motion_loss_history:
        motion_csv_path = os.path.join(output_directory, f"{stage}_motion_regularisation.csv")
        with open(motion_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Iteration',
                'Stage',
                'LocalSoftTissueLossWeighted',
                'LocalSoftTissuePoints',
                'LocalMotionCandidateCount',
                'LocalMotionRawEffectiveSupport',
                'LocalMotionEffectiveSupport',
                'LocalMotionSoftTau',
                'LocalMotionSoftTemperature',
                'LocalMotionWeightMean',
                'RGBLoss',
                'LocalSoftTissueLossRaw',
                'WeightedLocalToRGBRatio',
                'SigmaLoc',
                'DisplacementLossRaw',
                'DistanceLossRaw',
            ])
            writer.writerows(motion_loss_history)
        print(f"Motion-regularisation log saved to: {motion_csv_path}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
