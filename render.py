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
import os
import sys
from argparse import ArgumentParser
from os import makedirs
from time import perf_counter

import cv2
import imageio
import numpy as np
import open3d as o3d
import torch
import torchvision
from tqdm import tqdm

from arguments import (
    ModelHiddenParams, ModelParams, PipelineParams,
    build_render_parser, resolve_render_args,
)
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.depth_io import save_depth
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal


def measure_render_fps(views, gaussians, pipeline, background, stage):
    """Measure all views over ten rounds after ten untimed warm-up renders."""
    for index in range(10):
        render(views[index % len(views)], gaussians, pipeline, background, stage=stage)
    torch.cuda.synchronize(background.device)

    render_seconds = 0.0
    for _ in range(10):
        for view in views:
            torch.cuda.synchronize(background.device)
            start = perf_counter()
            rendering = render(view, gaussians, pipeline, background, stage=stage)
            torch.cuda.synchronize(background.device)
            render_seconds += perf_counter() - start
            del rendering
    return len(views) * 10 / render_seconds


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, coarse_only, reconstruct=False):
    if not views:
        return
    stage = 'coarse' if coarse_only else 'fine'
    fps = measure_render_fps(views, gaussians, pipeline, background, stage)
    print(f"[{name}] Rendering FPS: {fps:.2f} "
          f"(10 warm-up renders; {len(views)} frames x 10 repeats; "
          "excludes data loading, CPU transfer and file saving)")

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gtdepth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masks")
    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gtdepth_path, exist_ok=True)
    makedirs(masks_path, exist_ok=True)
    
    render_images = []
    render_depths = []
    gt_list = []
    gt_depths = []
    mask_list = []
    camera_parameters = []

    for view in tqdm(views, desc="Rendering progress"):
        rendering = render(view, gaussians, pipeline, background, stage=stage)
        render_depths.append(rendering["depth"].cpu())
        render_images.append(rendering["render"].cpu())
        if name in ["train", "test", "video"]:
            gt = view.original_image[0:3, :, :]
            gt_list.append(gt)
            mask = view.mask
            mask_list.append(mask)
            gt_depth = view.original_depth
            gt_depths.append(gt_depth)
        focal_y = fov2focal(view.FoVy, view.image_height)
        focal_x = fov2focal(view.FoVx, view.image_width)
        camera_parameters.append(
            (
                focal_x,
                focal_y,
                view.principal_x,
                view.principal_y,
                view.image_width,
                view.image_height,
            )
        )
    count = 0
    print("writing training images.")
    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count+=1
            
    count = 0
    print("writing rendering images.")
    if len(render_images) != 0:
        for image in tqdm(render_images):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing mask images.")
    if len(mask_list) != 0:
        for image in tqdm(mask_list):
            image = image.float()
            torchvision.utils.save_image(image, os.path.join(masks_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing rendered depth images.")
    if len(render_depths) != 0:
        for image in tqdm(render_depths):
            save_depth(
                os.path.join(depth_path, '{0:05d}'.format(count) + ".npy"),
                image,
            )
            count += 1
    
    count = 0
    print("writing gt depth images.")
    if len(gt_depths) != 0:
        for image in tqdm(gt_depths):
            save_depth(
                os.path.join(gtdepth_path, '{0:05d}'.format(count) + ".npy"),
                image,
            )
            count += 1
            
    render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'ours_video.mp4'), render_array, fps=30, quality=8)
    
    gt_array = torch.stack(gt_list, dim=0).permute(0, 2, 3, 1)
    gt_array = (gt_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'gt_video.mp4'), gt_array, fps=30, quality=8)
                    
    if reconstruct:
        reconstruction_path = os.path.join(
            model_path,
            name,
            f"ours_{iteration}",
            "reconstruction",
        )
        reconstruct_point_cloud(
            render_images,
            mask_list,
            render_depths,
            camera_parameters,
            reconstruction_path,
        )





def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, reconstruct: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            coarse_only=dataset.coarse_only,
        )

        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.coarse_only, reconstruct=False)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.coarse_only, reconstruct=reconstruct)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter, scene.getVideoCameras(),gaussians,pipeline,background, dataset.coarse_only, reconstruct=False)

def reconstruct_point_cloud(
    images,
    masks,
    depths,
    camera_parameters,
    output_directory,
):
    os.makedirs(output_directory, exist_ok=True)
    frames = np.arange(len(images))
    for i_frame in frames:
        focal_x, focal_y, principal_x, principal_y, width, height = (
            camera_parameters[i_frame]
        )
        rgb_tensor = images[i_frame]
        rgb_np = rgb_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to("cpu").numpy()
        depth_np = depths[i_frame].cpu().numpy()
        depth_np = depth_np.squeeze(0)
        mask = masks[i_frame].squeeze(0).cpu().numpy().astype(bool)

        depth_smoother = (128, 64, 64)
        depth_np = cv2.bilateralFilter(depth_np, depth_smoother[0], depth_smoother[1], depth_smoother[2])
        valid_depth = mask & np.isfinite(depth_np) & (depth_np > 0)
        if not np.any(valid_depth):
            continue
        close_depth, far_depth = np.percentile(depth_np[valid_depth], (5, 95))
        depth_np = np.where(
            valid_depth,
            np.clip(depth_np, close_depth, far_depth),
            0.0,
        ).astype(np.float32)

        rgb_im = o3d.geometry.Image(rgb_np.astype(np.uint8))
        depth_im = o3d.geometry.Image(depth_np)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_im,
            depth_im,
            depth_scale=1.0,
            depth_trunc=float(far_depth),
            convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                width,
                height,
                focal_x,
                focal_y,
                principal_x,
                principal_y,
            ),
            project_valid_depth_only=True
        )
        o3d.io.write_point_cloud(
            os.path.join(output_directory, f"frame_{i_frame}.ply"),
            pcd,
        )

if __name__ == "__main__":
    parser = build_render_parser()
    args = resolve_render_args(parser, sys.argv[1:])
    parameter_parser = ArgumentParser(add_help=False)
    model = ModelParams(parameter_parser, sentinel=True)
    pipeline = PipelineParams(parameter_parser, sentinel=True)
    hyperparam = ModelHiddenParams(parameter_parser, sentinel=True)
    print("Rendering ", args.model_path)
    if args.dataset_type == "scared":
        print(
            "Detected SCARED geometry schema v{}.".format(
                args.scared_geometry_version
            )
        )
    safe_state(args.quiet)
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.reconstruct)
