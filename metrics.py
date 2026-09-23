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

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import lpips
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm

from utils.depth_io import (
    directory_depth_temporal_instability,
    load_depth_for_frame,
    masked_depth_rmse,
)
from utils.image_utils import flip, psnr
from utils.loss_utils import ssim

def to_uint8(image):
    return (255 * image).to(torch.uint8)


class LPIPSMetric:
    """Compute Learned Perceptual Image Patch Similarity."""

    def __init__(self, device="cuda"):
        self.model = lpips.LPIPS(net='alex').to(device)

    def __call__(self, y_pred, y_true, normalised=True):
        if normalised:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        error = self.model.forward(y_pred, y_true)
        return torch.mean(error)


def cal_lpips(a, b, model, batch=2):
    """Compute LPIPS for image batches."""
    lpips_all = []
    for a_split, b_split in zip(a.split(split_size=batch, dim=0), b.split(split_size=batch, dim=0)):
        out = model(a_split, b_split)
        lpips_all.append(out)
    lpips_all = torch.stack(lpips_all)
    lpips_mean = lpips_all.mean()
    return lpips_mean


def readImages(renders_dir, gt_dir, depth_dir, gtdepth_dir, masks_dir):
    renders = []
    gts = []
    image_names = []
    depths = []
    gt_depths = []
    masks = []

    # Sort rendered image files to preserve their temporal order.
    sorted_fnames = sorted(
        fname
        for fname in os.listdir(renders_dir)
        if Path(fname).suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    for fname in sorted_fnames:
        render = np.array(Image.open(renders_dir / fname))
        gt = np.array(Image.open(gt_dir / fname))
        depth = load_depth_for_frame(depth_dir, fname)
        gt_depth = load_depth_for_frame(gtdepth_dir, fname)
        mask = np.array(Image.open(masks_dir / fname))

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        depths.append(torch.from_numpy(depth).unsqueeze(0).unsqueeze(1).cuda())
        gt_depths.append(torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(1).cuda())
        masks.append(tf.to_tensor(mask).unsqueeze(0)[:, 0:1, :, :].cuda())

        image_names.append(fname)
    return renders, gts, depths, gt_depths, masks, image_names


def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    print("")

    lpips_model = None
    raft_transforms = None
    raft_model = None

    with torch.no_grad():
        for scene_dir in model_paths:
            print("====================================")
            print("Evaluating Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            for split in ["train", "test", "video"]:
                split_dir = Path(scene_dir) / split

                if not split_dir.exists():
                    continue

                print(f"\n--- Processing split: [{split.upper()}] ---")

                for method in os.listdir(split_dir):
                    print("Method:", method)

                    # Prefix method keys so that split results remain distinct.
                    method_key = f"{split}_{method}"

                    full_dict[scene_dir][method_key] = {}
                    per_view_dict[scene_dir][method_key] = {}

                    method_dir = split_dir / method
                    if split == "video":
                        depth_temporal_instability, temporal_per_view = (
                            directory_depth_temporal_instability(
                                method_dir / "depth",
                                method_dir / "masks",
                            )
                        )
                        if depth_temporal_instability is None:
                            print(
                                "[VIDEO] Depth Temporal Instability: "
                                "n/a (no valid adjacent frame pairs)"
                            )
                        else:
                            print(
                                "[VIDEO] Depth Temporal Instability: "
                                f"{depth_temporal_instability:>12.7f}"
                            )
                        full_dict[scene_dir][method_key].update({
                            "Depth_Temporal_Instability": (
                                depth_temporal_instability
                            ),
                        })
                        per_view_dict[scene_dir][method_key].update({
                            "Depth_Temporal_Instability": temporal_per_view,
                        })
                        continue

                    if lpips_model is None:
                        print("Loading perceptual and optical-flow models...")
                        lpips_model = LPIPSMetric()
                        raft_weights = Raft_Large_Weights.DEFAULT
                        raft_transforms = raft_weights.transforms()
                        raft_model = raft_large(
                            weights=raft_weights,
                            progress=False,
                        ).to("cuda").eval()
                        print("Perceptual and optical-flow models loaded.")

                    gt_dir = method_dir / "gt"
                    renders_dir = method_dir / "renders"
                    depth_dir = method_dir / "depth"
                    gt_depth_dir = method_dir / "gt_depth"
                    masks_dir = method_dir / "masks"

                    renders, gts, depths, gt_depths, masks, image_names = readImages(
                        renders_dir,
                        gt_dir,
                        depth_dir,
                        gt_depth_dir,
                        masks_dir,
                    )

                    ssims = []
                    psnrs = []
                    psnrs_star = []
                    lpipss = []
                    rmses = []
                    rmse_per_view = {}

                    render_wmask = []
                    gt_wmask = []

                    flow_errors = []

                    # The first frame has no preceding frame for temporal metrics.
                    flow_errors_per_view = [0.0]

                    prev_gt = None
                    prev_render = None
                    prev_mask = None

                    for idx in tqdm(range(len(renders)), desc=f"Evaluating {split}/{method}"):
                        render, gt, depth, gt_depth, mask = (
                            renders[idx],
                            gts[idx],
                            depths[idx],
                            gt_depths[idx],
                            masks[idx],
                        )

                        psnrs_star.append(psnr(render, gt, mask))

                        render = render * mask
                        gt = gt * mask
                        render_wmask.append(render)
                        gt_wmask.append(gt)
                        psnrs.append(psnr(render, gt))
                        ssims.append(ssim(render, gt))
                        lpipss.append(cal_lpips(render, gt, lpips_model))

                        depth_rmse, valid_depth_count = masked_depth_rmse(
                            depth,
                            gt_depth,
                            mask,
                        )
                        if valid_depth_count > 0:
                            rmses.append(depth_rmse)
                            rmse_per_view[image_names[idx]] = depth_rmse

                        if prev_gt is not None:
                            # RAFT expects uint8 images in the range [0, 255].
                            gt_t1_img = (prev_gt * 255.0).to(torch.uint8)
                            gt_t2_img = (gt * 255.0).to(torch.uint8)
                            rend_t1_img = (prev_render * 255.0).to(torch.uint8)
                            rend_t2_img = (render * 255.0).to(torch.uint8)

                            gt_img1, gt_img2 = raft_transforms(gt_t1_img, gt_t2_img)
                            rend_img1, rend_img2 = raft_transforms(rend_t1_img, rend_t2_img)

                            # Optical-flow endpoint error.
                            flow_gt = raft_model(gt_img1, gt_img2)[-1]
                            flow_rend = raft_model(rend_img1, rend_img2)[-1]

                            valid_flow_mask = (prev_mask * mask) > 0.5
                            if valid_flow_mask.sum() > 0:
                                epe = torch.norm(flow_gt - flow_rend, p=2, dim=1, keepdim=True)
                                epe_mean = (epe * valid_flow_mask).sum() / valid_flow_mask.sum()
                                flow_errors.append(epe_mean.item())
                                flow_errors_per_view.append(epe_mean.item())
                            else:
                                flow_errors_per_view.append(0.0)

                        prev_gt = gt
                        prev_render = render
                        prev_mask = mask

                    flipped_metrics = flip(
                        [to_uint8(e) for e in render_wmask],
                        [to_uint8(g) for g in gt_wmask],
                    )
                    print(f"[{split.upper()}] SSIM : {torch.tensor(ssims).mean():>12.7f}")
                    print(f"[{split.upper()}] PSNR : {torch.tensor(psnrs).mean():>12.7f}")
                    print(f"[{split.upper()}] PSNR*: {torch.tensor(psnrs_star).mean():>12.7f}")
                    print(f"[{split.upper()}] LPIPS: {torch.tensor(lpipss).mean():>12.7f}")
                    print(f"[{split.upper()}] FLIP : {torch.tensor(flipped_metrics).mean():>12.7f}")
                    if len(rmses) > 0:
                        print(f"[{split.upper()}] RMSE : {torch.tensor(rmses).mean():>12.7f}")

                    if len(flow_errors) > 0:
                        print(
                            f"[{split.upper()}] Flow Error: "
                            f"{torch.tensor(flow_errors).mean():>12.7f}"
                        )


                    full_dict[scene_dir][method_key].update({
                        "SSIM": torch.tensor(ssims).mean().item(),
                        "PSNR": torch.tensor(psnrs).mean().item(),
                        "PSNR*": torch.tensor(psnrs_star).mean().item(),
                        "LPIPS": torch.tensor(lpipss).mean().item(),
                        "FLIP": torch.tensor(flipped_metrics).mean().item(),
                        "RMSE": torch.tensor(rmses).mean().item() if len(rmses) > 0 else 0.0,
                        "Flow_Error": (
                            torch.tensor(flow_errors).mean().item()
                            if len(flow_errors) > 0
                            else 0.0
                        ),
                    })

                    per_view_dict[scene_dir][method_key].update({
                        "SSIM": {
                            name: ssim
                            for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)
                        },
                        "PSNR": {
                            name: psnr
                            for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)
                        },
                        "PSNR*": {
                            name: psnr
                            for psnr, name in zip(
                                torch.tensor(psnrs_star).tolist(), image_names
                            )
                        },
                        "LPIPS": {
                            name: lp
                            for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)
                        },
                        "RMSES": rmse_per_view,
                        "Flow_Error": {
                            name: fe
                            for fe, name in zip(flow_errors_per_view, image_names)
                        },
                    })

            # Write both split summaries to the scene directory.
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)


def build_argument_parser():
    """Build the public evaluation command-line interface."""

    parser = ArgumentParser(description="Evaluate rendered EndoPrior-GS models")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str)
    return parser


if __name__ == "__main__":
    parser = build_argument_parser()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("metric evaluation requires a CUDA-capable PyTorch environment")
    torch.cuda.set_device(0)
    evaluate(args.model_paths)
