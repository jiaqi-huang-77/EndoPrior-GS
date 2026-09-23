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

import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, stage="fine"):
    """Render a view with a GPU-resident background tensor."""
    # Retain gradients for the projected Gaussian centres used during densification.
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    ) + 0
    try:
        screenspace_points.retain_grad()
    except RuntimeError:
        pass

    # Configure rasterisation.
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # World-space means.
    means3D = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)
    means2D = screenspace_points
    opacity = pc._opacity

    # Use activated scale and rotation in the fast rasterisation path.
    scales = pc._scaling
    rotations = pc._rotation
    cov3D_precomp = None

    deformation_point = pc._deformation_table

    use_deformation = (
        stage != "coarse"
        and not getattr(pc, "disable_deformation", False)
        and deformation_point.any()
    )

    if use_deformation:
        means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(
            means3D[deformation_point],
            scales[deformation_point],
            rotations[deformation_point],
            opacity[deformation_point],
            time[deformation_point],
        )

        # Accumulate displacement only for Gaussians processed by the deformation field.
        with torch.no_grad():
            pc._deformation_accum[deformation_point] += torch.abs(
                means3D_deform - means3D[deformation_point]
            )

        means3D_final = means3D.clone()
        rotations_final = rotations.clone()
        scales_final = scales.clone()
        opacity_final = opacity.clone()

        means3D_final[deformation_point] = means3D_deform
        rotations_final[deformation_point] = rotations_deform
        scales_final[deformation_point] = scales_deform
        opacity_final[deformation_point] = opacity_deform

    else:
        means3D_final = means3D
        rotations_final = rotations
        scales_final = scales
        opacity_final = opacity
    # Convert latent parameters to their physical values.
    if scales_final is not None:
        scales_final = pc.scaling_activation(scales_final)
    if rotations_final is not None:
        rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)

    if pipe.compute_covariance_python:
        cov3D_precomp = pc.covariance_activation(scales_final, scaling_modifier, rotations_final)
        scales_final = None
        rotations_final = None

    # Precompute colours from spherical harmonics when requested; otherwise the
    # rasteriser performs the conversion.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_spherical_harmonics_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = means3D_final - viewpoint_camera.camera_center.cuda().repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterise visible Gaussians and return their screen-space radii.
    rendered_image, radii, depth = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity_final,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "depth": depth,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "means3D_deform": means3D_final,
        "scales_deform": scales_final,
    }
