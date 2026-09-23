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

import torch
from torch import nn
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


class Camera(nn.Module):
    def __init__(
        self,
        R,
        T,
        FoVx,
        FoVy,
        image,
        depth,
        mask,
        image_name,
        uid,
        time=0,
        Znear=None,
        Zfar=None,
        motion_mask=None,
        principal_x=None,
        principal_y=None,
    ):
        super().__init__()

        self.uid = uid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        self.mask = mask
        # Optional stricter valid-motion region, such as a foreground mask.
        self.motion_mask = motion_mask if motion_mask is not None else mask
        self.original_image = image.clamp(0.0, 1.0)
        self.original_depth = depth
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.principal_x = (
            (self.image_width - 1.0) / 2.0
            if principal_x is None
            else float(principal_x)
        )
        self.principal_y = (
            (self.image_height - 1.0) / 2.0
            if principal_y is None
            else float(principal_y)
        )
        if Zfar is not None and Znear is not None:
            self.zfar = Zfar
            self.znear = Znear
        else:
            self.zfar = 120.0
            self.znear = 0.01

        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
            image_width=self.image_width,
            image_height=self.image_height,
            principal_x=self.principal_x,
            principal_y=self.principal_y,
        ).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]
