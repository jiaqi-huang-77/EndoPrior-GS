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

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


project_directory = os.path.dirname(os.path.abspath(__file__))
glm_directory = os.path.join(project_directory, "third_party", "glm")
glm_header = os.path.join(glm_directory, "glm", "glm.hpp")
if not os.path.isfile(glm_header):
    raise FileNotFoundError(
        "Vendored GLM header is missing: {}. Recreate the checkout before "
        "building the rasteriser.".format(glm_header)
    )

setup(
    name="diff_gaussian_rasterization",
    version="0.1.0",
    description="Depth-aware differentiable Gaussian rasterisation extension",
    python_requires=">=3.10",
    license_files=["LICENSE.md"],
    packages=["diff_gaussian_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            include_dirs=[glm_directory],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
