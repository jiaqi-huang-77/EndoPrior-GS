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

cxx_compiler_flags = []

if os.name == "nt":
    cxx_compiler_flags.append("/wd4624")

setup(
    name="simple_knn",
    version="0.1.0",
    description="CUDA nearest-neighbour extension for Gaussian splatting",
    python_requires=">=3.10",
    license_files=["LICENSE.md"],
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
                "spatial.cu",
                "simple_knn.cu",
                "ext.cpp",
            ],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags},
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
