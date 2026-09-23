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
from argparse import ArgumentParser, Namespace
from collections import namedtuple
from pathlib import Path

from arguments.presets import PRESET_NAMES, get_preset
import datasets.scared as scared_dataset


_PARAMETER_HELP = {
    'disable_prior_initialisation': 'Ablation: sample initial Gaussians without the texture prior.',
    'disable_prior_density_control': 'Ablation: densify and prune without texture-prior guidance.',
    'disable_prior_temporal_regularisation': 'Ablation: remove the local temporal loss.',
    'disable_prior_temporal_weighting': 'Ablation: retain temporal loss but select visible points without texture-prior weights.',
    'disable_deformation': 'Ablation: disable all time-dependent deformation.',
    'disable_grid': 'Ablation: disable HexPlane features.',
    'disable_position_deformation': 'Ablation: disable position offsets.',
    'disable_scale_deformation': 'Ablation: disable scale offsets.',
    'disable_rotation_deformation': 'Ablation: disable rotation offsets.',
    'disable_opacity_deformation': 'Ablation: disable opacity offsets.',
    'coarse_only': 'Train only the static coarse stage.',
    'local_motion_start_iteration': 'First fine-stage iteration eligible for temporal regularisation.',
    'local_motion_max_points': 'Maximum temporal-support Gaussians; zero removes the limit.',
    'local_motion_min_points': 'Minimum eligible Gaussians needed for temporal regularisation.',
    'local_motion_lambda': 'Weight of the local temporal loss.',
    'local_motion_sigma_scale': 'Multiplier of the adaptive median neighbour-distance scale.',
    'local_motion_k': 'Number of spatial neighbours in the temporal loss.',
    'motion_adaptation_evaluations': 'Maximum temporal-weight adaptation evaluations.',
    'motion_temperature_growth': 'Temperature multiplier during temporal-weight adaptation.',
    'motion_minimum_weight': 'Discard temporal candidates at or below this temporal weight.',
    'depth_loss_weight': 'Weight of the masked inverse-depth loss.',
    'tv_loss_weight': 'Weight of image and inverse-depth total variation.',
    'minimum_valid_depth_pixels': 'Minimum nonzero target depth pixels needed for the depth loss.',
    'sh_degree_interval': 'Iterations between increases of the active spherical-harmonic degree.',
    'pruning_screen_size': 'Screen-space radius threshold for pruning after the first opacity-reset interval.',
}


PriorComponentPolicy = namedtuple(
    "PriorComponentPolicy",
    ("initialisation", "density_control", "temporal_regularisation"),
)


def prior_component_policy(arguments):
    """Resolve independently ablated uses of the joint texture prior."""
    return PriorComponentPolicy(
        initialisation=not bool(getattr(arguments, "disable_prior_initialisation", False)),
        density_control=not bool(getattr(arguments, "disable_prior_density_control", False)),
        temporal_regularisation=not bool(
            getattr(arguments, "disable_prior_temporal_regularisation", False)
        ),
    )


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = key.startswith("_")
            if shorthand:
                key = key[1:]

            default = None if fill_none else value
            option_strings = ["--" + key]
            if shorthand:
                option_strings.append("-" + key[0])

            if isinstance(value, bool):
                if key.startswith("no_"):
                    inverse_option = "--enable-" + key[3:].replace("_", "-")
                elif key.startswith("disable_"):
                    inverse_option = "--enable-" + key[8:].replace("_", "-")
                else:
                    inverse_option = "--no-" + key.replace("_", "-")
                group.add_argument(
                    *option_strings,
                    dest=key,
                    default=default,
                    action="store_true",
                    help=_PARAMETER_HELP.get(key),
                )
                group.add_argument(
                    inverse_option,
                    dest=key,
                    default=default,
                    action="store_false",
                    help=f'Reverse --{key}.',
                )
            else:
                value_parser = json.loads if isinstance(value, (dict, list)) else type(value)
                group.add_argument(*option_strings, default=default, type=value_parser,
                                   help=_PARAMETER_HELP.get(key))

    def extract(self, args):
        group = GroupParams()
        for key, value in vars(args).items():
            if key in vars(self) or ("_" + key) in vars(self):
                setattr(group, key, value)
        return group


class ModelParams(ParamGroup):
    """Dataset, initial point sampling, texture prior and component ablations."""

    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self.dataset_type = "auto"
        # Independent component ablations; all components are enabled by default.
        self.disable_prior_initialisation = False
        self.disable_prior_density_control = False
        self.disable_prior_temporal_regularisation = False
        self.disable_prior_temporal_weighting = False
        self.camera_extent = 10.0
        self.coarse_only = False
        self.initial_point_budget = 20_000
        self.initialise_from_all_frames = True
        self.prior_uniform_mix = 0.45
        self.prior_erosion_kernel = 9
        self.prior_brightness_threshold = 0.85
        self.prior_brightness_percentile = 97.0
        self.prior_saturation_threshold = 0.35
        self.prior_gradient_percentile = 99.0
        self.initialisation_seed = 0

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.convert_spherical_harmonics_python = False
        self.compute_covariance_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters", sentinel)


class ModelHiddenParams(ParamGroup):
    """Deformation architecture and independent deformation-head ablations."""

    def __init__(self, parser, sentinel=False):
        self.deformation_width = 64
        self.deformation_depth = 1
        self.bounds = 1.6
        self.kplanes_config = {
            "grid_dimensions": 2,
            "input_coordinate_dim": 4,
            "output_coordinate_dim": 32,
            "resolution": [64, 64, 64, 25],
        }
        self.multires = [1, 2, 4, 8]
        self.disable_grid = False
        self.disable_position_deformation = False
        self.disable_scale_deformation = False
        self.disable_rotation_deformation = False
        self.disable_opacity_deformation = False
        self.disable_deformation = False
        super().__init__(parser, "Model Hidden Parameters", sentinel)


class OptimizationParams(ParamGroup):
    """Optimisation schedules, density control and loss hyperparameters."""

    def __init__(self, parser):
        self.iterations = 30_000
        self.coarse_iterations = 3_000
        self.sh_degree_interval = 500
        self.depth_loss_weight = 1.0
        self.tv_loss_weight = 0.03
        self.minimum_valid_depth_pixels = 10
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.opacity_reset_interval = 3_000
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 100
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        self.pruning_screen_size = 40.0
        # Persistent Gaussian prior scores and density control.
        self.prior_score_update_interval = 100
        self.prior_densification_floor = 0.35
        self.prior_pruning_threshold = 0.20
        self.prior_min_observations = 20
        self.prior_pruning_opacity_threshold = 0.0075
        self.prior_pruning_max_scale = 40.0
        # Local temporal regularisation and its support-selection budget.
        self.local_motion_start_iteration = 1_000
        self.local_motion_max_points = 1_024
        self.local_motion_min_points = 128
        self.local_motion_lambda = 1e-6
        self.local_motion_k = 8
        self.local_motion_sigma_scale = 1.0
        self.local_motion_distance_weight = 0.2
        self.local_motion_robust_delta = 0.01
        # Adaptive prior-derived weights for temporal support.
        self.motion_adaptation_evaluations = 8
        self.motion_temperature_growth = 1.5
        self.motion_minimum_weight = 1e-4
        self.motion_min_observations = 1
        self.motion_prior_threshold_initial = 0.45
        self.motion_prior_threshold_minimum = 0.25
        self.motion_prior_threshold_decay = 0.05
        self.motion_prior_temperature_initial = 0.08
        self.motion_prior_temperature_maximum = 0.30
        self.motion_target_effective_support = 256

        super().__init__(parser, "Optimisation Parameters")


# Training arguments and saved configuration.


def apply_preset_defaults(parser, preset_name):
    for group in get_preset(preset_name).values():
        parser.set_defaults(**group)
    return parser


def attach_scared_geometry_version(args):
    if args.dataset_type == "scared":
        args.scared_geometry_version = scared_dataset.detect_scared_geometry_version(args.source_path)
        return args.scared_geometry_version


def save_run_config(args, output_directory):
    """Record the exact training arguments for rendering."""
    path = Path(output_directory) / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "preset": getattr(args, "config", None),
               "arguments": vars(args)}
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_run_config(output_directory):
    path = Path(output_directory) / "run_config.json"
    with path.open(encoding="utf-8") as file:
        return Namespace(**json.load(file)["arguments"])


def resolve_render_args(parser, argv):
    command_line = parser.parse_args(argv)
    args = load_run_config(command_line.model_path)
    for name, value in vars(command_line).items():
        if value is not None:
            setattr(args, name, value)
    return args


def validate_training_output(output_directory):
    """Require a new or empty directory so previous experiments stay intact."""

    output_path = Path(output_directory).resolve()
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(f"Training output is not a directory: {output_path}")
    if output_path.is_dir() and any(output_path.iterdir()):
        raise FileExistsError(
            f"Refusing to use non-empty output directory: {output_path}"
        )


def build_render_parser():
    """Build the deliberately narrow rendering command-line interface."""

    parser = ArgumentParser(description="Render a trained EndoPrior-GS model")
    parser.add_argument("--model_path", "-m", required=True)
    parser.add_argument("--source_path", "-s")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--reconstruct", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def build_training_parser():
    parser = ArgumentParser(description="Train EndoPrior-GS: full model, ablations and hyperparameter studies")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--config", choices=PRESET_NAMES, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--expname", type=str, default="")
    return parser, lp, op, pp, hp


def parse_training_args(argv):
    if "-h" in argv or "--help" in argv:
        parser, _, _, _, _ = build_training_parser()
        parser.parse_args(argv)

    selector = ArgumentParser(add_help=False)
    selector.add_argument("--config", choices=PRESET_NAMES, required=True)
    selected, _ = selector.parse_known_args(argv)

    parser, lp, op, pp, hp = build_training_parser()
    apply_preset_defaults(parser, selected.config)
    parsed = parser.parse_args(argv)
    return parsed, lp, op, pp, hp
