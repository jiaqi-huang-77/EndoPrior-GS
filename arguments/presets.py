"""Canonical, publication-facing experiment presets.

Presets contain only dataset defaults. Runtime paths and explicit command-
line overrides remain command-line arguments, which keeps the published settings
reusable across machines and sequences.
"""

from copy import deepcopy


_ENDOPRIOR_MODEL = {
    "prior_uniform_mix": 0.45,
    "prior_erosion_kernel": 9,
    "prior_brightness_threshold": 0.85,
    "prior_brightness_percentile": 97.0,
    "prior_saturation_threshold": 0.35,
    "prior_gradient_percentile": 99.0,
    "initialisation_seed": 0,
}

_ENDOPRIOR_OPTIMISATION = {
    "coarse_iterations": 1_000,
    "iterations": 3_000,
    "percent_dense": 0.01,
    "opacity_reset_interval": 3_000,
    "position_lr_max_steps": 3_000,
    "pruning_interval": 100,
    "opacity_threshold_coarse": 0.005,
    "opacity_threshold_fine_init": 0.005,
    "opacity_threshold_fine_after": 0.005,
    "prior_score_update_interval": 100,
    "prior_densification_floor": 0.35,
    "prior_pruning_threshold": 0.20,
    "prior_min_observations": 20,
    "prior_pruning_opacity_threshold": 0.0075,
    "prior_pruning_max_scale": 40.0,
    "motion_min_observations": 1,
    "motion_prior_threshold_initial": 0.45,
    "motion_prior_threshold_minimum": 0.25,
    "motion_prior_threshold_decay": 0.05,
    "motion_prior_temperature_initial": 0.08,
    "motion_prior_temperature_maximum": 0.30,
    "motion_target_effective_support": 256,
    "local_motion_lambda": 1e-6,
    "local_motion_k": 8,
    "local_motion_sigma_scale": 1.0,
    "local_motion_distance_weight": 0.20,
    "local_motion_robust_delta": 0.01,
}

_ENDONERF_HIDDEN = {
    "kplanes_config": {
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 64,
        "resolution": [64, 64, 64, 100],
    },
    "multires": [1, 2, 4, 8],
    "deformation_depth": 0,
    "deformation_width": 32,
}

_SCARED_HIDDEN = {
    "kplanes_config": {
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 32,
        "resolution": [64, 64, 64, 100],
    },
    "multires": [1, 2, 4, 8],
    "deformation_depth": 0,
    "deformation_width": 32,
    "disable_scale_deformation": True,
    "disable_rotation_deformation": True,
}


_PRESETS = {
    "endoprior-endonerf": {
        "ModelParams": {
            **_ENDOPRIOR_MODEL,
            "dataset_type": "endonerf",
            "camera_extent": 10,
            "initial_point_budget": 0,
            "initialise_from_all_frames": True,
            "initial_points_per_frame_min": 210,
            "initial_points_per_frame_max": 430,
            "initial_points_ess_ratio": 0.26,
        },
        "OptimizationParams": {
            **_ENDOPRIOR_OPTIMISATION,
            "deformation_lr_init": 0.00016,
            "deformation_lr_final": 0.0000016,
            "deformation_lr_delay_mult": 0.01,
            "grid_lr_init": 0.0016,
            "grid_lr_final": 0.000016,
        },
        "ModelHiddenParams": _ENDONERF_HIDDEN,
    },
    "endoprior-scared": {
        "ModelParams": {
            **_ENDOPRIOR_MODEL,
            "dataset_type": "scared",
            "initial_point_budget": 20_000,
            "initialise_from_all_frames": True,
        },
        "OptimizationParams": {
            **_ENDOPRIOR_OPTIMISATION,
            "position_lr_init": 0.00016,
            "position_lr_final": 0.0000016,
            "position_lr_delay_mult": 0.01,
            "deformation_lr_init": 0.00016,
            "deformation_lr_final": 0.0000016,
            "deformation_lr_delay_mult": 0.01,
            "grid_lr_init": 0.0016,
            "grid_lr_final": 0.000016,
        },
        "ModelHiddenParams": _SCARED_HIDDEN,
    },
}


_PRESETS["endoprior-stereomis"] = deepcopy(_PRESETS["endoprior-endonerf"])
_PRESETS["endoprior-stereomis"]["ModelParams"]["dataset_type"] = "stereomis"

PRESET_NAMES = tuple(_PRESETS)


def get_preset(name):
    """Return an independent copy of a named public preset."""

    try:
        preset = _PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(PRESET_NAMES)
        raise KeyError("Unknown preset {!r}; choose one of: {}".format(name, choices)) from exc
    return deepcopy(preset)
