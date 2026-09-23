
import torch
import torch.nn as nn
import torch.nn.init as init

from scene.hexplane import HexPlaneField


class Deformation(nn.Module):
    def __init__(self, depth, width, args):
        super().__init__()
        self.depth = depth
        self.width = width
        self.disable_grid = args.disable_grid
        self.args = args
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)

        self.feature_out = self._create_feature_network()
        (
            self.pos_deform,
            self.scales_deform,
            self.rotations_deform,
            self.opacity_deform,
        ) = self._create_deformation_heads()

    def _create_feature_network(self):
        input_width = 4 if self.disable_grid else self.grid.feat_dim
        layers = [nn.Linear(input_width, self.width)]
        for _ in range(self.depth - 1):
            layers.extend((nn.ReLU(), nn.Linear(self.width, self.width)))
        return nn.Sequential(*layers)

    def _deformation_head(self, output_width):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, output_width),
        )

    def _create_deformation_heads(self):
        return (
            self._deformation_head(3),
            self._deformation_head(3),
            self._deformation_head(4),
            self._deformation_head(1),
        )

    def _query_features(self, points, times):
        if self.disable_grid:
            features = torch.cat((points[:, :3], times[:, :1]), dim=-1)
        else:
            features = self.grid(points[:, :3], times[:, :1])
        return self.feature_out(features)

    def forward(self, points, scales, rotations, opacity, times):
        hidden = self._query_features(points, times).float()

        if self.args.disable_position_deformation:
            deformed_points = points[:, :3]
        else:
            deformed_points = points[:, :3] + self.pos_deform(hidden)

        if self.args.disable_scale_deformation:
            deformed_scales = scales[:, :3]
        else:
            deformed_scales = scales[:, :3] + self.scales_deform(hidden)

        if self.args.disable_rotation_deformation:
            deformed_rotations = rotations[:, :4]
        else:
            deformed_rotations = rotations[:, :4] + self.rotations_deform(hidden)

        if self.args.disable_opacity_deformation:
            deformed_opacity = opacity[:, :1]
        else:
            deformed_opacity = opacity[:, :1] + self.opacity_deform(hidden)

        return (
            deformed_points,
            deformed_scales,
            deformed_rotations,
            deformed_opacity,
        )

    def get_mlp_parameters(self):
        return [
            parameter
            for name, parameter in self.named_parameters()
            if "grid" not in name
        ]

    def get_grid_parameters(self):
        return list(self.grid.parameters())


class DeformationNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.deformation_net = Deformation(
            depth=args.deformation_depth,
            width=args.deformation_width,
            args=args,
        )
        self.apply(initialize_weights)

    def forward(self, points, scales, rotations, opacity, times_sel):
        return self.deformation_net(
            points,
            scales,
            rotations,
            opacity,
            times_sel,
        )

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters()

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight, gain=1)
        if module.bias is not None:
            init.zeros_(module.bias)


_LEGACY_STATE_KEYS = {
    "time_poc",
    "pos_poc",
    "rotation_scaling_poc",
    "opacity_poc",
}


def load_deformation_state(module, state_dict):
    """Load current weights while discarding the old, unused time encoder."""

    filtered = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("timenet.") and key not in _LEGACY_STATE_KEYS
    }
    incompatible = module.load_state_dict(filtered, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Incompatible deformation checkpoint: missing={}, unexpected={}".format(
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )
        )
