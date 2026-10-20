"""CPT-dependent spatial gate for the existing IDW residual SchemaGAN V3."""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from synthetic3d.v2.core import Grid
from .models import Discriminator3D
from .v3_adapter import _safe
from .v3_idw_guided import IDWGuidedVolumeDataset, IDWResidualGenerator3D


def gate_features_from_mask(mask: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Return [CPT count / 20, nearest physical XY distance / grid diagonal].

    The output is 2,Z,Y,X. Only observation locations and grid coordinates are
    read; hidden Ic, geological units, and target normalization are not used.
    """
    if mask.shape != (1, grid.nz, grid.ny, grid.nx):
        raise ValueError('mask/grid shape mismatch')
    if not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('observation mask must be finite and binary')
    sites = mask[0].detach().cpu().numpy().astype(bool).any(axis=0)
    count = int(sites.sum())
    if count < 1 or count > 20:
        raise ValueError('gate supports 1–20 observed CPT locations')
    coordinates = np.array([grid.xy_m(x, y) for y in range(grid.ny) for x in range(grid.nx)],
                           dtype=np.float64).reshape(grid.ny, grid.nx, 2)
    observed_xy = coordinates[sites]
    distance_m = np.linalg.norm(coordinates[:, :, None, :] - observed_xy[None, None, :, :], axis=-1).min(axis=-1)
    diagonal_m = float(np.linalg.norm(coordinates[0, 0] - coordinates[-1, -1]))
    if not np.isfinite(diagonal_m) or diagonal_m <= 0:
        raise ValueError('invalid physical grid diagonal')
    features = np.stack((np.full((grid.ny, grid.nx), count / 20., dtype=np.float32),
                         (distance_m / diagonal_m).astype(np.float32)))
    return torch.from_numpy(np.ascontiguousarray(np.broadcast_to(features[:, None],
                                                                  (2, grid.nz, grid.ny, grid.nx))))


class GatedIDWVolumeDataset(Dataset):
    """Add gate features to the existing observed-only IDW adapter."""

    def __init__(self, base: IDWGuidedVolumeDataset):
        self.base = base
        self.shapes = base.shapes
        self.spacings = base.spacings

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        entry = self.base.base.entries[index]
        grid = Grid(**json.loads(_safe(self.base.base.root, entry['metadata']).read_text())['grid'])
        return {**item, 'gate_features': gate_features_from_mask(item['mask'], grid)}


class CPTGatedResidualGenerator3D(nn.Module):
    """Jointly trained residual U-Net and shallow spatial gate."""

    def __init__(self, base_channels: int = 4):
        super().__init__()
        self.residual_generator = IDWResidualGenerator3D(base_channels)
        self.gating_network = nn.Sequential(nn.Conv3d(2, 8, 1), nn.SiLU(), nn.Conv3d(8, 1, 1))
        with torch.no_grad():
            nn.init.normal_(self.gating_network[-1].weight, mean=0., std=1e-3)
            self.gating_network[-1].bias.fill_(math.log(.25 / .75))

    def forward_with_gate(self, sparse, mask, idw, gate_features):
        if gate_features.ndim != 5 or gate_features.shape[0] != sparse.shape[0] or gate_features.shape[1] != 2 or gate_features.shape[2:] != sparse.shape[2:]:
            raise ValueError('gate features must have B,2,Z,Y,X shape')
        if not torch.isfinite(gate_features).all():
            raise ValueError('nonfinite gate features')
        raw_residual, _ = self.residual_generator(sparse, mask, idw)
        gate = torch.sigmoid(self.gating_network(gate_features))
        gated_residual = gate * raw_residual
        reconstruction = torch.where(mask.bool(), sparse, idw + gated_residual)
        return gated_residual, reconstruction, gate, raw_residual

    def forward(self, sparse, mask, idw, gate_features):
        gated_residual, reconstruction, _, _ = self.forward_with_gate(sparse, mask, idw, gate_features)
        return gated_residual, reconstruction


def make_models(base_channels: int = 4):
    return CPTGatedResidualGenerator3D(base_channels), Discriminator3D(base_channels)
