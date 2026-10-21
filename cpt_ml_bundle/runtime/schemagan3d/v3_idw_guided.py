"""Experimental IDW-guided residual SchemaGAN V3; baseline modules stay intact."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from synthetic3d.v2.core import Grid
from .models import Discriminator3D, Generator3D
from .v3_adapter import V3Normalization, V3VolumeDataset, _safe, xyz_to_czyx
from .v3_eval_protocol import idw_physical
from .v3_losses import generator_objective


def idw_normalized_from_observed(
    observed_ic_physical: torch.Tensor,
    mask: torch.Tensor,
    grid: Grid,
    stats: V3Normalization,
) -> torch.Tensor:
    """Make one CZYX IDW channel using measured Ic alone, at physical XY sites.

    The baseline evaluator's depth-wise power-2 IDW is reused. Hidden entries
    are cleared before interpolation, even if the caller passes nonzero values.
    This CPU data preparation is intentionally outside the autograd graph.
    """
    shape = (1, grid.nz, grid.ny, grid.nx)
    if observed_ic_physical.shape != shape or mask.shape != shape:
        raise ValueError('expected one CZYX volume matching Grid')
    if not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('observed mask must be binary')
    sites = mask[0].detach().cpu().numpy().transpose(2, 1, 0).astype(bool)
    values = observed_ic_physical[0].detach().cpu().numpy().transpose(2, 1, 0)
    if not sites.any() or not np.isfinite(values[sites]).all():
        raise ValueError('finite observed Ic required')
    sparse = np.where(sites, values, 0.).astype(np.float32)
    physical = idw_physical(sparse, sites, grid, power=2.)
    normalized = stats.normalize(physical).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError('nonfinite IDW conditioning')
    return torch.from_numpy(xyz_to_czyx(normalized))


class IDWGuidedVolumeDataset(Dataset):
    """Wrap V3's validated split/observation policy with a separate IDW channel."""

    def __init__(self, base: V3VolumeDataset):
        self.base = base
        self.shapes = base.shapes
        self.spacings = base.spacings

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        batch = self.base[index]
        entry = self.base.entries[index]
        grid = Grid(**json.loads(_safe(self.base.root, entry['metadata']).read_text())['grid'])
        idw = idw_normalized_from_observed(batch['observed_ic_physical'], batch['mask'], grid, self.base.stats)
        # Target is added only for supervised loss; IDW never reads it.
        return {**batch, 'idw': idw, 'residual_target': batch['target'] - idw}


class IDWResidualGenerator3D(Generator3D):
    """Baseline U-Net width/depth with a three-channel input and residual output."""

    def __init__(self, base_channels: int = 4):
        super().__init__(base_channels)
        self.enc1[0] = nn.Conv3d(3, base_channels, 3, padding=1)

    def forward(self, sparse: torch.Tensor, mask: torch.Tensor, idw: torch.Tensor):
        if sparse.ndim != 5 or sparse.shape != mask.shape or sparse.shape != idw.shape or sparse.shape[1] != 1:
            raise ValueError('sparse, mask and IDW must share B,1,Z,Y,X shape')
        if not torch.isfinite(idw).all():
            raise ValueError('nonfinite IDW input')
        e1 = self.enc1(torch.cat((sparse, mask, idw), dim=1))
        e2 = self.enc2(self.down1(e1))
        bottleneck = self.bottleneck(self.down2(e2))
        d2 = self.dec2(torch.cat((self.up2(bottleneck), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        residual = self.output(d1)
        reconstruction = torch.where(mask.bool(), sparse, idw + residual)
        return residual, reconstruction


def residual_generator_objective(
    fake_logits: torch.Tensor,
    residual: torch.Tensor,
    target: torch.Tensor,
    idw: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    *,
    reconstruction_weight: float = 100.,
):
    """Existing adversarial term plus hidden-valid L1 on normalized residual.

    On hidden voxels, this L1 equals reconstruction L1 because
    (target-idw)-residual == target-(idw+residual).
    """
    if target.shape != idw.shape:
        raise ValueError('target and IDW shape mismatch')
    return generator_objective(fake_logits, residual, target - idw, mask, valid,
                               reconstruction_weight=reconstruction_weight)


def make_models(base_channels: int = 4):
    """The discriminator retains Pilot 3's three-channel conditioning contract."""
    return IDWResidualGenerator3D(base_channels), Discriminator3D(base_channels)
