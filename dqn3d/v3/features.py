"""Fixed feature schema for a variable number of legal XY candidates."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from synthetic3d.v2.core import Grid
from .adapter import EXPECTED_MEAN, EXPECTED_STD

CANDIDATE_NAMES = (
    'x_local_fraction', 'y_local_fraction', 'nearest_cpt_distance_fraction',
    'local_observed_density_4m', 'valid_depth_fraction',
    'reconstructed_ic_mean_z', 'reconstructed_ic_std_z',
    'depth_bin_0_ic_mean_z', 'depth_bin_1_ic_mean_z',
    'depth_bin_2_ic_mean_z', 'depth_bin_3_ic_mean_z',
    'depth_bin_0_empty', 'depth_bin_1_empty', 'depth_bin_2_empty', 'depth_bin_3_empty',
    'local_4m_reconstructed_ic_mean_z', 'local_4m_valid_fraction',
)
GLOBAL_NAMES = (
    'log_observed_cpt_fraction', 'extent_x_over_32m', 'extent_y_over_32m',
    'extent_z_over_16m', 'dx_over_2m', 'dy_over_2m', 'dz_over_2m',
    'global_reconstructed_ic_mean_z', 'global_reconstructed_ic_std_z',
    'valid_voxel_fraction', 'observed_valid_voxel_fraction',
)


@dataclass(frozen=True)
class CandidateState:
    candidate_xy: np.ndarray  # [N,2], lexicographic (x,y)
    features: np.ndarray      # [N,F]
    global_context: np.ndarray  # [G]

    def __post_init__(self):
        if (self.candidate_xy.ndim != 2 or self.candidate_xy.shape[1] != 2 or
                self.features.shape != (len(self.candidate_xy), len(CANDIDATE_NAMES)) or
                self.global_context.shape != (len(GLOBAL_NAMES),)):
            raise ValueError('candidate state shape mismatch')
        if not np.isfinite(self.features).all() or not np.isfinite(self.global_context).all():
            raise ValueError('nonfinite state features')
        if len({tuple(row) for row in self.candidate_xy}) != len(self.candidate_xy):
            raise ValueError('duplicate candidate')


def build_state(reconstruction: np.ndarray, observed_mask: np.ndarray, valid_mask: np.ndarray,
                grid: Grid, access_mask: np.ndarray | None = None) -> CandidateState:
    shape = (grid.nx, grid.ny, grid.nz)
    rec = np.asarray(reconstruction, dtype=np.float32)
    observed = np.asarray(observed_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if rec.shape != shape or observed.shape != shape or valid.shape != shape:
        raise ValueError('state grid shape mismatch')
    if np.any(observed & ~valid) or not valid.any() or not np.isfinite(rec[valid]).all():
        raise ValueError('invalid masks/reconstruction')
    observed_xy = observed.any(axis=2)
    if not observed_xy.any():
        raise ValueError('at least one observed CPT required')
    if access_mask is None:
        accessible = np.ones((grid.nx, grid.ny), dtype=bool)
    else:
        accessible = np.asarray(access_mask, dtype=bool)
        if accessible.shape != (grid.nx, grid.ny):
            raise ValueError('access mask shape mismatch')
    legal = valid.any(axis=2) & ~observed_xy & accessible
    candidates = np.argwhere(legal).astype(np.int16)
    sites = np.argwhere(observed_xy)
    coords = np.array([grid.xy_m(x, y) for x in range(grid.nx) for y in range(grid.ny)], dtype=np.float64).reshape(grid.nx, grid.ny, 2)
    site_coords = coords[observed_xy]
    diagonal = math.hypot(grid.nx * grid.dx_m, grid.ny * grid.dy_m)
    local_radius = 4.0
    features = np.empty((len(candidates), len(CANDIDATE_NAMES)), dtype=np.float32)
    zrec = (rec - EXPECTED_MEAN) / EXPECTED_STD
    for k, (x, y) in enumerate(candidates):
        xy = coords[x, y]
        distance_to_sites = np.linalg.norm(site_coords - xy, axis=1)
        distances_all = np.linalg.norm(coords - xy, axis=2)
        neighborhood = distances_all <= local_radius + 1e-9
        local_valid = valid[neighborhood]
        local_rec = zrec[neighborhood]
        profile_valid = valid[x, y]
        profile = zrec[x, y, profile_valid]
        bins = np.array_split(np.arange(grid.nz), 4)
        bin_means = []
        bin_empty = []
        for indices in bins:
            keep = profile_valid[indices]
            bin_empty.append(float(not keep.any()))
            bin_means.append(float(zrec[x, y, indices][keep].mean()) if keep.any() else 0.0)
        features[k] = (
            x / (grid.nx - 1), y / (grid.ny - 1),
            min(float(distance_to_sites.min() / diagonal), 1.0),
            float(observed_xy[neighborhood].sum() / max(int(neighborhood.sum()), 1)),
            float(profile_valid.mean()), float(profile.mean()), float(profile.std()),
            *bin_means, *bin_empty,
            float(local_rec[local_valid].mean()) if local_valid.any() else 0.0,
            float(local_valid.mean()),
        )
    global_z = zrec[valid]
    n_sites = int(observed_xy.sum())
    possible_sites = int(valid.any(axis=2).sum())
    context = np.array((
        math.log1p(n_sites) / math.log1p(possible_sites),
        grid.nx * grid.dx_m / 32.0, grid.ny * grid.dy_m / 32.0,
        grid.nz * grid.dz_m / 16.0,
        grid.dx_m / 2.0, grid.dy_m / 2.0, grid.dz_m / 2.0,
        float(global_z.mean()), float(global_z.std()),
        float(valid.mean()), float(observed.sum() / valid.sum()),
    ), dtype=np.float32)
    return CandidateState(candidates, features, context)
