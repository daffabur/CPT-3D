"""CSV loader and sparse full-profile CPT conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from synthetic3d.generate_csv import read_csv_volume


@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std: float

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


def sorted_csv_paths(directory: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(
        directory.glob("cs3d_*.csv"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    return paths if limit is None else paths[:limit]


def compute_normalization(paths: list[Path]) -> NormalizationStats:
    """Compute statistics from training files only, one volume at a time."""

    if not paths:
        raise ValueError("at least one training CSV is required")
    total = total_square = 0.0
    count = 0
    for path in paths:
        values = read_csv_volume(path).astype(np.float64)
        total += float(values.sum())
        total_square += float(np.square(values).sum())
        count += values.size
    mean = total / count
    variance = max(total_square / count - mean * mean, 1e-12)
    return NormalizationStats(mean=mean, std=variance**0.5)


def make_sparse_observation(
    normalized_volume: torch.Tensor,
    *,
    profile_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reveal complete Z profiles at selected X/Y locations only."""

    if normalized_volume.shape != (1, 8, 16, 16):
        raise ValueError("normalized_volume must have shape (1, 8, 16, 16)")
    if not 1 <= profile_count <= 16 * 16:
        raise ValueError("profile_count must be between 1 and 256")
    rng = np.random.default_rng(seed)
    flat_positions = rng.choice(16 * 16, size=profile_count, replace=False)
    positions = np.column_stack((flat_positions % 16, flat_positions // 16)).astype(
        np.int64
    )
    mask = torch.zeros_like(normalized_volume)
    for x, y in positions:
        mask[:, :, y, x] = 1.0
    sparse = normalized_volume * mask
    return sparse, mask, torch.from_numpy(positions)


class SoilVolumeDataset(Dataset):
    """Lazy dataset returning target separately from generator inputs."""

    def __init__(
        self,
        directory: Path,
        *,
        stats: NormalizationStats,
        limit: int | None = None,
        min_profiles: int = 3,
        max_profiles: int = 20,
        observation_seed: int = 1234,
    ):
        self.paths = sorted_csv_paths(directory, limit)
        if not self.paths:
            raise ValueError(f"no CSV files found in {directory}")
        if not 1 <= min_profiles <= max_profiles <= 256:
            raise ValueError("invalid profile-count range")
        self.stats = stats
        self.min_profiles = min_profiles
        self.max_profiles = max_profiles
        self.observation_seed = observation_seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        xyz = read_csv_volume(self.paths[index])
        # CSV axes X,Y,Z -> Conv3D axes channel,Z,Y,X.
        target = torch.from_numpy(xyz.transpose(2, 1, 0).copy()).unsqueeze(0)
        target = self.stats.normalize(target)
        rng = np.random.default_rng(self.observation_seed + index)
        count = int(rng.integers(self.min_profiles, self.max_profiles + 1))
        sparse, mask, _positions = make_sparse_observation(
            target, profile_count=count, seed=self.observation_seed + 10_000 + index
        )
        return {
            "sparse": sparse.float(),
            "mask": mask.float(),
            "target": target.float(),
            "profile_count": torch.tensor(count, dtype=torch.int64),
            "path": str(self.paths[index]),
        }
