"""Synthetic 3D CPT-placement environment with a compact horizontal action space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Reconstructor(Protocol):
    def predict(self, observed_ic: np.ndarray, observed_mask: np.ndarray) -> tuple[np.ndarray, float]: ...


@dataclass(frozen=True)
class EnvironmentConfig:
    budget: int = 8
    sampling_cost: float = 0.005
    reward_mode: str = "oracle_rmse"  # alternatives: oracle_mae, coverage_proxy
    action_cells: int = 4  # 4 x 4 action regions, 16 discrete actions

    def __post_init__(self) -> None:
        if not 1 <= self.budget <= 256:
            raise ValueError("budget must be 1..256")
        if self.action_cells < 1 or 16 % self.action_cells:
            raise ValueError("action_cells must divide 16")
        if self.sampling_cost < 0:
            raise ValueError("sampling_cost must be nonnegative")
        if self.reward_mode not in ("oracle_rmse", "oracle_mae", "coverage_proxy"):
            raise ValueError("unsupported reward mode")


class CPTPlacementEnv3D:
    """One action selects one XY site, which reveals all eight Z levels.

    The environment keeps ground truth private. The returned state contains
    only observed profiles, their mask, reconstruction, and remaining budget.
    """

    def __init__(self, reconstructor: Reconstructor, config: EnvironmentConfig = EnvironmentConfig()):
        self.reconstructor = reconstructor
        self.config = config
        self.action_count = config.action_cells**2
        self._truth: np.ndarray | None = None
        self.observed_ic = np.zeros((16, 16, 8), dtype=np.float32)
        self.observed_mask = np.zeros((16, 16, 8), dtype=bool)
        self.prediction = np.zeros((16, 16, 8), dtype=np.float32)
        self.positions: list[tuple[int, int]] = []
        self.last_reconstruction_seconds = 0.0

    def reset(self, volume: np.ndarray, *, initial_xy: tuple[int, int] | list[tuple[int, int]] = (8, 8)) -> tuple[dict, dict]:
        if volume.shape != (16, 16, 8) or not np.all(np.isfinite(volume)):
            raise ValueError("expected finite IC volume of shape (16,16,8)")
        initial_sites = [initial_xy] if len(initial_xy) == 2 and isinstance(initial_xy[0], (int, np.integer)) else list(initial_xy)
        if not initial_sites or len(initial_sites) > self.config.budget:
            raise ValueError("initial CPT count must be between one and the budget")
        for xy in initial_sites:
            self._validate_xy(xy)
        if len({tuple(xy) for xy in initial_sites}) != len(initial_sites):
            raise ValueError("initial CPT coordinates must be unique")
        self._truth = np.array(volume, dtype=np.float32, copy=True)
        self.observed_ic.fill(0)
        self.observed_mask.fill(False)
        self.positions = []
        for xy in initial_sites:
            self._sample(tuple(xy))
        self._reconstruct()
        return self.state(), self._info()

    @staticmethod
    def _validate_xy(xy: tuple[int, int]) -> None:
        if len(xy) != 2 or any(not isinstance(v, (int, np.integer)) or v < 0 or v >= 16 for v in xy):
            raise ValueError("XY coordinate must contain two integers in 0..15")

    def _sample(self, xy: tuple[int, int]) -> None:
        if self._truth is None:
            raise RuntimeError("reset must be called first")
        x, y = xy
        if self.observed_mask[x, y, 0]:
            raise ValueError(f"CPT site {xy} was already sampled")
        self.observed_ic[x, y, :] = self._truth[x, y, :]
        self.observed_mask[x, y, :] = True
        self.positions.append((x, y))

    def _reconstruct(self) -> None:
        result, seconds = self.reconstructor.predict(self.observed_ic.copy(), self.observed_mask.copy())
        if result.shape != (16, 16, 8) or not np.all(np.isfinite(result)):
            raise ValueError("reconstructor returned invalid volume")
        self.prediction = np.array(result, dtype=np.float32, copy=True)
        self.last_reconstruction_seconds = float(seconds)

    def candidate_xy(self, action: int) -> tuple[int, int] | None:
        """Return the nearest unsampled grid point to a cell's center."""
        if not 0 <= action < self.action_count:
            raise ValueError("action index out of range")
        width = 16 // self.config.action_cells
        cell_x, cell_y = divmod(action, self.config.action_cells)
        center = (cell_x * width + (width - 1) / 2, cell_y * width + (width - 1) / 2)
        candidates = [(x, y) for x in range(cell_x * width, (cell_x + 1) * width)
                      for y in range(cell_y * width, (cell_y + 1) * width)
                      if not self.observed_mask[x, y, 0]]
        if not candidates:
            return None
        return min(candidates, key=lambda p: ((p[0] - center[0])**2 + (p[1] - center[1])**2, p[0], p[1]))

    def valid_actions(self) -> np.ndarray:
        return np.array([self.candidate_xy(a) is not None for a in range(self.action_count)], dtype=bool)

    def state(self) -> dict:
        if self._truth is None:
            raise RuntimeError("reset must be called first")
        # Channel,Z,Y,X for a future convolutional Q-network.
        channels = np.stack((self.observed_ic, self.observed_mask.astype(np.float32), self.prediction))
        return {
            "volume": channels.transpose(0, 3, 2, 1).copy(),
            "remaining_budget": np.float32((self.config.budget - len(self.positions)) / self.config.budget),
            "valid_actions": self.valid_actions(),
        }

    def _rmse_all(self) -> float:
        assert self._truth is not None
        return float(np.sqrt(np.mean(np.square(self.prediction - self._truth))))

    def _mae_all(self) -> float:
        assert self._truth is not None
        return float(np.mean(np.abs(self.prediction - self._truth)))

    def _coverage_proxy(self) -> float:
        sites = np.asarray(self.positions, dtype=np.float32)
        grid = np.indices((16, 16)).reshape(2, -1).T.astype(np.float32)
        distances = np.linalg.norm(grid[:, None] - sites[None], axis=-1)
        return float(np.mean(np.min(distances, axis=1)) / np.hypot(15, 15))

    def _info(self) -> dict:
        assert self._truth is not None
        hidden = ~self.observed_mask
        error = self.prediction - self._truth
        return {
            "sampled_xy": tuple(self.positions),
            "cpt_count": len(self.positions),
            "rmse_all": self._rmse_all(),
            "rmse_hidden": float(np.sqrt(np.mean(np.square(error[hidden])))) if hidden.any() else 0.0,
            "mae_hidden": float(np.mean(np.abs(error[hidden]))) if hidden.any() else 0.0,
            "coverage_proxy": self._coverage_proxy(),
            "reconstruction_seconds": self.last_reconstruction_seconds,
        }

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        if self._truth is None:
            raise RuntimeError("reset must be called first")
        if len(self.positions) >= self.config.budget:
            raise RuntimeError("sampling budget exhausted")
        xy = self.candidate_xy(action)
        if xy is None:
            raise ValueError("action region has no unused CPT site")
        score = {"oracle_rmse": self._rmse_all, "oracle_mae": self._mae_all,
                 "coverage_proxy": self._coverage_proxy}[self.config.reward_mode]
        before = score()
        self._sample(xy)
        self._reconstruct()
        after = score()
        reward = before - after - self.config.sampling_cost
        done = len(self.positions) >= self.config.budget or not self.valid_actions().any()
        info = self._info()
        info.update(action=action, selected_xy=xy, reward_before=before, reward_after=after)
        return self.state(), float(reward), done, info
