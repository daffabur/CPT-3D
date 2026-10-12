"""Stateful environments shared by the UI and future agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_loader import CPTDataset, load_cpt_csv, profiles_on_common_rl
from .interpolator3d import horizontal_idw, volumetric_errors
from .synthetic_generator import SyntheticVolume, generate_synthetic_volume


@dataclass(frozen=True)
class SampleEvent:
    position_xy: tuple[float, float]
    profile: np.ndarray
    measured: bool
    message: str


class SyntheticEnvironment3D:
    """Grid environment where Space reveals one complete hidden IC profile."""

    parameter = "IC"
    unit = "-"

    def __init__(self, seed: int = 14, shape: tuple[int, int, int] = (16, 16, 8)):
        self.synthetic: SyntheticVolume = generate_synthetic_volume(shape, seed)
        self.truth = self.synthetic.values
        self.x = np.arange(shape[0], dtype=float)
        self.y = np.arange(shape[1], dtype=float)
        self.z = -np.arange(shape[2], dtype=float)
        self.vehicle = np.asarray([shape[0] // 2, shape[1] // 2], dtype=int)
        self.sample_positions: list[tuple[int, int]] = []
        self.sample_profiles: list[np.ndarray] = []
        self.prediction: np.ndarray | None = None
        self.uncertainty: np.ndarray | None = None
        self.rmse = float("nan")
        self.mae = float("nan")

    def move(self, dx: int, dy: int) -> None:
        self.vehicle += (dx, dy)
        self.vehicle[0] = np.clip(self.vehicle[0], 0, len(self.x) - 1)
        self.vehicle[1] = np.clip(self.vehicle[1], 0, len(self.y) - 1)

    def sample(self) -> SampleEvent:
        position = tuple(int(value) for value in self.vehicle)
        profile = self.truth[position[0], position[1], :].copy()
        if position not in self.sample_positions:
            self.sample_positions.append(position)
            self.sample_profiles.append(profile)
            result = horizontal_idw(
                np.asarray(self.sample_positions),
                np.asarray(self.sample_profiles),
                self.x,
                self.y,
            )
            self.prediction, self.uncertainty = result.volume, result.uncertainty
            self.rmse, self.mae = volumetric_errors(self.truth, self.prediction)
            message = f"Measured synthetic CPT at {position}."
        else:
            message = f"Synthetic CPT at {position} was already sampled."
        return SampleEvent(position, profile, True, message)

    @property
    def cpt_count(self) -> int:
        return len(self.sample_positions)


class RealCPTEnvironment3D:
    """Read-only real observations plus an interpolated qc volume."""

    parameter = "qc"
    unit = "kg/cm²"

    def __init__(self, csv_path: Path, grid_size: int = 16, margin_m: float = 12.0):
        self.dataset: CPTDataset = load_cpt_csv(csv_path)
        minimum = self.dataset.positions_xy.min(axis=0) - margin_m
        maximum = self.dataset.positions_xy.max(axis=0) + margin_m
        self.x = np.linspace(minimum[0], maximum[0], grid_size)
        self.y = np.linspace(minimum[1], maximum[1], grid_size)
        self.z, common_profiles = profiles_on_common_rl(self.dataset)
        result = horizontal_idw(
            self.dataset.positions_xy,
            common_profiles,
            self.x,
            self.y,
        )
        self.prediction, self.uncertainty = result.volume, result.uncertainty
        self.vehicle = np.asarray([grid_size // 2, grid_size // 2], dtype=int)
        self.last_prediction_event: SampleEvent | None = None

    def move(self, dx: int, dy: int) -> None:
        self.vehicle += (dx, dy)
        self.vehicle[0] = np.clip(self.vehicle[0], 0, len(self.x) - 1)
        self.vehicle[1] = np.clip(self.vehicle[1], 0, len(self.y) - 1)

    @property
    def vehicle_xy(self) -> tuple[float, float]:
        return float(self.x[self.vehicle[0]]), float(self.y[self.vehicle[1]])

    def sample(self) -> SampleEvent:
        """Return prediction only; never mutate the three measured CPT profiles."""

        x_index, y_index = self.vehicle
        profile = self.prediction[x_index, y_index, :].copy()
        event = SampleEvent(
            self.vehicle_xy,
            profile,
            False,
            "Prediction preview only - no real CPT measurement was added.",
        )
        self.last_prediction_event = event
        return event

    @property
    def cpt_count(self) -> int:
        return len(self.dataset.ids)
