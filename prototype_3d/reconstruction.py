"""Synthetic 3D soil volume generation and horizontal profile IDW.

This module is deliberately independent of the existing 2D environment and
agent.  Coordinates use array-index space: ``volume[x, y, z]``.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


GridShape = Tuple[int, int, int]


@dataclass(frozen=True)
class SyntheticSoilVolume:
    """A complete synthetic soil realization and its layer metadata."""

    ic: np.ndarray
    layer_ids: np.ndarray
    boundaries: np.ndarray
    spacing_xy: Tuple[float, float] = (1.0, 1.0)

    @property
    def shape(self) -> GridShape:
        return tuple(self.ic.shape)


def generate_synthetic_volume(
    shape: GridShape = (16, 16, 8),
    seed: int = 14,
    layer_ic: Sequence[float] = (1.65, 2.55, 3.45),
    heterogeneity: float = 0.12,
    noise_std: float = 0.015,
) -> SyntheticSoilVolume:
    """Generate a three-layer volume with smooth, non-planar boundaries.

    The two boundaries are functions of both X and Y.  IC also contains a
    smooth X/Y/Z-dependent component inside each layer, so the result is not
    an extrusion of a single 2D section.
    """

    if len(shape) != 3 or any(size < 3 for size in shape):
        raise ValueError("shape must contain three dimensions of size >= 3")
    if len(layer_ic) != 3:
        raise ValueError("layer_ic must contain exactly three values")
    if heterogeneity < 0 or noise_std < 0:
        raise ValueError("heterogeneity and noise_std must be non-negative")

    nx, ny, nz = shape
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, 1.0, nx, dtype=np.float64)[:, None]
    y = np.linspace(0.0, 1.0, ny, dtype=np.float64)[None, :]

    first = (
        0.30 * nz
        + 0.075 * nz * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
        + 0.025 * nz * np.sin(2.0 * np.pi * (x + y))
    )
    second = (
        0.66 * nz
        + 0.070 * nz * np.cos(2.0 * np.pi * x + 0.4)
        * np.sin(2.0 * np.pi * y + 0.2)
        + 0.020 * nz * np.cos(2.0 * np.pi * (x - y))
    )

    minimum_separation = max(1.0, 0.18 * nz)
    first = np.clip(first, 0.8, nz - minimum_separation - 0.8)
    second = np.clip(
        np.maximum(second, first + minimum_separation),
        first + minimum_separation,
        nz - 0.4,
    )
    boundaries = np.stack((first, second), axis=-1)

    z = np.arange(nz, dtype=np.float64)[None, None, :]
    layer_ids = np.where(
        z < first[:, :, None],
        0,
        np.where(z < second[:, :, None], 1, 2),
    ).astype(np.int8)

    base = np.take(np.asarray(layer_ic, dtype=np.float64), layer_ids)
    x3 = x[:, :, None]
    y3 = y[:, :, None]
    z3 = z / max(nz - 1, 1)
    smooth_variation = heterogeneity * (
        0.48 * np.sin(2.0 * np.pi * x3 + 0.65 * z3)
        + 0.34 * np.cos(2.0 * np.pi * y3 - 0.55 * z3)
        + 0.18 * np.sin(2.0 * np.pi * (x3 + y3) + 0.8 * z3)
    )
    noise = rng.normal(0.0, noise_std, size=shape)
    ic = (base + smooth_variation + noise).astype(np.float32)

    return SyntheticSoilVolume(
        ic=ic,
        layer_ids=layer_ids,
        boundaries=boundaries.astype(np.float32),
    )


def sample_cpt_profile(volume: np.ndarray, position: Sequence[int]) -> np.ndarray:
    """Return a defensive copy of the full Z profile at integer ``(X, Y)``."""

    if volume.ndim != 3:
        raise ValueError("volume must have shape (Nx, Ny, Nz)")
    if len(position) != 2:
        raise ValueError("position must contain X and Y")

    x, y = (int(position[0]), int(position[1]))
    nx, ny, _ = volume.shape
    if not (0 <= x < nx and 0 <= y < ny):
        raise IndexError(f"position {(x, y)} is outside horizontal grid {(nx, ny)}")
    return volume[x, y, :].copy()


def regular_cpt_positions(
    horizontal_shape: Tuple[int, int],
    count_per_axis: int = 3,
) -> np.ndarray:
    """Return a duplicate-free regular sampling design including grid edges."""

    if len(horizontal_shape) != 2 or any(size < 1 for size in horizontal_shape):
        raise ValueError("horizontal_shape must contain two positive sizes")
    if count_per_axis < 1:
        raise ValueError("count_per_axis must be positive")

    nx, ny = horizontal_shape
    xs = np.unique(np.rint(np.linspace(0, nx - 1, count_per_axis)).astype(int))
    ys = np.unique(np.rint(np.linspace(0, ny - 1, count_per_axis)).astype(int))
    return np.asarray([(x, y) for x in xs for y in ys], dtype=np.int64)


def space_filling_cpt_positions(
    horizontal_shape: Tuple[int, int],
    count: int,
    seed: int = 14,
    spacing_xy: Tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Select a reproducible, nested set of well-spaced X/Y locations.

    A seeded first point is followed by greedy farthest-point selection. Calls
    with the same seed and increasing counts share the same prefix.
    """

    if len(horizontal_shape) != 2 or any(size < 1 for size in horizontal_shape):
        raise ValueError("horizontal_shape must contain two positive sizes")
    nx, ny = horizontal_shape
    total = nx * ny
    if not (1 <= count <= total):
        raise ValueError(f"count must be between 1 and {total}")

    spacing = np.asarray(spacing_xy, dtype=np.float64)
    if spacing.shape != (2,) or np.any(spacing <= 0):
        raise ValueError("spacing_xy must contain two positive values")

    grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    candidates = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.int64)
    scaled = candidates.astype(np.float64) * spacing
    rng = np.random.default_rng(seed)
    selected_indexes = [int(rng.integers(total))]
    minimum_squared_distance = np.sum(
        np.square(scaled - scaled[selected_indexes[0]]), axis=1
    )

    for _ in range(1, count):
        minimum_squared_distance[selected_indexes] = -1.0
        next_index = int(np.argmax(minimum_squared_distance))
        selected_indexes.append(next_index)
        squared_distance = np.sum(np.square(scaled - scaled[next_index]), axis=1)
        minimum_squared_distance = np.minimum(minimum_squared_distance, squared_distance)

    return candidates[selected_indexes]


def horizontal_idw(
    sample_positions: np.ndarray,
    sample_profiles: np.ndarray,
    horizontal_shape: Tuple[int, int],
    power: float = 2.0,
    neighbors: int | None = None,
    spacing_xy: Tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Reconstruct ``(Nx, Ny, Nz)`` using horizontal distances only.

    Each sampled value is a complete Z profile.  Neighbor search occurs in
    scaled X/Y space and the same horizontal weights are applied to every Z
    value in a profile.
    """

    positions = np.asarray(sample_positions, dtype=np.float64)
    profiles = np.asarray(sample_profiles, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("sample_positions must have shape (N, 2)")
    if profiles.ndim != 2 or profiles.shape[0] != positions.shape[0]:
        raise ValueError("sample_profiles must have shape (N, Nz)")
    if positions.shape[0] == 0:
        raise ValueError("at least one CPT profile is required")
    if len(horizontal_shape) != 2 or any(size < 1 for size in horizontal_shape):
        raise ValueError("horizontal_shape must contain two positive sizes")
    if power <= 0:
        raise ValueError("power must be positive")

    spacing = np.asarray(spacing_xy, dtype=np.float64)
    if spacing.shape != (2,) or np.any(spacing <= 0):
        raise ValueError("spacing_xy must contain two positive values")

    nx, ny = horizontal_shape
    if np.any(positions[:, 0] < 0) or np.any(positions[:, 0] >= nx):
        raise ValueError("sample X coordinates are outside the grid")
    if np.any(positions[:, 1] < 0) or np.any(positions[:, 1] >= ny):
        raise ValueError("sample Y coordinates are outside the grid")
    if np.unique(positions, axis=0).shape[0] != positions.shape[0]:
        raise ValueError("sample_positions must not contain duplicates")

    if neighbors is None:
        k = positions.shape[0]
    else:
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        k = min(int(neighbors), positions.shape[0])

    grid_x, grid_y = np.meshgrid(
        np.arange(nx, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        indexing="ij",
    )
    prediction_positions = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    tree = cKDTree(positions * spacing)
    distances, indexes = tree.query(prediction_positions * spacing, k=k)
    if k == 1:
        distances = distances[:, None]
        indexes = indexes[:, None]

    exact = distances[:, 0] == 0.0
    safe_distances = np.maximum(distances, np.finfo(np.float64).eps)
    weights = 1.0 / np.power(safe_distances, power)
    weights /= np.sum(weights, axis=1, keepdims=True)
    predicted_profiles = np.sum(
        profiles[indexes] * weights[:, :, None],
        axis=1,
    )
    predicted_profiles[exact] = profiles[indexes[exact, 0]]

    return predicted_profiles.reshape(nx, ny, profiles.shape[1]).astype(np.float32)


def volumetric_rmse(true_volume: np.ndarray, predicted_volume: np.ndarray) -> float:
    """Calculate RMSE across every voxel in two equally shaped volumes."""

    true_values = np.asarray(true_volume, dtype=np.float64)
    predicted_values = np.asarray(predicted_volume, dtype=np.float64)
    if true_values.shape != predicted_values.shape:
        raise ValueError("true_volume and predicted_volume must have equal shapes")
    return float(np.sqrt(np.mean(np.square(true_values - predicted_values))))


def volumetric_mae(true_volume: np.ndarray, predicted_volume: np.ndarray) -> float:
    """Calculate MAE across every voxel in two equally shaped volumes."""

    true_values = np.asarray(true_volume, dtype=np.float64)
    predicted_values = np.asarray(predicted_volume, dtype=np.float64)
    if true_values.shape != predicted_values.shape:
        raise ValueError("true_volume and predicted_volume must have equal shapes")
    return float(np.mean(np.abs(true_values - predicted_values)))
