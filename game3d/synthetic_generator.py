"""Synthetic 3D soil volume used as hidden ground truth."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticVolume:
    values: np.ndarray
    layer_ids: np.ndarray
    boundaries: np.ndarray
    parameter: str = "IC"
    unit: str = "-"


def generate_synthetic_volume(
    shape: tuple[int, int, int] = (16, 16, 8), seed: int = 14
) -> SyntheticVolume:
    """Generate three layers with smooth X/Y boundaries and X/Y/Z variation."""

    if len(shape) != 3 or any(size < 3 for size in shape):
        raise ValueError("shape must contain three dimensions of size >= 3")
    nx, ny, nz = shape
    x = np.linspace(0.0, 1.0, nx)[:, None]
    y = np.linspace(0.0, 1.0, ny)[None, :]
    z = np.arange(nz, dtype=float)[None, None, :]

    first = 0.30 * nz + 0.08 * nz * np.sin(2 * np.pi * x) * np.cos(
        2 * np.pi * y
    ) + 0.025 * nz * np.sin(2 * np.pi * (x + y))
    second = 0.66 * nz + 0.07 * nz * np.cos(2 * np.pi * x + 0.4) * np.sin(
        2 * np.pi * y + 0.2
    ) + 0.02 * nz * np.cos(2 * np.pi * (x - y))
    second = np.maximum(second, first + max(1.0, 0.18 * nz))
    second = np.minimum(second, nz - 0.35)
    boundaries = np.stack((first, second), axis=-1)

    layers = np.where(
        z < first[:, :, None],
        0,
        np.where(z < second[:, :, None], 1, 2),
    ).astype(np.int8)
    base = np.take(np.asarray((1.65, 2.55, 3.45)), layers)
    x3, y3 = x[:, :, None], y[:, :, None]
    z3 = z / (nz - 1)
    smooth = 0.12 * (
        0.48 * np.sin(2 * np.pi * x3 + 0.65 * z3)
        + 0.34 * np.cos(2 * np.pi * y3 - 0.55 * z3)
        + 0.18 * np.sin(2 * np.pi * (x3 + y3) + 0.8 * z3)
    )
    noise = np.random.default_rng(seed).normal(0.0, 0.015, shape)
    return SyntheticVolume(
        values=(base + smooth + noise).astype(np.float32),
        layer_ids=layers,
        boundaries=boundaries.astype(np.float32),
    )
