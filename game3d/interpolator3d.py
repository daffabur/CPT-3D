"""Horizontal IDW for complete depth profiles."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterpolationResult:
    volume: np.ndarray
    uncertainty: np.ndarray


def horizontal_idw(
    positions_xy: np.ndarray,
    profiles: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    *,
    power: float = 2.0,
) -> InterpolationResult:
    """Interpolate profiles using X/Y distance, ignoring unavailable depths.

    ``profiles`` may contain NaN at elevations outside an individual real CPT.
    The interpolator has no ground-truth argument and cannot access hidden data.
    """

    positions = np.asarray(positions_xy, dtype=float)
    observed = np.asarray(profiles, dtype=float)
    xs, ys = np.asarray(target_x, dtype=float), np.asarray(target_y, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions_xy must have shape (N, 2)")
    if observed.ndim != 2 or observed.shape[0] != len(positions):
        raise ValueError("profiles must have shape (N, Nz)")
    if len(positions) == 0 or power <= 0:
        raise ValueError("at least one profile and positive power are required")
    if np.unique(positions, axis=0).shape[0] != len(positions):
        raise ValueError("positions_xy must not contain duplicates")

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    targets = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    distances = np.linalg.norm(targets[:, None, :] - positions[None, :, :], axis=2)
    exact = distances <= np.finfo(float).eps
    safe = np.maximum(distances, np.finfo(float).eps)
    base_weights = 1.0 / np.power(safe, power)

    predictions = np.full((len(targets), observed.shape[1]), np.nan)
    for depth_index in range(observed.shape[1]):
        available = np.isfinite(observed[:, depth_index])
        if not available.any():
            continue
        values = observed[available, depth_index]
        weights = base_weights[:, available]
        weights /= weights.sum(axis=1, keepdims=True)
        predictions[:, depth_index] = weights @ values
        available_indices = np.flatnonzero(available)
        for target_index in np.flatnonzero(exact[:, available].any(axis=1)):
            source_index = available_indices[
                np.flatnonzero(exact[target_index, available])[0]
            ]
            predictions[target_index, depth_index] = observed[source_index, depth_index]

    nearest = distances.min(axis=1)
    domain_diagonal = max(
        float(np.hypot(np.ptp(xs), np.ptp(ys))), np.finfo(float).eps
    )
    uncertainty = np.clip(nearest / domain_diagonal, 0.0, 1.0)
    return InterpolationResult(
        volume=predictions.reshape(len(xs), len(ys), observed.shape[1]).astype(
            np.float32
        ),
        uncertainty=np.broadcast_to(
            uncertainty.reshape(len(xs), len(ys), 1),
            (len(xs), len(ys), observed.shape[1]),
        ).copy().astype(np.float32),
    )


def volumetric_errors(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction must have equal shape")
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not valid.any():
        return float("nan"), float("nan")
    difference = truth[valid] - prediction[valid]
    return float(np.sqrt(np.mean(difference**2))), float(np.mean(np.abs(difference)))
