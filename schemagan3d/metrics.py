"""Physical-unit reconstruction metrics."""

import torch

from .data import NormalizationStats


def volume_metrics(
    prediction: torch.Tensor, target: torch.Tensor, stats: NormalizationStats
) -> dict[str, float]:
    error = stats.denormalize(prediction) - stats.denormalize(target)
    return {
        "rmse": float(torch.sqrt(torch.mean(error.square())).item()),
        "mae": float(torch.mean(torch.abs(error)).item()),
    }


def observed_max_error(
    prediction: torch.Tensor, sparse: torch.Tensor, mask: torch.Tensor
) -> float:
    selected = torch.abs(prediction - sparse) * mask
    return float(selected.max().item())
