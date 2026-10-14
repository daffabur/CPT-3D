"""Loss functions for conditional 3D reconstruction GAN training."""

import torch
from torch import nn


_ADVERSARIAL = nn.BCEWithLogitsLoss()


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    real = _ADVERSARIAL(real_logits, torch.ones_like(real_logits))
    fake = _ADVERSARIAL(fake_logits, torch.zeros_like(fake_logits))
    return 0.5 * (real + fake)


def generator_loss(
    fake_logits: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    reconstruction_weight: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, adversarial, and hidden-voxel L1 losses."""

    adversarial = _ADVERSARIAL(fake_logits, torch.ones_like(fake_logits))
    hidden = 1.0 - mask
    reconstruction = (torch.abs(prediction - target) * hidden).sum() / hidden.sum().clamp_min(1.0)
    total = adversarial + reconstruction_weight * reconstruction
    return total, adversarial, reconstruction
