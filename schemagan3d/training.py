"""Reusable training and evaluation operations."""

import torch

from .losses import discriminator_loss, generator_loss


def optimization_step(
    generator,
    discriminator,
    batch,
    generator_optimizer,
    discriminator_optimizer,
    *,
    reconstruction_weight: float = 100.0,
) -> dict[str, float]:
    sparse, mask, target = batch["sparse"], batch["mask"], batch["target"]

    discriminator_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        detached_prediction = generator(sparse, mask)
    real_logits = discriminator(target, sparse, mask)
    fake_logits = discriminator(detached_prediction, sparse, mask)
    d_loss = discriminator_loss(real_logits, fake_logits)
    d_loss.backward()
    discriminator_optimizer.step()

    generator_optimizer.zero_grad(set_to_none=True)
    prediction = generator(sparse, mask)
    fake_logits = discriminator(prediction, sparse, mask)
    g_loss, adversarial, reconstruction = generator_loss(
        fake_logits,
        prediction,
        target,
        mask,
        reconstruction_weight=reconstruction_weight,
    )
    g_loss.backward()
    generator_optimizer.step()
    return {
        "generator": float(g_loss.item()),
        "discriminator": float(d_loss.item()),
        "adversarial": float(adversarial.item()),
        "reconstruction": float(reconstruction.item()),
    }
