"""V3 loss components; V1 losses remain unchanged."""
import torch
from torch import nn


def hidden_valid_l1(prediction:torch.Tensor,target:torch.Tensor,observed_mask:torch.Tensor,valid_mask:torch.Tensor)->torch.Tensor:
    """Mean over volumes of each volume's mean normalized-Ic hidden-valid L1.

    Raises if any volume has no hidden valid voxels. Invalid/observed values are
    excluded with torch.where, avoiding 0*NaN propagation from excluded voxels.
    """
    if prediction.ndim!=5 or prediction.shape!=target.shape or prediction.shape!=observed_mask.shape or prediction.shape!=valid_mask.shape:
        raise ValueError('all tensors must share B,C,Z,Y,X shape')
    hidden=valid_mask.bool() & ~observed_mask.bool()
    counts=hidden.flatten(1).sum(1)
    if (counts==0).any():raise ValueError('each volume must contain hidden valid voxels')
    selected=torch.where(hidden,torch.abs(prediction-target),torch.zeros_like(prediction))
    losses=selected.flatten(1).sum(1)/counts
    if not torch.isfinite(losses).all():raise ValueError('nonfinite hidden-valid reconstruction loss')
    return losses.mean()


def hidden_valid_mse(prediction:torch.Tensor,target:torch.Tensor,observed_mask:torch.Tensor,valid_mask:torch.Tensor)->torch.Tensor:
    """Mean of per-volume hidden-valid squared normalized-Ic errors."""
    if prediction.ndim!=5 or prediction.shape!=target.shape or prediction.shape!=observed_mask.shape or prediction.shape!=valid_mask.shape:
        raise ValueError('all tensors must share B,C,Z,Y,X shape')
    hidden=valid_mask.bool() & ~observed_mask.bool()
    counts=hidden.flatten(1).sum(1)
    if (counts==0).any():raise ValueError('each volume must contain hidden valid voxels')
    selected=torch.where(hidden,(prediction-target).square(),torch.zeros_like(prediction))
    losses=selected.flatten(1).sum(1)/counts
    if not torch.isfinite(losses).all():raise ValueError('nonfinite hidden-valid reconstruction loss')
    return losses.mean()


def generator_objective(fake_logits,prediction,target,observed_mask,valid_mask,*,reconstruction_weight:float=100.0):
    if reconstruction_weight<=0:raise ValueError('positive reconstruction weight required')
    adversarial=nn.functional.binary_cross_entropy_with_logits(fake_logits,torch.ones_like(fake_logits))
    reconstruction=hidden_valid_l1(prediction,target,observed_mask,valid_mask)
    return adversarial+reconstruction_weight*reconstruction,adversarial,reconstruction
