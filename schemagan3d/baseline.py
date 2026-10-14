"""Horizontal inverse-distance weighting baseline for complete CPT profiles."""

import torch


def horizontal_idw(
    sparse: torch.Tensor, mask: torch.Tensor, *, power: float = 2.0
) -> torch.Tensor:
    """Interpolate each Z level using only horizontal X/Y distances."""

    if sparse.shape != mask.shape or sparse.ndim != 4 or sparse.shape[0] != 1:
        raise ValueError("sparse and mask must have shape (1, Z, Y, X)")
    if power <= 0:
        raise ValueError("power must be positive")
    sampled_yx = torch.nonzero(mask[0, 0] > 0.5, as_tuple=False)
    if sampled_yx.numel() == 0:
        raise ValueError("at least one observed profile is required")
    z_size, y_size, x_size = sparse.shape[1:]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(y_size, device=sparse.device, dtype=sparse.dtype),
        torch.arange(x_size, device=sparse.device, dtype=sparse.dtype),
        indexing="ij",
    )
    dy = grid_y[..., None] - sampled_yx[:, 0].to(sparse.dtype)
    dx = grid_x[..., None] - sampled_yx[:, 1].to(sparse.dtype)
    distance = torch.sqrt(dx.square() + dy.square())
    weights = 1.0 / distance.clamp_min(1e-6).pow(power)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    profiles = sparse[0, :, sampled_yx[:, 0], sampled_yx[:, 1]]
    prediction = torch.einsum("yxn,zn->zyx", weights, profiles).unsqueeze(0)
    return mask * sparse + (1.0 - mask) * prediction
