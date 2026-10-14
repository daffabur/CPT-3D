"""Experimental MPS-friendly models; not used by the default training path."""

import torch
from torch import nn

from .models import _block


class _UpsampleConv3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        )


class Generator3DUpsample(nn.Module):
    """Generator3D variant replacing ConvTranspose3d with resize + Conv3d."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.enc1 = _block(2, c)
        self.down1 = nn.Conv3d(c, c * 2, 4, stride=2, padding=1)
        self.enc2 = _block(c * 2, c * 2)
        self.down2 = nn.Conv3d(c * 2, c * 4, 4, stride=2, padding=1)
        self.bottleneck = _block(c * 4, c * 4)
        self.up2 = _UpsampleConv3D(c * 4, c * 2)
        self.dec2 = _block(c * 4, c * 2)
        self.up1 = _UpsampleConv3D(c * 2, c)
        self.dec1 = _block(c * 2, c)
        self.output = nn.Conv3d(c, 1, 1)

    def forward(self, sparse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if sparse.shape != mask.shape or sparse.ndim != 5:
            raise ValueError("sparse and mask must be equal 5D tensors")
        e1 = self.enc1(torch.cat((sparse, mask), dim=1))
        e2 = self.enc2(self.down1(e1))
        bottleneck = self.bottleneck(self.down2(e2))
        d2 = self.dec2(torch.cat((self.up2(bottleneck), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        raw = self.output(d1)
        return mask * sparse + (1.0 - mask) * raw
