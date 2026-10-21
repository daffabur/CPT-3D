"""Lightweight conditional generator and discriminator for 16x16x8 volumes."""

import torch
from torch import nn


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm3d(out_channels, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Generator3D(nn.Module):
    """Small 3D U-Net conditioned on sparse IC and its binary mask."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.enc1 = _block(2, c)
        self.down1 = nn.Conv3d(c, c * 2, 4, stride=2, padding=1)
        self.enc2 = _block(c * 2, c * 2)
        self.down2 = nn.Conv3d(c * 2, c * 4, 4, stride=2, padding=1)
        self.bottleneck = _block(c * 4, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 4, stride=2, padding=1)
        self.dec2 = _block(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, 4, stride=2, padding=1)
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
        # Hard data consistency: measured CPT profiles cannot change.
        return mask * sparse + (1.0 - mask) * raw


class Discriminator3D(nn.Module):
    """Conditional 3D PatchGAN discriminator."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.network = nn.Sequential(
            nn.Conv3d(3, c, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(c, c * 2, 4, stride=2, padding=1),
            nn.InstanceNorm3d(c * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(c * 2, c * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(c * 4, 1, 1),
        )

    def forward(
        self, candidate: torch.Tensor, sparse: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat((candidate, sparse, mask), dim=1))
