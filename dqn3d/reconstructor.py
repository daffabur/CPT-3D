"""Frozen SchemaGAN 3D inference adapter."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from schemagan3d.data import NormalizationStats
from schemagan3d.device import select_device
from schemagan3d.models import Generator3D


class SchemaGANReconstructor:
    """Reconstruct from observed IC and mask, without receiving ground truth."""

    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.stats = NormalizationStats(**saved["normalization"])
        self.device = select_device(device)
        self.model = Generator3D(saved["base_channels"]).to(self.device).eval()
        self.model.load_state_dict(saved["generator"])
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def predict(self, observed_ic: np.ndarray, observed_mask: np.ndarray) -> tuple[np.ndarray, float]:
        if observed_ic.shape != (16, 16, 8) or observed_mask.shape != observed_ic.shape:
            raise ValueError("expected X,Y,Z arrays of shape (16,16,8)")
        if not np.all(np.isfinite(observed_ic[observed_mask])):
            raise ValueError("observed IC contains nonfinite values")
        sparse = np.where(observed_mask, observed_ic, 0.0).astype(np.float32)
        normalized = np.where(observed_mask, (sparse - self.stats.mean) / self.stats.std, 0.0)
        model_input = torch.from_numpy(normalized.transpose(2, 1, 0).copy())[None, None].to(self.device)
        mask = torch.from_numpy(observed_mask.astype(np.float32).transpose(2, 1, 0).copy())[None, None].to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model(model_input, mask)
            if self.device.type == "mps":
                torch.mps.synchronize()
            predicted = output[0, 0].cpu().numpy().transpose(2, 1, 0)
        elapsed = time.perf_counter() - start
        physical = (predicted * self.stats.std + self.stats.mean).astype(np.float32)
        physical[observed_mask] = observed_ic[observed_mask]
        return physical, elapsed
