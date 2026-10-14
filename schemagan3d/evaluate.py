"""Evaluate a trained generator or the horizontal-IDW baseline on one split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .baseline import horizontal_idw
from .data import NormalizationStats, SoilVolumeDataset
from .device import select_device
from .metrics import volume_metrics
from .models import Generator3D


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic3d"))
    parser.add_argument("--split", choices=("vali", "test"), default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    if args.checkpoint is None:
        stats = NormalizationStats(0.0, 1.0)
        generator = None
    else:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        stats = NormalizationStats(**saved["normalization"])
        generator = Generator3D(saved["base_channels"]).to(device)
        generator.load_state_dict(saved["generator"])
        generator.eval()
    dataset = SoilVolumeDataset(args.data_root / args.split, stats=stats, limit=args.limit)
    loader = DataLoader(dataset, batch_size=1)
    totals = {"rmse": 0.0, "mae": 0.0}
    with torch.no_grad():
        for batch in loader:
            sparse, mask, target = (batch[name].to(device) for name in ("sparse", "mask", "target"))
            if generator is None:
                prediction = horizontal_idw(sparse[0], mask[0]).unsqueeze(0)
            else:
                prediction = generator(sparse, mask)
            result = volume_metrics(prediction, target, stats)
            for key in totals:
                totals[key] += result[key]
    print({key: value / len(dataset) for key, value in totals.items()})


if __name__ == "__main__":
    main()
