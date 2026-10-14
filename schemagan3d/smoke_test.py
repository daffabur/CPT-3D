"""Run a small end-to-end check without starting model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .baseline import horizontal_idw
from .data import SoilVolumeDataset, compute_normalization, sorted_csv_paths
from .device import select_device
from .metrics import observed_max_error, volume_metrics
from .models import Discriminator3D, Generator3D
from .training import optimization_step


def run_smoke_test(data_root: Path, requested_device: str = "auto") -> dict:
    torch.manual_seed(14)
    device = select_device(requested_device)
    train_paths = sorted_csv_paths(data_root / "train", limit=8)
    stats = compute_normalization(train_paths)
    dataset = SoilVolumeDataset(
        data_root / "train", stats=stats, limit=4, min_profiles=3, max_profiles=8
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    tensor_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    generator = Generator3D(base_channels=8).to(device)
    discriminator = Discriminator3D(base_channels=8).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    with torch.no_grad():
        before = generator(tensor_batch["sparse"], tensor_batch["mask"])
        logits = discriminator(before, tensor_batch["sparse"], tensor_batch["mask"])
    losses = optimization_step(
        generator,
        discriminator,
        tensor_batch,
        g_optimizer,
        d_optimizer,
    )
    with torch.no_grad():
        prediction = generator(tensor_batch["sparse"], tensor_batch["mask"])
    idw = horizontal_idw(tensor_batch["sparse"][0], tensor_batch["mask"][0])
    report = {
        "device": str(device),
        "input_shape": tuple(tensor_batch["sparse"].shape),
        "generator_output_shape": tuple(prediction.shape),
        "discriminator_output_shape": tuple(logits.shape),
        "generator_parameters": sum(p.numel() for p in generator.parameters()),
        "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()),
        "losses": losses,
        "all_losses_finite": all(torch.isfinite(torch.tensor(v)) for v in losses.values()),
        "observed_max_error": observed_max_error(
            prediction, tensor_batch["sparse"], tensor_batch["mask"]
        ),
        "untrained_generator_metrics": volume_metrics(
            prediction, tensor_batch["target"], stats
        ),
        "idw_metrics_first_sample": volume_metrics(
            idw, tensor_batch["target"][0], stats
        ),
    }
    if not report["all_losses_finite"]:
        raise RuntimeError("smoke-test loss is non-finite")
    if report["observed_max_error"] != 0.0:
        raise RuntimeError("observed CPT profile was not preserved")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic3d"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    for key, value in run_smoke_test(args.data_root, args.device).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
