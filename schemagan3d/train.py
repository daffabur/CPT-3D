"""Train the initial conditional SchemaGAN 3D prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SoilVolumeDataset, compute_normalization, sorted_csv_paths
from .device import select_device
from .metrics import volume_metrics
from .models import Discriminator3D, Generator3D
from .training import optimization_step


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def validate(generator, loader, stats, device) -> dict[str, float]:
    generator.eval()
    weighted_rmse = weighted_mae = 0.0
    count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        prediction = generator(batch["sparse"], batch["mask"])
        metrics = volume_metrics(prediction, batch["target"], stats)
        size = batch["target"].shape[0]
        weighted_rmse += metrics["rmse"] * size
        weighted_mae += metrics["mae"] * size
        count += size
    generator.train()
    return {"rmse": weighted_rmse / count, "mae": weighted_mae / count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic3d"))
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--val-limit", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--profiles-min", type=int, default=3)
    parser.add_argument("--profiles-max", type=int, default=20)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--reconstruction-weight", type=float, default=100.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--output", type=Path, default=Path("results/schemagan3d"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    train_paths = sorted_csv_paths(args.data_root / "train", args.train_limit)
    stats = compute_normalization(train_paths)
    common = dict(
        stats=stats,
        min_profiles=args.profiles_min,
        max_profiles=args.profiles_max,
        observation_seed=args.seed,
    )
    train_set = SoilVolumeDataset(args.data_root / "train", limit=args.train_limit, **common)
    validation_set = SoilVolumeDataset(
        args.data_root / "vali", limit=args.val_limit, observation_seed=args.seed + 1, **{k: v for k, v in common.items() if k != "observation_seed"}
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)
    generator = Generator3D(args.base_channels).to(device)
    discriminator = Discriminator3D(args.base_channels).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    print(f"device={device}; train={len(train_set)}; validation={len(validation_set)}")
    for epoch in range(1, args.epochs + 1):
        last_losses = {}
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            last_losses = optimization_step(
                generator,
                discriminator,
                _to_device(batch, device),
                g_optimizer,
                d_optimizer,
                reconstruction_weight=args.reconstruction_weight,
            )
        validation = validate(generator, validation_loader, stats, device)
        print(f"epoch={epoch} losses={last_losses} validation={validation}")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "checkpoint.pt"
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "normalization": {"mean": stats.mean, "std": stats.std},
            "base_channels": args.base_channels,
            "args": vars(args),
        },
        checkpoint,
    )
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
