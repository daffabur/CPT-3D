import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path("data/synthetic3d_v4_geosyn/full_24k")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for_training():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Block(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(
                channels, channels, 3,
                padding=dilation, dilation=dilation
            ),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return F.leaky_relu(x + self.net(x), 0.1)


class Generator(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(7, channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            Block(channels, 1),
            Block(channels, 2),
            Block(channels, 4),
            nn.Conv3d(channels, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        sparse = features[:, 0:1]
        mask = features[:, 1:2]
        idw = features[:, 2:3]

        prediction = idw + self.net(features)
        return torch.where(mask > 0, sparse, prediction)


class Discriminator(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(6, channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(
                channels, channels * 2, 3,
                stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.Conv3d(
                channels * 2, channels * 4, 3,
                stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            nn.Conv3d(channels * 4, 1, 3, padding=1),
        )

    def forward(self, volume, features):
        # volume + IDW + CPT mask + physical spacing (dx,dy,dz)
        conditioning = torch.cat([
            features[:, 2:3],
            features[:, 1:2],
            features[:, 4:7],
        ], dim=1)
        return self.net(torch.cat(
            [volume, conditioning], dim=1
        ))


def prepare(path, norm, n_cpt, seed):
    with np.load(path, allow_pickle=False) as d:
        ic = d["Ic"].astype(np.float32)
        meta = json.loads(str(d["metadata_json"]))

    nx, ny, nz = ic.shape
    grid = meta["grid"]
    dx = float(grid["dx_m"])
    dy = float(grid["dy_m"])
    dz = float(grid["dz_m"])

    rng = np.random.default_rng(seed)
    n_cpt = min(n_cpt, nx * ny)
    indices = rng.choice(
        nx * ny, size=n_cpt, replace=False
    )
    px, py = np.unravel_index(indices, (nx, ny))

    target = (ic - norm["mean"]) / norm["std"]
    sparse = np.zeros_like(target)
    mask = np.zeros_like(target)

    sparse[px, py, :] = target[px, py, :]
    mask[px, py, :] = 1.0

    xx, yy = np.meshgrid(
        np.arange(nx, dtype=np.float32) * dx,
        np.arange(ny, dtype=np.float32) * dy,
        indexing="ij"
    )

    dist = np.sqrt(
        (xx[None] - px[:, None, None] * dx) ** 2 +
        (yy[None] - py[:, None, None] * dy) ** 2
    )

    weights = 1.0 / np.maximum(dist, 1e-5) ** 2
    weights /= weights.sum(axis=0, keepdims=True)

    idw = np.einsum(
        "kxy,kz->xyz",
        weights,
        target[px, py, :],
        optimize=True
    ).astype(np.float32)

    idw = np.where(mask > 0, sparse, idw)

    distance_scale = max(nx * dx, ny * dy, 1.0)
    distance = np.broadcast_to(
        dist.min(axis=0)[:, :, None] / distance_scale,
        target.shape
    )

    features = np.stack([
        sparse,
        mask,
        idw,
        distance,
        np.full_like(target, np.log1p(dx)),
        np.full_like(target, np.log1p(dy)),
        np.full_like(target, np.log1p(dz)),
    ]).astype(np.float32)

    # NumPy layout X,Y,Z -> PyTorch layout Z,Y,X.
    features = np.transpose(features, (0, 3, 2, 1)).copy()
    target = np.transpose(target, (2, 1, 0)).copy()

    return (
        torch.from_numpy(features)[None],
        torch.from_numpy(target)[None, None],
        meta["scenario"]
    )


def masked_mae(pred, target, mask):
    hidden = 1.0 - mask
    return (
        ((pred - target).abs() * hidden).sum()
        / hidden.sum().clamp_min(1)
    )


@torch.no_grad()
def evaluate(generator, paths, norm, device, eval_cpts):
    generator.eval()

    totals = {
        n: {"model": 0.0, "idw": 0.0, "count": 0}
        for n in eval_cpts
    }

    for i, path in enumerate(paths):
        for n_cpt in eval_cpts:
            features, target, _ = prepare(
                path, norm, n_cpt,
                seed=900000 + i * 17 + n_cpt
            )

            features = features.to(device)
            target = target.to(device)

            predicted = generator(features)
            mask = features[:, 1:2]
            hidden = 1.0 - mask

            denom = hidden.sum().clamp_min(1)

            rmse = torch.sqrt(
                (((predicted - target) ** 2) * hidden).sum()
                / denom
            )

            idw_rmse = torch.sqrt(
                (((features[:, 2:3] - target) ** 2)
                 * hidden).sum() / denom
            )

            totals[n_cpt]["model"] += float(rmse.item())
            totals[n_cpt]["idw"] += float(idw_rmse.item())
            totals[n_cpt]["count"] += 1

    result = {}
    for n, values in totals.items():
        c = max(values["count"], 1)
        result[n] = {
            "model_rmse_ic": values["model"] / c * norm["std"],
            "idw_rmse_ic": values["idw"] / c * norm["std"],
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--limit-train", type=int, default=30)
    parser.add_argument("--limit-val", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/synthetic_v4/smoke")
    )
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--no-adversarial", action="store_true")
    parser.add_argument("--max-cpt", type=int, default=32)
    parser.add_argument(
        "--recon-loss",
        choices=("mae", "mse", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--mse-weight", type=float, default=0.5
    )
    parser.add_argument(
        "--adv-weight", type=float, default=0.005
    )
    parser.add_argument(
        "--eval-cpts",
        type=str,
        default="1,2,3,4,5,6,7,8,12,16,24,32",
    )
    parser.add_argument(
        "--init-generator", type=Path, default=None
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=5
    )
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.warmup_epochs < 1:
        raise ValueError("--warmup-epochs must be >= 1")

    if args.max_cpt < 1:
        raise ValueError("--max-cpt must be >= 1")
    if args.mse_weight < 0 or args.adv_weight < 0:
        raise ValueError("Loss weights must be nonnegative")

    eval_cpts = tuple(dict.fromkeys(
        int(v.strip())
        for v in args.eval_cpts.split(",")
    ))

    if not eval_cpts or any(n < 1 for n in eval_cpts):
        raise ValueError("--eval-cpts requires positive integers")

    seed_all(args.seed)
    device = device_for_training()

    norm = json.loads(
        (DATA / "normalization.json").read_text()
    )
    assert norm["std"] > 0

    train_paths = sorted((DATA / "train").glob("*.npz"))
    val_paths = sorted((DATA / "validation").glob("*.npz"))

    if args.limit_train > 0:
        train_paths = train_paths[:args.limit_train]
    if args.limit_val > 0:
        val_paths = val_paths[:args.limit_val]

    if not train_paths or not val_paths:
        raise RuntimeError("Dataset split is empty")

    args.output.mkdir(parents=True, exist_ok=True)

    gen = Generator().to(device)
    disc = Discriminator().to(device)

    if args.init_generator is not None:
        checkpoint = torch.load(
            args.init_generator,
            map_location="cpu",
            weights_only=True,
        )
        gen.load_state_dict(
            checkpoint["generator_state"],
            strict=True,
        )
        print(
            "Loaded pretrained generator:",
            args.init_generator,
            flush=True,
        )

    optimizer_g = torch.optim.Adam(
        gen.parameters(), lr=1e-4, betas=(0.5, 0.999)
    )
    optimizer_d = torch.optim.Adam(
        disc.parameters(), lr=1e-4, betas=(0.5, 0.999)
    )

    best = float("inf")
    print(
        f"Device={device} | train={len(train_paths)}"
        f" | val={len(val_paths)}"
        f" | adversarial={not args.no_adversarial}",
        flush=True
    )

    for epoch in range(args.epochs):
        gen.train()
        disc.train()

        current_adv_weight = (
            args.adv_weight
            * min(1.0, (epoch + 1) / args.warmup_epochs)
        )
        print(
            f"EPOCH {epoch+1} "
            f"adversarial_weight={current_adv_weight:.6f}",
            flush=True,
        )

        order = np.random.default_rng(
            args.seed + epoch
        ).permutation(len(train_paths))

        losses = []
        d_losses = []
        adv_losses = []

        for position, index in enumerate(order):
            path = train_paths[int(index)]

            rng = np.random.default_rng(
                args.seed * 100000
                + epoch * len(train_paths)
                + int(index)
            )
            n_cpt = int(rng.integers(1, args.max_cpt + 1))

            features, target, _ = prepare(
                path, norm, n_cpt,
                seed=int(rng.integers(0, 2**32 - 1))
            )

            features = features.to(device)
            target = target.to(device)
            mask = features[:, 1:2]
            idw = features[:, 2:3]

            fake = gen(features)

            if not args.no_adversarial:
                optimizer_d.zero_grad(set_to_none=True)

                score_real = disc(
                    target, features
                )
                score_fake = disc(
                    fake.detach(), features
                )

                loss_d = (
                    F.relu(1 - score_real).mean()
                    + F.relu(1 + score_fake).mean()
                ) * 0.5

                loss_d.backward()
                optimizer_d.step()
                d_losses.append(float(loss_d.item()))

            optimizer_g.zero_grad(set_to_none=True)
            if not args.no_adversarial:
                for parameter in disc.parameters():
                    parameter.requires_grad_(False)

            fake = gen(features)
            hidden = 1.0 - mask
            denominator = hidden.sum().clamp_min(1)

            mae = (
                ((fake - target).abs() * hidden).sum()
                / denominator
            )
            mse = (
                (((fake - target) ** 2) * hidden).sum()
                / denominator
            )

            if args.recon_loss == "mae":
                reconstruction = mae
            elif args.recon_loss == "mse":
                reconstruction = mse
            else:
                reconstruction = (
                    mae + args.mse_weight * mse
                )

            loss_g = reconstruction

            if not args.no_adversarial:
                adversarial = -disc(
                    fake, features
                ).mean()
                adv_losses.append(float(adversarial.item()))
                loss_g = (
                    reconstruction
                    + current_adv_weight * adversarial
                )

            loss_g.backward()
            optimizer_g.step()

            if not args.no_adversarial:
                for parameter in disc.parameters():
                    parameter.requires_grad_(True)

            losses.append(float(reconstruction.item()))

            if (position + 1) % 10 == 0:
                print(
                    f"epoch={epoch+1}/{args.epochs}"
                    f" sample={position+1}/{len(order)}"
                    f" hidden_mae_norm={np.mean(losses):.5f}",
                    flush=True
                )

        print(
            f"TRAIN LOSSES | epoch={epoch+1}"
            f" | reconstruction={np.mean(losses):.6f}"
            f" | discriminator="
            f"{np.mean(d_losses) if d_losses else float('nan'):.6f}"
            f" | adversarial="
            f"{np.mean(adv_losses) if adv_losses else float('nan'):.6f}",
            flush=True,
        )

        metrics = evaluate(
            gen, val_paths, norm, device, eval_cpts
        )
        score = float(np.mean([
            values["model_rmse_ic"]
            for values in metrics.values()
        ]))

        print(
            f"EPOCH {epoch+1} | "
            f"val={json.dumps(metrics)}",
            flush=True
        )

        if score < best:
            best = score
            torch.save({
                "generator_state": gen.state_dict(),
                "discriminator_state": disc.state_dict(),
                "normalization": norm,
                "validation": metrics,
                "epoch": epoch + 1,
                "seed": args.seed,
                "architecture": "SGAN_V4_conditional_residual_3D",
                "input_channels": 7,
                "adversarial": not args.no_adversarial,
                "recon_loss": args.recon_loss,
                "mse_weight": args.mse_weight,
                "adv_weight": args.adv_weight,
                "max_cpt": args.max_cpt,
                "eval_cpts": list(eval_cpts),
                "version": "V4.2",
                "discriminator_input_channels": 6,
                "warmup_epochs": args.warmup_epochs,
                "current_adv_weight": current_adv_weight,
                "init_generator": (
                    str(args.init_generator)
                    if args.init_generator else None
                ),
            }, args.output / "best.pt")

    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "best_validation_mean_rmse_ic": best,
                "last_validation": metrics,
                "train_volumes": len(train_paths),
                "validation_volumes": len(val_paths),
                "epochs": args.epochs,
            },
            indent=2
        ) + "\n"
    )

    print("TRAINING COMPLETE")
    print("Checkpoint:", args.output / "best.pt")


if __name__ == "__main__":
    main()
