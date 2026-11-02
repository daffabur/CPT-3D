import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sgan_v42 import DATA, Generator, prepare, device_for_training


def seed_for(path, n_cpt, seed):
    identity = f"{seed}|{path.name}|{n_cpt}"
    digest = hashlib.sha256(identity.encode()).digest()
    return int.from_bytes(digest[:4], "little")


def summarize(rows):
    model_rmse = np.array([r["model_rmse"] for r in rows])
    idw_rmse = np.array([r["idw_rmse"] for r in rows])
    model_mae = np.array([r["model_mae"] for r in rows])
    idw_mae = np.array([r["idw_mae"] for r in rows])
    diff = idw_rmse - model_rmse

    return {
        "samples": len(rows),
        "model_rmse": float(model_rmse.mean()),
        "idw_rmse": float(idw_rmse.mean()),
        "model_mae": float(model_mae.mean()),
        "idw_mae": float(idw_mae.mean()),
        "rmse_improvement": float(diff.mean()),
        "rmse_improvement_pct": float(
            100 * diff.mean() / max(idw_rmse.mean(), 1e-12)
        ),
        "win_rate_pct": float(100 * (diff > 0).mean()),
        "preservation_max_error": float(
            max(r["preservation_error"] for r in rows)
        ),
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/synthetic_v4/full_gan_seed14/best.pt"
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--cpts",
        default="1,2,3,4,5,6,7,8,9,10,16,24,32",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/synthetic_v4/final_test"),
    )
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    counts = list(dict.fromkeys(
        int(item.strip()) for item in args.cpts.split(",")
    ))
    if not counts or any(n < 1 for n in counts):
        raise ValueError("CPT counts must be positive")

    paths = sorted((DATA / "test").glob("*.npz"))
    if not paths:
        raise FileNotFoundError("Test NPZ files not found")
    if args.limit:
        paths = paths[:args.limit]

    norm = json.loads((DATA / "normalization.json").read_text())
    if "std" not in norm:
        raise KeyError("Expected 'std' in normalization.json")

    std = float(norm["std"])
    device = device_for_training()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model = Generator().to(device)
    model.load_state_dict(
        checkpoint["generator_state"], strict=True
    )
    model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    print(
        f"DEVICE={device} | TEST={len(paths)} "
        f"| CPT_COUNTS={counts}",
        flush=True,
    )

    with torch.inference_mode():
        for i, path in enumerate(paths, 1):
            metadata_path = path.with_suffix(".json")
            if not metadata_path.exists():
                raise FileNotFoundError(metadata_path)

            metadata = json.loads(metadata_path.read_text())
            scenario = str(metadata["scenario"])
            family = str(metadata["family"])

            for n in counts:
                with np.load(path, allow_pickle=False) as volume:
                    nx, ny = volume["Ic"].shape[:2]

                if n > nx * ny:
                    raise ValueError(
                        f"{n} CPT exceeds {nx*ny} XY cells "
                        f"in {path.name}"
                    )

                features, target, _ = prepare(
                    path, norm, n_cpt=n,
                    seed=seed_for(path, n, args.seed),
                )

                features = features.to(device)
                target = target.to(device)

                mask = features[:, 1:2]
                hidden = 1.0 - mask
                idw = features[:, 2:3]
                pred = model(features)

                denominator = hidden.sum().clamp_min(1)
                e_model = (pred - target) * hidden
                e_idw = (idw - target) * hidden

                rmse_model = (
                    (e_model.square().sum() / denominator)
                    .sqrt().item() * std
                )
                rmse_idw = (
                    (e_idw.square().sum() / denominator)
                    .sqrt().item() * std
                )
                mae_model = (
                    e_model.abs().sum().item()
                    / denominator.item() * std
                )
                mae_idw = (
                    e_idw.abs().sum().item()
                    / denominator.item() * std
                )

                observed_error = (
                    ((pred - target).abs() * mask)
                    .max().item() * std
                )

                rows.append({
                    "volume": path.stem,
                    "scenario": scenario,
                    "family": family,
                    "cpt": n,
                    "model_rmse": rmse_model,
                    "idw_rmse": rmse_idw,
                    "model_mae": mae_model,
                    "idw_mae": mae_idw,
                    "preservation_error": observed_error,
                })

            if i % 100 == 0 or i == len(paths):
                print(
                    f"EVALUATED {i}/{len(paths)} volumes",
                    flush=True,
                )

    grouped = defaultdict(list)
    for row in rows:
        grouped[("overall", "all")].append(row)
        grouped[("cpt", str(row["cpt"]))].append(row)
        grouped[("scenario", row["scenario"])].append(row)
        grouped[("family", row["family"])].append(row)
        grouped[
            ("scenario_family",
             f'{row["scenario"]}_{row["family"]}')
        ].append(row)
        grouped[
            ("cpt_family",
             f'{row["cpt"]}_{row["family"]}')
        ].append(row)

    summary_rows = []
    for (group_type, group_value), group_rows in sorted(
        grouped.items()
    ):
        summary_rows.append({
            "group": group_type,
            "value": group_value,
            **summarize(group_rows),
        })

    write_csv(args.output / "details.csv", rows)
    write_csv(args.output / "summary.csv", summary_rows)

    (args.output / "run_config.json").write_text(
        json.dumps({
            "checkpoint": str(args.checkpoint),
            "test_count": len(paths),
            "cpt_counts": counts,
            "seed": args.seed,
            "normalization": norm,
        }, indent=2)
    )

    print("\n===== OVERALL =====")
    print(json.dumps(
        summarize(rows), indent=2
    ))

    print("\n===== PER CPT =====")
    for n in counts:
        r = summarize(grouped[("cpt", str(n))])
        print(
            f"CPT={n:3d} "
            f"model={r['model_rmse']:.6f} "
            f"idw={r['idw_rmse']:.6f} "
            f"win={r['win_rate_pct']:.2f}%"
        )

    print("\nEVALUATION COMPLETE")
    print("Output:", args.output)


if __name__ == "__main__":
    main()
