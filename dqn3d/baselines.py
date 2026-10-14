"""Action-matched random and spatially dispersed baseline policies."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from synthetic3d.generate_csv import read_csv_volume
from schemagan3d.data import sorted_csv_paths

from .environment import CPTPlacementEnv3D, EnvironmentConfig
from .reconstructor import SchemaGANReconstructor


def choose_action(env: CPTPlacementEnv3D, policy: str, rng: np.random.Generator) -> int:
    valid = np.flatnonzero(env.valid_actions())
    if len(valid) == 0:
        raise RuntimeError("no valid action")
    if policy == "random":
        return int(rng.choice(valid))
    if policy == "uniform":
        sites = np.asarray(env.positions)
        candidates = np.asarray([env.candidate_xy(int(action)) for action in valid])
        nearest = np.min(np.sum((candidates[:, None] - sites[None]) ** 2, axis=-1), axis=1)
        return int(valid[np.argmax(nearest)])
    raise ValueError(f"unknown policy {policy}")


def run_episode(env: CPTPlacementEnv3D, volume: np.ndarray, policy: str, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    initial_xy = (int(rng.integers(16)), int(rng.integers(16)))
    _, initial_info = env.reset(volume, initial_xy=initial_xy)
    records = [dict(step=0, reward=0.0, selected_xy=initial_xy, **initial_info)]
    done = env.config.budget == 1
    while not done:
        action = choose_action(env, policy, rng)
        _, reward, done, info = env.step(action)
        records.append(dict(step=len(records), reward=reward, **info))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/schemagan3d/checkpoint.pt"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/synthetic3d/test"))
    parser.add_argument("--volumes", type=int, default=6)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted_csv_paths(args.test_dir)
    if len(paths) < args.volumes:
        raise ValueError("not enough test volumes")
    selected = np.random.default_rng(args.seed).choice(len(paths), size=args.volumes, replace=False)
    reconstructor = SchemaGANReconstructor(args.checkpoint, args.device)
    env = CPTPlacementEnv3D(reconstructor, EnvironmentConfig(budget=args.budget))
    destination = args.output or Path("results") / f"dqn3d_baselines_{datetime.now():%Y%m%d_%H%M%S}"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    rows = []
    for index in selected:
        path = paths[int(index)]
        volume = read_csv_volume(path)
        # Same initial site for both policies; policy RNG branches after reset.
        for policy in ("random", "uniform"):
            episode = run_episode(env, volume, policy, args.seed + int(index))
            for record in episode:
                rows.append({
                    "test_file": path.name, "policy": policy,
                    "step": record["step"], "cpt_count": record["cpt_count"],
                    "selected_xy": str(record["selected_xy"]),
                    "rmse_hidden": record["rmse_hidden"], "mae_hidden": record["mae_hidden"],
                    "rmse_all": record["rmse_all"], "reward": record["reward"],
                    "reconstruction_seconds": record["reconstruction_seconds"],
                })
        print(f"evaluated {path.name}")
    with (destination / "per_step.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    final = [row for row in rows if row["cpt_count"] == args.budget]
    summary = {}
    for policy in ("random", "uniform"):
        selected_rows = [row for row in final if row["policy"] == policy]
        steps = [row for row in rows if row["policy"] == policy and row["step"] > 0]
        summary[policy] = {
            "episodes": len(selected_rows),
            "mean_hidden_rmse": float(np.mean([row["rmse_hidden"] for row in selected_rows])),
            "mean_hidden_mae": float(np.mean([row["mae_hidden"] for row in selected_rows])),
            "mean_all_rmse": float(np.mean([row["rmse_all"] for row in selected_rows])),
            "mean_reconstruction_seconds_per_step": float(np.mean([row["reconstruction_seconds"] for row in steps])) if steps else 0.0,
        }
    (destination / "summary.json").write_text(json.dumps({
        "config": {"budget_including_initial": args.budget, "initial_cpt_seed": args.seed,
                   "action_count": env.action_count, "reward_mode": env.config.reward_mode,
                   "sampling_cost": env.config.sampling_cost, "test_files": [paths[int(i)].name for i in selected]},
        "summary": summary,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
