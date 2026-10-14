"""Aggregate independently saved DQN validation runs without reading test data."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np


def summarize(root: Path) -> dict:
    runs = []
    for directory in sorted(root.iterdir()):
        summary_path = directory / "validation_summary.json"
        if not summary_path.exists():
            continue
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        records = list(csv.DictReader((directory / "validation_per_volume.csv").open(encoding="utf-8")))
        by_volume: dict[str, dict[str, dict]] = {}
        for record in records:
            by_volume.setdefault(record["test_file"], {})[record["policy"]] = record
        if len(by_volume) != 100 or any(set(policies) != {"dqn", "random", "uniform"} for policies in by_volume.values()):
            raise ValueError(f"incomplete validation set in {directory}")
        if any(len({record["initial_xy"] for record in policies.values()}) != 1
               for policies in by_volume.values()):
            raise ValueError(f"mismatched warm starts in {directory}")
        for policies in by_volume.values():
            for record in policies.values():
                positions = ast.literal_eval(record["sampled_xy"])
                initial = ast.literal_eval(record["initial_xy"])
                if len(positions) != 8 or len(set(positions)) != 8 or list(positions[:3]) != initial:
                    raise ValueError(f"invalid episode in {directory}")
        policy_summary = report["validation"]
        wins_uniform = sum(float(p["dqn"]["final_rmse_hidden"]) < float(p["uniform"]["final_rmse_hidden"])
                           for p in by_volume.values())
        wins_random = sum(float(p["dqn"]["final_rmse_hidden"]) < float(p["random"]["final_rmse_hidden"])
                          for p in by_volume.values())
        overlap_uniform = np.mean([
            len(set(ast.literal_eval(p["dqn"]["sampled_xy"])[3:]) &
                set(ast.literal_eval(p["uniform"]["sampled_xy"])[3:]))
            for p in by_volume.values()
        ])
        arguments = config["arguments"]
        runs.append({
            "run": directory.name,
            "seed": int(arguments["seed"]),
            "train_limit": int(arguments["train_limit"]),
            "episodes": int(arguments["episodes"]),
            "action_count": config["agent"]["action_count"],
            "coverage_channel": config["agent"]["coverage_channel"],
            "reward_mode": config["environment"]["reward_mode"],
            "validation_seed": config["validation_seed"],
            "training_seconds": report["training_seconds"],
            "dqn_hidden_rmse_mean": policy_summary["dqn"]["final_rmse_hidden"]["mean"],
            "dqn_hidden_rmse_median": policy_summary["dqn"]["final_rmse_hidden"]["median"],
            "dqn_hidden_rmse_volume_sd": policy_summary["dqn"]["final_rmse_hidden"]["std"],
            "dqn_hidden_mae_mean": policy_summary["dqn"]["final_mae_hidden"]["mean"],
            "dqn_all_rmse_mean": policy_summary["dqn"]["final_rmse_all"]["mean"],
            "random_hidden_rmse_mean": policy_summary["random"]["final_rmse_hidden"]["mean"],
            "uniform_hidden_rmse_mean": policy_summary["uniform"]["final_rmse_hidden"]["mean"],
            "wins_vs_random": wins_random,
            "wins_vs_uniform": wins_uniform,
            "mean_additional_site_overlap_vs_uniform": float(overlap_uniform),
            "dqn_policy_inference_ms_per_action": policy_summary["dqn"]["policy_inference_seconds"]["mean"] * 200,
            "dqn_reconstruction_ms_per_action": policy_summary["dqn"]["reconstruction_seconds"]["mean"] * 200,
        })
    groups: dict[str, list[dict]] = {}
    for run in runs:
        label = run["run"].rsplit("_seed", 1)[0]
        groups.setdefault(label, []).append(run)
    aggregated = {}
    for label, items in groups.items():
        metrics = ("dqn_hidden_rmse_mean", "dqn_hidden_mae_mean", "dqn_all_rmse_mean",
                   "wins_vs_uniform", "wins_vs_random", "training_seconds")
        aggregated[label] = {
            "seeds": [run["seed"] for run in items],
            **{metric: {"mean": float(np.mean([item[metric] for item in items])),
                        "seed_sd": float(np.std([item[metric] for item in items], ddof=1)) if len(items) > 1 else None}
               for metric in metrics},
        }
    return {"runs": runs, "groups": aggregated,
            "notes": ["Validation only; test split not read.",
                      "256-action baselines use the same 256-action space, so compare within action count.",
                      "Best checkpoints were selected on these same validation volumes; reported scores are selection-biased."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/dqn3d_opt"))
    args = parser.parse_args()
    result = summarize(args.root)
    (args.root / "aggregate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.root / "aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["runs"][0]))
        writer.writeheader()
        writer.writerows(result["runs"])
    print(json.dumps(result["groups"], indent=2))


if __name__ == "__main__":
    main()
