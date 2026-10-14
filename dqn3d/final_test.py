"""Lock a validation-selected DQN checkpoint, then evaluate the untouched test split once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from schemagan3d.data import sorted_csv_paths

from .agent import MaskedDQNAgent
from .environment import CPTPlacementEnv3D, EnvironmentConfig
from .reconstructor import SchemaGANReconstructor
from .train import evaluate, write_rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_model(root: Path, *, seeds: tuple[int, ...] = (14, 37, 73)) -> dict:
    """Choose by validation alone; do not inspect the test directory."""
    lock_path = root / "locked_model.json"
    if lock_path.exists():
        raise FileExistsError(lock_path)
    candidates = []
    for seed in seeds:
        run = root / f"seed{seed}"
        configuration = json.loads((run / "config.json").read_text(encoding="utf-8"))
        report = json.loads((run / "validation_summary.json").read_text(encoding="utf-8"))
        args = configuration["arguments"]
        if (int(args["train_limit"]) != 16000 or int(args["episodes"]) != 16000 or
                int(args["action_cells"]) != 4 or int(args["budget"]) != 8 or
                configuration["agent"]["action_count"] != 16 or
                configuration["environment"]["reward_mode"] != "oracle_rmse"):
            raise ValueError(f"unexpected full-training configuration in {run}")
        checkpoint = run / "best.pt"
        candidates.append({
            "seed": seed, "run": str(run), "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "validation_hidden_rmse": report["best_validation_hidden_rmse"],
            "schemagan_sha256": configuration["schemagan_sha256"],
        })
    winner = min(candidates, key=lambda item: (item["validation_hidden_rmse"], item["seed"]))
    manifest = {"selection_rule": "minimum mean validation hidden-voxel RMSE",
                "candidates": candidates, "selected": winner,
                "test_seed": 2_000_014, "test_count": 4000,
                "test_evaluated": False}
    lock_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def bootstrap_paired_ci(differences: np.ndarray, *, seed: int = 9901,
                        replicates: int = 5000) -> dict:
    """Percentile bootstrap CI for mean paired RMSE difference; negative favors DQN."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("need at least two finite paired differences")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 100):
        stop = min(start + 100, replicates)
        indices = rng.integers(len(values), size=(stop - start, len(values)))
        samples[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {"mean_difference": float(values.mean()),
            "sd_difference": float(values.std(ddof=1)),
            "ci95_lower": float(lower), "ci95_upper": float(upper),
            "method": "paired percentile bootstrap, 5000 resamples, seed 9901"}


def evaluate_locked(root: Path, test_dir: Path, schemagan_checkpoint: Path) -> dict:
    manifest = json.loads((root / "locked_model.json").read_text(encoding="utf-8"))
    output = root / "final_test"
    if output.exists():
        raise FileExistsError(f"refusing to repeat or overwrite final test: {output}")
    selected = manifest["selected"]
    checkpoint = Path(selected["checkpoint"])
    if sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("locked DQN checkpoint hash changed")
    if sha256(schemagan_checkpoint) != selected["schemagan_sha256"]:
        raise ValueError("frozen SchemaGAN checkpoint hash changed")
    paths = sorted_csv_paths(test_dir)
    if len(paths) != manifest["test_count"]:
        raise ValueError("expected exactly 4000 test CSVs")
    agent = MaskedDQNAgent.load(checkpoint)
    reconstructor = SchemaGANReconstructor(schemagan_checkpoint, device="cpu")
    env = CPTPlacementEnv3D(reconstructor, EnvironmentConfig(budget=8, action_cells=4))
    records, summary = evaluate(agent, env, paths, seed=manifest["test_seed"],
                                policies=("dqn", "uniform", "random"), progress_every=500)
    by_file: dict[str, dict[str, dict]] = {}
    for record in records:
        by_file.setdefault(record["test_file"], {})[record["policy"]] = record
    paired = {}
    for baseline in ("uniform", "random"):
        differences = np.array([pair["dqn"]["final_rmse_hidden"] - pair[baseline]["final_rmse_hidden"]
                                for pair in by_file.values()], dtype=np.float64)
        paired[baseline] = {
            **bootstrap_paired_ci(differences),
            "dqn_wins": int(np.count_nonzero(differences < 0)),
            "ties": int(np.count_nonzero(differences == 0)),
            "volume_count": len(differences),
        }
    output.mkdir(parents=True)
    write_rows(output / "per_volume.csv", records)
    report = {"locked_seed": selected["seed"], "locked_checkpoint_sha256": selected["checkpoint_sha256"],
              "test_seed": manifest["test_seed"], "test_volume_count": len(paths),
              "validation_selection_rmse": selected["validation_hidden_rmse"],
              "metrics": summary, "paired_hidden_rmse": paired}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("lock", "evaluate"))
    parser.add_argument("--root", type=Path, default=Path("results/dqn3d_full"))
    parser.add_argument("--test-dir", type=Path, default=Path("data/synthetic3d/test"))
    parser.add_argument("--schemagan-checkpoint", type=Path,
                        default=Path("results/schemagan3d/checkpoint.pt"))
    args = parser.parse_args()
    result = (lock_model(args.root) if args.command == "lock" else
              evaluate_locked(args.root, args.test_dir, args.schemagan_checkpoint))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
