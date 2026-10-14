"""Train a masked spatial DQN on synthetic volumes with frozen SchemaGAN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from synthetic3d.generate_csv import read_csv_volume
from schemagan3d.data import sorted_csv_paths

from .agent import AgentConfig, MaskedDQNAgent
from .baselines import choose_action
from .environment import CPTPlacementEnv3D, EnvironmentConfig
from .reconstructor import SchemaGANReconstructor


def initial_sites(index: int, seed: int, *, count: int = 3) -> list[tuple[int, int]]:
    choices = np.random.default_rng(np.random.SeedSequence([seed, index])).choice(256, size=count, replace=False)
    return [(int(choice // 16), int(choice % 16)) for choice in choices]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_episode(env: CPTPlacementEnv3D, volume: np.ndarray, sites: list[tuple[int, int]],
                policy: str, agent: MaskedDQNAgent | None, seed: int, *, epsilon: float = 0.0,
                learn: bool = False) -> dict:
    state, info = env.reset(volume, initial_xy=sites)
    initial_rmse = info["rmse_all"]
    total_reward = 0.0
    losses = []
    inference_seconds = 0.0
    reconstruction_seconds = 0.0
    baseline_stream = {"random": 0, "uniform": 1}.get(policy, 2)
    rng = np.random.default_rng(np.random.SeedSequence([seed, baseline_stream]))
    while len(env.positions) < env.config.budget:
        if policy == "dqn":
            assert agent is not None
            decision_start = time.perf_counter()
            action = agent.act(state, epsilon=epsilon)
            inference_seconds += time.perf_counter() - decision_start
        else:
            decision_start = time.perf_counter()
            action = choose_action(env, policy, rng)
            inference_seconds += time.perf_counter() - decision_start
        next_state, reward, done, info = env.step(action)
        reconstruction_seconds += info["reconstruction_seconds"]
        if learn:
            assert agent is not None
            loss = agent.observe(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)
        total_reward += reward
        state = next_state
        if done:
            break
    assert len(env.positions) == env.config.budget
    return {
        "reward": total_reward,
        "initial_rmse_all": initial_rmse,
        "final_rmse_all": info["rmse_all"],
        "final_rmse_hidden": info["rmse_hidden"],
        "final_mae_hidden": info["mae_hidden"],
        "episode_length": len(env.positions) - len(sites),
        "sampling_cost_total": (len(env.positions) - len(sites)) * env.config.sampling_cost,
        "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        "sampled_xy": str(info["sampled_xy"]),
        "policy_inference_seconds": inference_seconds,
        "reconstruction_seconds": reconstruction_seconds,
    }


def evaluate(agent: MaskedDQNAgent, env: CPTPlacementEnv3D, validation_paths: list[Path],
             *, seed: int, policies: tuple[str, ...] = ("dqn",),
             progress_every: int = 0) -> tuple[list[dict], dict]:
    records = []
    for volume_number, path in enumerate(validation_paths, start=1):
        index = int(path.stem.split("_")[-1])
        volume = read_csv_volume(path)
        sites = initial_sites(index, seed)
        for policy in policies:
            record = run_episode(env, volume, sites, policy, agent if policy == "dqn" else None,
                                 seed + index, epsilon=0.0, learn=False)
            records.append({"test_file": path.name, "policy": policy, "initial_xy": str(sites), **record})
        if progress_every and volume_number % progress_every == 0:
            print(f"evaluated {volume_number}/{len(validation_paths)} volumes", flush=True)
    summary = {}
    for policy in policies:
        group = [item for item in records if item["policy"] == policy]
        summary[policy] = {}
        for metric in ("reward", "final_rmse_hidden", "final_mae_hidden", "final_rmse_all", "episode_length", "sampling_cost_total", "policy_inference_seconds", "reconstruction_seconds"):
            values = np.asarray([item[metric] for item in group], dtype=np.float64)
            summary[policy][metric] = {"mean": float(values.mean()), "median": float(np.median(values)),
                                       "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                                       "min": float(values.min()), "max": float(values.max())}
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/schemagan3d/checkpoint.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic3d"))
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--val-limit", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--sampling-cost", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--validation-seed", type=int, default=1000014)
    parser.add_argument("--action-cells", type=int, choices=(4, 16), default=4)
    parser.add_argument("--coverage-channel", action="store_true")
    parser.add_argument("--reward-mode", choices=("oracle_rmse", "oracle_mae"), default="oracle_rmse")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("results/dqn3d"))
    args = parser.parse_args()
    if min(args.train_limit, args.val_limit, args.episodes, args.eval_every, args.progress_every) < 1:
        raise ValueError("limits and intervals must be positive")
    train_paths = sorted_csv_paths(args.data_root / "train", args.train_limit)
    val_paths = sorted_csv_paths(args.data_root / "vali", args.val_limit)
    if len(train_paths) != args.train_limit or len(val_paths) != args.val_limit:
        raise ValueError("requested dataset count is unavailable")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing results: {args.output}")
    args.output.mkdir(parents=True)
    reconstructor = SchemaGANReconstructor(args.checkpoint, device="cpu")
    env_config = EnvironmentConfig(budget=args.budget, sampling_cost=args.sampling_cost,
                                   action_cells=args.action_cells, reward_mode=args.reward_mode)
    if args.budget < 3:
        raise ValueError("budget must allow three warm-start CPTs")
    env = CPTPlacementEnv3D(reconstructor, env_config)
    agent_config = AgentConfig(seed=args.seed, epsilon_decay_episodes=args.episodes,
                               action_count=env.action_count, coverage_channel=args.coverage_channel)
    agent = MaskedDQNAgent(agent_config, ic_mean=reconstructor.stats.mean, ic_std=reconstructor.stats.std)
    configuration = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": asdict(env_config), "agent": asdict(agent_config),
        "schemagan_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "validation_seed": args.validation_seed,
    }
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    started = time.perf_counter()
    train_rows = []
    checkpoint_scores = []
    best_rmse = float("inf")
    for episode in range(args.episodes):
        source = train_paths[episode % len(train_paths)]
        index = int(source.stem.split("_")[-1])
        cycle = episode // len(train_paths)
        sites = initial_sites(index, args.seed + cycle * 1_000_000)
        epsilon = agent.epsilon(episode)
        result = run_episode(env, read_csv_volume(source), sites, "dqn", agent,
                             args.seed + episode, epsilon=epsilon, learn=True)
        train_rows.append({"episode": episode + 1, "train_file": source.name, "epsilon": epsilon, **result})
        if (episode + 1) % args.progress_every == 0 or episode == 0:
            recent = train_rows[-args.progress_every:]
            print(f"episode {episode+1}/{args.episodes} epsilon={epsilon:.3f} "
                  f"reward={np.mean([r['reward'] for r in recent]):.4f} "
                  f"hidden_RMSE={np.mean([r['final_rmse_hidden'] for r in recent]):.4f}", flush=True)
        if (episode + 1) % args.eval_every == 0 or episode + 1 == args.episodes:
            val_records, val_summary = evaluate(agent, env, val_paths,
                                                seed=configuration["validation_seed"],
                                                policies=("dqn", "uniform"))
            score = val_summary["dqn"]["final_rmse_hidden"]["mean"]
            by_file: dict[str, dict[str, dict]] = {}
            for record in val_records:
                by_file.setdefault(record["test_file"], {})[record["policy"]] = record
            wins_uniform = sum(pair["dqn"]["final_rmse_hidden"] < pair["uniform"]["final_rmse_hidden"]
                               for pair in by_file.values())
            checkpoint_scores.append({
                "episode": episode + 1,
                "validation_hidden_rmse": score,
                "validation_hidden_mae": val_summary["dqn"]["final_mae_hidden"]["mean"],
                "validation_full_rmse": val_summary["dqn"]["final_rmse_all"]["mean"],
                "validation_wins_vs_uniform": wins_uniform,
                "validation_count": len(by_file),
            })
            print(f"validation episode={episode+1} hidden_RMSE={score:.5f} best={best_rmse:.5f}", flush=True)
            if score < best_rmse:
                best_rmse = score
                agent.save(args.output / "best.pt", metadata={"episode": episode + 1, "validation_hidden_rmse": score})
    training_seconds = time.perf_counter() - started
    write_rows(args.output / "training_metrics.csv", train_rows)
    best_agent = MaskedDQNAgent.load(args.output / "best.pt")
    comparison_rows, comparison = evaluate(
        best_agent, env, val_paths, seed=configuration["validation_seed"],
        policies=("dqn", "random", "uniform"),
    )
    write_rows(args.output / "validation_per_volume.csv", comparison_rows)
    report = {"training_seconds": training_seconds, "best_validation_hidden_rmse": best_rmse,
              "checkpoint_candidates": checkpoint_scores, "validation": comparison}
    (args.output / "validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"results saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
