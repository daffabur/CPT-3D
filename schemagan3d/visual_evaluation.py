"""Evaluate a fixed 3D checkpoint across CPT counts and placement patterns.

This module only performs inference. It never modifies the checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

from synthetic3d.generate_csv import read_csv_volume

from .baseline import horizontal_idw
from .data import NormalizationStats, sorted_csv_paths
from .device import select_device
from .models import Generator3D


GRID = 16
DEPTH = 8
COUNTS = (2, 4, 8, 16)
PATTERNS = ("random", "uniform", "clustered")


def sampling_order(pattern: str, seed: int) -> np.ndarray:
    """Return one deterministic ordering of all XY sites; prefixes are nested."""

    rng = np.random.default_rng(seed)
    xy = np.stack(np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij"), -1).reshape(-1, 2)
    if pattern == "random":
        return xy[rng.permutation(len(xy))]
    if pattern == "clustered":
        anchor = xy[rng.integers(len(xy))]
        distance = np.sum((xy - anchor) ** 2, axis=1)
        return xy[np.lexsort((rng.random(len(xy)), distance))]
    if pattern == "uniform":
        # Greedy farthest-site design: expand coverage with every extra CPT.
        chosen = [int(rng.integers(len(xy)))]
        best_distance = np.sum((xy - xy[chosen[0]]) ** 2, axis=1)
        for _ in range(1, len(xy)):
            best_distance[chosen[-1]] = -1
            next_index = int(np.argmax(best_distance))
            chosen.append(next_index)
            best_distance = np.minimum(best_distance, np.sum((xy - xy[next_index]) ** 2, axis=1))
        return xy[chosen]
    raise ValueError(f"unknown pattern: {pattern}")


def make_observation(target: torch.Tensor, xy: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Target is normalized in Conv3D layout (1,Z,Y,X); reveal full Z profiles."""

    if target.shape != (1, DEPTH, GRID, GRID):
        raise ValueError("expected normalized target shape (1,8,16,16)")
    mask = torch.zeros_like(target)
    for x, y in xy:
        mask[:, :, int(y), int(x)] = 1
    return target * mask, mask


def error_metrics(prediction: np.ndarray, truth: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    hidden = ~observed
    return {
        "rmse_unobserved": float(np.sqrt(np.mean(np.square(error[hidden])))),
        "mae_unobserved": float(np.mean(np.abs(error[hidden]))),
        "rmse_all": float(np.sqrt(np.mean(np.square(error)))),
        "observed_max_error": float(np.max(np.abs(error[observed]))),
    }


def _slices(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return images with horizontal X and vertical Y or Z axes."""

    return volume[:, :, 4].T, volume[:, 8, :].T, volume[8, :, :].T


def plot_slices(
    path: Path, truth: np.ndarray, idw: np.ndarray, gan: np.ndarray, xy: np.ndarray, title: str
) -> None:
    ic_limits = (float(min(truth.min(), idw.min(), gan.min())), float(max(truth.max(), idw.max(), gan.max())))
    error_limit = float(max(np.abs(idw - truth).max(), np.abs(gan - truth).max()))
    rows = (truth, idw, gan, np.abs(idw - truth), np.abs(gan - truth))
    labels = ("Ground truth IC", "IDW IC", "SchemaGAN IC", "IDW absolute error", "SchemaGAN absolute error")
    slice_names = ("X–Y at Z=4", "X–Z at Y=8", "Y–Z at X=8")
    fig, axes = plt.subplots(5, 3, figsize=(13, 16), constrained_layout=True)
    for row, (volume, label) in enumerate(zip(rows, labels)):
        for column, image in enumerate(_slices(volume)):
            error_row = row >= 3
            axes[row, column].imshow(
                image,
                origin="lower",
                interpolation="nearest",
                aspect="auto",
                cmap="magma" if error_row else "viridis",
                vmin=0 if error_row else ic_limits[0],
                vmax=error_limit if error_row else ic_limits[1],
            )
            axes[row, column].set(title=f"{label} · {slice_names[column]}")
            axes[row, column].set_xlabel("X" if column != 2 else "Y")
            axes[row, column].set_ylabel("Y" if column == 0 else "Z")
            if row == 0 and column == 0:
                axes[row, column].scatter(xy[:, 0], xy[:, 1], marker="x", s=42, c="red", linewidths=1.4)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(*ic_limits), cmap="viridis"),
        ax=axes[:3, :], shrink=0.55, label="IC (common scale)",
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(0, error_limit), cmap="magma"),
        ax=axes[3:, :], shrink=0.7, label="Absolute IC error (common scale)",
    )
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_volume(
    path: Path, truth: np.ndarray, idw: np.ndarray, gan: np.ndarray, xy: np.ndarray, title: str
) -> None:
    """3D cutaway: show sampled columns and a clipped volume to expose interior."""

    arrays = (truth, idw, gan, np.abs(idw - truth), np.abs(gan - truth))
    labels = ("Ground truth", "IDW", "SchemaGAN", "IDW error", "SchemaGAN error")
    ic_limits = (float(min(a.min() for a in arrays[:3])), float(max(a.max() for a in arrays[:3])))
    error_limit = float(max(np.abs(idw - truth).max(), arrays[3].max()))
    x, y, z = np.indices(truth.shape)
    # Outer surface plus an interior cut in the positive X/Y corner.
    cutaway = (x < 8) | (y < 8)
    visible = cutaway & ((x == 0) | (y == 0) | (z == 0) | (z == DEPTH - 1) | (x == 7) | (y == 7))
    fig = plt.figure(figsize=(16, 9))
    for index, (volume, label) in enumerate(zip(arrays, labels), 1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        is_error = index >= 4
        norm = colors.Normalize(0, error_limit) if is_error else colors.Normalize(*ic_limits)
        color_map = plt.get_cmap("magma" if is_error else "viridis")
        ax.scatter(x[visible], y[visible], -z[visible], c=color_map(norm(volume[visible])), s=11, alpha=0.52, linewidths=0)
        for cx, cy in xy:
            ax.plot([cx, cx], [cy, cy], [0, -(DEPTH - 1)], c="red", linewidth=1.2, alpha=0.9)
        ax.set(xlim=(0, 15), ylim=(0, 15), zlim=(-7, 0), xlabel="X", ylabel="Y", zlabel="−Z", title=label)
        ax.view_init(elev=23, azim=-55)
        ax.set_box_aspect((1, 1, 0.55))
    ic_bar = fig.add_axes((0.91, 0.56, 0.015, 0.28))
    error_bar = fig.add_axes((0.91, 0.15, 0.015, 0.28))
    fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(*ic_limits), cmap="viridis"), cax=ic_bar, label="IC (common scale)")
    fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, error_limit), cmap="magma"), cax=error_bar, label="Absolute IC error")
    fig.suptitle(title + " · red columns = sampled CPT")
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.04, top=0.91, wspace=0.10, hspace=0.10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for pattern in PATTERNS:
        for count in COUNTS:
            subset = [row for row in rows if row["pattern"] == pattern and row["cpt_count"] == count]
            for method in ("IDW", "SchemaGAN"):
                selected = [row for row in subset if row["method"] == method]
                item = {"pattern": pattern, "cpt_count": count, "method": method, "volumes": len(selected),
                        "outside_training_distribution": selected[0]["outside_training_distribution"]}
                for name in ("rmse_unobserved", "mae_unobserved", "rmse_all"):
                    item[name] = float(np.mean([row[name] for row in selected]))
                summary.append(item)
    return summary


def plot_trends(path: Path, summary: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, constrained_layout=True)
    for axis, pattern in zip(axes, PATTERNS):
        for method, marker in (("IDW", "o"), ("SchemaGAN", "s")):
            subset = [row for row in summary if row["pattern"] == pattern and row["method"] == method]
            axis.plot([row["cpt_count"] for row in subset], [row["rmse_unobserved"] for row in subset],
                      marker=marker, label=method)
        axis.set(title=pattern.capitalize(), xlabel="Number of CPT profiles", xticks=COUNTS)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean RMSE on unobserved voxels (IC)")
    axes[-1].legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training = saved["args"]
    training_min = training["profiles_min"]
    training_max = training["profiles_max"]
    stats = NormalizationStats(**saved["normalization"])
    device = select_device(args.device)
    model = Generator3D(saved["base_channels"]).to(device)
    model.load_state_dict(saved["generator"])
    model.eval()
    test_paths = sorted_csv_paths(args.data_root / "test", args.test_pool)
    if len(test_paths) < args.volumes:
        raise ValueError("requested more volumes than available in test pool")
    rng = np.random.default_rng(args.seed)
    selected_indices = sorted(rng.choice(len(test_paths), size=args.volumes, replace=False).tolist())
    destination = args.output or Path("results") / f"schemagan3d_visual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.mkdir(parents=True)
    figures = destination / "figures"
    figures.mkdir()
    rows = []
    for volume_number, index in enumerate(selected_indices):
        source = test_paths[index]
        truth = read_csv_volume(source)
        normalized = stats.normalize(torch.from_numpy(truth.transpose(2, 1, 0).copy()).unsqueeze(0))
        for pattern in PATTERNS:
            order = sampling_order(pattern, args.seed + index * 37 + PATTERNS.index(pattern))
            for count in COUNTS:
                xy = order[:count]
                sparse, mask = make_observation(normalized, xy)
                with torch.inference_mode():
                    gan_tensor = model(sparse.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))[0].cpu()
                    idw_tensor = horizontal_idw(sparse, mask)
                gan = stats.denormalize(gan_tensor)[0].numpy().transpose(2, 1, 0)
                idw = stats.denormalize(idw_tensor)[0].numpy().transpose(2, 1, 0)
                observed = mask[0].numpy().transpose(2, 1, 0).astype(bool)
                outside = count < training_min or count > training_max or pattern != "random"
                for method, prediction in (("IDW", idw), ("SchemaGAN", gan)):
                    rows.append({
                        "test_file": source.name, "test_index": index, "pattern": pattern,
                        "cpt_count": count, "method": method,
                        "outside_training_distribution": outside,
                        "cpt_xy": ";".join(f"{x}:{y}" for x, y in xy),
                        **error_metrics(prediction, truth, observed),
                    })
                if volume_number == 0 and (count == 8 or (pattern == "random" and count in (2, 16))):
                    stem = f"{source.stem}_{pattern}_{count:02d}cpt"
                    label = f"{source.name} · {pattern} · {count} CPT" + (" · OOD" if outside else "")
                    plot_slices(figures / f"{stem}_slices.png", truth, idw, gan, xy, label)
                    plot_volume(figures / f"{stem}_volume.png", truth, idw, gan, xy, label)
        print(f"evaluated {volume_number + 1}/{args.volumes}: {source.name}", flush=True)
    summary = summarize(rows)
    write_csv(destination / "metrics_per_volume.csv", rows)
    write_csv(destination / "metrics_summary.csv", summary)
    plot_trends(destination / "error_vs_cpt_count.png", summary)
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha256,
        "data_root": str(args.data_root.resolve()), "test_indices": selected_indices,
        "test_files": [test_paths[i].name for i in selected_indices],
        "seed": args.seed, "device": str(device), "training_profiles_range": [training_min, training_max],
        "training_location_pattern": "random", "counts": COUNTS, "patterns": PATTERNS,
        "notes": ["CPT locations are identical for IDW and SchemaGAN per comparison.",
                  "Sampling locations are nested within each volume and pattern.",
                  "Ground truth only builds observations and computes evaluation metrics."],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved results: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/schemagan3d/checkpoint.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/synthetic3d"))
    parser.add_argument("--volumes", type=int, default=12)
    parser.add_argument("--test-pool", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
