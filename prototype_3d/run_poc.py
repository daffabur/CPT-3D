"""Run the standalone synthetic-volume reconstruction proof of concept."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from prototype_3d.reconstruction import (
    SyntheticSoilVolume,
    generate_synthetic_volume,
    horizontal_idw,
    sample_cpt_profile,
    space_filling_cpt_positions,
    volumetric_mae,
    volumetric_rmse,
)


@dataclass(frozen=True)
class ReconstructionResult:
    """One reconstruction produced from a specified number of CPT profiles."""

    cpt_count: int
    positions: np.ndarray
    observed_profiles: np.ndarray
    reconstructed_volume: np.ndarray
    rmse: float
    mae: float


def run(
    output_dir: Path,
    seed: int = 14,
    cpt_counts: Sequence[int] = (3, 5, 10, 20),
) -> list[ReconstructionResult]:
    """Run all CPT-count scenarios against one fixed synthetic ground truth."""

    counts = tuple(int(count) for count in cpt_counts)
    if not counts or any(count < 1 for count in counts):
        raise ValueError("cpt_counts must contain positive integers")
    if len(set(counts)) != len(counts):
        raise ValueError("cpt_counts must not contain duplicates")

    soil = generate_synthetic_volume(seed=seed)
    all_positions = space_filling_cpt_positions(
        soil.shape[:2],
        count=max(counts),
        seed=seed,
        spacing_xy=soil.spacing_xy,
    )
    results: list[ReconstructionResult] = []

    # The full volume remains evaluation-only. The interpolator receives only
    # positions and profile values that were explicitly sampled here.
    for count in counts:
        positions = all_positions[:count].copy()
        observed_profiles = np.stack(
            [sample_cpt_profile(soil.ic, position) for position in positions]
        )
        reconstructed = horizontal_idw(
            positions,
            observed_profiles,
            soil.shape[:2],
            neighbors=min(8, count),
            spacing_xy=soil.spacing_xy,
        )
        results.append(
            ReconstructionResult(
                cpt_count=count,
                positions=positions,
                observed_profiles=observed_profiles,
                reconstructed_volume=reconstructed,
                rmse=volumetric_rmse(soil.ic, reconstructed),
                mae=volumetric_mae(soil.ic, reconstructed),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_arrays(soil, results, output_dir / "reconstruction_3d.npz")
    _save_metrics(results, output_dir / "metrics.csv")
    _plot_metric_curve(results, output_dir / "metrics_by_cpt_count.png")
    for result in results:
        _plot_slices(
            soil.ic,
            result,
            output_dir / f"slices_{result.cpt_count:02d}_cpt.png",
        )
    return results


def _save_arrays(
    soil: SyntheticSoilVolume,
    results: Sequence[ReconstructionResult],
    output_path: Path,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "true_volume": soil.ic,
        "layer_ids": soil.layer_ids,
        "boundaries": soil.boundaries,
    }
    for result in results:
        suffix = f"{result.cpt_count:02d}_cpt"
        arrays[f"positions_{suffix}"] = result.positions
        arrays[f"observed_profiles_{suffix}"] = result.observed_profiles
        arrays[f"reconstruction_{suffix}"] = result.reconstructed_volume
    np.savez_compressed(output_path, **arrays)


def _save_metrics(
    results: Sequence[ReconstructionResult], output_path: Path
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("cpt_count", "rmse", "mae"))
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "cpt_count": result.cpt_count,
                    "rmse": f"{result.rmse:.8f}",
                    "mae": f"{result.mae:.8f}",
                }
            )


def _plot_metric_curve(
    results: Sequence[ReconstructionResult], output_path: Path
) -> None:
    counts = [result.cpt_count for result in results]
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(counts, [result.rmse for result in results], "o-", label="RMSE")
    axis.plot(counts, [result.mae for result in results], "s-", label="MAE")
    axis.set(xlabel="Number of CPT profiles", ylabel="Volumetric IC error")
    axis.set_xticks(counts)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_slices(
    true_volume: np.ndarray,
    result: ReconstructionResult,
    output_path: Path,
) -> None:
    reconstructed = result.reconstructed_volume
    nx, ny, nz = true_volume.shape
    x_mid, y_mid, z_mid = nx // 2, ny // 2, nz // 2
    figure, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    slices = (
        (true_volume[:, :, z_mid], reconstructed[:, :, z_mid], "X-Y", f"Z={z_mid}"),
        (true_volume[:, y_mid, :], reconstructed[:, y_mid, :], "X-Z", f"Y={y_mid}"),
        (true_volume[x_mid, :, :], reconstructed[x_mid, :, :], "Y-Z", f"X={x_mid}"),
    )
    vmin = min(float(true_volume.min()), float(reconstructed.min()))
    vmax = max(float(true_volume.max()), float(reconstructed.max()))
    for row, (truth, prediction, plane, location) in enumerate(slices):
        error = np.abs(truth - prediction)
        axes[row, 0].imshow(truth.T, origin="lower", vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(prediction.T, origin="lower", vmin=vmin, vmax=vmax)
        axes[row, 2].imshow(error.T, origin="lower", vmin=0.0)
        axes[row, 0].set_ylabel(f"{plane} ({location})")
        if row == 0:
            for column in (0, 1):
                axes[row, column].scatter(
                    result.positions[:, 0],
                    result.positions[:, 1],
                    c="red",
                    edgecolors="white",
                    linewidths=0.35,
                    s=24,
                )
    for column, title in enumerate(("Ground truth IC", "IDW reconstruction", "Absolute error")):
        axes[0, column].set_title(title)
    figure.suptitle(
        f"{result.cpt_count} CPT profiles | RMSE={result.rmse:.4f} | MAE={result.mae:.4f}"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("prototype_3d/output"))
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--cpt-counts", nargs="+", type=int, default=(3, 5, 10, 20))
    args = parser.parse_args()
    results = run(args.output_dir, args.seed, args.cpt_counts)
    for result in results:
        print(
            f"{result.cpt_count:>2} CPT | RMSE={result.rmse:.6f} | "
            f"MAE={result.mae:.6f}"
        )
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
