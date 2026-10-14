"""Generate resumable synthetic 3D soil datasets as ``x,y,z,IC`` CSV files.

The generator keeps only one 16 x 16 x 8 volume in memory. Existing files are
validated and skipped, so an interrupted run can safely be resumed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SHAPE = (16, 16, 8)
COLUMNS = ("x", "y", "z", "IC")
SPLITS = ("train", "vali", "test")
SPLIT_SEED_OFFSETS = {"test": 0, "train": 1_000_000, "vali": 2_000_000}
TARGET_COUNTS = {"train": 16_000, "vali": 4_000, "test": 4_000}
DEFAULT_SEED = 14
DEFAULT_COUNT = 5
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "synthetic3d"
DEFAULT_VISUALIZATION_DIR = ROOT / "synthetic3d" / "visualizations"


def seed_for(split: str, index: int, base_seed: int = DEFAULT_SEED) -> int:
    """Return a deterministic, non-overlapping seed namespace per split."""

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if index < 0:
        raise ValueError("index must be non-negative")
    return int(base_seed + SPLIT_SEED_OFFSETS[split] + index)


def generate_volume(
    *, shape: tuple[int, int, int] = SHAPE, seed: int = DEFAULT_SEED
) -> np.ndarray:
    """Return a three-layer IC volume with genuine X/Y/Z variation."""

    if len(shape) != 3 or any(size < 3 for size in shape):
        raise ValueError("shape must contain three dimensions with size >= 3")
    nx, ny, nz = shape
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, nx, dtype=np.float64)[:, None]
    y = np.linspace(0.0, 1.0, ny, dtype=np.float64)[None, :]
    z = np.arange(nz, dtype=np.float64)[None, None, :]
    phase_1, phase_2 = rng.uniform(0.0, 2.0 * np.pi, size=2)

    boundary_1 = (
        0.29 * nz
        + 0.075 * nz * np.sin(2.0 * np.pi * x + phase_1)
        * np.cos(2.0 * np.pi * y - 0.35 * phase_1)
        + 0.025 * nz * np.sin(2.0 * np.pi * (x + y))
    )
    boundary_2 = (
        0.66 * nz
        + 0.070 * nz * np.cos(2.0 * np.pi * x + phase_2)
        * np.sin(2.0 * np.pi * y + 0.25 * phase_2)
        + 0.020 * nz * np.cos(2.0 * np.pi * (x - y))
    )
    separation = max(1.0, 0.18 * nz)
    boundary_1 = np.clip(boundary_1, 0.75, nz - separation - 0.75)
    boundary_2 = np.clip(
        np.maximum(boundary_2, boundary_1 + separation),
        boundary_1 + separation,
        nz - 0.35,
    )
    layer_ids = np.where(
        z < boundary_1[:, :, None],
        0,
        np.where(z < boundary_2[:, :, None], 1, 2),
    )
    base_ic = np.asarray((1.65, 2.55, 3.45))[layer_ids]
    x3, y3, z3 = x[:, :, None], y[:, :, None], z / max(nz - 1, 1)
    amplitudes = np.asarray((0.10, 0.14, 0.18))[layer_ids]
    depth_slopes = np.asarray((0.06, -0.04, 0.09))[layer_ids]
    spatial_pattern = (
        0.50 * np.sin(2.0 * np.pi * x3 + 0.55 * z3 + phase_1)
        + 0.32 * np.cos(2.0 * np.pi * y3 - 0.65 * z3 + phase_2)
        + 0.18 * np.sin(2.0 * np.pi * (x3 + y3) + 0.85 * z3)
    )
    noise = rng.normal(0.0, 0.012, size=shape)
    return (base_ic + amplitudes * spatial_pattern + depth_slopes * z3 + noise).astype(
        np.float32
    )


def volume_digest(volume: np.ndarray) -> str:
    """Stable identity used to prevent duplicate volumes across all splits."""

    return hashlib.sha256(np.asarray(volume, dtype=np.float32).tobytes()).hexdigest()


def write_csv(volume: np.ndarray, path: Path) -> None:
    """Atomically write one voxel per row without an index column."""

    if volume.shape != SHAPE:
        raise ValueError(f"volume must have shape {SHAPE}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for x, y, z in np.ndindex(volume.shape):
            writer.writerow((x, y, z, f"{float(volume[x, y, z]):.8f}"))
    temporary.replace(path)


def read_csv_volume(path: Path, *, shape: tuple[int, int, int] = SHAPE) -> np.ndarray:
    """Read and strictly validate schema, coordinates, coverage, and IC values."""

    expected_count = int(np.prod(shape))
    expected_coordinates = set(np.ndindex(shape))
    volume = np.empty(shape, dtype=np.float32)
    seen: set[tuple[int, int, int]] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path.name}: expected columns {COLUMNS}")
        for row_number, row in enumerate(reader, start=2):
            try:
                coordinate = (int(row["x"]), int(row["y"]), int(row["z"]))
                value = float(row["IC"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}: invalid row {row_number}") from exc
            if coordinate not in expected_coordinates:
                raise ValueError(f"{path.name}: out-of-range coordinate {coordinate}")
            if coordinate in seen:
                raise ValueError(f"{path.name}: duplicate coordinate {coordinate}")
            if not np.isfinite(value):
                raise ValueError(f"{path.name}: non-finite IC on row {row_number}")
            seen.add(coordinate)
            volume[coordinate] = value
    if len(seen) != expected_count:
        raise ValueError(
            f"{path.name}: expected {expected_count} voxels, found {len(seen)}"
        )
    return volume


def validate_csv(
    path: Path,
    *,
    shape: tuple[int, int, int] = SHAPE,
    expected_seed: int | None = None,
) -> dict:
    """Validate one CSV, including deterministic reproduction when requested."""

    volume = read_csv_volume(path, shape=shape)
    axis_ranges = {
        "x": float(np.ptp(volume, axis=0).max()),
        "y": float(np.ptp(volume, axis=1).max()),
        "z": float(np.ptp(volume, axis=2).max()),
    }
    if any(value <= 0.0 for value in axis_ranges.values()):
        raise ValueError(f"{path.name}: IC does not vary along every axis")
    digest = volume_digest(volume)
    if expected_seed is not None:
        expected = generate_volume(shape=shape, seed=expected_seed)
        if digest != volume_digest(expected):
            raise ValueError(
                f"{path.name}: content does not match deterministic seed {expected_seed}"
            )
    return {
        "path": str(path),
        "rows": int(np.prod(shape)),
        "unique_coordinates": int(np.prod(shape)),
        "ic_min": float(volume.min()),
        "ic_max": float(volume.max()),
        "axis_ranges": axis_ranges,
        "digest": digest,
    }


def plot_slices(volume: np.ndarray, path: Path, *, title: str) -> None:
    """Save central X-Y, X-Z, and Y-Z diagnostic slices."""

    nx, ny, nz = volume.shape
    slices = (
        (volume[:, :, nz // 2].T, f"X-Y at Z={nz // 2}", "X", "Y"),
        (volume[:, ny // 2, :].T, f"X-Z at Y={ny // 2}", "X", "Z"),
        (volume[nx // 2, :, :].T, f"Y-Z at X={nx // 2}", "Y", "Z"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    vmin, vmax = float(volume.min()), float(volume.max())
    for axis, (values, subtitle, xlabel, ylabel) in zip(axes, slices):
        image = axis.imshow(
            values, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
        )
        axis.set(title=subtitle, xlabel=xlabel, ylabel=ylabel)
    figure.colorbar(image, ax=axes, label="IC")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _collect_existing_digests(data_root: Path, base_seed: int) -> dict[str, Path]:
    """Validate existing generated files and reject duplicates across splits."""

    digests: dict[str, Path] = {}
    for split in SPLITS:
        for path in sorted((data_root / split).glob("cs3d_*.csv")):
            try:
                index = int(path.stem.removeprefix("cs3d_"))
            except ValueError as exc:
                raise ValueError(f"unexpected dataset filename: {path.name}") from exc
            report = validate_csv(path, expected_seed=seed_for(split, index, base_seed))
            previous = digests.get(report["digest"])
            if previous:
                raise ValueError(f"duplicate volumes: {previous} and {path}")
            digests[report["digest"]] = path
    return digests


def generate_split(
    *,
    split: str,
    count: int,
    data_root: Path = DATA_ROOT,
    output_dir: Path | None = None,
    visualization_dir: Path = DEFAULT_VISUALIZATION_DIR,
    base_seed: int = DEFAULT_SEED,
    batch_size: int = 100,
    visualize_count: int = 0,
    validate_all_splits: bool = True,
) -> dict:
    """Generate one split incrementally, validating and resuming existing files."""

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if count < 1 or batch_size < 1 or visualize_count < 0:
        raise ValueError("count/batch_size must be positive; visualize_count >= 0")
    destination = output_dir or data_root / split
    destination.mkdir(parents=True, exist_ok=True)
    known = (
        _collect_existing_digests(data_root, base_seed)
        if validate_all_splits and output_dir is None
        else {}
    )
    generated = skipped = 0
    for index in range(count):
        path = destination / f"cs3d_{index}.csv"
        dataset_seed = seed_for(split, index, base_seed)
        if path.exists():
            report = validate_csv(path, expected_seed=dataset_seed)
            known.setdefault(report["digest"], path)
            skipped += 1
        else:
            volume = generate_volume(seed=dataset_seed)
            digest = volume_digest(volume)
            duplicate = known.get(digest)
            if duplicate:
                raise ValueError(f"new volume duplicates {duplicate}")
            write_csv(volume, path)
            validate_csv(path, expected_seed=dataset_seed)
            known[digest] = path
            generated += 1
            if index < visualize_count:
                plot_slices(
                    volume,
                    visualization_dir / split / f"cs3d_{index}_slices.png",
                    title=f"{split} dataset {index} (seed={dataset_seed})",
                )
        completed = index + 1
        if completed % batch_size == 0 or completed == count:
            print(
                f"[{split}] {completed}/{count} complete "
                f"(generated={generated}, resumed={skipped})"
            )
    return {"split": split, "count": count, "generated": generated, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--visualization-dir", type=Path, default=DEFAULT_VISUALIZATION_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--visualize-count", type=int, default=0)
    parser.add_argument(
        "--no-cross-split-validation",
        action="store_true",
        help="Skip scanning other split folders (not recommended for final generation).",
    )
    args = parser.parse_args()
    report = generate_split(
        split=args.split,
        count=args.count,
        data_root=args.data_root,
        output_dir=args.output_dir,
        visualization_dir=args.visualization_dir,
        base_seed=args.seed,
        batch_size=args.batch_size,
        visualize_count=args.visualize_count,
        validate_all_splits=not args.no_cross_split_validation,
    )
    print(
        f"Finished {report['split']}: target={report['count']}, "
        f"generated={report['generated']}, resumed={report['skipped']}"
    )


if __name__ == "__main__":
    main()
