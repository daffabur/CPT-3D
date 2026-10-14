"""Tests for deterministic, resumable 3D CSV generation."""

import numpy as np

from synthetic3d.generate_csv import (
    SHAPE,
    generate_split,
    generate_volume,
    read_csv_volume,
    seed_for,
    validate_csv,
    volume_digest,
)


def test_split_seed_namespaces_are_distinct_and_reproducible():
    seeds = {seed_for(split, 0, 14) for split in ("train", "vali", "test")}
    assert len(seeds) == 3
    first = generate_volume(seed=seed_for("train", 3, 14))
    second = generate_volume(seed=seed_for("train", 3, 14))
    np.testing.assert_array_equal(first, second)


def test_generated_csv_is_complete_and_resumable(tmp_path):
    output = tmp_path / "train"
    first = generate_split(
        split="train",
        count=3,
        output_dir=output,
        batch_size=2,
        validate_all_splits=False,
    )
    second = generate_split(
        split="train",
        count=3,
        output_dir=output,
        batch_size=2,
        validate_all_splits=False,
    )

    assert first == {"split": "train", "count": 3, "generated": 3, "skipped": 0}
    assert second == {"split": "train", "count": 3, "generated": 0, "skipped": 3}
    for index in range(3):
        path = output / f"cs3d_{index}.csv"
        report = validate_csv(path, expected_seed=seed_for("train", index))
        assert report["rows"] == 2048
        assert report["unique_coordinates"] == 2048
        assert read_csv_volume(path).shape == SHAPE


def test_neighboring_split_volumes_are_not_duplicates():
    digests = {
        volume_digest(generate_volume(seed=seed_for(split, index)))
        for split in ("train", "vali", "test")
        for index in range(3)
    }
    assert len(digests) == 9
