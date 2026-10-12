import numpy as np
import pytest

from prototype_3d.reconstruction import (
    generate_synthetic_volume,
    horizontal_idw,
    regular_cpt_positions,
    sample_cpt_profile,
    space_filling_cpt_positions,
    volumetric_mae,
    volumetric_rmse,
)


def test_synthetic_volume_has_three_dimensional_variation():
    soil = generate_synthetic_volume(seed=14)

    assert soil.ic.shape == (16, 16, 8)
    assert soil.layer_ids.shape == soil.ic.shape
    assert soil.boundaries.shape == (16, 16, 2)
    assert set(np.unique(soil.layer_ids)) == {0, 1, 2}
    assert np.ptp(soil.boundaries[:, :, 0], axis=0).max() > 0
    assert np.ptp(soil.boundaries[:, :, 0], axis=1).max() > 0
    assert np.ptp(soil.ic, axis=0).max() > 0
    assert np.ptp(soil.ic, axis=1).max() > 0
    assert np.ptp(soil.ic, axis=2).max() > 0


def test_sampling_returns_complete_independent_profile():
    soil = generate_synthetic_volume(seed=7)
    profile = sample_cpt_profile(soil.ic, (3, 11))

    assert profile.shape == (8,)
    np.testing.assert_array_equal(profile, soil.ic[3, 11, :])
    profile[0] = -999
    assert soil.ic[3, 11, 0] != -999


def test_horizontal_idw_preserves_samples_and_beats_constant_baseline():
    soil = generate_synthetic_volume(seed=21)
    positions = regular_cpt_positions(soil.shape[:2], count_per_axis=3)
    profiles = np.stack([sample_cpt_profile(soil.ic, point) for point in positions])

    reconstructed = horizontal_idw(
        positions,
        profiles,
        soil.shape[:2],
        neighbors=8,
    )

    assert reconstructed.shape == soil.ic.shape
    for position, profile in zip(positions, profiles):
        np.testing.assert_array_equal(reconstructed[tuple(position)], profile)

    baseline = np.broadcast_to(profiles.mean(axis=0), soil.ic.shape)
    assert volumetric_rmse(soil.ic, reconstructed) < volumetric_rmse(soil.ic, baseline)


def test_horizontal_idw_rejects_duplicate_positions():
    positions = np.asarray([[0, 0], [0, 0]])
    profiles = np.ones((2, 8))

    with pytest.raises(ValueError, match="duplicates"):
        horizontal_idw(positions, profiles, (16, 16))


def test_space_filling_design_is_reproducible_unique_and_nested():
    first = space_filling_cpt_positions((16, 16), 20, seed=14)
    second = space_filling_cpt_positions((16, 16), 20, seed=14)
    smaller = space_filling_cpt_positions((16, 16), 5, seed=14)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:5], smaller)
    assert np.unique(first, axis=0).shape == (20, 2)


def test_idw_has_no_access_to_hidden_ground_truth():
    soil = generate_synthetic_volume(seed=14)
    positions = space_filling_cpt_positions(soil.shape[:2], 5, seed=14)
    observed_profiles = np.stack(
        [sample_cpt_profile(soil.ic, position) for position in positions]
    )

    first = horizontal_idw(positions, observed_profiles, soil.shape[:2])
    hidden_truth_changed = soil.ic.copy()
    sampled_mask = np.zeros(soil.shape[:2], dtype=bool)
    sampled_mask[positions[:, 0], positions[:, 1]] = True
    hidden_truth_changed[~sampled_mask] = -12345.0
    second = horizontal_idw(positions, observed_profiles, hidden_truth_changed.shape[:2])

    np.testing.assert_array_equal(first, second)


def test_volumetric_metrics_are_exact_for_known_difference():
    truth = np.zeros((2, 2, 2), dtype=np.float32)
    prediction = np.ones((2, 2, 2), dtype=np.float32)

    assert volumetric_rmse(truth, prediction) == pytest.approx(1.0)
    assert volumetric_mae(truth, prediction) == pytest.approx(1.0)
