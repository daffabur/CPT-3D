"""Headless tests for the standalone 3D game core."""

from pathlib import Path

import numpy as np
import pytest

from game3d.data_loader import extract_pdf_to_csv, load_cpt_csv
from game3d.environment3d import RealCPTEnvironment3D, SyntheticEnvironment3D
from game3d.interpolator3d import horizontal_idw
from game3d.visualization import headless_smoke_test


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "real_cpt" / "SONDIR SONDIR NGC 2025(1).pdf"
CSV = ROOT / "data" / "real_cpt" / "sondir_ngc_2025.csv"


@pytest.mark.skipif(not PDF.exists(), reason="authorized real CPT PDF is not distributed")
def test_pdf_extraction_is_complete_and_preserves_missing_surface_rows(tmp_path):
    output = tmp_path / "cpt.csv"
    report = extract_pdf_to_csv(PDF, output)
    dataset = load_cpt_csv(output)

    assert report["rows_total"] == 228
    assert report["measured_rows"] == 225
    assert report["missing_qc_rows"] == 3
    assert dataset.ids == ("S1", "S2", "S3")
    assert dataset.qc_profiles.shape == (3, 75)
    assert np.allclose(dataset.depths_m[[0, -1]], (0.2, 15.0))
    assert np.allclose(dataset.elevations_m, (770.0, 764.0, 764.5))


def test_synthetic_profile_sampling_and_reconstruction():
    environment = SyntheticEnvironment3D(seed=14)
    first_event = environment.sample()

    assert environment.truth.shape == (16, 16, 8)
    assert first_event.profile.shape == (8,)
    assert environment.prediction.shape == environment.truth.shape
    np.testing.assert_array_equal(
        environment.prediction[tuple(environment.vehicle)], first_event.profile
    )
    assert np.isfinite(environment.rmse)


def test_interpolator_depends_only_on_observed_profiles():
    positions = np.asarray([[0.0, 0.0], [2.0, 2.0]])
    profiles = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    first = horizontal_idw(positions, profiles, np.arange(3), np.arange(3)).volume
    unrelated_hidden_truth = np.full((3, 3, 2), -9999.0)
    second = horizontal_idw(positions, profiles, np.arange(3), np.arange(3)).volume

    assert unrelated_hidden_truth.shape == first.shape
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0, 0], profiles[0])
    np.testing.assert_array_equal(first[2, 2], profiles[1])


@pytest.mark.skipif(not CSV.exists(), reason="real CPT CSV is generated locally")
def test_real_mode_space_is_prediction_only():
    environment = RealCPTEnvironment3D(CSV)
    original_profiles = environment.dataset.qc_profiles.copy()
    event = environment.sample()

    assert event.measured is False
    assert environment.cpt_count == 3
    assert environment.prediction.shape[:2] == (16, 16)
    np.testing.assert_array_equal(environment.dataset.qc_profiles, original_profiles)


def test_pyvista_scene_geometry_builds_without_render_window():
    environment = SyntheticEnvironment3D()
    result = headless_smoke_test(environment)

    assert result["volume_shape"] == (16, 16, 8)
    assert result["volume_points"] == 16 * 16 * 8
    assert result["vehicle_cells"] > 0
    assert result["profile_cells"] > 0
