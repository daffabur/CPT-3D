"""Selection lock and paired uncertainty calculations."""

import json

import numpy as np
import pytest

from .final_test import bootstrap_paired_ci, lock_model


def test_bootstrap_ci_is_reproducible_and_paired():
    values = np.array([-0.04, -0.03, -0.02, -0.01])
    first = bootstrap_paired_ci(values, replicates=300)
    second = bootstrap_paired_ci(values, replicates=300)
    assert first == second
    assert first["mean_difference"] == pytest.approx(-0.025)
    assert first["ci95_upper"] < 0


def test_lock_selects_only_validation_and_refuses_relock(tmp_path):
    for seed, score in ((14, 0.10), (37, 0.09), (73, 0.11)):
        run = tmp_path / f"seed{seed}"
        run.mkdir()
        (run / "best.pt").write_bytes(bytes([seed]))
        (run / "config.json").write_text(json.dumps({
            "arguments": {"train_limit": 16000, "episodes": 16000,
                          "action_cells": 4, "budget": 8},
            "agent": {"action_count": 16},
            "environment": {"reward_mode": "oracle_rmse"},
            "schemagan_sha256": "frozen",
        }))
        (run / "validation_summary.json").write_text(json.dumps({"best_validation_hidden_rmse": score}))
    result = lock_model(tmp_path)
    assert result["selected"]["seed"] == 37
    assert result["test_evaluated"] is False
    with pytest.raises(FileExistsError):
        lock_model(tmp_path)
