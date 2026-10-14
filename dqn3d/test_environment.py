"""Core invariants for the action, state, and reward interface."""

import numpy as np
import pytest
from pathlib import Path

from .environment import CPTPlacementEnv3D, EnvironmentConfig
from .reconstructor import SchemaGANReconstructor


class MeanReconstructor:
    def predict(self, observed_ic, observed_mask):
        average = float(observed_ic[observed_mask].mean())
        result = np.full((16, 16, 8), average, dtype=np.float32)
        result[observed_mask] = observed_ic[observed_mask]
        return result, 0.001


def volume():
    x, y, z = np.indices((16, 16, 8))
    return (1.5 + 0.04 * x + 0.03 * y + 0.1 * z).astype(np.float32)


def test_state_contains_only_observations_mask_prediction_budget():
    env = CPTPlacementEnv3D(MeanReconstructor(), EnvironmentConfig(budget=3))
    state, info = env.reset(volume(), initial_xy=(3, 4))
    assert set(state) == {"volume", "remaining_budget", "valid_actions"}
    assert state["volume"].shape == (3, 8, 16, 16)
    assert state["valid_actions"].shape == (16,)
    assert np.count_nonzero(state["volume"][1]) == 8
    assert np.count_nonzero(state["volume"][0]) == 8
    assert np.allclose(state["volume"][0, :, 4, 3], volume()[3, 4])
    assert state["remaining_budget"] == pytest.approx(2 / 3)
    assert info["cpt_count"] == 1


def test_action_profile_budget_and_fixed_domain_reward():
    env = CPTPlacementEnv3D(MeanReconstructor(), EnvironmentConfig(budget=2, sampling_cost=0.01))
    env.reset(volume(), initial_xy=(3, 4))
    action = 15
    chosen = env.candidate_xy(action)
    before = env._rmse_all()
    state, reward, done, info = env.step(action)
    assert done
    assert info["selected_xy"] == chosen
    assert env.observed_mask[chosen[0], chosen[1]].all()
    assert env.observed_mask.sum() == 16
    assert reward == pytest.approx(before - info["rmse_all"] - 0.01, abs=1e-6)
    assert state["remaining_budget"] == 0
    with pytest.raises(RuntimeError):
        env.step(action)


def test_hidden_ground_truth_is_not_in_state_and_duplicate_is_rejected():
    first = volume()
    second = first.copy()
    second[5, 5, :] += 10.0
    env = CPTPlacementEnv3D(MeanReconstructor())
    first_state, first_info = env.reset(first, initial_xy=(3, 4))
    second_state, second_info = env.reset(second, initial_xy=(3, 4))
    assert np.array_equal(first_state["volume"], second_state["volume"])
    assert first_info["rmse_all"] != second_info["rmse_all"]
    with pytest.raises(ValueError):
        env._sample((3, 4))


def test_coverage_proxy_uses_no_oracle():
    env = CPTPlacementEnv3D(MeanReconstructor(), EnvironmentConfig(budget=2, reward_mode="coverage_proxy"))
    env.reset(volume(), initial_xy=(3, 4))
    _, reward, _, info = env.step(15)
    assert reward == pytest.approx(info["reward_before"] - info["reward_after"] - env.config.sampling_cost)
    assert reward > -env.config.sampling_cost


def test_exhausted_cell_is_masked_out():
    env = CPTPlacementEnv3D(MeanReconstructor(), EnvironmentConfig(budget=20))
    env.reset(volume(), initial_xy=(8, 8))
    env.observed_mask[:4, :4, :] = True
    assert env.candidate_xy(0) is None
    assert not env.valid_actions()[0]
    with pytest.raises(ValueError):
        env.step(0)


def test_frozen_schemagan_cannot_observe_hidden_truth():
    checkpoint = Path(__file__).resolve().parents[1] / "results/schemagan3d/checkpoint.pt"
    if not checkpoint.exists():
        pytest.skip("trained local checkpoint unavailable")
    reconstructor = SchemaGANReconstructor(checkpoint)
    first = volume()
    second = first.copy()
    second[5, 5, :] += 10
    env = CPTPlacementEnv3D(reconstructor)
    state_a, info_a = env.reset(first, initial_xy=(3, 4))
    state_b, info_b = env.reset(second, initial_xy=(3, 4))
    assert np.array_equal(state_a["volume"], state_b["volume"])
    assert info_a["rmse_all"] != info_b["rmse_all"]
