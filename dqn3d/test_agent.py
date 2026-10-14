"""Masked DQN and three-CPT warm-start regression tests."""

import numpy as np
import pytest
import torch
from torch import nn

from .agent import AgentConfig, MaskedDQNAgent, masked_next_values
from .environment import CPTPlacementEnv3D, EnvironmentConfig
from .test_environment import MeanReconstructor, volume
from .train import initial_sites, run_episode


class FixedQ(nn.Module):
    def forward(self, volume, remaining_budget):
        return torch.arange(16, dtype=torch.float32).repeat(volume.shape[0], 1)


def test_epsilon_greedy_masks_both_branches():
    agent = MaskedDQNAgent(AgentConfig(seed=4), ic_mean=2.5, ic_std=0.7)
    agent.online = FixedQ()
    state = {"volume": np.zeros((3, 8, 16, 16), dtype=np.float32),
             "remaining_budget": 0.5,
             "valid_actions": np.array([True, False, True] + [False] * 13)}
    assert {agent.act(state, epsilon=1.0) for _ in range(100)} == {0, 2}
    assert agent.act(state, epsilon=0.0) == 2


def test_bellman_max_ignores_invalid_and_terminal():
    q = torch.tensor([[100.0, 2.0, -1.0], [9.0, 3.0, 1.0], [8.0, 7.0, 6.0]])
    valid = torch.tensor([[False, True, True], [True, True, True], [False, False, False]])
    result = masked_next_values(q, valid, torch.tensor([False, True, False]))
    assert result.tolist() == [2.0, 0.0, 0.0]


def test_warm_start_uniqueness_budget_and_no_oracle_state():
    sites = initial_sites(7, 14)
    assert sites == initial_sites(7, 14)
    assert len(sites) == len(set(sites)) == 3
    env = CPTPlacementEnv3D(MeanReconstructor(), EnvironmentConfig(budget=8))
    state, info = env.reset(volume(), initial_xy=sites)
    assert info["cpt_count"] == 3
    assert np.count_nonzero(state["volume"][1]) == 3 * 8
    assert state["remaining_budget"] == pytest.approx(5 / 8)
    assert set(state) == {"volume", "remaining_budget", "valid_actions"}
    assert "rmse_all" not in state
    with pytest.raises(ValueError):
        env.reset(volume(), initial_xy=[sites[0]] * 3)
    result = run_episode(env, volume(), sites, "random", None, 14)
    assert result["episode_length"] == 5
    assert len(env.positions) == len(set(env.positions)) == 8
    assert result["sampling_cost_total"] == pytest.approx(5 * env.config.sampling_cost)


def test_checkpoint_save_load_and_rng_reproducibility(tmp_path):
    agent = MaskedDQNAgent(AgentConfig(seed=17), ic_mean=2.5, ic_std=0.7)
    path = tmp_path / "agent.pt"
    agent.save(path, metadata={"episode": 1})
    restored = MaskedDQNAgent.load(path)
    state = {"volume": np.zeros((3, 8, 16, 16), dtype=np.float32),
             "remaining_budget": 0.5, "valid_actions": np.ones(16, dtype=bool)}
    assert [agent.act(state, epsilon=1.0) for _ in range(20)] == [
        restored.act(state, epsilon=1.0) for _ in range(20)]
    with torch.no_grad():
        tensor = torch.zeros((1, 3, 8, 16, 16))
        budget = torch.tensor([0.5])
        assert torch.equal(agent.online(tensor, budget), restored.online(tensor, budget))


def test_coverage_channel_depends_only_on_observed_mask():
    agent = MaskedDQNAgent(AgentConfig(coverage_channel=True), ic_mean=2.5, ic_std=0.7)
    inputs = torch.zeros((1, 3, 8, 16, 16))
    inputs[:, 1, :, 4, 5] = 1
    first = agent._normalize(inputs)
    inputs[:, 2] = 999
    second = agent._normalize(inputs)
    assert first.shape == (1, 4, 8, 16, 16)
    assert torch.equal(first[:, 3], second[:, 3])
    assert first[0, 3, 0, 4, 5] == 0


def test_direct_coordinate_action_space_and_mae_reward():
    env = CPTPlacementEnv3D(MeanReconstructor(), EnvironmentConfig(
        budget=4, action_cells=16, reward_mode="oracle_mae", sampling_cost=0.005))
    env.reset(volume(), initial_xy=[(0, 0), (1, 1), (2, 2)])
    assert env.action_count == 256
    assert not env.valid_actions()[0]
    assert not env.valid_actions()[17]
    before = env._mae_all()
    _, reward, done, info = env.step(255)
    assert done and info["selected_xy"] == (15, 15)
    assert reward == pytest.approx(before - env._mae_all() - 0.005)


def test_validation_frequency_does_not_change_training_actions_or_weights():
    def train_with_validation_interval(interval):
        agent = MaskedDQNAgent(AgentConfig(seed=23, batch_size=2), ic_mean=2.5, ic_std=0.7)
        train_env = CPTPlacementEnv3D(MeanReconstructor())
        validation_env = CPTPlacementEnv3D(MeanReconstructor())
        sampled = []
        for episode in range(6):
            sites = initial_sites(episode, 23)
            result = run_episode(train_env, volume(), sites, "dqn", agent, 23 + episode,
                                 epsilon=0.6, learn=True)
            sampled.append(result["sampled_xy"])
            if (episode + 1) % interval == 0:
                run_episode(validation_env, volume(), initial_sites(0, 999),
                            "dqn", agent, 999, epsilon=0.0, learn=False)
        return sampled, {key: value.clone() for key, value in agent.online.state_dict().items()}

    actions_a, weights_a = train_with_validation_interval(1)
    actions_b, weights_b = train_with_validation_interval(3)
    assert actions_a == actions_b
    assert all(torch.equal(weights_a[key], weights_b[key]) for key in weights_a)
