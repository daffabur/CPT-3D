"""Masked, spatial DQN for 16 horizontal CPT action regions."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class AgentConfig:
    action_count: int = 16
    learning_rate: float = 1e-4
    gamma: float = 0.99
    replay_capacity: int = 5000
    batch_size: int = 32
    target_update_steps: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 500
    seed: int = 14
    coverage_channel: bool = False


class SpatialQNetwork(nn.Module):
    """Two 3D convolution layers followed by a 16-value Q head."""

    def __init__(self, action_count: int = 16, *, coverage_channel: bool = False) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(4 if coverage_channel else 3, 8, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(16 * 2 * 4 * 4 + 1, 64), nn.ReLU(), nn.Linear(64, action_count))

    def forward(self, volume: torch.Tensor, remaining_budget: torch.Tensor) -> torch.Tensor:
        features = self.encoder(volume)
        return self.head(torch.cat((features, remaining_budget.reshape(-1, 1)), dim=1))


def masked_next_values(q_values: torch.Tensor, valid_actions: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    """Bellman max over legal actions; terminal/all-invalid transitions yield zero."""

    if q_values.shape != valid_actions.shape or done.shape != (q_values.shape[0],):
        raise ValueError("invalid Q/action-mask/done shapes")
    has_action = valid_actions.any(dim=1)
    max_q = q_values.masked_fill(~valid_actions, -torch.inf).max(dim=1).values
    return torch.where(done | ~has_action, torch.zeros_like(max_q), max_q)


def _compact_state(state: dict) -> tuple[np.ndarray, float, np.ndarray]:
    return (np.asarray(state["volume"], dtype=np.float16).copy(),
            float(state["remaining_budget"]),
            np.asarray(state["valid_actions"], dtype=bool).copy())


class MaskedDQNAgent:
    def __init__(self, config: AgentConfig, *, ic_mean: float, ic_std: float) -> None:
        self.config = config
        self.ic_mean = ic_mean
        self.ic_std = ic_std
        # Evaluation never advances either training stream. Exploration and
        # replay sampling also have independent generators.
        self.exploration_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0]))
        self.replay_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
        torch.manual_seed(config.seed)
        self.device = torch.device("cpu")
        self.online = SpatialQNetwork(config.action_count, coverage_channel=config.coverage_channel).to(self.device)
        self.target = SpatialQNetwork(config.action_count, coverage_channel=config.coverage_channel).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay: deque = deque(maxlen=config.replay_capacity)
        self.learn_steps = 0

    def _normalize(self, volume: torch.Tensor) -> torch.Tensor:
        result = volume.clone()
        observed = result[:, 1:2]
        result[:, 0:1] = torch.where(observed > 0.5, (result[:, 0:1] - self.ic_mean) / self.ic_std, 0.0)
        result[:, 2:3] = (result[:, 2:3] - self.ic_mean) / self.ic_std
        if self.config.coverage_channel:
            # The map depends only on observed CPT coordinates, never hidden IC.
            observed_xy = result[:, 1, 0] > 0.5
            grid_y, grid_x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
            grid = torch.stack((grid_y, grid_x), dim=-1).to(result.device)
            maps = []
            for locations in observed_xy:
                distances = torch.cdist(grid.reshape(1, 256, 2).float(),
                                        torch.nonzero(locations).reshape(1, -1, 2).float())[0]
                maps.append((distances.min(dim=1).values.reshape(16, 16) / (15 * 2**0.5)).clamp(0, 1))
            coverage = torch.stack(maps)[:, None, None].expand(-1, 1, 8, -1, -1)
            result = torch.cat((result, coverage), dim=1)
        return result

    def epsilon(self, episode: int) -> float:
        progress = min(episode / max(self.config.epsilon_decay_episodes, 1), 1.0)
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    def act(self, state: dict, *, epsilon: float = 0.0) -> int:
        valid = np.flatnonzero(state["valid_actions"])
        if len(valid) == 0:
            raise ValueError("no legal actions")
        if epsilon > 0 and self.exploration_rng.random() < epsilon:
            return int(self.exploration_rng.choice(valid))
        with torch.no_grad():
            volume = self._normalize(torch.as_tensor(state["volume"], dtype=torch.float32).unsqueeze(0))
            budget = torch.tensor([state["remaining_budget"]], dtype=torch.float32)
            q = self.online(volume, budget)[0]
            q = q.masked_fill(~torch.as_tensor(state["valid_actions"]), -torch.inf)
        return int(q.argmax().item())

    def observe(self, state: dict, action: int, reward: float, next_state: dict, done: bool) -> float | None:
        self.replay.append((_compact_state(state), action, reward, _compact_state(next_state), done))
        if len(self.replay) < self.config.batch_size:
            return None
        return self._learn()

    def _learn(self) -> float:
        indices = self.replay_rng.choice(len(self.replay), self.config.batch_size, replace=False)
        batch = [self.replay[int(index)] for index in indices]
        states = [item[0] for item in batch]
        next_states = [item[3] for item in batch]
        volumes = self._normalize(torch.from_numpy(np.stack([s[0] for s in states]).astype(np.float32)))
        next_volumes = self._normalize(torch.from_numpy(np.stack([s[0] for s in next_states]).astype(np.float32)))
        budgets = torch.tensor([s[1] for s in states], dtype=torch.float32)
        next_budgets = torch.tensor([s[1] for s in next_states], dtype=torch.float32)
        next_masks = torch.from_numpy(np.stack([s[2] for s in next_states]))
        actions = torch.tensor([item[1] for item in batch], dtype=torch.int64)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        dones = torch.tensor([item[4] for item in batch], dtype=torch.bool)
        predicted = self.online(volumes, budgets).gather(1, actions[:, None])[:, 0]
        with torch.no_grad():
            next_q = self.target(next_volumes, next_budgets)
            targets = rewards + self.config.gamma * masked_next_values(next_q, next_masks, dones)
        loss = nn.functional.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.config.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "config": asdict(self.config),
            "ic_mean": self.ic_mean, "ic_std": self.ic_std,
            "exploration_rng_state": self.exploration_rng.bit_generator.state,
            "replay_rng_state": self.replay_rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "learn_steps": self.learn_steps, "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: Path) -> "MaskedDQNAgent":
        saved = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(AgentConfig(**saved["config"]), ic_mean=saved["ic_mean"], ic_std=saved["ic_std"])
        agent.online.load_state_dict(saved["online"])
        agent.target.load_state_dict(saved["target"])
        agent.optimizer.load_state_dict(saved["optimizer"])
        if "exploration_rng_state" in saved:
            agent.exploration_rng.bit_generator.state = saved["exploration_rng_state"]
            agent.replay_rng.bit_generator.state = saved["replay_rng_state"]
        elif "rng_state" in saved:  # Load checkpoints written before RNG separation.
            agent.exploration_rng.bit_generator.state = saved["rng_state"]
            agent.replay_rng.bit_generator.state = saved["rng_state"]
        torch.set_rng_state(saved["torch_rng_state"])
        agent.learn_steps = saved["learn_steps"]
        return agent
