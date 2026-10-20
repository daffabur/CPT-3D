"""Shared candidate Q network, variable-action replay, and masked Bellman update."""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import numpy as np
import torch
from torch import nn
from .features import CandidateState, CANDIDATE_NAMES, GLOBAL_NAMES


@dataclass(frozen=True)
class AgentConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    replay_capacity: int = 512
    batch_size: int = 8
    target_update_steps: int = 50
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 200
    seed: int = 14
    hidden_width: int = 64

    def __post_init__(self):
        if self.learning_rate <= 0 or not 0 <= self.gamma <= 1 or self.replay_capacity < 1 or self.batch_size < 1 or self.target_update_steps < 1 or self.hidden_width < 1:
            raise ValueError('invalid DQN configuration')


class SharedCandidateQ(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(CANDIDATE_NAMES) + len(GLOBAL_NAMES), width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != len(CANDIDATE_NAMES) or context.shape != (len(GLOBAL_NAMES),):
            raise ValueError('candidate Q tensor shape mismatch')
        if len(features) == 0:
            return features.new_empty((0,))
        return self.net(torch.cat((features, context[None].expand(len(features), -1)), dim=1)).squeeze(1)


@dataclass(frozen=True)
class Transition:
    state: CandidateState
    action_index: int
    action_xy: tuple[int, int]
    reward: float
    next_state: CandidateState
    done: bool


def _copy_state(state: CandidateState) -> CandidateState:
    return CandidateState(state.candidate_xy.copy(), state.features.copy(), state.global_context.copy())


class CandidateDQNAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        torch.manual_seed(config.seed)
        self.online = SharedCandidateQ(config.hidden_width)
        self.target = SharedCandidateQ(config.hidden_width)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay: deque[Transition] = deque(maxlen=config.replay_capacity)
        self.exploration_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0]))
        self.replay_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
        self.learn_steps = 0

    @staticmethod
    def _q(model: SharedCandidateQ, state: CandidateState) -> torch.Tensor:
        return model(torch.from_numpy(state.features.astype(np.float32)),
                     torch.from_numpy(state.global_context.astype(np.float32)))

    def epsilon(self, episode: int) -> float:
        t = min(max(episode, 0) / self.config.epsilon_decay_episodes, 1.0)
        return self.config.epsilon_start + t * (self.config.epsilon_end - self.config.epsilon_start)

    def scores(self, state: CandidateState) -> np.ndarray:
        with torch.inference_mode():
            return self._q(self.online, state).numpy().copy()

    def act(self, state: CandidateState, epsilon: float = 0.0) -> int:
        if len(state.candidate_xy) == 0:
            raise ValueError('no legal candidate')
        if not 0 <= epsilon <= 1:
            raise ValueError('epsilon outside [0,1]')
        if epsilon > 0 and self.exploration_rng.random() < epsilon:
            return int(self.exploration_rng.integers(len(state.candidate_xy)))
        # np.argwhere is lexicographic; torch.argmax returns the first exact tie.
        return int(np.argmax(self.scores(state)))

    def observe(self, state: CandidateState, action_index: int, reward: float,
                next_state: CandidateState, done: bool) -> float | None:
        if not 0 <= action_index < len(state.candidate_xy):
            raise ValueError('replay action index not legal in state')
        if not np.isfinite(reward):
            raise ValueError('nonfinite reward')
        if not done and len(next_state.candidate_xy) == 0:
            raise ValueError('nonterminal transition has no next legal action')
        xy = tuple(map(int, state.candidate_xy[action_index]))
        self.replay.append(Transition(_copy_state(state), int(action_index), xy, float(reward), _copy_state(next_state), bool(done)))
        if len(self.replay) < self.config.batch_size:
            return None
        return self.learn_step()

    def learn_step(self) -> float:
        if len(self.replay) < self.config.batch_size:
            raise ValueError('insufficient replay')
        picks = self.replay_rng.choice(len(self.replay), self.config.batch_size, replace=False)
        batch = [self.replay[int(i)] for i in picks]
        predictions = []
        targets = []
        for tr in batch:
            if tuple(map(int, tr.state.candidate_xy[tr.action_index])) != tr.action_xy:
                raise ValueError('replay candidate/action mapping corrupted')
            predictions.append(self._q(self.online, tr.state)[tr.action_index])
            with torch.no_grad():
                future = (self._q(self.target, tr.next_state).max() if not tr.done else torch.tensor(0.0))
                targets.append(torch.tensor(tr.reward, dtype=torch.float32) + self.config.gamma * future)
        loss = nn.functional.smooth_l1_loss(torch.stack(predictions), torch.stack(targets))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.config.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'config': asdict(self.config), 'online': self.online.state_dict(),
                   'target': self.target.state_dict(), 'optimizer': self.optimizer.state_dict(),
                   'replay': list(self.replay), 'learn_steps': self.learn_steps,
                   'exploration_rng_state': self.exploration_rng.bit_generator.state,
                   'replay_rng_state': self.replay_rng.bit_generator.state,
                   'torch_rng_state': torch.get_rng_state(), 'metadata': metadata or {}}
        temp = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temp)
        os.replace(temp, path)

    @classmethod
    def load(cls, path: Path) -> tuple['CandidateDQNAgent', dict]:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        agent = cls(AgentConfig(**payload['config']))
        agent.online.load_state_dict(payload['online'])
        agent.target.load_state_dict(payload['target'])
        agent.optimizer.load_state_dict(payload['optimizer'])
        agent.replay.extend(payload['replay'])
        agent.learn_steps = int(payload['learn_steps'])
        agent.exploration_rng.bit_generator.state = payload['exploration_rng_state']
        agent.replay_rng.bit_generator.state = payload['replay_rng_state']
        torch.set_rng_state(payload['torch_rng_state'])
        return agent, payload['metadata']
