"""Inference scorer copied unchanged from dqn3d/v3/agent.py."""
from __future__ import annotations
import torch
from torch import nn
from .features import CANDIDATE_NAMES, GLOBAL_NAMES

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

