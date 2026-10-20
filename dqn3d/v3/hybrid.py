"""Frozen, ground-truth-free one-step SchemaGAN V3 / DQN V3 hybrid policy."""
from __future__ import annotations

import numpy as np
from synthetic3d.v2.core import Grid
from .baselines import uniform_action
from .features import build_state

DQN_COUNTS = frozenset((1, 2, 3, 5, 8))
UNIFORM_COUNTS = frozenset((12, 20))


def recommend_next_cpt(reconstructor, agent, observed_ic: np.ndarray,
                       observed_mask: np.ndarray, valid_mask: np.ndarray,
                       grid: Grid, access_mask: np.ndarray | None = None) -> dict:
    """Recommend one legal XY location using only observed CPT profiles."""
    observed = np.asarray(observed_mask, dtype=bool)
    if observed.shape != (grid.nx, grid.ny, grid.nz):
        raise ValueError('observed mask/grid shape mismatch')
    count = int(observed.any(axis=2).sum())
    if count < 1:
        raise ValueError('at least one observed CPT required')
    if count not in DQN_COUNTS | UNIFORM_COUNTS:
        return {'status': 'unsupported_cpt_count', 'policy_used': None,
                'observed_cpt_count': count, 'candidate_count': None,
                'x_index': None, 'y_index': None, 'x_m': None, 'y_m': None,
                'q_score': None}
    reconstruction, seconds = reconstructor.predict(observed_ic, observed, valid_mask, grid)
    state = build_state(reconstruction, observed, valid_mask, grid, access_mask)
    policy = 'scaled_dqn' if count in DQN_COUNTS else 'uniform'
    if len(state.candidate_xy) == 0:
        return {'status': 'no_candidate', 'policy_used': policy,
                'observed_cpt_count': count, 'candidate_count': 0,
                'x_index': None, 'y_index': None, 'x_m': None, 'y_m': None,
                'q_score': None, 'reconstruction_seconds': seconds}
    if policy == 'scaled_dqn':
        scores = agent.scores(state)
        action = int(np.argmax(scores))
        q_score = float(scores[action])
    else:
        action = uniform_action(state, observed, grid)
        q_score = None
    x, y = map(int, state.candidate_xy[action])
    xm, ym = grid.xy_m(x, y)
    return {'status': 'ok', 'policy_used': policy,
            'observed_cpt_count': count, 'candidate_count': len(state.candidate_xy),
            'x_index': x, 'y_index': y, 'x_m': float(xm), 'y_m': float(ym),
            'q_score': q_score, 'reconstruction_seconds': seconds}
