"""Field-facing one-step recommendation API with no ground-truth argument."""
from __future__ import annotations
import numpy as np
from synthetic3d.v2.core import Grid
from .agent import CandidateDQNAgent
from .features import build_state


def recommend(reconstructor, agent: CandidateDQNAgent, observed_ic: np.ndarray,
              observed_mask: np.ndarray, valid_mask: np.ndarray, grid: Grid,
              access_mask: np.ndarray | None = None, *, top_k: int = 0) -> dict:
    if top_k < 0:
        raise ValueError('top_k must be nonnegative')
    reconstruction, seconds = reconstructor.predict(observed_ic, observed_mask, valid_mask, grid)
    state = build_state(reconstruction, observed_mask, valid_mask, grid, access_mask)
    if len(state.candidate_xy) == 0:
        return {'status': 'no_candidate', 'reconstruction_seconds': seconds, 'ranked_candidates': []}
    scores = agent.scores(state)
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), int(state.candidate_xy[i, 0]), int(state.candidate_xy[i, 1])))
    def row(index):
        x, y = map(int, state.candidate_xy[index]);xm, ym = grid.xy_m(x, y)
        return {'x_index': x, 'y_index': y, 'x_m': float(xm), 'y_m': float(ym),
                'q_score': float(scores[index]),
                'nearest_cpt_distance_fraction': float(state.features[index, 2])}
    return {'status': 'ok', **row(order[0]), 'reconstruction_seconds': seconds,
            'ranked_candidates': [row(i) for i in order[:top_k]]}
