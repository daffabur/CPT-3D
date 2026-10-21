"""Random and physical-distance Uniform policies on the same legal XY candidates."""
from __future__ import annotations
import numpy as np
from synthetic3d.v2.core import Grid
from .features import CandidateState


def random_action(state: CandidateState, rng: np.random.Generator) -> int:
    if len(state.candidate_xy) == 0:
        raise ValueError('no legal candidate')
    return int(rng.integers(len(state.candidate_xy)))


def uniform_action(state: CandidateState, observed_mask: np.ndarray, grid: Grid) -> int:
    if len(state.candidate_xy) == 0:
        raise ValueError('no legal candidate')
    sites = np.argwhere(np.asarray(observed_mask, dtype=bool).any(axis=2))
    if len(sites) == 0:
        raise ValueError('no observed CPT')
    site_m = np.array([grid.xy_m(int(x), int(y)) for x, y in sites])
    candidate_m = np.array([grid.xy_m(int(x), int(y)) for x, y in state.candidate_xy])
    nearest = np.linalg.norm(candidate_m[:, None, :] - site_m[None, :, :], axis=2).min(axis=1)
    # Candidates are lexicographic; np.argmax chooses first equal-distance candidate.
    return int(np.argmax(nearest))
