"""Synthetic-only CPT reveal and fixed-common-mask reward."""
from __future__ import annotations
import math
import numpy as np
from synthetic3d.v2.core import Grid
from .features import CandidateState, build_state


class SyntheticCPTSimulator:
    def __init__(self, reconstructor, truth_ic: np.ndarray, valid_mask: np.ndarray,
                 grid: Grid, access_mask: np.ndarray | None = None):
        shape = (grid.nx, grid.ny, grid.nz)
        self._truth = np.asarray(truth_ic, dtype=np.float32).copy()
        self.valid = np.asarray(valid_mask, dtype=bool).copy()
        if self._truth.shape != shape or self.valid.shape != shape or not self.valid.any() or not np.isfinite(self._truth[self.valid]).all():
            raise ValueError('invalid synthetic truth/valid mask/grid')
        self.grid = grid
        self.reconstructor = reconstructor
        self.access_mask = access_mask
        self.observed_ic = np.zeros(shape, dtype=np.float32)
        self.observed_mask = np.zeros(shape, dtype=bool)
        self.prediction = None
        self.state: CandidateState | None = None
        self.reconstruction_seconds = 0.0

    def _reveal(self, xy: tuple[int, int]):
        x, y = map(int, xy)
        self.grid.xy_m(x, y)  # validates bounds
        if not self.valid[x, y].any() or self.observed_mask[x, y].any():
            raise ValueError('invalid or duplicate CPT site')
        # Existing measured CPTs may predate the current candidate access area.
        # New recommendations are restricted by build_state's legal mask.
        self.observed_mask[x, y] = self.valid[x, y]
        self.observed_ic[x, y, self.valid[x, y]] = self._truth[x, y, self.valid[x, y]]

    def _reconstruct(self):
        self.prediction, self.reconstruction_seconds = self.reconstructor.predict(
            self.observed_ic, self.observed_mask, self.valid, self.grid)
        self.state = build_state(self.prediction, self.observed_mask, self.valid, self.grid, self.access_mask)

    def reset(self, initial_xy: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> CandidateState:
        if not initial_xy:
            raise ValueError('at least one initial CPT required')
        self.observed_ic.fill(0)
        self.observed_mask.fill(False)
        for xy in initial_xy:
            self._reveal(xy)
        self._reconstruct()
        return self.state

    def step(self, action_index: int) -> tuple[CandidateState, float, bool, dict]:
        if self.state is None:
            raise RuntimeError('call reset first')
        if not 0 <= action_index < len(self.state.candidate_xy):
            raise ValueError('illegal candidate action index')
        x, y = map(int, self.state.candidate_xy[action_index])
        before_prediction = self.prediction.copy()
        self._reveal((x, y))
        self._reconstruct()
        common = self.valid & ~self.observed_mask
        if common.any():
            before_rmse = float(np.sqrt(np.mean((before_prediction[common] - self._truth[common]) ** 2)))
            after_rmse = float(np.sqrt(np.mean((self.prediction[common] - self._truth[common]) ** 2)))
            reward = before_rmse - after_rmse
        else:
            before_rmse = after_rmse = reward = 0.0
        done = not common.any() or len(self.state.candidate_xy) == 0
        info = {'selected_xy': (x, y), 'x_m': self.grid.xy_m(x, y)[0], 'y_m': self.grid.xy_m(x, y)[1],
                'common_hidden_voxels': int(common.sum()), 'rmse_before_common': before_rmse,
                'rmse_after_common': after_rmse, 'rmse_improvement': reward,
                'reconstruction_seconds': self.reconstruction_seconds}
        assert math.isfinite(reward)
        return self.state, reward, done, info
