"""Portable, CPU-only inference entry point for frozen SchemaGAN V3 and DQN V3."""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CPTModel:
    def __init__(self, root, grid_type, stats, generator, q_model):
        self.root = root
        self.Grid = grid_type
        self.stats = stats
        self.generator = generator
        self.q_model = q_model

    @classmethod
    def load(cls, bundle_path):
        root = Path(bundle_path).resolve()
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest.get('schema_version') != 'cpt-ml-bundle-1':
            raise ValueError('unsupported bundle version')
        for relative, expected in manifest['files'].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or _sha(path) != expected['sha256']:
                raise ValueError(f'bundle file missing or modified: {relative}')
        runtime = str(root / 'runtime')
        if runtime not in sys.path:
            sys.path.insert(0, runtime)
        from synthetic3d.v2.core import Grid
        from schemagan3d.v3_adapter import V3Normalization
        from schemagan3d.v3_idw_guided import IDWResidualGenerator3D
        from dqn3d.v3.q_scorer import SharedCandidateQ
        from dqn3d.v3.features import CANDIDATE_NAMES, GLOBAL_NAMES
        from dqn3d.v3.hybrid import DQN_COUNTS, UNIFORM_COUNTS
        model_cfg = json.loads((root / 'configs/model.json').read_text())
        norm = json.loads((root / 'configs/normalization.json').read_text())
        schema = json.loads((root / 'configs/feature_schema.json').read_text())
        policy = json.loads((root / 'configs/hybrid_policy.json').read_text())
        if norm.get('source_split') != 'train':
            raise ValueError('normalization is not train-only')
        if tuple(schema['candidate_names_ordered']) != tuple(CANDIDATE_NAMES) or tuple(schema['global_names_ordered']) != tuple(GLOBAL_NAMES):
            raise ValueError('feature schema mismatch')
        routes = {item['policy']: set(item['unique_observed_xy_cpt_counts']) for item in policy['routes']}
        if routes != {'scaled_dqn': set(DQN_COUNTS), 'uniform_spatial_coverage': set(UNIFORM_COUNTS)}:
            raise ValueError('hybrid policy mismatch')
        stats = V3Normalization(float(norm['mean']), float(norm['std']))
        generator = IDWResidualGenerator3D(base_channels=model_cfg['schemagan']['base_channels'])
        q_model = SharedCandidateQ(width=model_cfg['dqn']['hidden_width'])
        sg_saved = torch.load(root / 'models/schemagan_v3.pt', map_location='cpu', weights_only=True)
        q_saved = torch.load(root / 'models/dqn_v3.pt', map_location='cpu', weights_only=True)
        generator.load_state_dict(sg_saved['state_dict'], strict=True)
        q_model.load_state_dict(q_saved['state_dict'], strict=True)
        generator.eval().requires_grad_(False)
        q_model.eval().requires_grad_(False)
        return cls(root, Grid, stats, generator, q_model)

    def _reconstruct(self, observed_ic, observed_mask, valid_mask, grid):
        from schemagan3d.v3_adapter import assert_model_shape, restore_observed, xyz_to_czyx
        from schemagan3d.v3_idw_guided import idw_normalized_from_observed
        shape = (grid.nx, grid.ny, grid.nz)
        assert_model_shape(shape)
        obs = np.asarray(observed_ic, dtype=np.float32)
        mask = np.asarray(observed_mask, dtype=bool)
        valid = np.asarray(valid_mask, dtype=bool)
        if obs.shape != shape or mask.shape != shape or valid.shape != shape:
            raise ValueError('observation, mask, valid mask and grid shapes differ')
        if not mask.any() or np.any(mask & ~valid) or not np.isfinite(obs[mask]).all():
            raise ValueError('invalid observed CPT values/mask')
        sites = mask.any(axis=2)
        if not np.array_equal(mask, sites[:, :, None] & valid):
            raise ValueError('each observed CPT must include every valid depth at its XY site')
        sparse_physical = np.where(mask, obs, 0.0).astype(np.float32)
        normalized = np.zeros(shape, dtype=np.float32)
        normalized[mask] = self.stats.normalize(sparse_physical[mask])
        sparse = torch.from_numpy(xyz_to_czyx(normalized))[None]
        mask_tensor = torch.from_numpy(xyz_to_czyx(mask.astype(np.float32)))[None]
        observed_tensor = torch.from_numpy(xyz_to_czyx(sparse_physical))[None]
        started = time.perf_counter()
        with torch.inference_mode():
            idw = idw_normalized_from_observed(observed_tensor[0], mask_tensor[0], grid, self.stats)[None]
            _, prediction = self.generator(sparse, mask_tensor, idw)
            physical = restore_observed(prediction, mask_tensor, self.stats, observed_tensor)
        seconds = time.perf_counter() - started
        result = physical[0, 0].numpy().transpose(2, 1, 0).copy()
        if not np.isfinite(result[valid]).all() or not np.array_equal(result[mask], obs[mask]):
            raise ValueError('nonfinite reconstruction or observed value changed')
        return result, seconds

    def predict(self, grid, cpt_data, valid_mask=None, access_mask=None):
        """Return physical XYZ Ic reconstruction and one legal next CPT recommendation."""
        from dqn3d.v3.hybrid import recommend_next_cpt
        if isinstance(grid, dict):
            grid = self.Grid(**grid)
        if not isinstance(grid, self.Grid):
            raise TypeError('grid must be Grid or a Grid field dictionary')
        if not isinstance(cpt_data, dict) or set(cpt_data) != {'observed_ic', 'observed_mask'}:
            raise ValueError('cpt_data requires observed_ic and observed_mask')
        shape = (grid.nx, grid.ny, grid.nz)
        observed_ic = np.asarray(cpt_data['observed_ic'], dtype=np.float32)
        observed_mask = np.asarray(cpt_data['observed_mask'], dtype=bool)
        valid = np.ones(shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if access_mask is not None:
            access_mask = np.asarray(access_mask, dtype=bool)
        reconstruction, seconds = self._reconstruct(observed_ic, observed_mask, valid, grid)
        class CachedReconstructor:
            def predict(self, *args):
                return reconstruction, seconds
        class Scorer:
            def __init__(self, model):
                self.model = model
            def scores(self, state):
                with torch.inference_mode():
                    return self.model(torch.from_numpy(state.features.astype(np.float32)),
                                      torch.from_numpy(state.global_context.astype(np.float32))).numpy().copy()
        recommendation = recommend_next_cpt(CachedReconstructor(), Scorer(self.q_model),
                                             observed_ic, observed_mask, valid, grid, access_mask)
        return {'ic_3d': reconstruction, 'recommended_xy': None if recommendation['x_m'] is None else
                (recommendation['x_m'], recommendation['y_m']),
                'recommended_index': None if recommendation['x_index'] is None else
                (recommendation['x_index'], recommendation['y_index']),
                **recommendation}
