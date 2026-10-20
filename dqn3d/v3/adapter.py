"""Frozen SchemaGAN V3 reconstruction from observed CPT profiles only."""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from synthetic3d.v2.core import Grid
from schemagan3d.train_v3 import _models
from schemagan3d.v3_adapter import V3Normalization, assert_model_shape, restore_observed, xyz_to_czyx
from schemagan3d.v3_dataset_production import digest
from schemagan3d.v3_idw_guided import idw_normalized_from_observed

EXPECTED_SHA = '5694bb7b43efd7054a99fac8b74dadc5d74ebe6d291f42a8393b57fe9da52b64'
EXPECTED_MEAN = 2.314278231778639
EXPECTED_STD = 0.27927494076233944


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenV3Reconstructor:
    def __init__(self, checkpoint: Path, config_path: Path, dataset_root: Path, *, device: str = 'cpu'):
        if device != 'cpu':
            raise ValueError('V3 DQN adapter uses CPU until MPS ConvTranspose3D support is verified')
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = _sha(self.checkpoint)
        if self.checkpoint_sha256 != EXPECTED_SHA:
            raise ValueError('unexpected SchemaGAN V3 checkpoint SHA-256')
        self.config = json.loads(Path(config_path).read_text())
        # This trusted, SHA-verified checkpoint was written with NumPy 2. Its
        # pickle module path needs an alias in the repository's NumPy 1.23 venv.
        if not hasattr(np, '_core'):
            sys.modules.setdefault('numpy._core', np.core)
            sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
            sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
        saved = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        if saved.get('epoch') != 26 or saved.get('model_variant') != 'idw_guided_residual' or saved.get('objective_variant') != 'mse_only':
            raise ValueError('unexpected checkpoint epoch/model/objective')
        if saved.get('configuration_sha256') != digest(self.config):
            raise ValueError('checkpoint configuration hash mismatch')
        identity = saved.get('dataset_identity', {})
        if _sha(Path(dataset_root) / 'manifest.json') != identity.get('manifest_file_sha256'):
            raise ValueError('dataset manifest identity mismatch')
        if identity.get('plan_sha256') != 'ff2884e0370e9597cb3afa3f8c0b613f19670835fc96ff2dc26cdebf0990d663':
            raise ValueError('unexpected dataset plan identity')
        norm = saved.get('normalization', {})
        if norm.get('source_split') != 'train' or norm.get('mean') != EXPECTED_MEAN or norm.get('std') != EXPECTED_STD:
            raise ValueError('train-only normalization mismatch')
        if identity.get('normalization', {}).get('mean') != EXPECTED_MEAN or identity.get('normalization', {}).get('std') != EXPECTED_STD:
            raise ValueError('dataset normalization mismatch')
        if (identity.get('normalization', {}).get('source_split') != 'train' or
                identity.get('normalization', {}).get('source_train_ids_sha256') != identity.get('train_ids_sha256')):
            raise ValueError('train-only normalization provenance mismatch')
        self.stats = V3Normalization(EXPECTED_MEAN, EXPECTED_STD)
        self.model, _ = _models(self.config)
        self.model.load_state_dict(saved['generator_state'], strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = torch.device('cpu')
        self.dataset_identity = identity

    def predict(self, observed_ic: np.ndarray, observed_mask: np.ndarray, valid_mask: np.ndarray,
                grid: Grid) -> tuple[np.ndarray, float]:
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
            _, prediction = self.model(sparse, mask_tensor, idw)
            physical = restore_observed(prediction, mask_tensor, self.stats, observed_tensor)
        seconds = time.perf_counter() - started
        result = physical[0, 0].numpy().transpose(2, 1, 0).copy()
        if not np.isfinite(result[valid]).all() or not np.array_equal(result[mask], obs[mask]):
            raise ValueError('nonfinite reconstruction or observed value changed')
        return result, seconds
