from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .baseline import horizontal_idw
from .data import SoilVolumeDataset, compute_normalization, make_sparse_observation, sorted_csv_paths
from .losses import discriminator_loss, generator_loss
from .models import Discriminator3D, Generator3D


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "synthetic3d"


def test_sparse_observation_reveals_complete_profiles_only():
    target = torch.arange(2048, dtype=torch.float32).reshape(1, 8, 16, 16)
    sparse, mask, positions = make_sparse_observation(target, profile_count=5, seed=7)
    assert positions.shape == (5, 2)
    assert int(mask.sum()) == 5 * 8
    assert torch.equal(sparse[mask.bool()], target[mask.bool()])
    assert torch.count_nonzero(sparse[~mask.bool()]) == 0


def test_loader_model_losses_and_idw():
    paths = sorted_csv_paths(DATA_ROOT / "train", limit=2)
    if len(paths) < 2:
        raise RuntimeError("full synthetic3d training data is required")
    stats = compute_normalization(paths)
    dataset = SoilVolumeDataset(
        DATA_ROOT / "train", stats=stats, limit=2, min_profiles=3, max_profiles=6
    )
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    generator = Generator3D(base_channels=4)
    discriminator = Discriminator3D(base_channels=4)
    prediction = generator(batch["sparse"], batch["mask"])
    assert prediction.shape == (2, 1, 8, 16, 16)
    assert torch.equal(prediction[batch["mask"].bool()], batch["sparse"][batch["mask"].bool()])
    real_logits = discriminator(batch["target"], batch["sparse"], batch["mask"])
    fake_logits = discriminator(prediction, batch["sparse"], batch["mask"])
    d_loss = discriminator_loss(real_logits, fake_logits)
    g_loss, adversarial, reconstruction = generator_loss(
        fake_logits, prediction, batch["target"], batch["mask"]
    )
    assert all(torch.isfinite(value) for value in (d_loss, g_loss, adversarial, reconstruction))
    idw = horizontal_idw(batch["sparse"][0], batch["mask"][0])
    assert idw.shape == (1, 8, 16, 16)
    assert torch.equal(idw[batch["mask"][0].bool()], batch["sparse"][0][batch["mask"][0].bool()])
