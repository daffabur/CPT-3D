"""Read-only random-weight smoke test for the experimental IDW residual variant."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .losses import discriminator_loss
from .models import Generator3D
from .v3_adapter import V3Normalization, V3VolumeDataset, load_manifest, masked_hidden_metrics, restore_observed
from .v3_adapter import HomogeneousShapeBatchSampler
from .v3_eval_protocol import PROFILE_COUNTS
from .v3_idw_guided import IDWGuidedVolumeDataset, make_models, residual_generator_objective


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    if config['mode'] != 'planned_not_executed' or config['idw_power'] != 2.:
        raise ValueError('only the frozen dry-run plan is supported')
    if (config['train_split'], config['validation_split']) != ('train', 'validation_id'):
        raise ValueError('test and holdout splits are forbidden in dry run')
    root = Path(config['dataset_root'])
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['plan_sha256'] != config['expected_plan_sha256']:
        raise ValueError('frozen plan mismatch')
    entries = load_manifest(root)
    train = [e for e in entries if e['split'] == 'train']
    validation = [e for e in entries if e['split'] == 'validation_id']
    if len(train) != 16000 or len(validation) != 4000:
        raise ValueError('unexpected source split size')
    train_ids = tuple(sorted(e['realization_id'] for e in train))
    metadata = json.loads(Path(config['normalization_metadata']).read_text())
    ids_sha = hashlib.sha256(json.dumps(train_ids, separators=(',', ':')).encode()).hexdigest()
    if (metadata['plan_sha256'] != manifest['plan_sha256'] or metadata['source_split'] != 'train'
            or metadata['source_volume_count'] != len(train_ids) or metadata['source_train_ids_sha256'] != ids_sha):
        raise ValueError('train-only normalization provenance mismatch')
    stats = V3Normalization(metadata['mean'], metadata['std'], source_train_ids=train_ids)
    subset = json.loads(Path(config['pilot3_subset_ids_path']).read_text())
    selected = subset['ids']
    if (subset['selection_seed'] != config['selection_seed'] or
            len(selected['train']) != config['training_total'] or
            len(selected['validation_id']) != config['validation_total']):
        raise ValueError('Pilot 3 subset policy mismatch')
    for key, source in (('train', train), ('validation_id', validation)):
        ids = selected[key]
        if not set(ids).issubset({e['realization_id'] for e in source}):
            raise ValueError('Pilot 3 subset crosses splits')
        if hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() != subset['sha256'][key]:
            raise ValueError('Pilot 3 subset ID hash mismatch')
    indices = {
        'train': next(i for i, e in enumerate(train) if e['realization_id'] == selected['train'][0]),
        'validation_id': next(i for i, e in enumerate(validation) if e['realization_id'] == selected['validation_id'][0]),
    }
    count = config['primary_evaluation_cpt_count']
    datasets = {split: IDWGuidedVolumeDataset(V3VolumeDataset(root, split=split, stats=stats,
                                  indices=[indices[split]], evaluation_cpt_count=count))
                for split in ('train', 'validation_id')}
    batches = {split: next(iter(DataLoader(dataset, batch_size=1))) for split, dataset in datasets.items()}
    torch.set_num_threads(1)
    torch.manual_seed(config['random_seed'])
    generator, discriminator = make_models(config['base_channels'])
    generator.eval(); discriminator.eval()
    before_g = [p.detach().clone() for p in generator.parameters()]
    before_d = [p.detach().clone() for p in discriminator.parameters()]
    batch = batches['train']
    sparse, mask, idw, target, valid = (batch[k] for k in ('sparse', 'mask', 'idw', 'target', 'valid_mask'))
    with torch.no_grad():
        _, detached = generator(sparse, mask, idw)
    real_logits = discriminator(target, sparse, mask)
    fake_logits = discriminator(detached, sparse, mask)
    d_loss = discriminator_loss(real_logits, fake_logits)
    d_loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in discriminator.parameters()):
        raise ValueError('nonfinite discriminator gradient')
    discriminator.zero_grad(set_to_none=True)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    residual, prediction = generator(sparse, mask, idw)
    g_loss, adv, rec = residual_generator_objective(discriminator(prediction, sparse, mask), residual,
                       target, idw, mask, valid, reconstruction_weight=config['reconstruction_weight'])
    g_loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in generator.parameters()):
        raise ValueError('nonfinite generator gradient')
    if not all(torch.isfinite(x) for x in (d_loss, g_loss, adv, rec)):
        raise ValueError('nonfinite loss')
    if not all(torch.equal(a, b) for a, b in zip(before_g, generator.parameters())):
        raise ValueError('generator weights changed')
    if not all(torch.equal(a, b) for a, b in zip(before_d, discriminator.parameters())):
        raise ValueError('discriminator weights changed')
    physical = restore_observed(prediction.detach(), mask, stats, batch['observed_ic_physical'])
    observed_exact = torch.equal(physical[mask.bool()], batch['observed_ic_physical'][mask.bool()])
    if not observed_exact:
        raise ValueError('observed restoration failed')
    hidden = masked_hidden_metrics(physical, stats.denormalize(target), mask, valid)
    with torch.inference_mode():
        val = batches['validation_id']
        val_residual, val_prediction = generator(val['sparse'], val['mask'], val['idw'])
        if not torch.isfinite(val_prediction).all():
            raise ValueError('nonfinite validation forward pass')
    # Real validation-ID smoke across all three grids and seven nested CPT counts.
    val_by_id = {e['realization_id']: (i, e) for i, e in enumerate(validation)}
    selected_by_grid = {name: [] for name in ('g12', 'g16', 'g24')}
    for rid in selected['validation_id']:
        i, entry = val_by_id[rid]
        group = selected_by_grid.get(entry['grid_name'])
        if group is not None and len(group) < len(PROFILE_COUNTS):
            group.append(i)
    if any(len(group) != len(PROFILE_COUNTS) for group in selected_by_grid.values()):
        raise ValueError('Pilot 3 validation subset lacks grid/CPT smoke coverage')
    smoke_indices = [i for group in selected_by_grid.values() for i in group]
    smoke_counts = list(PROFILE_COUNTS) * len(selected_by_grid)
    smoke_base = V3VolumeDataset(root, split='validation_id', stats=stats,
                                 indices=smoke_indices, cpt_counts_by_index=smoke_counts)
    smoke = IDWGuidedVolumeDataset(smoke_base)
    observed_counts = []
    for i in range(len(smoke)):
        item = smoke[i]
        count_found = int(item['mask'][0].bool().any(dim=0).sum())
        if count_found != smoke_counts[i] or not torch.isfinite(item['idw']).all():
            raise ValueError('real validation CPT/grid conditioning failed')
        observed_counts.append(count_found)
    grouped_batch_shapes = [list(x['idw'].shape) for x in DataLoader(
        smoke, batch_sampler=HomogeneousShapeBatchSampler(smoke, config['batch_size'], group_spacing=True))]
    if any(shape[0] > config['batch_size'] or shape[1] != 1 for shape in grouped_batch_shapes):
        raise ValueError('homogeneous batching failed')
    # Verify legacy Pilot 3 checkpoint still loads into the unchanged two-channel class.
    checkpoint = Path('results/schemagan3d_v3_pilot3_scaled_20e/best.pt')
    checkpoint_sha_before = file_sha(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    baseline = Generator3D(config['base_channels']).eval()
    baseline.load_state_dict(saved['generator_state'])
    with torch.inference_mode():
        baseline_output = baseline(sparse, mask)
    if not torch.isfinite(baseline_output).all() or file_sha(checkpoint) != checkpoint_sha_before:
        raise ValueError('baseline checkpoint compatibility/integrity failed')
    return {
        'mode': 'random_weight_dry_run_only', 'optimizer_created': False, 'optimizer_step': False,
        'checkpoint_written': False, 'weights_unchanged': True,
        'random_weight_loss_not_accuracy': True, 'dataset_plan_sha256': manifest['plan_sha256'],
        'normalization_train_only': {'mean': stats.mean, 'std': stats.std, 'train_ids_sha256': ids_sha},
        'pilot3_subset_sha256': {k: subset['sha256'][k] for k in ('train', 'validation_id')},
        'used_splits': ['train', 'validation_id'], 'evaluation_cpt_count': count,
        'train_realization_id': selected['train'][0], 'validation_realization_id': selected['validation_id'][0],
        'input_shapes': {k: list(batch[k].shape) for k in ('sparse', 'mask', 'idw')},
        'residual_shape': list(residual.shape), 'reconstruction_shape': list(prediction.shape),
        'validation_residual_shape': list(val_residual.shape), 'validation_reconstruction_shape': list(val_prediction.shape),
        'generator_loss_random': float(g_loss.detach()), 'discriminator_loss_random': float(d_loss.detach()),
        'adversarial_loss_random': float(adv.detach()), 'residual_l1_normalized_random': float(rec.detach()),
        'generator_gradients_finite': True, 'discriminator_gradients_finite': True,
        'observed_values_exact': observed_exact, 'hidden_metrics_random_do_not_interpret': hidden,
        'baseline_pilot3_checkpoint_loads': True, 'baseline_checkpoint_sha256_unchanged': checkpoint_sha_before,
        'validation_grid_cpt_smoke_samples': len(smoke), 'validation_grid_cpt_smoke_counts': observed_counts,
        'homogeneous_batch_shapes': grouped_batch_shapes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('schemagan3d/v3_idw_guided_pilot_config.json'))
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == '__main__':
    main()
