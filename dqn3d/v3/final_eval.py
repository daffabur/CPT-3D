"""Frozen one-step final evaluation; never used for model or route selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from synthetic3d.v2.core import Grid
from schemagan3d.v3_adapter import _safe, load_manifest
from .adapter import FrozenV3Reconstructor
from .agent import CandidateDQNAgent
from .baselines import uniform_action, random_action
from .features import build_state
from .hybrid import DQN_COUNTS, UNIFORM_COUNTS
from .runner import _initial, _paired

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('test_id', 'test_family', 'test_geometry_parameter', 'test_response_parameter')
COUNTS = (1, 2, 3, 5, 8, 12, 20)
POLICIES = ('hybrid', 'uniform', 'random')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(entries: list[dict]) -> dict:
    by_split = {}
    ids = set()
    lineages = {}
    for e in entries:
        split, rid, lineage = e['split'], e['realization_id'], e['lineage_group']
        if rid in ids:
            raise ValueError('duplicate realization ID')
        ids.add(rid)
        if lineage in lineages and lineages[lineage] != split:
            raise ValueError('lineage crosses splits')
        lineages[lineage] = split
        by_split.setdefault(split, []).append(e)
    for split in SPLITS:
        if len(by_split.get(split, [])) != 1000:
            raise ValueError(f'{split} does not have 1000 manifest entries')
    if len(by_split.get('train', [])) != 16000 or len(by_split.get('validation_id', [])) != 4000:
        raise ValueError('development split count mismatch')
    return by_split


def common_hidden_mask(valid: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    if set(masks) != set(POLICIES):
        raise ValueError('three policy masks required')
    common = np.asarray(valid, dtype=bool) & ~np.logical_or.reduce(list(masks.values()))
    if not common.any():
        raise ValueError('empty common hidden-valid mask')
    return common


def summarize(rows: list[dict], seed: int) -> dict:
    grouped = {}
    for row in rows:
        key = (row['realization_id'], row['initial_cpt_count'])
        if row['policy'] in grouped.setdefault(key, {}):
            raise ValueError('duplicate policy row')
        grouped[key][row['policy']] = row
    if any(set(p) != set(POLICIES) for p in grouped.values()):
        raise ValueError('incomplete policy triplet')
    out = {'n_cases': len(grouped), 'n_volumes': len({k[0] for k in grouped}), 'by_policy': {}, 'paired_hybrid_minus': {}}
    cases = list(grouped.values())
    for policy in POLICIES:
        pr = [c[policy] for c in cases]
        out['by_policy'][policy] = {
            'mean_hidden_valid_ic_rmse_after': float(np.mean([r['rmse_after_common'] for r in pr])),
            'mean_rmse_improvement': float(np.mean([r['rmse_improvement_common'] for r in pr])),
            'valid_action_rate': float(np.mean([r['valid_action'] for r in pr])),
            'duplicate_action_rate': float(np.mean([r['duplicate_action'] for r in pr])),
            'no_candidate_rate': float(np.mean([r['no_candidate'] for r in pr])),
        }
    for baseline in ('uniform', 'random'):
        volume_deltas = {}
        for key, p in grouped.items():
            volume_deltas.setdefault(key[0], []).append(p['hybrid']['rmse_after_common']-p[baseline]['rmse_after_common'])
        delta = np.asarray([np.mean(v) for _, v in sorted(volume_deltas.items())])
        out['paired_hybrid_minus'][baseline] = {**_paired(delta, seed=seed),
            'hybrid_win_rate': float(np.mean(delta < 0)), 'bootstrap_unit': 'realization_id'}
    return out


def evaluate_split(split: str, entries: list[dict], dataset: Path, recon, agent,
                   output: Path, *, seed: int, protocol_sha: str) -> list[dict]:
    """Rows are written incrementally; truth is held only inside this scoring function."""
    path = output / f'{split}_per_case.csv'
    if path.exists():
        raise FileExistsError(path)
    rows = []
    with path.open('w', newline='') as f:
        writer = None
        for idx, entry in enumerate(entries):
            truth = np.load(_safe(dataset, entry['arrays']['Ic']), allow_pickle=False).astype(np.float32)
            valid = np.load(_safe(dataset, entry['arrays']['valid_mask']), allow_pickle=False).astype(bool)
            grid = Grid(**json.loads(_safe(dataset, entry['metadata']).read_text())['grid'])
            if truth.shape != valid.shape or truth.shape != (grid.nx, grid.ny, grid.nz):
                raise ValueError('case shape mismatch')
            for count in COUNTS:
                initial = _initial(grid, valid, entry, count, seed)
                observed = np.zeros_like(valid)
                observed_ic = np.zeros_like(truth)
                for x, y in initial:
                    observed[x, y] = valid[x, y]
                    observed_ic[x, y, valid[x, y]] = truth[x, y, valid[x, y]]
                before, _ = recon.predict(observed_ic, observed, valid, grid)
                state = build_state(before, observed, valid, grid)
                if len(state.candidate_xy) == 0:
                    raise ValueError('no legal action in final evaluation')
                token = int.from_bytes(hashlib.sha256(f'{seed}|{entry["realization_id"]}|{count}|random'.encode()).digest()[:8], 'big')
                selected = {}
                for policy in POLICIES:
                    if policy == 'hybrid' and count in DQN_COUNTS:
                        action = agent.act(state, 0.0)
                    elif policy == 'random':
                        action = random_action(state, np.random.default_rng(token))
                    else:
                        action = uniform_action(state, observed, grid)
                    selected[policy] = tuple(map(int, state.candidate_xy[action]))
                predictions = {}
                masks = {}
                for policy, (x, y) in selected.items():
                    mask = observed.copy()
                    obs = observed_ic.copy()
                    mask[x, y] = valid[x, y]
                    obs[x, y, valid[x, y]] = truth[x, y, valid[x, y]]
                    masks[policy] = mask
                    predictions[policy], _ = recon.predict(obs, mask, valid, grid)
                common = common_hidden_mask(valid, masks)
                before_rmse = float(np.sqrt(np.mean((before[common]-truth[common])**2)))
                for policy in POLICIES:
                    x, y = selected[policy]
                    after_rmse = float(np.sqrt(np.mean((predictions[policy][common]-truth[common])**2)))
                    xm, ym = grid.xy_m(x, y)
                    row = {'split':split, 'realization_id':entry['realization_id'], 'grid_id':entry['grid_id'],
                           'family':entry['family'], 'initial_cpt_count':count,
                           'initial_xy':json.dumps(initial), 'policy':policy,
                           'policy_used':'scaled_dqn' if policy=='hybrid' and count in DQN_COUNTS else ('uniform' if policy=='hybrid' else policy),
                           'x_index':x, 'y_index':y, 'x_m':xm, 'y_m':ym,
                           'candidate_count':len(state.candidate_xy), 'common_hidden_voxels':int(common.sum()),
                           'rmse_before_common':before_rmse, 'rmse_after_common':after_rmse,
                           'rmse_improvement_common':before_rmse-after_rmse,
                           'valid_action':True, 'duplicate_action':False, 'no_candidate':False,
                           'protocol_sha256':protocol_sha}
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row));writer.writeheader()
                    writer.writerow(row)
                    rows.append(row)
            if (idx+1) % 50 == 0:
                f.flush()
                print(f'{split}: {idx+1}/{len(entries)}', flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    args = parser.parse_args()
    out, protocol_path = args.output, args.protocol
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = sha(protocol_path)
    if protocol['schema_version'] != 'dqn-v3-frozen-final-evaluation-1':
        raise ValueError('unexpected protocol')
    if tuple(protocol['test_splits']) != SPLITS or tuple(protocol['initial_cpt_counts']) != COUNTS:
        raise ValueError('protocol splits/CPT count mismatch')
    dataset = ROOT / protocol['dataset']['path']
    manifest = dataset/'manifest.json'
    if sha(manifest) != protocol['dataset']['manifest_sha256']:
        raise ValueError('manifest changed')
    by_split = verify_manifest(load_manifest(dataset))
    sg_path = ROOT/protocol['checkpoints']['schemagan']['path']
    dq_path = ROOT/protocol['checkpoints']['dqn']['path']
    schema = ROOT/protocol['feature_schema']['path']
    for p, expected in ((sg_path,protocol['checkpoints']['schemagan']['sha256']),
                        (dq_path,protocol['checkpoints']['dqn']['sha256']),
                        (schema,protocol['feature_schema']['sha256'])):
        if sha(p) != expected:
            raise ValueError(f'identity mismatch: {p}')
    recon = FrozenV3Reconstructor(sg_path, ROOT/protocol['checkpoints']['schemagan']['config'], dataset)
    agent, meta = CandidateDQNAgent.load(dq_path)
    if meta.get('schemagan_sha256') != sha(sg_path) or meta.get('manifest_sha256') != sha(manifest) or meta.get('feature_schema_sha256') != sha(schema):
        raise ValueError('DQN checkpoint identity mismatch')
    agent.online.eval()
    seed = protocol['sampling']['seed']
    summaries = {}
    by_cpt = {}
    intervals = {}
    for split in SPLITS:
        rows = evaluate_split(split, by_split[split], dataset, recon, agent, out, seed=seed, protocol_sha=protocol_sha)
        summaries[split] = {'overall':summarize(rows, seed),
            'by_grid':{g:summarize([r for r in rows if r['grid_id']==g],seed) for g in sorted({r['grid_id'] for r in rows})},
            'by_family':{g:summarize([r for r in rows if r['family']==g],seed) for g in sorted({r['family'] for r in rows})}}
        by_cpt[split] = {str(c):summarize([r for r in rows if r['initial_cpt_count']==c],seed) for c in COUNTS}
        intervals[split] = {'overall':summaries[split]['overall']['paired_hybrid_minus'],
                            'by_cpt':{c:v['paired_hybrid_minus'] for c,v in by_cpt[split].items()}}
        (out/'final_evaluation_by_split.json').write_text(json.dumps({'protocol_sha256':protocol_sha,'splits':summaries},indent=2)+'\n')
        (out/'final_evaluation_by_cpt.json').write_text(json.dumps({'protocol_sha256':protocol_sha,'splits':by_cpt},indent=2)+'\n')
        (out/'final_paired_confidence_intervals.json').write_text(json.dumps({'protocol_sha256':protocol_sha,'splits':intervals},indent=2)+'\n')
    with (out/'final_policy_comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(('split','initial_cpt_count','policy','n_cases','n_volumes','mean_hidden_valid_ic_rmse_after','mean_rmse_improvement','paired_hybrid_minus_uniform','paired_hybrid_minus_random'))
        for split in SPLITS:
            for c, group in by_cpt[split].items():
                for policy, metrics in group['by_policy'].items():
                    writer.writerow((split,c,policy,group['n_cases'],group['n_volumes'],metrics['mean_hidden_valid_ic_rmse_after'],metrics['mean_rmse_improvement'],group['paired_hybrid_minus']['uniform']['mean'],group['paired_hybrid_minus']['random']['mean']))

if __name__ == '__main__':
    main()
