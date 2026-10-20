"""DQN V3 train and validation-only runners; never open test/holdout arrays."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from synthetic3d.v2.core import Grid
from schemagan3d.v3_adapter import _safe, load_manifest
from schemagan3d.v3_eval_protocol import nested_sites
from .adapter import FrozenV3Reconstructor
from .agent import AgentConfig, CandidateDQNAgent
from .baselines import random_action, uniform_action
from .simulator import SyntheticCPTSimulator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'results/dqn_v3_implementation/dqn_v3_training_config.json'


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entries(root: Path, split: str, limit: int) -> list[dict]:
    if split not in ('train', 'validation_id'):
        raise ValueError('DQN V3 development runner only allows train/validation_id')
    entries = [e for e in load_manifest(root) if e['split'] == split]
    if not 1 <= limit <= len(entries):
        raise ValueError('requested split count unavailable')
    return entries[:limit]


def _case(root: Path, entry: dict):
    if entry['split'] not in ('train', 'validation_id'):
        raise ValueError('test/holdout case forbidden')
    truth = np.load(_safe(root, entry['arrays']['Ic']), allow_pickle=False).astype(np.float32)
    valid = np.load(_safe(root, entry['arrays']['valid_mask']), allow_pickle=False).astype(bool)
    grid = Grid(**json.loads(_safe(root, entry['metadata']).read_text())['grid'])
    if truth.shape != valid.shape or truth.shape != (grid.nx, grid.ny, grid.nz):
        raise ValueError('case shape mismatch')
    return truth, valid, grid


def _initial(grid: Grid, valid: np.ndarray, entry: dict, count: int, seed: int, episode: int = 0):
    # Fixed per-volume seed; episode offset varies training observations only.
    site_seed = int(entry['evaluation_site_seed']) + int(seed) + int(episode)
    sites = nested_sites(grid, site_seed, valid, count)
    if len(sites) != count:
        raise ValueError('insufficient valid initial CPT sites')
    return sites


def _paired(values: np.ndarray, seed: int = 1414, resamples: int = 2000) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2:
        return {'mean': float(x.mean()) if len(x) else None, 'ci95': None, 'n': len(x)}
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 100):
        stop = min(start + 100, resamples)
        means[start:stop] = x[rng.integers(len(x), size=(stop-start, len(x)))].mean(axis=1)
    return {'mean': float(x.mean()), 'sd': float(x.std(ddof=1)),
            'ci95': [float(np.quantile(means, .025)), float(np.quantile(means, .975))], 'n': len(x)}


def evaluate_validation(agent: CandidateDQNAgent, reconstructor: FrozenV3Reconstructor,
                        dataset_root: Path, *, volume_count: int, initial_counts: list[int], seed: int,
                        reference_agent: CandidateDQNAgent | None = None) -> tuple[list[dict], dict]:
    entries = _entries(dataset_root, 'validation_id', volume_count)
    rows = []
    policies = ('dqn', 'scaled_dqn', 'uniform', 'random') if reference_agent is not None else ('dqn', 'uniform', 'random')
    for entry in entries:
        truth, valid, grid = _case(dataset_root, entry)
        for count in initial_counts:
            initial = _initial(grid, valid, entry, count, seed)
            episodes = {}
            for policy in policies:
                sim = SyntheticCPTSimulator(reconstructor, truth, valid, grid)
                state = sim.reset(initial)
                if len(state.candidate_xy) == 0:
                    raise ValueError('no legal validation action')
                before = sim.prediction.copy()
                if policy == 'dqn':
                    action = agent.act(state, 0.0)
                elif policy == 'scaled_dqn':
                    action = reference_agent.act(state, 0.0)
                elif policy == 'uniform':
                    action = uniform_action(state, sim.observed_mask, grid)
                else:
                    token = int.from_bytes(hashlib.sha256(f'{seed}|{entry["realization_id"]}|{count}|random'.encode()).digest()[:8], 'big')
                    action = random_action(state, np.random.default_rng(token))
                selected = tuple(map(int, state.candidate_xy[action]))
                _, reward, _, info = sim.step(action)
                episodes[policy] = (before, sim.prediction.copy(), sim.observed_mask.copy(), selected, reward, info)
            # All three policies are evaluated on exactly the same still-hidden voxels.
            union_observed = np.logical_or.reduce([v[2] for v in episodes.values()])
            common = valid & ~union_observed
            if not common.any():
                raise ValueError('empty cross-policy common hidden mask')
            for policy, (before, after, mask, selected, reward, info) in episodes.items():
                b = float(np.sqrt(np.mean((before[common]-truth[common])**2)))
                a = float(np.sqrt(np.mean((after[common]-truth[common])**2)))
                rows.append({'realization_id': entry['realization_id'], 'grid_id': entry['grid_id'],
                             'family': entry['family'], 'initial_cpt_count': count,
                             'initial_xy': json.dumps(initial), 'policy': policy,
                             'x_index': selected[0], 'y_index': selected[1],
                             'x_m': grid.xy_m(*selected)[0], 'y_m': grid.xy_m(*selected)[1],
                             'common_hidden_voxels': int(common.sum()),
                             'rmse_before_common': b, 'rmse_after_common': a,
                             'rmse_improvement_common': b-a, 'own_mask_reward': reward,
                             'valid_action': True, 'duplicate_action': False})
    groups = {}
    for row in rows:
        groups.setdefault((row['realization_id'], row['initial_cpt_count']), {})[row['policy']] = row
    def summarize(selected):
        pairs = [groups[k] for k in selected]
        out = {'n': len(pairs), 'by_policy': {}, 'paired_dqn_minus': {}}
        for policy in policies:
            after = np.array([p[policy]['rmse_after_common'] for p in pairs])
            improvement = np.array([p[policy]['rmse_improvement_common'] for p in pairs])
            out['by_policy'][policy] = {'mean_after_rmse': float(after.mean()),
                                        'mean_improvement': float(improvement.mean()),
                                        'valid_action_rate': 1.0, 'duplicate_action_rate': 0.0}
        for baseline in policies[1:]:
            by_volume = {}
            for p in pairs:
                rid = p['dqn']['realization_id']
                by_volume.setdefault(rid, []).append(p['dqn']['rmse_after_common']-p[baseline]['rmse_after_common'])
            delta = np.array([np.mean(v) for _, v in sorted(by_volume.items())])
            out['paired_dqn_minus'][baseline] = {**_paired(delta), 'dqn_win_rate': float((delta < 0).mean()),
                'bootstrap_unit': 'realization_id', 'paired_case_count': len(pairs)}
        return out
    keys = list(groups)
    summary = {'split': 'validation_id', 'evaluation': 'one recommendation per policy, shared initial CPT and cross-policy common hidden mask',
               'policies': list(policies),
               'overall': summarize(keys), 'by_grid': {}, 'by_initial_cpt_count': {}}
    for grid_id in sorted({groups[k]['dqn']['grid_id'] for k in keys}):
        selected = [k for k in keys if groups[k]['dqn']['grid_id'] == grid_id]
        summary['by_grid'][grid_id] = summarize(selected)
    for count in initial_counts:
        summary['by_initial_cpt_count'][str(count)] = summarize([k for k in keys if k[1] == count])
    return rows, summary


def _write_csv(path: Path, rows: list[dict]):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)



def _validation_settings(cfg: dict) -> tuple[list[int], list[int], str]:
    training_counts = cfg['initial_cpt_counts']
    validation_counts = cfg.get('validation_initial_cpt_counts', training_counts)
    selection_counts = cfg.get('selection_cpt_counts', [])
    if (not training_counts or not validation_counts or
            any(not isinstance(x, int) or x < 1 for x in training_counts + validation_counts) or
            len(set(training_counts)) != len(training_counts) or len(set(validation_counts)) != len(validation_counts) or
            len(set(selection_counts)) != len(selection_counts) or
            any(x not in validation_counts for x in selection_counts)):
        raise ValueError('invalid training/validation/selection CPT counts')
    selection_metric = ('mean_validation_dqn_rmse_after_common_cpt_' + '_'.join(map(str, selection_counts)) + '_equal'
                        if selection_counts else 'mean_validation_dqn_rmse_after_common')
    return validation_counts, selection_counts, selection_metric


def _reference_agent(cfg: dict, recon: FrozenV3Reconstructor, manifest_sha: str,
                     feature_schema_sha: str) -> CandidateDQNAgent | None:
    name = cfg.get('reference_checkpoint')
    if name is None:
        return None
    path = ROOT / name
    expected = cfg.get('reference_checkpoint_sha256')
    if not expected or _hash(path) != expected:
        raise ValueError('reference checkpoint SHA-256 mismatch')
    reference, meta = CandidateDQNAgent.load(path)
    if (meta.get('schemagan_sha256') != recon.checkpoint_sha256 or
            meta.get('manifest_sha256') != manifest_sha or
            meta.get('feature_schema_sha256') != feature_schema_sha):
        raise ValueError('reference checkpoint identity mismatch')
    return reference


def run_training(config_path: Path, output: Path, *, resume: bool = False) -> dict:
    cfg = json.loads(Path(config_path).read_text())
    if cfg.get('schema_version') != 'dqn-v3-pilot-config-1' or cfg.get('development_splits') != ['train','validation_id']:
        raise ValueError('invalid development config/splits')
    root = ROOT / cfg['dataset_root']
    recon = FrozenV3Reconstructor(ROOT/cfg['schemagan_checkpoint'], ROOT/cfg['schemagan_config'], root)
    train_entries = _entries(root, 'train', cfg['train_volume_count'])
    _entries(root, 'validation_id', cfg['validation_volume_count'])
    validation_counts, selection_counts, selection_metric = _validation_settings(cfg)
    if cfg['episode_horizon'] < 1:
        raise ValueError('invalid CPT counts/horizon')
    agent_cfg = AgentConfig(**cfg['agent'])
    manifest_sha = _hash(root/'manifest.json')
    feature_schema_sha = _hash(ROOT/'results/dqn_v3_implementation/dqn_v3_feature_schema.json')
    reference_agent = _reference_agent(cfg, recon, manifest_sha, feature_schema_sha)
    output = Path(output)
    if resume:
        if not output.exists() or json.loads((output/'config.json').read_text()) != cfg:
            raise ValueError('resume output/config mismatch')
        agent, meta = CandidateDQNAgent.load(output/'last.pt')
        if (meta['schemagan_sha256'] != recon.checkpoint_sha256 or meta['manifest_sha256'] != manifest_sha or
                meta['config_sha256'] != _hash(config_path) or meta['feature_schema_sha256'] != feature_schema_sha):
            raise ValueError('resume identity mismatch')
        start = meta['next_episode'];best = meta['best_validation_rmse']
        train_rows = json.loads((output/'training_history.json').read_text())
        val_history = json.loads((output/'validation_history.json').read_text())
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        agent = CandidateDQNAgent(agent_cfg);start=0;best=float('inf');train_rows=[];val_history=[]
    for episode in range(start, cfg['episode_count']):
        episode_started = time.perf_counter()
        entry = train_entries[episode % len(train_entries)]
        count = cfg['initial_cpt_counts'][episode % len(cfg['initial_cpt_counts'])]
        truth, valid, grid = _case(root, entry)
        sites = _initial(grid, valid, entry, count, cfg['seed'], episode)
        sim = SyntheticCPTSimulator(recon, truth, valid, grid)
        state = sim.reset(sites)
        losses=[];rewards=[];selected=[];step_info=None;gradient_norm=None;epsilon=agent.epsilon(episode)
        q_min=[];q_median=[];q_max=[];q_selected=[]
        for step in range(cfg['episode_horizon']):
            if len(state.candidate_xy) == 0:break
            scores=agent.scores(state)
            if not np.isfinite(scores).all():
                raise ValueError('nonfinite Q values')
            action = agent.act(state, epsilon)
            q_min.append(float(scores.min()));q_median.append(float(np.median(scores)))
            q_max.append(float(scores.max()));q_selected.append(float(scores[action]))
            chosen = tuple(map(int,state.candidate_xy[action]));selected.append(chosen)
            next_state, reward, exhausted, info = sim.step(action)
            step_info=info
            done = exhausted or step+1 == cfg['episode_horizon']
            loss = agent.observe(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)
                gradient_norm = float(np.sqrt(sum(float(p.grad.detach().square().sum())
                    for p in agent.online.parameters() if p.grad is not None)))
            rewards.append(reward);state=next_state
            if done:break
        train_rows.append({'episode': episode+1,'realization_id':entry['realization_id'],'grid_id':entry['grid_id'],
                           'initial_cpt_count':count,'epsilon':epsilon,'selected_xy':json.dumps(selected),
                           'steps':len(rewards),'reward_sum':float(sum(rewards)),
                           'rmse_before_common':step_info['rmse_before_common'] if step_info else None,
                           'rmse_after_common':step_info['rmse_after_common'] if step_info else None,
                           'last_common_rmse_improvement':float(rewards[-1]) if rewards else 0.0,
                           'mean_td_loss':float(np.mean(losses)) if losses else None,
                           'replay_size':len(agent.replay),'learning_steps':agent.learn_steps,
                           'gradient_norm_after_clip':gradient_norm,
                           'q_min':float(min(q_min)) if q_min else None,
                           'q_median':float(np.mean(q_median)) if q_median else None,
                           'q_max':float(max(q_max)) if q_max else None,
                           'q_selected':float(np.mean(q_selected)) if q_selected else None,
                           'edge_action':bool(selected and any(x in (0,grid.nx-1) or y in (0,grid.ny-1)
                               for x,y in selected)),
                           'valid_action':bool(selected),'duplicate_action':len(selected)!=len(set(selected)),
                           'episode_seconds':time.perf_counter()-episode_started})
        if (episode+1) % cfg['validation_every_episodes'] == 0 or episode+1 == cfg['episode_count']:
            rows, summary = evaluate_validation(agent,recon,root,volume_count=cfg['validation_volume_count'],
                                                 initial_counts=validation_counts,seed=cfg['validation_seed'],
                                                 reference_agent=reference_agent)
            score = (float(np.mean([summary['by_initial_cpt_count'][str(count)]['by_policy']['dqn']['mean_after_rmse']
                                    for count in selection_counts])) if selection_counts else
                     summary['overall']['by_policy']['dqn']['mean_after_rmse'])
            baseline=summary['overall']['by_policy']
            paired=summary['overall']['paired_dqn_minus']
            validation_row={'episode':episode+1,'selection_metric':selection_metric,
              'selection_score':score,'dqn_mean_after_rmse':baseline['dqn']['mean_after_rmse'],
              'uniform_mean_after_rmse':baseline['uniform']['mean_after_rmse'],
              'random_mean_after_rmse':baseline['random']['mean_after_rmse'],
              'dqn_mean_improvement':baseline['dqn']['mean_improvement'],
              'uniform_mean_improvement':baseline['uniform']['mean_improvement'],
              'random_mean_improvement':baseline['random']['mean_improvement'],
              'dqn_win_rate_uniform':paired['uniform']['dqn_win_rate'],
              'dqn_win_rate_random':paired['random']['dqn_win_rate'],
              'dqn_minus_uniform_mean':paired['uniform']['mean'],
              'dqn_minus_random_mean':paired['random']['mean'],
              'validation_cases':summary['overall']['n'],
              'bootstrap_volume_count':paired['uniform']['n'],
              'valid_action_rate':baseline['dqn']['valid_action_rate'],
              'duplicate_action_rate':baseline['dqn']['duplicate_action_rate']}
            if reference_agent is not None:
                validation_row.update(scaled_dqn_mean_after_rmse=baseline['scaled_dqn']['mean_after_rmse'],
                    dqn_minus_scaled_dqn_mean=paired['scaled_dqn']['mean'],
                    dqn_win_rate_scaled_dqn=paired['scaled_dqn']['dqn_win_rate'])
            val_history.append(validation_row)
            (output/'validation_history.json').write_text(json.dumps(val_history,indent=2)+'\n')
            _write_csv(output/'validation_history.csv',val_history)
            (output/f'validation_summary_episode_{episode+1:04d}.json').write_text(json.dumps(summary,indent=2)+'\n')
            _write_csv(output/f'validation_per_volume_episode_{episode+1:04d}.csv',rows)
            if score < best:
                best=score
                agent.save(output/'best.pt',metadata={'next_episode':episode+1,'best_validation_rmse':best,
                           'schemagan_sha256':recon.checkpoint_sha256,'manifest_sha256':manifest_sha,
                           'config_sha256':_hash(config_path),'feature_schema_sha256':feature_schema_sha,
                           'training_config':cfg,'selection_metric':selection_metric})
            (output/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            _write_csv(output/'validation_per_volume.csv',rows)
        meta={'next_episode':episode+1,'best_validation_rmse':best,
              'schemagan_sha256':recon.checkpoint_sha256,'manifest_sha256':manifest_sha,
              'config_sha256':_hash(config_path),'feature_schema_sha256':feature_schema_sha,
              'training_config':cfg,'selection_metric':selection_metric}
        agent.save(output/'last.pt',metadata=meta)
        (output/'training_history.json').write_text(json.dumps(train_rows,indent=2)+'\n')
    _write_csv(output/'training_history.csv',train_rows)
    return {'status':'complete','episodes':cfg['episode_count'],'best_validation_rmse':best,'output':str(output)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('train','validate'))
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    if args.command=='train':
        if args.output is None:raise ValueError('--output required')
        result=run_training(args.config,args.output,resume=args.resume)
    else:
        if args.checkpoint is None or args.output is None:raise ValueError('--checkpoint and --output required')
        if args.output.exists():raise FileExistsError(args.output)
        agent,meta=CandidateDQNAgent.load(args.checkpoint)
        root=ROOT/cfg['dataset_root'];recon=FrozenV3Reconstructor(ROOT/cfg['schemagan_checkpoint'],ROOT/cfg['schemagan_config'],root)
        if (meta.get('schemagan_sha256') != recon.checkpoint_sha256 or
                meta.get('manifest_sha256') != _hash(root/'manifest.json') or
                meta.get('feature_schema_sha256') != _hash(ROOT/'results/dqn_v3_implementation/dqn_v3_feature_schema.json')):
            raise ValueError('validation checkpoint identity mismatch')
        rows,summary=evaluate_validation(agent,recon,root,volume_count=cfg['validation_volume_count'],
                                         initial_counts=_validation_settings(cfg)[0],seed=cfg['validation_seed'],
                                         reference_agent=_reference_agent(cfg,recon,_hash(root/'manifest.json'),
                                             _hash(ROOT/'results/dqn_v3_implementation/dqn_v3_feature_schema.json')))
        args.output.mkdir(parents=True);_write_csv(args.output/'per_volume.csv',rows)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        result=summary
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
