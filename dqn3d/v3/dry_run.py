"""Small integration audit: real train/validation inference, fixture-only optimizer."""
from __future__ import annotations
import json
import resource
import tempfile
import time
from pathlib import Path
import numpy as np
from .adapter import FrozenV3Reconstructor
from .agent import AgentConfig, CandidateDQNAgent
from .baselines import uniform_action
from .features import CandidateState, CANDIDATE_NAMES, GLOBAL_NAMES
from .runner import ROOT, _case, _entries, _initial, evaluate_validation
from .service import recommend
from .simulator import SyntheticCPTSimulator


def run() -> tuple[dict,dict]:
    root=ROOT/'data/synthetic3d_v2/schemagan_v3_24k'
    recon=FrozenV3Reconstructor(ROOT/'results/schemagan3d_v3_training_continuation/mse_continuation/best.pt',
                                ROOT/'results/schemagan3d_v3_training_continuation/mse_continuation_config.json',root)
    agent=CandidateDQNAgent(AgentConfig(batch_size=8,seed=14))
    train=_entries(root,'train',9)
    val=_entries(root,'validation_id',3)
    result={'schema_version':'dqn-v3-integration-dry-run-1','checkpoint_sha256':recon.checkpoint_sha256,
            'real_dataset_optimizer_steps':0,'train_cases':[],'validation_cases':[],
            'validation_comparison':None,'fixture_optimizer':None}
    bench={'schema_version':'dqn-v3-light-benchmark-1','device':'cpu','repeat_count':3,
           'note':'Short end-to-end inference on 3 real train volumes; no DQN training or full-pilot throughput claim.',
           'by_grid':{}}
    selected={}
    for e in train:
        selected.setdefault(e['grid_id'],e)
    if set(selected)!={'g12','g16','g24'}:
        raise ValueError('dry-run sample missing one V3 grid')
    for grid_id,e in selected.items():
        truth,valid,grid=_case(root,e)
        initial=_initial(grid,valid,e,1,14)
        t0=time.perf_counter();sim=SyntheticCPTSimulator(recon,truth,valid,grid)
        state=sim.reset(initial);reset_seconds=time.perf_counter()-t0
        t1=time.perf_counter();recommendation=recommend(recon,agent,sim.observed_ic,sim.observed_mask,valid,grid,top_k=3)
        recommend_seconds=time.perf_counter()-t1
        if recommendation['status']!='ok':raise ValueError('unexpected no_candidate')
        xy=(recommendation['x_index'],recommendation['y_index'])
        action=next(i for i,a in enumerate(state.candidate_xy) if tuple(a)==xy)
        t2=time.perf_counter();next_state,reward,done,info=sim.step(action)
        step_seconds=time.perf_counter()-t2
        if not np.isfinite(reward) or info['common_hidden_voxels']<1:
            raise ValueError('invalid common-mask reward')
        result['train_cases'].append({'realization_id':e['realization_id'],'grid_id':grid_id,
          'shape':list(truth.shape),'initial_cpt_count':1,'legal_candidates_before':len(state.candidate_xy),
          'recommended_xy':list(xy),'recommended_xy_m':[recommendation['x_m'],recommendation['y_m']],
          'q_score':recommendation['q_score'],'reward_common_mask':reward,
          'rmse_before_common':info['rmse_before_common'],'rmse_after_common':info['rmse_after_common'],
          'legal_candidates_after':len(next_state.candidate_xy),'fixture_only_optimizer':True})
        # Repeated recommendation timing from the same valid observed state.
        timings=[]
        for _ in range(3):
            start=time.perf_counter();recommend(recon,agent,sim.observed_ic,sim.observed_mask,valid,grid)
            timings.append(time.perf_counter()-start)
        bench['by_grid'][grid_id]={'shape':list(truth.shape),'reset_seconds':reset_seconds,
          'recommend_seconds_first':recommend_seconds,'simulation_step_seconds':step_seconds,
          'recommend_seconds_repeats':timings,'recommend_seconds_median':float(np.median(timings))}
    for e in val:
        truth,valid,grid=_case(root,e)
        initial=_initial(grid,valid,e,2,1000014)
        sim=SyntheticCPTSimulator(recon,truth,valid,grid);state=sim.reset(initial)
        rec=recommend(recon,agent,sim.observed_ic,sim.observed_mask,valid,grid)
        uniform_idx=uniform_action(state,sim.observed_mask,grid)
        result['validation_cases'].append({'realization_id':e['realization_id'],'grid_id':e['grid_id'],
            'initial_cpt_count':2,'dqn_xy':[rec['x_index'],rec['y_index']],
            'uniform_xy':list(map(int,state.candidate_xy[uniform_idx]))})
    rows,summary=evaluate_validation(agent,recon,root,volume_count=3,initial_counts=[1,2],seed=1000014)
    result['validation_comparison']={'rows':len(rows),'paired_volume_count':summary['overall']['n'],
       'mean_rmse_after':{p:v['mean_after_rmse'] for p,v in summary['overall']['by_policy'].items()},
       'dqn_minus_uniform':summary['overall']['paired_dqn_minus']['uniform'],
       'all_valid_actions':all(r['valid_action'] for r in rows),
       'all_unique_actions':all(not r['duplicate_action'] for r in rows)}
    # The only optimizer step in this dry run uses synthetic feature arrays, not a dataset case.
    fixture=CandidateState(np.array([[0,0],[0,1]],np.int16),
        np.zeros((2,len(CANDIDATE_NAMES)),np.float32),np.zeros(len(GLOBAL_NAMES),np.float32))
    empty=CandidateState(np.empty((0,2),np.int16),np.empty((0,len(CANDIDATE_NAMES)),np.float32),
        np.zeros(len(GLOBAL_NAMES),np.float32))
    fixture_agent=CandidateDQNAgent(AgentConfig(batch_size=1,seed=99))
    before=[p.clone() for p in fixture_agent.online.parameters()]
    loss=fixture_agent.observe(fixture,1,1.0,empty,True)
    changed=any(not np.array_equal(a.detach().numpy(),b.detach().numpy()) for a,b in zip(before,fixture_agent.online.parameters()))
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'fixture.pt';fixture_agent.save(path,metadata={'fixture_only':True})
        loaded,metadata=CandidateDQNAgent.load(path)
        valid_resume=(loaded.learn_steps==fixture_agent.learn_steps and len(loaded.replay)==1 and metadata['fixture_only'])
    result['fixture_optimizer']={'steps':1,'loss':loss,'online_weights_changed':changed,'checkpoint_resume_valid':valid_resume}
    bench['peak_process_rss_raw']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    bench['peak_process_rss_unit']='bytes on macOS; process peak includes Python/PyTorch imports and dataset cache'
    return result,bench


if __name__=='__main__':
    a,b=run();out=ROOT/'results/dqn_v3_implementation'
    (out/'dqn_v3_integration_dry_run.json').write_text(json.dumps(a,indent=2)+'\n')
    (out/'dqn_v3_benchmark.json').write_text(json.dumps(b,indent=2)+'\n')
    print(json.dumps({'train_cases':len(a['train_cases']),'validation_cases':len(a['validation_cases']),
                      'comparison_rows':a['validation_comparison']['rows'],
                      'fixture_optimizer_step':a['fixture_optimizer']['steps'],'benchmarked_grids':list(b['by_grid'])}))
