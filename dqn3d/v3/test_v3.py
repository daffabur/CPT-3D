"""Small, inference-only and fixture-only tests for separate DQN V3 package."""
from __future__ import annotations
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from synthetic3d.v2.core import Grid
from .adapter import FrozenV3Reconstructor, EXPECTED_MEAN, EXPECTED_STD
from .agent import AgentConfig, CandidateDQNAgent, SharedCandidateQ, Transition
from .baselines import random_action, uniform_action
from .features import CANDIDATE_NAMES, GLOBAL_NAMES, CandidateState, build_state
from .runner import _entries, _initial, _validation_settings, evaluate_validation, run_training
from .service import recommend
from .simulator import SyntheticCPTSimulator

ROOT=Path(__file__).resolve().parents[2]


class FakeReconstructor:
    def predict(self, observed_ic, observed_mask, valid_mask, grid):
        # Deliberately has no truth parameter; switches between 0 and 1 after a second CPT.
        count=int(observed_mask.any(axis=2).sum())
        p=np.full(observed_ic.shape,1.0 if count>=2 else 0.0,dtype=np.float32)
        p[observed_mask]=observed_ic[observed_mask]
        return p,0.001


def small_case(n=12):
    g=Grid(n,n,12,2.,2.,1.,origin_x_m=10.,origin_y_m=20.,azimuth_deg=30.)
    truth=np.ones((n,n,12),np.float32)
    valid=np.ones_like(truth,bool)
    return g,truth,valid


def state_fixture(n=3):
    return CandidateState(np.array([[i,0] for i in range(n)],np.int16).reshape(n,2),
                          np.zeros((n,len(CANDIDATE_NAMES)),np.float32),
                          np.zeros(len(GLOBAL_NAMES),np.float32))


@pytest.fixture(scope='module')
def real_adapter():
    return FrozenV3Reconstructor(ROOT/'results/schemagan3d_v3_training_continuation/mse_continuation/best.pt',
        ROOT/'results/schemagan3d_v3_training_continuation/mse_continuation_config.json',
        ROOT/'data/synthetic3d_v2/schemagan_v3_24k')


def test_checkpoint_load_and_freeze(real_adapter):
    assert real_adapter.config['model_variant']=='idw_guided_residual'
    assert all(not p.requires_grad for p in real_adapter.model.parameters())
    assert not real_adapter.model.training


def test_locked_normalization(real_adapter):
    assert real_adapter.stats.mean==EXPECTED_MEAN and real_adapter.stats.std==EXPECTED_STD


@pytest.mark.parametrize('shape,spacing', [((12,12,12),2.),((16,16,16),2.),((24,24,12),1.)])
def test_real_grid_shapes_and_reconstruction(real_adapter,shape,spacing):
    nx,ny,nz=shape;g=Grid(nx,ny,nz,spacing,spacing,1.)
    o=np.zeros(shape,np.float32);m=np.zeros(shape,bool);v=np.ones(shape,bool)
    o[0,0]=2.5;m[0,0]=1
    p,_=real_adapter.predict(o,m,v,g)
    assert p.shape==shape and np.isfinite(p).all() and np.array_equal(p[m],o[m])
    assert np.array_equal(p,real_adapter.predict(o,m,v,g)[0])


def test_adapter_rejects_incomplete_profile(real_adapter):
    g=Grid(12,12,12,2,2,1);o=np.zeros((12,12,12),np.float32);m=np.zeros_like(o,bool);v=np.ones_like(o,bool)
    m[0,0,0]=1
    with pytest.raises(ValueError,match='every valid depth'):real_adapter.predict(o,m,v,g)


def test_physical_coordinate_roundtrip():
    g,_,_=small_case();xy=g.xy_m(4,7)
    assert g.index_xy(*xy)==(4,7)
    assert xy!=(4,7)


def test_one_initial_cpt_and_feature_dimensions():
    g,t,v=small_case();s=SyntheticCPTSimulator(FakeReconstructor(),t,v,g).reset([(0,0)])
    assert s.features.shape==(143,17) and s.global_context.shape==(11,)
    assert (0,0) not in map(tuple,s.candidate_xy)


def test_multiple_initial_cpt_and_legal_mask():
    g,t,v=small_case();sim=SyntheticCPTSimulator(FakeReconstructor(),t,v,g);s=sim.reset([(0,0),(1,1),(2,2)])
    assert len(s.candidate_xy)==141 and not set(map(tuple,s.candidate_xy))&{(0,0),(1,1),(2,2)}


def test_access_mask_and_no_candidate():
    g,t,v=small_case();sim=SyntheticCPTSimulator(FakeReconstructor(),t,v,g,np.zeros((12,12),bool));s=sim.reset([(0,0)])
    assert len(s.candidate_xy)==0
    agent=CandidateDQNAgent(AgentConfig())
    result=recommend(FakeReconstructor(),agent,sim.observed_ic,sim.observed_mask,v,g,np.zeros((12,12),bool))
    assert result['status']=='no_candidate' and 'x_m' not in result


def test_no_duplicate_action():
    g,t,v=small_case();sim=SyntheticCPTSimulator(FakeReconstructor(),t,v,g);sim.reset([(0,0)])
    sim.step(0)
    with pytest.raises(ValueError):sim._reveal((0,1))


def test_feature_ic_normalization():
    g,t,v=small_case();m=np.zeros_like(v);m[0,0]=1
    rec=np.full_like(t,EXPECTED_MEAN+EXPECTED_STD)
    s=build_state(rec,m,v,g)
    assert np.isclose(s.features[0,5],1.,atol=1e-5)
    assert np.isclose(s.global_context[7],1.,atol=1e-5)


def test_feature_schema_file_matches_code():
    schema=json.loads((ROOT/'results/dqn_v3_implementation/dqn_v3_feature_schema.json').read_text())
    assert schema['candidate_names_ordered']==list(CANDIDATE_NAMES)
    assert schema['global_names_ordered']==list(GLOBAL_NAMES)


def test_no_hidden_truth_in_state_or_inference_signature():
    g,t,v=small_case();sim=SyntheticCPTSimulator(FakeReconstructor(),t,v,g);s=sim.reset([(0,0)])
    assert set(vars(s))=={'candidate_xy','features','global_context'}
    assert 'truth' not in inspect.signature(recommend).parameters
    assert np.isfinite(s.features).all()


def test_reward_common_mask_correctness_and_reveal():
    g,t,v=small_case();sim=SyntheticCPTSimulator(FakeReconstructor(),t,v,g);sim.reset([(0,0)])
    _,r,_,info=sim.step(0)
    assert info['selected_xy']==(0,1) and sim.observed_mask[0,1].all()
    assert np.array_equal(sim.observed_ic[0,1],t[0,1])
    assert r==pytest.approx(1.) and info['common_hidden_voxels']==(12*12-2)*12
    assert info['rmse_before_common']==pytest.approx(1.) and info['rmse_after_common']==pytest.approx(0.)


def test_empty_common_mask_finite_reward():
    g=Grid(8,8,8,1,1,1);t=np.ones((8,8,8),np.float32);v=np.zeros_like(t,bool);v[0,0]=1;v[0,1]=1
    sim=SyntheticCPTSimulator(FakeReconstructor(),t,v,g);sim.reset([(0,0)])
    _,r,done,info=sim.step(0)
    assert done and r==0 and info['common_hidden_voxels']==0


def test_variable_candidate_q_output():
    net=SharedCandidateQ()
    for n in (0,1,143,255,575):
        assert net(torch.zeros((n,17)),torch.zeros(11)).shape==(n,)


def test_epsilon_greedy_legal():
    agent=CandidateDQNAgent(AgentConfig());s=state_fixture(3)
    assert all(0<=agent.act(s,epsilon=1.)<3 for _ in range(20))


def test_lexicographic_tie_break():
    agent=CandidateDQNAgent(AgentConfig());s=state_fixture(3)
    for p in agent.online.parameters():p.data.zero_()
    assert agent.act(s)==0


def test_replay_action_mapping_and_one_fixture_optimizer_step():
    agent=CandidateDQNAgent(AgentConfig(batch_size=1,target_update_steps=100));s=state_fixture(2)
    before=[p.clone() for p in agent.online.parameters()];target=[p.clone() for p in agent.target.parameters()]
    loss=agent.observe(s,1,1.,state_fixture(1),True)
    assert np.isfinite(loss) and agent.replay[0].action_xy==(1,0)
    assert any(not torch.equal(a,b) for a,b in zip(before,agent.online.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(target,agent.target.parameters()))
    assert all(p.grad is None for p in agent.target.parameters())


def test_masked_bellman_terminal_and_nonterminal():
    agent=CandidateDQNAgent(AgentConfig(batch_size=1));s=state_fixture(1)
    assert np.isfinite(agent.observe(s,0,0.5,state_fixture(0),True))
    agent2=CandidateDQNAgent(AgentConfig(batch_size=1))
    assert np.isfinite(agent2.observe(s,0,0.5,state_fixture(3),False))
    with pytest.raises(ValueError,match='nonterminal'):agent2.observe(s,0,0.,state_fixture(0),False)


def test_replay_rejects_invalid_action_and_detects_mapping_corruption():
    agent=CandidateDQNAgent(AgentConfig(batch_size=1));s=state_fixture(2)
    with pytest.raises(ValueError):agent.observe(s,2,0.,s,True)
    agent.replay.append(Transition(s,1,(0,0),0.,s,True))
    with pytest.raises(ValueError,match='mapping'):agent.learn_step()


def test_checkpoint_save_load_and_resume_rng(tmp_path):
    cfg=AgentConfig(batch_size=2,seed=77);agent=CandidateDQNAgent(cfg);s=state_fixture(2)
    agent.observe(s,0,1.,s,False);agent.observe(s,1,.3,s,True)
    path=tmp_path/'agent.pt';agent.save(path,metadata={'episode':2});copy,meta=CandidateDQNAgent.load(path)
    assert meta=={'episode':2} and len(copy.replay)==2 and copy.learn_steps==agent.learn_steps
    assert copy.act(s,epsilon=1.)==agent.act(s,epsilon=1.)
    assert copy.learn_step()==pytest.approx(agent.learn_step(),rel=1e-7,abs=1e-7)
    for a,b in zip(copy.online.parameters(),agent.online.parameters()):assert torch.equal(a,b)


def test_uniform_physical_distance_and_random():
    g,t,v=small_case();sim=SyntheticCPTSimulator(FakeReconstructor(),t,v,g);s=sim.reset([(0,0)])
    assert tuple(s.candidate_xy[uniform_action(s,sim.observed_mask,g)])==(11,11)
    assert 0<=random_action(s,np.random.default_rng(14))<len(s.candidate_xy)


def test_service_inference_without_truth():
    g,t,v=small_case();sim=SyntheticCPTSimulator(FakeReconstructor(),t,v,g);sim.reset([(0,0)])
    agent=CandidateDQNAgent(AgentConfig());result=recommend(FakeReconstructor(),agent,sim.observed_ic,sim.observed_mask,v,g,top_k=3)
    assert result['status']=='ok' and len(result['ranked_candidates'])==3
    assert (result['x_index'],result['y_index'])!=(0,0)
    assert g.index_xy(result['x_m'],result['y_m'])==(result['x_index'],result['y_index'])


def test_development_runner_refuses_test_split():
    with pytest.raises(ValueError,match='only allows'):_entries(ROOT/'data/synthetic3d_v2/schemagan_v3_24k','test_id',1)


def test_high_cpt_training_and_validation_configuration_roundtrip(tmp_path, monkeypatch):
    from . import runner
    import hashlib
    root=tmp_path/'dataset';root.mkdir()
    (root/'manifest.json').write_text('{}')
    schema=tmp_path/'results/dqn_v3_implementation/dqn_v3_feature_schema.json'
    schema.parent.mkdir(parents=True);schema.write_text('{}')
    checkpoint=tmp_path/'schemagan.pt';checkpoint.write_bytes(b'fixture')
    schemagan_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reference=tmp_path/'reference.pt'
    CandidateDQNAgent(AgentConfig()).save(reference,metadata={
        'schemagan_sha256':schemagan_sha,
        'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        'feature_schema_sha256':hashlib.sha256(schema.read_bytes()).hexdigest()})
    cfg={'schema_version':'dqn-v3-pilot-config-1','development_splits':['train','validation_id'],
         'dataset_root':'dataset','schemagan_checkpoint':'schemagan.pt','schemagan_config':'unused.json',
         'train_volume_count':2,'validation_volume_count':1,'episode_count':2,'validation_every_episodes':2,
         'initial_cpt_counts':[12,20],'validation_initial_cpt_counts':[1,2,3,5,8,12,20],
         'selection_cpt_counts':[12,20],'episode_horizon':1,'seed':28,'validation_seed':1000014,
         'agent':vars(AgentConfig(batch_size=2,seed=28)),
         'reference_checkpoint':'reference.pt','reference_checkpoint_sha256':hashlib.sha256(reference.read_bytes()).hexdigest()}
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    class Recon(FakeReconstructor):
        def __init__(self,*args):self.checkpoint_sha256=schemagan_sha
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'FrozenV3Reconstructor',Recon)
    entries=[{'split':'train','realization_id':f'train-{j}','grid_id':'g12','evaluation_site_seed':j} for j in range(2)]
    val=[{'split':'validation_id','realization_id':'val-0','grid_id':'g12','evaluation_site_seed':3}]
    monkeypatch.setattr(runner,'_entries',lambda root,split,limit:(entries if split=='train' else val)[:limit])
    monkeypatch.setattr(runner,'_case',lambda root,entry:(small_case()[1],small_case()[2],small_case()[0]))
    received=[]
    def fake_evaluate(agent,reconstructor,dataset_root,*,volume_count,initial_counts,seed,reference_agent):
        received.append((initial_counts,reference_agent is not None))
        policies={k:{'mean_after_rmse':.1,'mean_improvement':.01,'valid_action_rate':1.,'duplicate_action_rate':0.}
                  for k in ('dqn','scaled_dqn','uniform','random')}
        paired={k:{'mean':0.,'dqn_win_rate':.5,'n':1} for k in ('scaled_dqn','uniform','random')}
        return [{'realization_id':'val-0','policy':'dqn'}],{'overall':{'n':7,'by_policy':policies,'paired_dqn_minus':paired},
            'by_initial_cpt_count':{str(n):{'by_policy':{'dqn':{'mean_after_rmse':float(n)/100}}} for n in initial_counts}}
    monkeypatch.setattr(runner,'evaluate_validation',fake_evaluate)
    assert _validation_settings(cfg)==([1,2,3,5,8,12,20],[12,20],
        'mean_validation_dqn_rmse_after_common_cpt_12_20_equal')
    result=run_training(path,tmp_path/'run')
    assert result['status']=='complete' and result['best_validation_rmse']==pytest.approx(.16)
    history=json.loads((tmp_path/'run/training_history.json').read_text())
    assert [x['initial_cpt_count'] for x in history]==[12,20]
    assert all(np.isfinite(x['q_min']) and np.isfinite(x['q_median']) and
               np.isfinite(x['q_max']) and np.isfinite(x['q_selected']) for x in history)
    assert all(isinstance(x['edge_action'],bool) for x in history)
    assert received==[([1,2,3,5,8,12,20],True)]
    _,meta=CandidateDQNAgent.load(tmp_path/'run/best.pt')
    assert meta['training_config']==cfg and meta['selection_metric'].endswith('cpt_12_20_equal')


def test_validation_four_policies_share_initial_sites_and_hidden_mask(monkeypatch):
    from . import runner
    entry={'split':'validation_id','realization_id':'fixture','grid_id':'g12','family':'horizontal'}
    monkeypatch.setattr(runner,'_entries',lambda root,split,limit:[entry])
    monkeypatch.setattr(runner,'_case',lambda root,entry:(small_case()[1],small_case()[2],small_case()[0]))
    monkeypatch.setattr(runner,'_initial',lambda grid,valid,entry,count,seed:[(0,0)])
    agent=CandidateDQNAgent(AgentConfig())
    reference=CandidateDQNAgent(AgentConfig(seed=28))
    rows,summary=evaluate_validation(agent,FakeReconstructor(),Path('.'),volume_count=1,
        initial_counts=[1],seed=14,reference_agent=reference)
    assert len(rows)==4 and set(r['policy'] for r in rows)=={'dqn','scaled_dqn','uniform','random'}
    assert len({r['initial_xy'] for r in rows})==1
    assert len({r['common_hidden_voxels'] for r in rows})==1
    assert set(summary['overall']['paired_dqn_minus'])=={'scaled_dqn','uniform','random'}


def test_legacy_modules_unchanged_contract():
    from dqn3d.agent import SpatialQNetwork
    from dqn3d.environment import EnvironmentConfig
    assert SpatialQNetwork().head[-1].out_features==16
    assert EnvironmentConfig().budget==8
