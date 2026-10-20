from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from synthetic3d.v2.core import Grid
from .hybrid import recommend_next_cpt
from .final_eval import common_hidden_mask, verify_manifest, ROOT
from .features import CANDIDATE_NAMES, GLOBAL_NAMES


class FakeRecon:
    def predict(self, observed_ic, observed_mask, valid_mask, grid):
        assert observed_ic.shape == observed_mask.shape == valid_mask.shape
        return np.full(observed_ic.shape, 2.3, dtype=np.float32), 0.0


class FakeAgent:
    def scores(self, state):
        return np.ones(len(state.candidate_xy), dtype=np.float32)


@pytest.mark.parametrize('side,nz', [(12, 8), (16, 8), (24, 12)])
@pytest.mark.parametrize('count', [1, 2, 3, 5, 8, 12, 20])
def test_route_legal_physical_deterministic(side, nz, count):
    grid = Grid(side, side, nz, 1.25, 2.0, 1.0, origin_x_m=10., origin_y_m=-4., azimuth_deg=17.)
    valid = np.ones((side,side,nz), dtype=bool)
    observed = np.zeros_like(valid)
    observed.reshape(side*side,nz)[:count] = True
    access = np.zeros((side,side), dtype=bool);access[-2:,-2:] = True
    x = recommend_next_cpt(FakeRecon(), FakeAgent(), np.ones_like(valid,dtype=np.float32), observed, valid, grid, access)
    y = recommend_next_cpt(FakeRecon(), FakeAgent(), np.ones_like(valid,dtype=np.float32), observed, valid, grid, access)
    assert x == y
    assert x['status'] == 'ok'
    assert x['policy_used'] == ('scaled_dqn' if count <= 8 else 'uniform')
    assert access[x['x_index'], x['y_index']]
    assert not observed[x['x_index'],x['y_index']].any()
    assert (x['x_m'],x['y_m']) == grid.xy_m(x['x_index'],x['y_index'])


def test_unsupported_and_no_candidate():
    grid = Grid(8,8,8,1.,1.,1.)
    valid = np.ones((8,8,8),dtype=bool)
    observed = np.zeros_like(valid);observed.reshape(64,8)[:4] = True
    assert recommend_next_cpt(FakeRecon(),FakeAgent(),np.zeros_like(valid,dtype=np.float32),observed,valid,grid)['status']=='unsupported_cpt_count'
    observed.reshape(64,8)[:]=False;observed[0,0]=True
    access=np.zeros((8,8),dtype=bool)
    assert recommend_next_cpt(FakeRecon(),FakeAgent(),np.zeros_like(valid,dtype=np.float32),observed,valid,grid,access)['status']=='no_candidate'


def test_no_truth_argument_and_minimum_one_cpt():
    assert 'truth' not in inspect.signature(recommend_next_cpt).parameters
    grid=Grid(8,8,8,1.,1.,1.)
    valid=np.ones((8,8,8),dtype=bool)
    with pytest.raises(ValueError,match='at least one'):
        recommend_next_cpt(FakeRecon(),FakeAgent(),np.zeros_like(valid,dtype=np.float32),np.zeros_like(valid),valid,grid)


def test_common_mask_and_split_isolation():
    valid=np.ones((4,4,4),dtype=bool)
    masks={p:np.zeros_like(valid) for p in ('hybrid','uniform','random')}
    masks['hybrid'][0,0]=True;masks['uniform'][0,1]=True;masks['random'][0,2]=True
    common=common_hidden_mask(valid,masks)
    assert common.sum()==valid.size-12
    assert not common[0,:3].any()
    entries=[{'split':'train','realization_id':'a','lineage_group':'one'},
             {'split':'test_id','realization_id':'b','lineage_group':'one'}]
    with pytest.raises(ValueError,match='lineage'):
        verify_manifest(entries)


def test_checkpoint_sha():
    paths=[('results/schemagan3d_v3_training_continuation/mse_continuation/best.pt','5694bb7b43efd7054a99fac8b74dadc5d74ebe6d291f42a8393b57fe9da52b64'),
           ('results/dqn_v3_scaled_pilot_2000e/run/best.pt','687539d5ad2606b198f2aadfaa6761bc9d9968ec8ce1fd2c915350413530345a')]
    for path,expected in paths:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected


def test_feature_schema_order():
    schema=json.loads((ROOT/'results/dqn_v3_implementation/dqn_v3_feature_schema.json').read_text())
    assert tuple(schema['candidate_names_ordered']) == CANDIDATE_NAMES
    assert tuple(schema['global_names_ordered']) == GLOBAL_NAMES
