"""Construct nested CPT observation variants from one frozen V3 truth volume."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from synthetic3d.v2.core import Grid
from .v3_eval_protocol import nested_sites

POLICY_PATH=Path(__file__).with_name('v3_observation_policy.json')

def variants(ic,valid_mask,grid:Grid,seed:int,counts=(1,2,3,5,8,12,20)):
    if ic.shape!=valid_mask.shape or ic.shape!=(grid.nx,grid.ny,grid.nz):raise ValueError('shape mismatch')
    if not np.all(np.isfinite(ic[valid_mask])):raise ValueError('nonfinite valid Ic')
    sites=nested_sites(grid,seed,valid_mask,max(counts))
    out={}
    for n in counts:
        if n<1 or n>len(sites):raise ValueError('requested CPT count exceeds valid XY capacity')
        mask=np.zeros(ic.shape,dtype=bool)
        for x,y in sites[:n]:mask[x,y,:]=valid_mask[x,y,:]
        sparse=np.zeros(ic.shape,dtype=np.float32)
        sparse[mask]=ic[mask]
        out[n]={'sites':sites[:n],'observed_mask':mask,'observed_ic':sparse}
    return out

def policy():return json.loads(POLICY_PATH.read_text())
