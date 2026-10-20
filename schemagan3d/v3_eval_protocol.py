"""Evaluation contracts only; no trained model evaluation is run here."""
from __future__ import annotations
import numpy as np
from synthetic3d.v2.core import Grid

PROFILE_COUNTS=(1,2,3,5,8,12,20)

def nested_sites(grid:Grid,seed:int,valid_mask:np.ndarray,max_profiles:int=20):
    if valid_mask.shape!=(grid.nx,grid.ny,grid.nz):raise ValueError('valid mask shape mismatch')
    candidates=[(x,y) for x in range(grid.nx) for y in range(grid.ny) if valid_mask[x,y].any()]
    if not candidates:raise ValueError('no valid CPT sites')
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(candidates))[:min(max_profiles,len(candidates))]
    return [candidates[i] for i in order]

def sparse_from_sites(observed_ic:np.ndarray,observed_mask:np.ndarray):
    if observed_ic.shape!=observed_mask.shape:raise ValueError('shape mismatch')
    if not observed_mask.any() or not np.all(np.isfinite(observed_ic[observed_mask])):raise ValueError('invalid observations')
    return np.where(observed_mask,observed_ic,0.).astype(np.float32)

def idw_physical(observed_ic:np.ndarray,observed_mask:np.ndarray,grid:Grid,power:float=2.):
    """IDW at each depth using observed CPT data only and physical XY distances."""
    shape=(grid.nx,grid.ny,grid.nz)
    if observed_ic.shape!=shape or observed_mask.shape!=shape or power<=0:raise ValueError('invalid IDW input')
    if not np.all(np.isfinite(observed_ic[observed_mask])):raise ValueError('nonfinite observed Ic')
    coordinates=np.array([grid.xy_m(x,y) for x in range(grid.nx) for y in range(grid.ny)]).reshape(grid.nx,grid.ny,2)
    out=np.empty(shape,dtype=np.float32)
    for z in range(grid.nz):
        pts=coordinates[observed_mask[:,:,z]];values=observed_ic[:,:,z][observed_mask[:,:,z]]
        if not len(pts):raise ValueError(f'no observed CPT at depth index {z}')
        d=np.linalg.norm(coordinates[:,:,None,:]-pts[None,None,:,:],axis=-1)
        exact=d==0
        weights=np.where(exact,0.,np.maximum(d,1e-12)**(-power))
        pred=(weights*values).sum(axis=-1)/np.maximum(weights.sum(axis=-1),1e-12)
        if exact.any():pred[exact.any(axis=-1)]=values[np.argmax(exact,axis=-1)[exact.any(axis=-1)]]
        out[:,:,z]=pred
    out[observed_mask]=observed_ic[observed_mask]
    return out

def valid_metrics(prediction:np.ndarray,target:np.ndarray,observed_mask:np.ndarray,valid_mask:np.ndarray):
    if not all(a.shape==target.shape for a in (prediction,observed_mask,valid_mask)):raise ValueError('shape mismatch')
    selected=valid_mask & ~observed_mask
    if not selected.any():raise ValueError('no hidden valid voxels')
    if not np.all(np.isfinite(prediction[valid_mask])):raise ValueError('nonfinite prediction')
    hidden=prediction[selected]-target[selected];full=prediction[valid_mask]-target[valid_mask]
    return {'hidden_rmse':float(np.sqrt(np.mean(hidden**2))),'hidden_mae':float(np.mean(np.abs(hidden))),
            'full_valid_rmse':float(np.sqrt(np.mean(full**2))),'hidden_count':int(selected.sum())}
