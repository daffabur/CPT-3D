"""Generate ten controlled, validation-only examples; refuse overwrites."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from .core import Grid, extract_cpt
from .geotechnical_v3 import DEFAULT_RESPONSE, VERSION, generate, validate, save, fingerprint

BASE=Grid(16,16,16,2,2,1)

def specs():
    isotropic=copy.deepcopy(DEFAULT_RESPONSE)
    flat=copy.deepcopy(DEFAULT_RESPONSE);flat['kernel_sigma_m']=[2.,2.,2.]
    vertical=copy.deepcopy(DEFAULT_RESPONSE);vertical['kernel_sigma_m']=[2.,2.,5.]
    return [
      ('material_changed','base','horizontal',BASE,101,201,isotropic,1,'train'),
      ('material_changed','new_material','horizontal',BASE,101,202,isotropic,2,'train'),
      ('geology_changed','horizontal','horizontal',BASE,111,211,isotropic,1,'train'),
      ('geology_changed','inclined','inclined',BASE,112,211,isotropic,3,'train'),
      ('seed_changed','seed_a','undulating',BASE,121,221,isotropic,1,'validation'),
      ('seed_changed','seed_b','undulating',BASE,122,222,isotropic,2,'validation'),
      ('anisotropy_changed','lateral','lens',BASE,131,231,flat,1,'test_family'),
      ('anisotropy_changed','vertical','lens',BASE,131,231,vertical,4,'test_family'),
      ('resolution_changed','coarse','horizontal',Grid(12,12,12,2,2,1),141,241,isotropic,1,'test_parameter'),
      ('resolution_changed','fine','horizontal',Grid(24,24,12,1,1,1),141,241,isotropic,3,'test_parameter'),
    ]

def build(i,s):
    pair,role,family,grid,geo_seed,mat_seed,response,n_cpt,split=s
    volume=generate(grid,family,3,1000+i,scenario_id=f'{pair}-{role}',realization_id=f'geo_v3_{i:02d}',
                    response=response,geology_seed=geo_seed,material_seed=mat_seed)
    sites=[(grid.nx//2,grid.ny//2),(2,2),(grid.nx-3,grid.ny-3),(2,grid.ny-3)][:n_cpt]
    obs=extract_cpt(volume,sites)
    err=validate(volume,obs)
    if err:raise ValueError((i,err))
    return volume,obs,split,pair,role

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('data/synthetic3d_v2/prototype_geotechnical_v3'))
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    planned=specs()
    if len(planned)>12:raise ValueError('too many examples')
    built=[build(i,s) for i,s in enumerate(planned)]
    # Paired derivatives stay in one split; exact generated outcomes cannot cross splits.
    by_pair={};by_hash={}
    for v,_,split,pair,_ in built:
        if pair in by_pair and by_pair[pair]!=split:raise ValueError('pair split leakage')
        by_pair[pair]=split
        fp=fingerprint(v)
        if fp in by_hash and by_hash[fp]!=split:raise ValueError('identical cross-split realization')
        by_hash[fp]=split
    args.output.mkdir(parents=True)
    entries=[save(v,o,args.output,s,p,r) for v,o,s,p,r in built]
    (args.output/'manifest.json').write_text(json.dumps({'schema_version':VERSION,'purpose':'controlled examples only; no model training',
      'split_policy':'pairs grouped; lens family held out; resolution parameter held out; validation seed pair held out',
      'entries':entries},indent=2)+'\n')
    print(f'created and validated {len(entries)} {VERSION} examples at {args.output}')
if __name__=='__main__':main()
