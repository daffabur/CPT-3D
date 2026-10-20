"""Read-only on-disk validation of prototype-3 examples."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .core import Grid
from .geotechnical_v3 import VERSION, generate, validate, fingerprint, direction_autocorrelation

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('data/synthetic3d_v2/prototype_geotechnical_v3'))
    args=p.parse_args();manifest=json.loads((args.root/'manifest.json').read_text())
    if manifest['schema_version']!=VERSION:raise ValueError('schema version mismatch')
    seen={};pairs={}
    for e in manifest['entries']:
        m=json.loads((args.root/e['metadata']).read_text())
        v={k:np.load(args.root/path,allow_pickle=False) for k,path in e['arrays'].items()};v['metadata']=m
        obs={'profiles':json.loads((args.root/e['profiles']).read_text()),'observed_mask':np.load(args.root/e['observed_mask'],allow_pickle=False)}
        errors=validate(v,obs)
        if errors:raise ValueError((e['realization_id'],errors))
        reproduced=generate(Grid(**m['grid']),m['family'],m['n_background_layers'],m['seed'],scenario_id=m['scenario_id'],
              realization_id=m['realization_id'],geometry=m['geology_params'],response=m['response_settings'],
              mixtures=m['unit_mixtures'],geology_seed=m['geology_seed'],material_seed=m['material_seed'])
        for k in e['arrays']:
            if not np.array_equal(reproduced[k],v[k]):raise ValueError((e['realization_id'],k,'repro mismatch'))
        fp=fingerprint(v)
        if fp!=e['fingerprint']:raise ValueError('fingerprint mismatch')
        if fp in seen and seen[fp]!=e['split']:raise ValueError('identical cross-split')
        seen[fp]=e['split']
        if e['pair_id'] in pairs and pairs[e['pair_id']]!=e['split']:raise ValueError('pair cross-split')
        pairs[e['pair_id']]=e['split']
        for i,name in enumerate(('x','y','z')):
            value=direction_autocorrelation(np.log(v['qc_mpa']),i)
            if not np.isclose(value,e['directional_log_qc'][name]['lag1_autocorrelation'],atol=1e-8):raise ValueError('directional diagnostic mismatch')
    print(f"validated {len(manifest['entries'])} saved examples; {len(pairs)} paired experiments; no identical cross-split fingerprint")
if __name__=='__main__':main()
