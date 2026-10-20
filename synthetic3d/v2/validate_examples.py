"""Read-only validator for a saved V2 prototype manifest and its artifacts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .core import validate, validate_reproducibility, validate_manifest, assign_split

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('data/synthetic3d_v2/prototype'))
    args=parser.parse_args()
    manifest=json.loads((args.root/'manifest.json').read_text())
    entries=manifest['examples']
    errors=validate_manifest(entries)
    for e in entries:
        metadata=json.loads((args.root/e['metadata']).read_text())
        volume={k:np.load(args.root/path,allow_pickle=False) for k,path in e['arrays'].items() if k!='observed_mask'}
        volume['metadata']=metadata
        observation={'observed_mask':np.load(args.root/e['arrays']['observed_mask'],allow_pickle=False),
                     'profiles':json.loads((args.root/e['profiles']).read_text())}
        errors += [f"{e['realization_id']}: {x}" for x in validate(volume,observation)+validate_reproducibility(volume)]
        if assign_split(metadata)!=e['split']: errors.append(f"{e['realization_id']}: split mismatch")
        if tuple(e['shape'])!=volume['Ic'].shape: errors.append(f"{e['realization_id']}: manifest shape mismatch")
        if e['n_cpt']!=len(observation['profiles']): errors.append(f"{e['realization_id']}: manifest CPT count mismatch")
    report={'examples':len(entries),'passed':len(entries) if not errors else None,'errors':errors,
            'splits':{s:sum(e['split']==s for e in entries) for s in sorted({e['split'] for e in entries})}}
    print(json.dumps(report,indent=2))
    if errors: raise SystemExit(1)

if __name__=='__main__': main()
