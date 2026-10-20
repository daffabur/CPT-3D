"""Read-only validation and diagnostics for saved refined examples."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .refined import SCHEMA_VERSION, validate, validate_reproducibility, validate_manifest, diagnostics, response_diagnostics, split

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('data/synthetic3d_v2/prototype_refined'))
    args=parser.parse_args()
    manifest=json.loads((args.root/'manifest.json').read_text())
    entries=manifest['examples']; errors=validate_manifest(entries)
    digests={}
    for e in entries:
        rid=e['realization_id']; metadata=json.loads((args.root/e['metadata']).read_text())
        volume={k:np.load(args.root/path,allow_pickle=False) for k,path in e['arrays'].items() if k!='observed_mask'}
        volume['metadata']=metadata
        observation={'observed_mask':np.load(args.root/e['arrays']['observed_mask'],allow_pickle=False),
                     'profiles':json.loads((args.root/e['profiles']).read_text())}
        errors += [f'{rid}: {x}' for x in validate(volume,observation)+validate_reproducibility(volume)]
        if metadata['schema_version']!=SCHEMA_VERSION or e['split']!=split(metadata): errors.append(f'{rid}: version/split mismatch')
        if e['shape']!=list(volume['Ic'].shape): errors.append(f'{rid}: shape metadata mismatch')
        if e['n_cpt']!=len(observation['profiles']): errors.append(f'{rid}: CPT count mismatch')
        if e['response_diagnostics']!=response_diagnostics(volume): errors.append(f'{rid}: response diagnostics mismatch')
        for name,path in e['arrays'].items():
            if not path.startswith(f'{rid}/'): errors.append(f'{rid}: derived artifact outside realization directory: {name}')
        for name in ('profiles','metadata'):
            if not e[name].startswith(f'{rid}/'): errors.append(f'{rid}: {name} outside realization directory')
        digest=hashlib.sha256(volume['Ic'].tobytes()).hexdigest()
        if digest in digests and digests[digest]!=e['split']: errors.append(f'{rid}: exact IC content across splits')
        digests[digest]=e['split']
    similarity=diagnostics(entries,args.root)
    if similarity['exact_cross_split_duplicates']: errors.append('exact IC duplicate across splits')
    report={'examples':len(entries),'passed':len(entries) if not errors else None,'errors':errors,
            'splits':{s:sum(e['split']==s for e in entries) for s in sorted({e['split'] for e in entries})},
            'similarity':{k:v for k,v in similarity.items() if k!='pairs'}}
    print(json.dumps(report,indent=2))
    if errors: raise SystemExit(1)

if __name__=='__main__': main()
