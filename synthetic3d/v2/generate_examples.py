"""Generate eight small, non-training V2 examples in a fresh directory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .core import Grid, generate, extract_cpt, validate, validate_reproducibility, assign_split, validate_manifest, save_volume

SPECS = [
 ('horizontal',2,Grid(16,16,8,2,2,1),1,'standard'),
 ('horizontal',3,Grid(12,10,12,3,2,0.75,100,200,15,45),3,'standard'),
 ('inclined',3,Grid(18,12,12,2,2,1),1,'standard'),
 ('inclined',4,Grid(14,14,16,1.5,2,0.8),4,'high_geometry'),
 ('undulating',3,Grid(20,12,12,2,3,1),1,'standard'),
 ('undulating',4,Grid(12,16,16,2,1.5,0.7),3,'high_geometry'),
 ('lens',3,Grid(18,18,12,2,2,1),1,'standard'),
 ('lens',4,Grid(16,12,16,2,2.5,0.8),4,'standard'),
]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('data/synthetic3d_v2/prototype'))
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    entries=[]
    for i,(family,n_layers,grid,n_cpt,parameter_family) in enumerate(SPECS):
        params={'thickness_weights':[1.0]*n_layers,'slope_fraction':.18 if parameter_family=='high_geometry' else .12,
                'direction_rad':.5,'amplitude_fraction':.055 if parameter_family=='high_geometry' else .045,
                'wavelength_fraction':.8,'phase_rad':.3}
        realization_id = f'v2_example_{i:02d}'
        volume=generate(grid,family,n_layers,seed=100+i,scenario_id=f'{family}-{n_layers}',
                        parameter_family_id=parameter_family,realization_id=realization_id,params=params)
        sites=[(grid.nx//2,grid.ny//2),(2,2),(grid.nx-3,grid.ny-3),(2,grid.ny-3)][:n_cpt]
        observation=extract_cpt(volume,sites)
        errors=validate(volume,observation)+validate_reproducibility(volume)
        if errors: raise ValueError(f'example {i}: {errors}')
        split=assign_split(volume['metadata'])
        entries.append(save_volume(volume,observation,args.output,split))
        print(i,family,grid.nx,grid.ny,grid.nz,'Ic',round(float(np.mean(volume['Ic'])),3),'split',split,flush=True)
    errors=validate_manifest(entries)
    if errors: raise ValueError(errors)
    (args.output/'manifest.json').write_text(json.dumps({'schema_version':'v2-prototype-1','purpose':'functional validation only; no model training',
                                                          'split_policy':{'family_holdout':'lens','parameter_holdout':'high_geometry',
                                                                          'remaining':'SHA-256 version|scenario_id|realization_id buckets: 70% train,20% validation,10% test_id'},
                                                          'examples':entries},indent=2)+'\n')
    print('validated',len(entries),'examples')

if __name__=='__main__': main()
