"""Create at most ten small version-2 examples; never overwrite earlier prototype."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .core import Grid, extract_cpt, save_volume
from .refined import (SCHEMA_VERSION, default_geometry, default_response, generate,
                      validate, validate_reproducibility, split, validate_manifest,
                      diagnostics, response_diagnostics)

SPECS=[
 ('horizontal',2,Grid(16,16,8,2,2,1),'standard',1,'common_shift'),
 ('horizontal',4,Grid(12,14,16,2.5,1.5,.8),'high_response',3,'varying'),
 ('inclined',3,Grid(18,12,14,2,2,1),'standard',1,'varying'),
 ('inclined',5,Grid(16,16,20,1.5,2,.8),'high_geometry',4,'varying'),
 ('undulating',3,Grid(20,12,12,2,3,1),'standard',1,'varying'),
 ('undulating',5,Grid(16,16,20,2,2,.8),'high_geometry',3,'varying'),
 ('lens',3,Grid(18,18,12,2,2,1),'standard',1,'varying'),
 ('lens',4,Grid(20,16,16,2,2.5,1),'standard',4,'varying'),
 ('lens',5,Grid(20,16,20,2,2,.8),'standard',3,'varying'),
 ('horizontal',3,Grid(16,16,12,2,2,1),'standard',2,'varying'),
]

def build(i,spec):
    family,n_layers,grid,parameter_family,n_cpt,mode=spec
    geom=default_geometry(n_layers,family,mode=mode)
    geom['thickness_weights']=[1.0+.13*((j+i)%3) for j in range(n_layers)]
    geom['slope_direction_rad']+=.15*i
    geom['boundary_phase_rad']=[v+.12*i for v in geom['boundary_phase_rad']]
    if parameter_family=='high_geometry':
        geom['common_slope_fraction']*=1.08
        geom['common_wave_fraction']*=1.15
        geom['boundary_wave_fraction']=[v*1.15 for v in geom['boundary_wave_fraction']]
    if family=='lens':
        lens=geom['lens']; lens['parent_unit']=[0,2,1][i-6]
        lens['center_x_fraction']=[.42,.57,.48][i-6]
        lens['center_y_fraction']=[.53,.43,.55][i-6]
        lens['center_z_fraction_in_parent']=[.50,.52,.48][i-6]
        lens['radius_x_fraction_domain']=[.16,.20,.18][i-6]
        lens['radius_y_fraction_domain']=[.17,.16,.19][i-6]
        lens['radius_z_fraction_parent']=[.34,.31,.33][i-6]
        lens['azimuth_deg']=[0,35,-25][i-6]
    response=default_response(n_layers+(family=='lens'),seed=800+i)
    if parameter_family=='high_response':
        response['log_qc_field_sd']=.32
        response['log_fr_field_sd']=.30
    # Different realizations draw their own unit parameters, not a fixed unit-id lookup.
    volume=generate(grid,family,n_layers,seed=200+i,scenario_id=f'{family}-{n_layers}',
                    parameter_family_id=parameter_family,realization_id=f'refined_{i:02d}',
                    geometry=geom,response=response)
    sites=[(grid.nx//2,grid.ny//2),(2,2),(grid.nx-3,grid.ny-3),(2,grid.ny-3)][:n_cpt]
    observation=extract_cpt(volume,sites)
    errors=validate(volume,observation)+validate_reproducibility(volume)
    if errors: raise ValueError(f'example {i}: {errors}')
    return volume,observation

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('data/synthetic3d_v2/prototype_refined'))
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if len(SPECS)>12: raise ValueError('prototype limit exceeded')
    # Preflight all configurations before creating the output directory.
    built=[build(i,spec) for i,spec in enumerate(SPECS)]
    args.output.mkdir(parents=True)
    entries=[]
    for i,(volume,observation) in enumerate(built):
        e=save_volume(volume,observation,args.output,split(volume['metadata']))
        e['response_diagnostics']=response_diagnostics(volume)
        boundaries=volume['boundaries_m']; zmax=volume['metadata']['domain_m'][2]
        thickness=np.concatenate((boundaries[:1],np.diff(boundaries,axis=0),zmax-boundaries[-1:]),axis=0)
        e['thickness_m_range']=[float(thickness.min()),float(thickness.max())]
        e['lateral_thickness_range_m']=float(max(np.ptp(layer) for layer in thickness))
        e['qc_mpa_range']=[float(volume['qc_mpa'].min()),float(volume['qc_mpa'].max())]
        e['fs_mpa_range']=[float(volume['fs_mpa'].min()),float(volume['fs_mpa'].max())]
        e['lens']=volume['metadata']['lens']
        entries.append(e)
    errors=validate_manifest(entries)
    if errors: raise ValueError(errors)
    (args.output/'manifest.json').write_text(json.dumps({'schema_version':SCHEMA_VERSION,'purpose':'functional validation only; no model training',
        'split_policy':{'family_holdout':'lens','parameter_holdout':['high_geometry','high_response'],
                        'remaining':'stable SHA-256 bucket of schema/scenario/realization'},
        'examples':entries},indent=2)+'\n')
    (args.output/'similarity_diagnostics.json').write_text(json.dumps(diagnostics(entries,args.output),indent=2)+'\n')
    print(f'created and validated {len(entries)} refined examples')

if __name__=='__main__': main()
