"""Create a non-overwriting, small six-split V3 contract-test dataset only."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from synthetic3d.v2.core import Grid,extract_cpt
from synthetic3d.v2.geotechnical_v3 import DEFAULT_RESPONSE,generate,save,validate,fingerprint
from synthetic3d.v2.refined import default_geometry
from .v3_split import prepare_split_manifest,SPLITS
from .v3_eval_protocol import nested_sites

DEFAULT_ROOT=Path('data/synthetic3d_v2/schemagan_v3_prep_small')
SEED=20261019

def design_records():
    out=[]
    for fam_idx,family in enumerate(('horizontal','inclined','undulating')):
        for rep in range(3):
            gid=f'id-{family}-{rep}'
            count=2 if family=='horizontal' and rep==0 else 1
            for derivative in range(count):
                i=len(out);out.append({'realization_id':f'prep_v3_{i:02d}','lineage_group':gid,'family':family,
                  'geometry_tag':'base','response_tag':'base','geology_seed':100+10*fam_idx+rep,
                  'material_seed':300+10*fam_idx+rep+derivative*1000,'seed':500+i,
                  'shape':[12,12,12] if rep!=2 else [16,16,16],
                  'spacing_m':[2.,2.,1.] if rep!=1 else [1.5,1.5,1.],
                  'n_cpt':1+(i%5),'site_seed':7000+i})
    for family,tag,group,geo_seed,mat_seed in (
        ('lens','family','holdout-lens',801,901),
        ('inclined','geometry','holdout-geometry',802,902),
        ('horizontal','response','holdout-response',803,903)):
        for derivative in range(2):
            i=len(out);out.append({'realization_id':f'prep_v3_{i:02d}','lineage_group':group,'family':family,
             'geometry_tag':'high_geometry' if tag=='geometry' else 'base',
             'response_tag':'high_response' if tag=='response' else 'base',
             'geology_seed':geo_seed,'material_seed':mat_seed+derivative,'seed':500+i,
             'shape':[12,12,12],'spacing_m':[2.,2.,1.],'n_cpt':1+(i%5),'site_seed':7000+i})
    return out

def _custom_settings(r):
    response=copy.deepcopy(DEFAULT_RESPONSE)
    if r['response_tag']=='high_response':
        # Predeclared variance holdout, not selected after viewing test data.
        response['log_covariance']=[[.32**2,-.012],[-.012,.30**2]]
    geometry=None
    if r['geometry_tag']=='high_geometry':
        geometry=default_geometry(3,r['family'])
        geometry['common_slope_fraction']*=1.6
        geometry['boundary_slope_fraction']=[v*1.6 for v in geometry['boundary_slope_fraction']]
        geometry['boundary_wave_fraction']=[v*1.6 for v in geometry['boundary_wave_fraction']]
    return geometry,response

def build():
    records=design_records()
    plan=prepare_split_manifest(records,heldout_family='lens',heldout_geometry_tag='high_geometry',heldout_response_tag='high_response')
    built=[];seen={}
    for r in plan['entries']:
        grid=Grid(*r['shape'],*r['spacing_m'])
        geometry,response=_custom_settings(r)
        v=generate(grid,r['family'],3,r['seed'],scenario_id=r['lineage_group'],realization_id=r['realization_id'],
                   geometry=geometry,response=response,geology_seed=r['geology_seed'],material_seed=r['material_seed'])
        sites=nested_sites(grid,r['site_seed'],v['valid_mask'],r['n_cpt'])
        obs=extract_cpt(v,sites)
        errors=validate(v,obs)
        if errors:raise ValueError((r['realization_id'],errors))
        fp=fingerprint(v)
        if fp in seen:raise ValueError(f'duplicate volume fingerprint: {r["realization_id"]}, {seen[fp]}')
        seen[fp]=r['realization_id'];built.append((r,v,obs))
    return plan,built

def validate_small_manifest(root:Path):
    manifest=json.loads((root/'manifest.json').read_text());entries=manifest['entries']
    if manifest['schema_version']!='v2-prototype-3' or set(e['split'] for e in entries)!=set(SPLITS):raise ValueError('six-split coverage missing')
    groups={};ids=set();fps=set();seed_splits={};train_families={e['family'] for e in entries if e['split']=='train'}
    for e in entries:
        if e['realization_id'] in ids or e['fingerprint'] in fps:raise ValueError('duplicate ID or fingerprint')
        ids.add(e['realization_id']);fps.add(e['fingerprint']);groups.setdefault(e['lineage_group'],set()).add(e['split'])
        seed_key=(e['geology_seed'],e['material_seed'])
        if seed_key in seed_splits and seed_splits[seed_key]!=e['split']:raise ValueError('related seed tuple crosses splits')
        seed_splits[seed_key]=e['split']
        if e['split'] in ('validation_id','test_id') and e['family'] not in train_families:raise ValueError('ID family not in train')
        if e['family']=='lens' and e['split']!='test_family':raise ValueError('lens family leakage')
        if e['split']=='test_response_parameter' and e['response_tag']!='high_response':raise ValueError('response holdout tag mismatch')
        if e['split']=='test_geometry_parameter' and e['geometry_tag']!='high_geometry':raise ValueError('geometry holdout tag mismatch')
        volume={k:np.load(root/p,allow_pickle=False) for k,p in e['arrays'].items()}
        volume['metadata']=json.loads((root/e['metadata']).read_text())
        observation={'profiles':json.loads((root/e['profiles']).read_text()),
                     'observed_mask':np.load(root/e['observed_mask'],allow_pickle=False)}
        if fingerprint(volume)!=e['fingerprint'] or validate(volume,observation):raise ValueError('stored volume failed fingerprint/physics/observation checks')
        m=volume['metadata']
        if (m['geology_seed'],m['material_seed'],m['seed'])!=(e['geology_seed'],e['material_seed'],e['seed']):raise ValueError('seed metadata mismatch')
        geometry,response=_custom_settings(e)
        replica=generate(Grid(**m['grid']),m['family'],m['n_background_layers'],m['seed'],scenario_id=m['scenario_id'],
                         realization_id=m['realization_id'],geometry=geometry,response=response,
                         geology_seed=m['geology_seed'],material_seed=m['material_seed'])
        if any(not np.array_equal(replica[k],volume[k]) for k in e['arrays']):raise ValueError('seed/config reproduction mismatch')
    if any(len(x)!=1 for x in groups.values()):raise ValueError('lineage group crosses splits')
    return {'volumes':len(entries),'groups':len(groups),'splits':{s:sum(e['split']==s for e in entries) for s in SPLITS}}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT_ROOT);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    plan,built=build()
    if len(built)>60:raise ValueError('small-data limit exceeded')
    args.output.mkdir(parents=True)
    entries=[]
    for r,v,o in built:
        e=save(v,o,args.output,r['split'],r['lineage_group'],r['realization_id'])
        e.update({k:r[k] for k in ('lineage_group','geometry_tag','response_tag','geology_seed','material_seed','seed','site_seed')})
        entries.append(e)
    manifest={'schema_version':'v2-prototype-3','purpose':'training pipeline contract tests only; no training',
      'design_seed':SEED,'split_plan_version':plan['schema_version'],'holdouts':{'family':'lens','geometry_tag':'high_geometry','response_tag':'high_response'},
      'response_holdout_definition':'log covariance SD changes from 0.24/0.23 to 0.32/0.30; offdiagonal stays -0.012',
      'geometry_holdout_definition':'common and boundary slope/wave fractions multiplied by 1.6 on inclined family',
      'entries':entries}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary=validate_small_manifest(args.output)
    print(json.dumps(summary))
if __name__=='__main__':main()
