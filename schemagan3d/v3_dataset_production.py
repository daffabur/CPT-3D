"""Frozen SchemaGAN V3 dataset plan, atomic pilot generation, validation and resume.

No full-dataset generation mode is exposed. No model training occurs here.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import numpy as np
from synthetic3d.v2.core import Grid,extract_cpt
from synthetic3d.v2.geotechnical_v3 import DEFAULT_RESPONSE,GROUPS,generate,save,validate,fingerprint
from synthetic3d.v2.refined import default_geometry
from .v3_adapter import assert_model_shape
from .v3_eval_protocol import nested_sites
from .v3_split import SPLITS

CONFIG_PATH=Path(__file__).with_name('v3_dataset_config.json')
ARRAY_KEYS=('Ic','unit_id','material_group','sbt_id','valid_mask','qc_mpa','fs_mpa','qt_mpa','Fr_percent','Qt','sigma_v0_mpa','sigma_v0_eff_mpa')

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'))
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def config_checks(c):
    if c['schema_version']!='schemagan-v3-dataset-pilot-1' or c['generator_version']!='v2-prototype-3':raise ValueError('unsupported version')
    if tuple(c['full_target_counts'])!=SPLITS or tuple(c['pilot_counts'])!=SPLITS:raise ValueError('split order mismatch')
    if sum(c['full_target_counts'].values())!=3400 or sum(c['pilot_counts'].values())>60:raise ValueError('count limit')
    if c.get('pilot_family_grid_cross') and tuple(c['pilot_counts'].values())!=(18,9,9,6,6,6):
        raise ValueError('final pilot requires 18/9/9 ID and 6/6/6 holdout volumes')
    if min(c['pilot_counts'].values())<3 or tuple(c['id_families'])!=('horizontal','inclined','undulating') or c['family_holdout']!='lens':raise ValueError('family design')
    if len(c['grid_catalog'])!=3:raise ValueError('three grids required')
    for g in c['grid_catalog']:
        assert_model_shape(tuple(g['shape']))
        if min(g['spacing_m'])<=0:raise ValueError('spacing')
    cov=np.array([[c['response_holdout']['log_qc_sd']**2,c['response_holdout']['log_cross_covariance']],
                  [c['response_holdout']['log_cross_covariance'],c['response_holdout']['log_fr_sd']**2]])
    if np.linalg.eigvalsh(cov).min()<=0:raise ValueError('response covariance not positive definite')
    if c['geometry_holdout']['slope_wave_multiplier']!=1.6:raise ValueError('geometry design changed')
    if tuple(c['stored_cpt_counts'])!=(1,2,3,5,8,12,20) or tuple(c['evaluation_cpt_counts'])!=(1,2,3,5,8,12,20):raise ValueError('CPT design changed')
    return cov

def build_plan(c,kind='pilot'):
    config_checks(c)
    if kind not in ('pilot','full'):raise ValueError('unknown plan')
    counts=c['pilot_counts'] if kind=='pilot' else c['full_target_counts']
    entries=[];serial=0
    for split,n in counts.items():
        for i in range(n):
            if split in ('train','validation_id','test_id'):
                if kind=='pilot' and c.get('pilot_family_grid_cross'):
                    repeats=2 if split=='train' else 1
                    pair=i//repeats
                    family=c['id_families'][pair//3]
                    grid_index=pair%3
                else:
                    family=c['id_families'][i%3]
                    grid_index=(i//3)%3 if kind=='full' else i%3
                gtag=rtag='base'
            elif split=='test_family':family='lens';gtag=rtag='base';grid_index=i%3
            elif split=='test_geometry_parameter':family='inclined';gtag='high_geometry';rtag='base';grid_index=i%3
            else:family='horizontal';gtag='base';rtag='high_response';grid_index=i%3
            grid=c['grid_catalog'][grid_index]
            # Each lineage is an independent generator realization; all seed namespaces are disjoint.
            rid=(f'finalpilot_v3_{serial:05d}' if kind=='pilot' and c.get('pilot_family_grid_cross')
                 else f'{kind}_v3_{serial:05d}')
            base=int(c['seed_namespace'])*10000+(100000000 if kind=='full' else 0)+serial*10
            n_cpt=c['stored_cpt_counts'][serial%len(c['stored_cpt_counts'])]
            scenario_prefix='finalpilot' if kind=='pilot' and c.get('pilot_family_grid_cross') else kind
            entry={'realization_id':rid,'scenario_id':f'{scenario_prefix}-{split}',
              'lineage_group':rid,'family':family,'geometry_tag':gtag,'response_tag':rtag,'split':split,
              'geology_seed':base+1,'material_seed':base+2,'seed':base+3,'site_seed':base+4,
              'evaluation_site_seed':int(c['evaluation_site_seed_namespace'])+serial,
              'grid_name':grid['name'],'shape':grid['shape'],'spacing_m':grid['spacing_m'],
              'origin_xy_m':grid['origin_xy_m'],'n_cpt':n_cpt,'generator_version':c['generator_version']}
            entries.append(entry);serial+=1
    return {'schema_version':'schemagan-v3-frozen-plan-1','kind':kind,'config_sha256':digest(c),
            'count':len(entries),'entries':entries}

def settings(e,c):
    geometry=None;response=copy.deepcopy(DEFAULT_RESPONSE)
    if e['geometry_tag']=='high_geometry':
        geometry=default_geometry(c['n_background_layers'],e['family'])
        m=c['geometry_holdout']['slope_wave_multiplier']
        geometry['common_slope_fraction']*=m
        geometry['boundary_slope_fraction']=[x*m for x in geometry['boundary_slope_fraction']]
        geometry['boundary_wave_fraction']=[x*m for x in geometry['boundary_wave_fraction']]
    if e['response_tag']=='high_response':
        s=c['response_holdout'];response['log_covariance']=[[s['log_qc_sd']**2,s['log_cross_covariance']],
                                                               [s['log_cross_covariance'],s['log_fr_sd']**2]]
    return geometry,response

def grid_for(e):
    return Grid(*e['shape'],*e['spacing_m'],*e['origin_xy_m'])

def make_volume(e,c):
    g=grid_for(e);geometry,response=settings(e,c)
    v=generate(g,e['family'],c['n_background_layers'],e['seed'],scenario_id=e['scenario_id'],
               realization_id=e['realization_id'],geology_seed=e['geology_seed'],material_seed=e['material_seed'],
               geometry=geometry,response=response)
    sites=nested_sites(g,e['site_seed'],v['valid_mask'],e['n_cpt'])
    obs=extract_cpt(v,sites)
    if len(sites)!=e['n_cpt'] or len(set(sites))!=len(sites):raise ValueError('CPT site count/uniqueness')
    errors=validate(v,obs)
    if errors:raise ValueError((e['realization_id'],errors))
    return v,obs

def _atomic_json(path,value):
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        temp.write_text(json.dumps(value,indent=2)+'\n')
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)

def _files(entry):
    return list(entry['arrays'].values())+[entry['metadata'],entry['profiles'],entry['observed_mask']]

def write_one(root,e,c):
    rid=e['realization_id'];final=root/rid
    if final.exists():raise FileExistsError(final)
    stage=root/('.staging-'+rid+'-'+uuid.uuid4().hex)
    stage.mkdir()
    started=time.perf_counter()
    try:
        v,obs=make_volume(e,c)
        saved=save(v,obs,stage,e['split'],e['lineage_group'],rid)
        staged=stage/rid
        # Move complete directory atomically; manifest entry follows only after rename.
        entry={**e,**saved}
        entry.update({k:e[k] for k in ('lineage_group','geometry_tag','response_tag','geology_seed','material_seed','seed','site_seed','evaluation_site_seed','generator_version','origin_xy_m','grid_name','scenario_id')})
        entry['file_sha256']={p:file_sha(stage/p) for p in _files(entry)}
        entry['generation_seconds']=time.perf_counter()-started
        entry['stored_bytes']=sum((stage/p).stat().st_size for p in _files(entry))
        (staged/'_entry.json').write_text(json.dumps(entry,indent=2)+'\n')
        os.rename(staged,final)
        return entry
    finally:shutil.rmtree(stage,ignore_errors=True)

def _read_saved(root,e):
    for rel,sha in e['file_sha256'].items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file() or file_sha(p)!=sha:raise ValueError(f'file hash mismatch: {rel}')
    v={k:np.load(root/p,allow_pickle=False) for k,p in e['arrays'].items()}
    v['metadata']=json.loads((root/e['metadata']).read_text())
    o={'profiles':json.loads((root/e['profiles']).read_text()),'observed_mask':np.load(root/e['observed_mask'],allow_pickle=False)}
    return v,o

def validate_entry(root,e,planned,c,reproduce=True):
    for key,value in planned.items():
        if e.get(key)!=value:raise ValueError(f'planned field changed: {key}')
    if set(e['arrays'])<set(ARRAY_KEYS):raise ValueError('required array missing')
    v,o=_read_saved(root,e)
    if fingerprint(v)!=e['fingerprint'] or validate(v,o):raise ValueError('volume physics or fingerprint invalid')
    if tuple(v['Ic'].shape)!=tuple(e['shape']) or len(o['profiles'])!=e['n_cpt']:raise ValueError('shape/profile count mismatch')
    if len({tuple(p['grid_xy']) for p in o['profiles']})!=len(o['profiles']):raise ValueError('duplicate CPT site')
    if reproduce:
        replica,robs=make_volume(planned,c)
        if fingerprint(replica)!=e['fingerprint'] or any(not np.array_equal(replica[k],v[k]) for k in e['arrays']):raise ValueError('seed reproduction mismatch')
        if not np.array_equal(robs['observed_mask'],o['observed_mask']) or canonical(robs['profiles'])!=canonical(o['profiles']):raise ValueError('observation reproduction mismatch')
    return v,o

def validate_dataset(root,c,reproduce=True):
    plan=build_plan(c,'pilot');saved_plan=json.loads((root/'plan.json').read_text())
    if saved_plan!=plan:raise ValueError('frozen plan/config mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['config_sha256']!=plan['config_sha256'] or manifest['plan_sha256']!=digest(plan):raise ValueError('manifest provenance mismatch')
    by_id={e['realization_id']:e for e in plan['entries']}
    if len(manifest['entries'])!=len({e['realization_id'] for e in manifest['entries']}):raise ValueError('duplicate ID')
    groups={};fps=set();seeds=set();totals={s:0 for s in SPLITS}
    for e in manifest['entries']:
        p=by_id.get(e['realization_id'])
        if p is None:raise ValueError('unplanned volume')
        v,o=validate_entry(root,e,p,c,reproduce)
        groups.setdefault(e['lineage_group'],set()).add(e['split'])
        if e['fingerprint'] in fps:raise ValueError('duplicate fingerprint')
        fps.add(e['fingerprint'])
        key=(e['geology_seed'],e['material_seed'])
        if key in seeds:raise ValueError('reused latent seed pair')
        seeds.add(key);totals[e['split']]+=1
    if any(len(s)!=1 for s in groups.values()):raise ValueError('cross-split lineage')
    orphan=[p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.staging-') and p.name not in {e['realization_id'] for e in manifest['entries']}]
    if orphan:raise ValueError(f'unindexed directory: {orphan[:3]}')
    return {'volumes':len(manifest['entries']),'splits':totals,'reproduced':reproduce,'fingerprints_unique':len(fps)}

def initialize(root,c):
    if root.exists():raise FileExistsError(root)
    config_checks(c);plan=build_plan(c,'pilot')
    root.mkdir(parents=True)
    _atomic_json(root/'config.json',c)
    _atomic_json(root/'plan.json',plan)
    _atomic_json(root/'manifest.json',{'schema_version':'schemagan-v3-pilot-manifest-1','config_sha256':digest(c),
                                      'plan_sha256':digest(plan),'entries':[]})
    return plan

def _recover_orphans(root,manifest,plan,c):
    planned={e['realization_id']:e for e in plan['entries']}
    indexed={e['realization_id'] for e in manifest['entries']}
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.staging-')):
        if folder.name in indexed:continue
        if folder.name not in planned:raise ValueError(f'unplanned directory: {folder.name}')
        sidecar=folder/'_entry.json'
        if not sidecar.is_file():raise ValueError(f'unindexed directory has no recovery record: {folder.name}')
        entry=json.loads(sidecar.read_text())
        validate_entry(root,entry,planned[folder.name],c,reproduce=False)
        manifest['entries'].append(entry)
        _atomic_json(root/'manifest.json',manifest)
    return manifest

def produce(root,c,limit=None):
    started=time.perf_counter();manifest_write_seconds=0.0
    if root.exists():
        frozen=json.loads((root/'config.json').read_text())
        if frozen!=c:raise ValueError('cannot resume with altered config')
        plan=json.loads((root/'plan.json').read_text())
        if plan!=build_plan(c,'pilot'):raise ValueError('cannot resume with altered plan')
        manifest=json.loads((root/'manifest.json').read_text())
        _recover_orphans(root,manifest,plan,c)
        validate_dataset(root,c,reproduce=False)
    else:plan=initialize(root,c)
    manifest=json.loads((root/'manifest.json').read_text())
    complete={e['realization_id'] for e in manifest['entries']}
    remaining=[e for e in plan['entries'] if e['realization_id'] not in complete]
    if limit is not None:remaining=remaining[:limit]
    for e in remaining:
        if (root/e['realization_id']).exists():raise ValueError('unindexed complete directory; manual recovery required')
        saved=write_one(root,e,c)
        manifest['entries'].append(saved)
        write_started=time.perf_counter()
        _atomic_json(root/'manifest.json',manifest)
        manifest_write_seconds+=time.perf_counter()-write_started
    result=validate_dataset(root,c,reproduce=False)
    result['end_to_end_seconds']=time.perf_counter()-started
    result['manifest_write_seconds']=manifest_write_seconds
    result['new_volumes']=len(remaining)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('plan','pilot','validate'))
    p.add_argument('--config',type=Path,default=CONFIG_PATH)
    p.add_argument('--root',type=Path)
    p.add_argument('--limit',type=int)
    a=p.parse_args();c=json.loads(a.config.read_text());config_checks(c)
    root=a.root or Path(c['output_root'])
    if a.action=='plan':
        plan=build_plan(c,'full');pilot=build_plan(c,'pilot')
        print(json.dumps({'full_counts':c['full_target_counts'],'full_plan_sha256':digest(plan),
                          'pilot_counts':c['pilot_counts'],'pilot_plan_sha256':digest(pilot)}))
    elif a.action=='pilot':
        if a.limit is not None and (a.limit<0 or a.limit>60):raise ValueError('invalid pilot limit')
        print(json.dumps(produce(root,c,a.limit)))
    else:
        frozen=json.loads((root/'config.json').read_text())
        if frozen!=c:raise ValueError('config mismatch')
        print(json.dumps(validate_dataset(root,c,reproduce=True)))
if __name__=='__main__':main()
