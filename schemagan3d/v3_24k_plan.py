"""Plan and validate 24,000 SchemaGAN V3 realizations; never generate volumes."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from synthetic3d.v2.core import Grid
from synthetic3d.v2.geotechnical_v3 import DEFAULT_RESPONSE
from .v3_adapter import assert_model_shape
from .v3_dataset_production import build_plan as old_build_plan,digest
from .prepare_v3_small_data import design_records
from .v3_split import SPLITS

CONFIG_PATH=Path(__file__).with_name('v3_24k_dataset_config.json')
EXPECTED={'train':16000,'validation_id':4000,'test_id':1000,'test_family':1000,
          'test_geometry_parameter':1000,'test_response_parameter':1000}
RESERVED_ROOTS={'schemagan_v3_prep_small','schemagan_v3_experiment_pilot','schemagan_v3_final_pilot'}
SEED_KEYS=('geology_seed','material_seed','seed','site_seed','evaluation_site_seed')


def _hashint(namespace,label):
    raw=hashlib.sha256(f'{namespace}|{label}'.encode()).digest()
    return int.from_bytes(raw[:8],'big') & ((1<<63)-1)


def existing_ids_seeds():
    old=json.loads(Path(__file__).with_name('v3_dataset_config.json').read_text())
    final=json.loads(Path(__file__).with_name('v3_final_pilot_config.json').read_text())
    groups=[old_build_plan(old,'pilot')['entries'],old_build_plan(old,'full')['entries'],
            old_build_plan(final,'pilot')['entries'],design_records()]
    ids=set();seeds={k:set() for k in SEED_KEYS}
    for group in groups:
        for e in group:
            ids.add(e['realization_id']);ids.add(e['lineage_group'])
            for k in SEED_KEYS:
                if k in e:seeds[k].add(e[k])
    return ids,seeds


def validate_config(c):
    if c.get('schema_version')!='schemagan-v3-24k-config-1' or c.get('plan_schema_version')!='schemagan-v3-24k-plan-1':raise ValueError('schema version')
    if c.get('generator_version')!='v2-prototype-3':raise ValueError('generator version')
    if c.get('target_counts')!=EXPECTED or tuple(c['target_counts'])!=SPLITS:raise ValueError('split counts/order')
    if sum(c['target_counts'].values())!=24000:raise ValueError('total count')
    if tuple(c.get('id_families',()))!=('horizontal','inclined','undulating') or c.get('family_holdout')!='lens':raise ValueError('family design')
    if c.get('realizations_per_lineage_group')!=1:raise ValueError('derivative lineage needs separate review')
    if c['geometry_holdout']!={'family':'inclined','tag':'high_geometry','slope_wave_multiplier':1.6}:raise ValueError('geometry holdout changed')
    if c['response_holdout']!={'family':'horizontal','tag':'high_response','log_qc_sd':0.32,'log_fr_sd':0.30,'log_cross_covariance':-0.012}:raise ValueError('response holdout changed')
    cov=np.array([[.32**2,-.012],[-.012,.30**2]])
    if np.linalg.eigvalsh(cov).min()<=0:raise ValueError('nonpositive response covariance')
    if not np.allclose(DEFAULT_RESPONSE['log_covariance'],[[.24**2,-.012],[-.012,.23**2]],rtol=0,atol=1e-12):raise ValueError('base response prior changed')
    if tuple(c.get('stored_cpt_counts',()))!=(1,2,3,5,8,12,20) or tuple(c.get('evaluation_cpt_counts',()))!=(1,2,3,5,8,12,20):raise ValueError('CPT counts')
    if c.get('n_background_layers')!=3:raise ValueError('layer design')
    grids=c.get('grid_catalog',[])
    if len(grids)!=3 or [g.get('name') for g in grids]!=['g12','g16','g24']:raise ValueError('grid catalog')
    expected_grids=[([12,12,12],[2.,2.,1.]),([16,16,16],[2.,2.,1.]),([24,24,12],[1.,1.,1.])]
    for g,(shape,spacing) in zip(grids,expected_grids):
        if g['shape']!=shape or g['spacing_m']!=spacing or g['origin_xy_m']!=[0.,0.]:raise ValueError('grid changed')
        assert_model_shape(tuple(g['shape']));Grid(*g['shape'],*g['spacing_m'],*g['origin_xy_m'])
        if max(c['stored_cpt_counts'])>g['shape'][0]*g['shape'][1]:raise ValueError('CPT capacity')
    root=Path(c.get('output_root',''))
    if root.is_absolute() or len(root.parts)!=3 or root.parts[:2]!=('data','synthetic3d_v2') or root.name in RESERVED_ROOTS or root.name!='schemagan_v3_24k':raise ValueError('unsafe output path')
    if not isinstance(c.get('seed_namespace'),str) or c['seed_namespace']!='schemagan-v3-24k-independent-realizations-v1':raise ValueError('seed namespace')
    if c.get('write_policy')!='plan_only; production writer disabled pending explicit approval and scale validation':raise ValueError('write policy')
    return {'base_covariance_min_eigenvalue':float(np.linalg.eigvalsh(DEFAULT_RESPONSE['log_covariance']).min()),
            'holdout_covariance_min_eigenvalue':float(np.linalg.eigvalsh(cov).min())}


def build_plan(c):
    validate_config(c)
    namespace=c['seed_namespace'];grids=c['grid_catalog'];families=c['id_families'];entries=[]
    for split_index,(split,n) in enumerate(c['target_counts'].items()):
        # Interleave all nine ID cells, then permute within each nine-cell cycle.
        # Counts differ by at most one, while IDs/seeds are SHA-derived and not arithmetic labels.
        cells=[(f,g) for f in families for g in grids]
        for i in range(n):
            if split in ('train','validation_id','test_id'):
                cycle,position=divmod(i,9)
                order=sorted(range(9),key=lambda j:_hashint(namespace,f'cell|{split}|{cycle}|{j}'))
                family,grid=cells[order[position]];geometry_tag=response_tag='base'
            elif split=='test_family':
                family='lens';grid=grids[i%3];geometry_tag=response_tag='base'
            elif split=='test_geometry_parameter':
                family='inclined';grid=grids[i%3];geometry_tag='high_geometry';response_tag='base'
            else:
                family='horizontal';grid=grids[i%3];geometry_tag='base';response_tag='high_response'
            serial=len(entries)
            rid=f'v3_24k_{_hashint(namespace,f"rid|{serial}"):016x}'
            seeds={k:_hashint(namespace,f'{k}|{rid}') for k in SEED_KEYS}
            entries.append({'realization_id':rid,'scenario_id':f'v3-24k-{split}',
              'lineage_group':rid,'split':split,'family':family,'grid_id':grid['name'],'grid_name':grid['name'],
              'shape':grid['shape'],'spacing_m':grid['spacing_m'],'origin_xy_m':grid['origin_xy_m'],
              'geometry_tag':geometry_tag,'response_tag':response_tag,'generator_version':c['generator_version'],
              'n_cpt':c['stored_cpt_counts'][_hashint(namespace,f'cpt|{rid}')%7],**seeds})
    return {'schema_version':c['plan_schema_version'],'purpose':'frozen plan only; no volume generation',
            'config_sha256':digest(c),'seed_namespace':namespace,'output_root':c['output_root'],
            'target_counts':c['target_counts'],'holdout_definitions':{'family':c['family_holdout'],
              'geometry':c['geometry_holdout'],'response':c['response_holdout']},
            'lineage_rule':'one independent latent realization per lineage group; observation variants remain in the same group',
            'entries':entries}


def validate_plan(plan,c,*,expected_hash=None,check_determinism=True):
    prior=validate_config(c)
    if plan.get('schema_version')!=c['plan_schema_version'] or plan.get('config_sha256')!=digest(c) or plan.get('seed_namespace')!=c['seed_namespace'] or plan.get('output_root')!=c['output_root']:raise ValueError('plan header/config mismatch')
    if plan.get('target_counts')!=EXPECTED or plan.get('holdout_definitions')!={'family':'lens','geometry':c['geometry_holdout'],'response':c['response_holdout']}:raise ValueError('holdout/count metadata mismatch')
    entries=plan.get('entries')
    if not isinstance(entries,list) or len(entries)!=24000:raise ValueError('entry count')
    counts=Counter();cells=defaultdict(Counter);grid_counts=defaultdict(Counter);families=defaultdict(set)
    ids=set();lineages={};seed_sets={k:set() for k in SEED_KEYS};all_seeds=set();tuple_seeds=set();cpt_counts=Counter()
    old_ids,old_seeds=existing_ids_seeds();grid_map={g['name']:g for g in c['grid_catalog']}
    prior_any_seed=set().union(*old_seeds.values())
    for e in entries:
        split=e.get('split');rid=e.get('realization_id');gid=e.get('lineage_group');family=e.get('family');grid_id=e.get('grid_id')
        if split not in EXPECTED:raise ValueError('unknown split')
        if not isinstance(rid,str) or not rid.startswith('v3_24k_') or rid in ids or rid in old_ids:raise ValueError('duplicate or prior realization ID')
        if gid!=rid or gid in old_ids:raise ValueError('lineage mismatch or collision')
        ids.add(rid);lineages.setdefault(gid,set()).add(split)
        if e.get('scenario_id')!=f'v3-24k-{split}' or e.get('generator_version')!=c['generator_version']:raise ValueError('scenario/generator metadata')
        grid=grid_map.get(grid_id)
        if grid is None or e.get('grid_name')!=grid_id or e.get('shape')!=grid['shape'] or e.get('spacing_m')!=grid['spacing_m'] or e.get('origin_xy_m')!=grid['origin_xy_m']:raise ValueError('grid metadata')
        assert_model_shape(tuple(e['shape']))
        if e.get('n_cpt') not in c['stored_cpt_counts'] or e['n_cpt']>grid['shape'][0]*grid['shape'][1]:raise ValueError('CPT count/capacity')
        for k in SEED_KEYS:
            seed=e.get(k)
            if not isinstance(seed,int) or seed<0 or seed in all_seeds or seed in prior_any_seed:raise ValueError(f'seed collision: {k}')
            seed_sets[k].add(seed);all_seeds.add(seed)
        seed_tuple=(e['geology_seed'],e['material_seed'],e['seed'],e['site_seed'])
        if seed_tuple in tuple_seeds:raise ValueError('seed tuple collision')
        tuple_seeds.add(seed_tuple)
        if split in ('train','validation_id','test_id'):
            if family not in c['id_families'] or e.get('geometry_tag')!='base' or e.get('response_tag')!='base':raise ValueError('ID family/tag leakage')
        elif split=='test_family':
            if family!='lens' or e.get('geometry_tag')!='base' or e.get('response_tag')!='base':raise ValueError('family holdout leakage')
        elif split=='test_geometry_parameter':
            if family!='inclined' or e.get('geometry_tag')!='high_geometry' or e.get('response_tag')!='base':raise ValueError('geometry holdout leakage')
        else:
            if family!='horizontal' or e.get('geometry_tag')!='base' or e.get('response_tag')!='high_response':raise ValueError('response holdout leakage')
        counts[split]+=1;cells[split][(family,grid_id)]+=1;grid_counts[split][grid_id]+=1;families[split].add(family);cpt_counts[e['n_cpt']]+=1
    if dict(counts)!=EXPECTED or len(lineages)!=24000 or any(len(x)!=1 for x in lineages.values()):raise ValueError('split/lineage count')
    wanted={(f,g['name']) for f in c['id_families'] for g in c['grid_catalog']}
    for split in ('train','validation_id','test_id'):
        observed=cells[split]
        if set(observed)!=wanted or max(observed.values())-min(observed.values())>1:raise ValueError(f'family-grid imbalance: {split}')
    for split,family in (('test_family','lens'),('test_geometry_parameter','inclined'),('test_response_parameter','horizontal')):
        if families[split]!={family} or set(grid_counts[split])!=set(grid_map) or max(grid_counts[split].values())-min(grid_counts[split].values())>1:raise ValueError(f'holdout distribution: {split}')
    if families['train']!=set(c['id_families']) or families['validation_id']!=families['train'] or families['test_id']!=families['train']:raise ValueError('ID family mismatch')
    actual_hash=digest(plan)
    if expected_hash is not None and actual_hash!=expected_hash:raise ValueError('canonical SHA-256 mismatch')
    if check_determinism and plan!=build_plan(c):raise ValueError('plan is not deterministic from frozen config')
    return {'schema_version':'schemagan-v3-24k-validation-1','valid':True,'total':len(entries),
      'split_counts':{s:counts[s] for s in SPLITS},
      'family_grid_by_split':{s:{f'{f}|{g}':n for (f,g),n in sorted(cells[s].items())} for s in SPLITS},
      'grid_counts_by_split':{s:{g:grid_counts[s][g] for g in grid_map} for s in SPLITS},
      'stored_cpt_counts':{str(k):cpt_counts[k] for k in c['stored_cpt_counts']},
      'lineage_groups':len(lineages),'unique_seed_counts':{k:len(v) for k,v in seed_sets.items()},
      'unique_seed_values_across_all_fields':len(all_seeds),
      'no_collision_with_previous_plans':True,'positive_covariance':prior,
      'output_root':c['output_root'],'output_root_exists':Path(c['output_root']).exists(),
      'canonical_plan_sha256':actual_hash,'deterministic_rebuild_equal':check_determinism,
      'volume_generation_performed':False}


def estimate_resources(plan,c,benchmark,prior_benchmark=None,shape_benchmark=None):
    grids=Counter(e['grid_id'] for e in plan['entries'])
    per=benchmark['per_grid'];payload=sum(grids[g]*per[g]['stored_bytes_mean'] for g in grids)
    core_seconds=sum(grids[g]*per[g]['generation_seconds_mean'] for g in grids)
    pilot_count=54;pilot_allocated=benchmark['actual_allocated_kib_du']*1024;pilot_payload=benchmark['payload_bytes_54']
    allocated=payload*(pilot_allocated/pilot_payload)
    n=len(plan['entries']);manifest_size=benchmark['metadata_only_manifest_simulation'][-1]['final_bytes']/800*n
    plan_size=len((json.dumps(plan,indent=2)+'\n').encode())
    sim=benchmark['metadata_only_manifest_simulation'][-1]
    manifest_quadratic=sim['seconds']*(n/sim['entries'])**2
    validation=benchmark['full_validation_seconds_separate_process']*n/pilot_count
    prior_projection=None
    if prior_benchmark:
        prior_per=prior_benchmark['benchmark_by_grid']
        prior_projection={'payload_bytes':sum(grids[g]*prior_per[g]['stored_bytes_mean'] for g in grids),
                          'generation_and_save_seconds_only':sum(grids[g]*prior_per[g]['generation_seconds_mean'] for g in grids)}
    return {'schema_version':'schemagan-v3-24k-resource-estimate-1','plan_sha256':digest(plan),
       'pilot_benchmark_sha256':digest(benchmark),'prior_pilot_benchmark_sha256':digest(prior_benchmark) if prior_benchmark else None,
       'grid_counts':dict(grids),'per_grid_pilot_measured':{g:per[g] for g in grids},
       'measured_final_pilot':{'volumes':54,'payload_bytes':pilot_payload,'allocated_bytes':pilot_allocated,
         'file_count':benchmark['actual_file_count'],'writer_function_end_to_end_seconds':benchmark['production_function_end_to_end_seconds_including_manifest_and_integrity_check'],
         'writer_process_wall_seconds':benchmark['process_wall_seconds_from_time_utility'],
         'manifest_write_seconds_54':benchmark['manifest_write_seconds_for_54'],
         'full_validation_seconds':benchmark['full_validation_seconds_separate_process'],
         'temporary_g24_writer_peak_rss_bytes':benchmark['temporary_g24_writer_peak_rss_bytes'],
         'full_regeneration_peak_rss_bytes':benchmark['peak_rss_bytes_during_full_regeneration_and_validation']},
       'projection_24k':{'payload_bytes':payload,'allocated_bytes_ratio_based':allocated,
         'volume_data_files':n*18,'volume_sidecar_files':n,
         'total_volume_files_including_sidecars':n*19,
         'final_manifest_bytes_linear_from_800_metadata_entries':manifest_size,
         'frozen_plan_bytes_actual':plan_size,
         'approximate_total_dataset_bytes_including_manifest_and_plan':allocated+manifest_size+plan_size,
         'backup_copy_bytes_ratio_based':allocated+manifest_size+plan_size,
         'minimum_free_bytes_for_dataset_plus_one_backup_and_manifest_temp':2*(allocated+manifest_size+plan_size)+manifest_size,
         'generation_and_save_seconds_per_grid_only':core_seconds,
         'full_validation_seconds_linear_hot_cache':validation,
         'manifest_rewrite_seconds_quadratic_metadata_simulation':manifest_quadratic,
         'manifest_cumulative_serialized_bytes_quadratic':manifest_size*(n+1)/2,
         'indicative_generation_validation_manifest_seconds_not_reliable':core_seconds+validation+manifest_quadratic},
       'manifest_simulation_measured':benchmark['metadata_only_manifest_simulation'],
       'prior_45_volume_pilot_sensitivity':prior_projection,
       'memory_assessment':{'largest_grid':'g24','temporary_g24_writer_peak_rss_bytes':benchmark['temporary_g24_writer_peak_rss_bytes'],
          'full_regeneration_peak_rss_bytes':benchmark['peak_rss_bytes_during_full_regeneration_and_validation'],
          'temporary_writer_per_grid':shape_benchmark,
          'large_run_peak_unmeasured':True,'per_shape_peak_available':{g:bool(shape_benchmark and g in shape_benchmark) for g in grids}},
       'limits':['All 24k figures are extrapolations; no production volumes were generated.',
         'Metadata-only manifest simulation is not a 24k production benchmark; O(n^2) rewrite likely dominates.',
         'Cold IO, filesystem allocation, backup/snapshot, long-run RAM, and interruption cost are not reliably estimated.',
         'Existing CLI cannot produce 24k; single-writer incremental manifest/journal needs implementation and crash-resume tests before production.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=CONFIG_PATH)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--prior-benchmark',type=Path)
    p.add_argument('--shape-benchmark',type=Path)
    a=p.parse_args();c=json.loads(a.config.read_text());validate_config(c)
    names=('SCHEMAGAN_V3_24K_SPLIT_PLAN.json','SCHEMAGAN_V3_24K_PLAN_VALIDATION.json','SCHEMAGAN_V3_24K_RESOURCE_ESTIMATE.json')
    if any((a.output_dir/name).exists() for name in names):raise FileExistsError('24K deliverable already exists')
    if Path(c['output_root']).exists():raise FileExistsError('24K output root already exists')
    benchmark=json.loads(a.benchmark.read_text())
    prior=json.loads(a.prior_benchmark.read_text()) if a.prior_benchmark else None
    shape=json.loads(a.shape_benchmark.read_text()) if a.shape_benchmark else None
    plan=build_plan(c);validation=validate_plan(plan,c,check_determinism=True)
    resources=estimate_resources(plan,c,benchmark,prior,shape)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for name,value in zip(names,(plan,validation,resources)):
        (a.output_dir/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'total':validation['total'],'canonical_plan_sha256':validation['canonical_plan_sha256'],
                      'output_root_created':False,'production_volumes_created':False}))
if __name__=='__main__':main()
