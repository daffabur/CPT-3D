"""Read-only pilot distribution/resource audit and random-weight training compatibility dry run."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import platform
from pathlib import Path
import resource
import time
import numpy as np
from .v3_dataset_production import CONFIG_PATH,build_plan,config_checks,validate_dataset,_read_saved
from .train_v3 import dry_run


def _summary(a):
    x=np.concatenate(a).astype(float)
    return {'count':int(x.size),'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),
            'p05':float(np.percentile(x,5)),'median':float(np.median(x)),'p95':float(np.percentile(x,95)),
            'max':float(x.max())}

def audit(root,c):
    started=time.perf_counter();integrity=validate_dataset(root,c,reproduce=True);validation_seconds=time.perf_counter()-started
    entries=json.loads((root/'manifest.json').read_text())['entries']
    if len(entries)!=sum(c['pilot_counts'].values()):raise ValueError('pilot incomplete')
    by_grid=defaultdict(list);grid_counts=Counter();family_counts=Counter();cross=Counter();cpt_counts=Counter()
    sbt_counts=Counter();material_counts=Counter();all_ic=[];all_qc=[];all_fs=[];response_by_tag=defaultdict(lambda:{'log_qc':[],'log_fr':[]})
    for e in entries:
        v,o=_read_saved(root,e);valid=v['valid_mask'].astype(bool)
        ic=v['Ic'][valid];qc=v['qc_mpa'][valid];fs=v['fs_mpa'][valid]
        all_ic.append(ic);all_qc.append(qc);all_fs.append(fs)
        response_by_tag[e['response_tag']]['log_qc'].append(np.log(qc));response_by_tag[e['response_tag']]['log_fr'].append(np.log(fs))
        sbt_counts.update(dict(zip(*np.unique(v['sbt_id'][valid],return_counts=True))))
        material_counts.update(dict(zip(*np.unique(v['material_group'][valid],return_counts=True))))
        family_counts[e['family']]+=1;grid_counts[e['grid_name']]+=1;cross[(e['split'],e['grid_name'])]+=1;cpt_counts[e['n_cpt']]+=1
        by_grid[e['grid_name']].append(e)
    full=build_plan(c,'full')
    full_grid=Counter(e['grid_name'] for e in full['entries'])
    benchmark={}
    for g,items in by_grid.items():
        secs=np.array([e['generation_seconds'] for e in items]);sizes=np.array([e['stored_bytes'] for e in items])
        benchmark[g]={'pilot_count':len(items),'generation_seconds_mean':float(secs.mean()),
                      'generation_seconds_std':float(secs.std()),'generation_seconds_min':float(secs.min()),
                      'generation_seconds_max':float(secs.max()),'stored_bytes_mean':float(sizes.mean()),
                      'stored_bytes_min':int(sizes.min()),'stored_bytes_max':int(sizes.max()),
                      'full_count':full_grid[g],'projected_generation_seconds':float(secs.mean()*full_grid[g]),
                      'projected_stored_bytes':float(sizes.mean()*full_grid[g])}
    usage=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes=int(usage if platform.system()=='Darwin' else usage*1024)
    return {'integrity':integrity,'split_family_counts':dict(Counter(e['split']+' / '+e['family'] for e in entries)),
            'family_counts':dict(family_counts),'grid_counts':dict(grid_counts),
            'split_grid_counts':{f'{s} / {g}':n for (s,g),n in cross.items()},
            'stored_cpt_counts':{str(k):v for k,v in sorted(cpt_counts.items())},
            'material_counts':{str(k):int(v) for k,v in sorted(material_counts.items())},
            'sbt_counts':{str(k):int(v) for k,v in sorted(sbt_counts.items())},
            'Ic':_summary(all_ic),'qc_mpa':_summary(all_qc),'fs_mpa':_summary(all_fs),
            'response_by_tag':{tag:{k:_summary(arr) for k,arr in vals.items()} for tag,vals in response_by_tag.items()},
            'benchmark_by_grid':benchmark,'pilot_stored_bytes':sum(e['stored_bytes'] for e in entries),
            'projected_full_stored_bytes':sum(x['projected_stored_bytes'] for x in benchmark.values()),
            'projected_full_generation_seconds':sum(x['projected_generation_seconds'] for x in benchmark.values()),
            'pilot_file_count':sum(len(e['file_sha256']) for e in entries),
            'projected_full_file_count':int(round(sum(len(e['file_sha256']) for e in entries)/len(entries)*len(full['entries']))),
            'validation_seconds_with_regeneration':validation_seconds,
            'validation_seconds_projected_linear':validation_seconds*len(full['entries'])/len(entries),
            'audit_process_peak_rss_bytes':rss_bytes,
            'generation_peak_ram_measured':False,
            'generation_ram_comment':'Peak RSS measured for audit/validation process, not original pilot generation process; per-shape generation peak remains unmeasured.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=CONFIG_PATH)
    p.add_argument('--root',type=Path)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text());config_checks(c)
    root=a.root or Path(c['output_root'])
    if a.report.exists():raise FileExistsError(a.report)
    result=audit(root,c)
    dry_config={'mode':'dry_run_only','dataset_root':str(root),'train_split':'train','validation_split':'validation_id',
                'base_channels':4,'batch_size':2,'reconstruction_weight':100.,'torch_seed':19,'batch_seed':20261019,
                'evaluation':{'profile_counts':c['evaluation_cpt_counts']}}
    result['training_compatibility']=dry_run(dry_config,validator=lambda _:result['integrity'])
    a.report.write_text(json.dumps({'config':c,'audit':result},indent=2)+'\n')
    print(json.dumps({'volumes':result['integrity']['volumes'],'all_reproduced':True,
                      'projected_full_gib':result['projected_full_stored_bytes']/1024**3,
                      'dry_run_weights_unchanged':result['training_compatibility']['model_weights_unchanged']}))
if __name__=='__main__':main()
