"""Lossless, resumable, one-volume-at-a-time export of frozen SchemaGAN V3 data."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,re,shutil,time,uuid
from collections import Counter
from pathlib import Path
import numpy as np
from synthetic3d.v2.core import Grid
from .v3_eval_protocol import PROFILE_COUNTS,nested_sites
from .v3_dataset_production import file_sha

SOURCE=Path('data/synthetic3d_v2/schemagan_v3_24k')
OUTPUT=Path('data/synthetic3d_v2/schemagan_v3_24k_csv')
EXPECTED_PLAN='ff2884e0370e9597cb3afa3f8c0b613f19670835fc96ff2dc26cdebf0990d663'
SPLITS=('train','validation_id','test_id','test_family','test_geometry_parameter','test_response_parameter')
SPLIT_COUNTS={'train':16000,'validation_id':4000,'test_id':1000,'test_family':1000,
              'test_geometry_parameter':1000,'test_response_parameter':1000}
# First 13 fields are the requested schema. Extra 3D source fields preserve all voxel arrays.
COLUMNS=('x','y','z','x_m','y_m','z_m','Ic','qc','fs','unit_id','sbt_id','valid_mask','observed_mask',
         'Fr_percent','Qt','qt_mpa','gamma_kn_m3','material_group','sigma_v0_eff_mpa','sigma_v0_mpa','u0_mpa')
ARRAY_FIELDS={'Ic':'Ic','qc':'qc_mpa','fs':'fs_mpa','unit_id':'unit_id','sbt_id':'sbt_id',
              'valid_mask':'valid_mask','Fr_percent':'Fr_percent','Qt':'Qt','qt_mpa':'qt_mpa',
              'gamma_kn_m3':'gamma_kn_m3','material_group':'material_group',
              'sigma_v0_eff_mpa':'sigma_v0_eff_mpa','sigma_v0_mpa':'sigma_v0_mpa','u0_mpa':'u0_mpa'}
FLOAT_COLUMNS={'Ic','qc','fs','Fr_percent','Qt','qt_mpa','gamma_kn_m3',
               'sigma_v0_eff_mpa','sigma_v0_mpa','u0_mpa'}
INT_COLUMNS={'unit_id','sbt_id','material_group'}
MASK_COLUMNS={'valid_mask','observed_mask'}
MANIFEST_COLUMNS=('realization_id','split','family','grid_id','shape_x','shape_y','shape_z','dx_m','dy_m','dz_m',
                  'stored_cpt_count','evaluation_site_seed','lineage_group','csv_path','sidecar_path',
                  'voxel_rows','csv_sha256','sidecar_sha256')

def atomic_text(path:Path,value:str):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with temp.open('w',encoding='utf-8',newline='') as f:
            f.write(value);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)

def safe_source(root:Path,relative:str):
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('source path escapes dataset root')
    return p

def source_manifest(root:Path):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('plan_sha256')!=EXPECTED_PLAN:raise ValueError('frozen plan SHA-256 mismatch')
    entries=manifest['entries'];ids=[e['realization_id'] for e in entries]
    if len(entries)!=24000 or len(set(ids))!=len(ids):raise ValueError('source volume count/ID mismatch')
    if Counter(e['split'] for e in entries)!=Counter(SPLIT_COUNTS):raise ValueError('source split counts mismatch')
    for e in entries:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',e['realization_id']):raise ValueError('unsafe realization ID')
    return manifest,entries,file_sha(root/'manifest.json')

def source_volume(root:Path,entry:dict,verify_hash=True):
    arrays={}
    for key,relative in entry['arrays'].items():
        path=safe_source(root,relative)
        if verify_hash and file_sha(path)!=entry['file_sha256'][relative]:raise ValueError(f'source checksum mismatch: {relative}')
        arrays[key]=np.load(path,allow_pickle=False)
        if arrays[key].dtype.kind=='f' and not np.isfinite(arrays[key]).all():raise ValueError(f'nonfinite source array: {relative}')
    shape=tuple(entry['shape'])
    if set(arrays)!=(set(ARRAY_FIELDS.values())|{'boundaries_m'}):raise ValueError('unexpected source arrays')
    if any(arrays[key].shape!=shape for key in ARRAY_FIELDS.values()):raise ValueError('voxel array shape mismatch')
    if arrays['boundaries_m'].shape[1:]!=shape[:2]:raise ValueError('boundary array shape mismatch')
    meta_path=safe_source(root,entry['metadata']);profile_path=safe_source(root,entry['profiles'])
    mask_path=safe_source(root,entry['observed_mask'])
    if verify_hash:
        for path,relative in ((meta_path,entry['metadata']),(profile_path,entry['profiles']),(mask_path,entry['observed_mask'])):
            if file_sha(path)!=entry['file_sha256'][relative]:raise ValueError(f'source checksum mismatch: {relative}')
    metadata=json.loads(meta_path.read_text());profiles=json.loads(profile_path.read_text())
    observed=np.load(mask_path,allow_pickle=False)
    if observed.shape!=shape or observed.dtype!=np.dtype(bool):raise ValueError('observed mask shape/dtype mismatch')
    grid=Grid(**metadata['grid'])
    if shape!=(grid.nx,grid.ny,grid.nz):raise ValueError('grid metadata shape mismatch')
    valid=arrays['valid_mask']
    if valid.dtype!=np.dtype(bool) or np.any(observed&~valid):raise ValueError('invalid source masks')
    expected=np.zeros(shape,dtype=bool);seen=set()
    for p in profiles:
        x,y=p['grid_xy']
        if (x,y) in seen or grid.index_xy(*p['xy_m'])!=(x,y):raise ValueError('duplicate/invalid CPT site')
        seen.add((x,y));expected[x,y,:]=valid[x,y,:]
        if not np.array_equal(np.asarray(p['Ic'],dtype=np.float32),arrays['Ic'][x,y,valid[x,y,:]]):
            raise ValueError('CPT Ic profile disagrees with source')
        for profile_key,array_key in (('qc_mpa','qc_mpa'),('fs_mpa','fs_mpa')):
            if not np.array_equal(np.asarray(p[profile_key],dtype=np.float32),arrays[array_key][x,y,valid[x,y,:]]):
                raise ValueError(f'CPT {profile_key} profile disagrees with source')
    if len(seen)!=entry['n_cpt'] or not np.array_equal(expected,observed):
        raise ValueError('stored observed mask disagrees with CPT profiles')
    for count in PROFILE_COUNTS:
        if len(nested_sites(grid,int(entry['evaluation_site_seed']),valid,count))!=count:
            raise ValueError('evaluation CPT protocol unsupported')
    return arrays,observed,metadata,profiles,grid

def format_value(column,value):
    if column in FLOAT_COLUMNS:return repr(float(value))
    if column in INT_COLUMNS or column in MASK_COLUMNS:return str(int(value))
    raise ValueError(f'unsupported field {column}')

def write_csv(path:Path,arrays,observed,grid:Grid):
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    shape=arrays['Ic'].shape
    xy={(x,y):grid.xy_m(x,y) for x in range(grid.nx) for y in range(grid.ny)}
    z_m=[(z+.5)*grid.dz_m for z in range(grid.nz)]
    try:
        with temp.open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f,lineterminator='\n');writer.writerow(COLUMNS)
            for x in range(shape[0]):
                for y in range(shape[1]):
                    xm,ym=xy[x,y]
                    for z in range(shape[2]):
                        writer.writerow((x,y,z,repr(xm),repr(ym),repr(z_m[z]),
                            *(format_value(column,arrays[ARRAY_FIELDS[column]][x,y,z])
                              for column in ('Ic','qc','fs','unit_id','sbt_id','valid_mask')),
                            int(observed[x,y,z]),
                            *(format_value(column,arrays[ARRAY_FIELDS[column]][x,y,z]) for column in COLUMNS[13:])))
            f.flush();os.fsync(f.fileno())
        return temp
    except Exception:
        temp.unlink(missing_ok=True);raise

def validate_csv(path:Path,arrays,observed,grid:Grid):
    shape=arrays['Ic'].shape;reconstructed={key:np.empty(shape,dtype=arrays[key].dtype) for key in ARRAY_FIELDS.values()}
    observed_copy=np.empty(shape,dtype=bool);count=0
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        if tuple(reader.fieldnames or ())!=COLUMNS:raise ValueError('CSV column schema mismatch')
        for x in range(shape[0]):
            for y in range(shape[1]):
                xm,ym=grid.xy_m(x,y)
                for z in range(shape[2]):
                    try:row=next(reader)
                    except StopIteration:raise ValueError('CSV has missing rows')
                    if tuple(int(row[k]) for k in ('x','y','z'))!=(x,y,z):raise ValueError('CSV axis order/index mismatch')
                    if (float(row['x_m']),float(row['y_m']),float(row['z_m']))!=(xm,ym,(z+.5)*grid.dz_m):
                        raise ValueError('CSV physical coordinate mismatch')
                    for column,key in ARRAY_FIELDS.items():
                        value=row[column]
                        if column in MASK_COLUMNS:
                            if value not in ('0','1'):raise ValueError('invalid mask value')
                            reconstructed[key][x,y,z]=bool(int(value))
                        elif column in INT_COLUMNS:reconstructed[key][x,y,z]=int(value)
                        else:reconstructed[key][x,y,z]=float(value)
                    if row['observed_mask'] not in ('0','1'):raise ValueError('invalid observed mask value')
                    observed_copy[x,y,z]=bool(int(row['observed_mask']));count+=1
        try:next(reader);raise ValueError('CSV has extra rows')
        except StopIteration:pass
    for key,original in arrays.items():
        if key=='boundaries_m':continue
        if not np.array_equal(reconstructed[key],original):raise ValueError(f'CSV round-trip mismatch: {key}')
    if not np.array_equal(observed_copy,observed):raise ValueError('CSV observed mask round-trip mismatch')
    return count

def file_paths(output:Path,entry):
    split=entry['split'];rid=entry['realization_id']
    return output/split/(rid+'.csv'),output/'sidecars'/split/(rid+'.json')

def valid_receipt(output:Path,entry,manifest_sha):
    csv_path,sidecar_path=file_paths(output,entry)
    if not csv_path.is_file() or not sidecar_path.is_file():return False
    try:receipt=json.loads(sidecar_path.read_text())
    except (json.JSONDecodeError,OSError):return False
    return (receipt.get('realization_id')==entry['realization_id'] and receipt.get('split')==entry['split'] and
            receipt.get('source_manifest_sha256')==manifest_sha and receipt.get('columns')==list(COLUMNS) and
            receipt.get('row_count')==int(np.prod(entry['shape'])) and receipt.get('source_file_sha256')==entry['file_sha256'] and
            receipt.get('csv_sha256')==file_sha(csv_path))

def export_one(root:Path,output:Path,entry,manifest_sha):
    csv_path,sidecar_path=file_paths(output,entry)
    arrays,observed,metadata,profiles,grid=source_volume(root,entry)
    csv_path.parent.mkdir(parents=True,exist_ok=True);sidecar_path.parent.mkdir(parents=True,exist_ok=True)
    temp=write_csv(csv_path,arrays,observed,grid)
    try:
        count=validate_csv(temp,arrays,observed,grid)
        csv_sha=file_sha(temp)
        boundary=arrays['boundaries_m']
        sidecar={'schema_version':'schemagan-v3-csv-sidecar-1','source_manifest_sha256':manifest_sha,
                 'realization_id':entry['realization_id'],'split':entry['split'],
                 'manifest_entry':entry,'metadata':metadata,'profiles':profiles,
                 'boundaries_m':{'dtype':str(boundary.dtype),'shape':list(boundary.shape),
                                 'data':boundary.astype(float).tolist()},
                 'columns':list(COLUMNS),'source_file_sha256':entry['file_sha256'],
                 'csv_sha256':csv_sha,'row_count':count,'csv_bytes':temp.stat().st_size,
                 'round_trip_exact':True}
        if csv_path.exists() and valid_receipt(output,entry,manifest_sha):
            raise FileExistsError('validated CSV already exists')
        os.replace(temp,csv_path)
        atomic_text(sidecar_path,json.dumps(sidecar,separators=(',',':'),ensure_ascii=False)+'\n')
        return sidecar
    finally:temp.unlink(missing_ok=True)

def metadata_payload(root,manifest,manifest_sha,entries,pilot):
    norm=json.loads(Path('schemagan3d/v3_24k_normalization.json').read_text())
    if norm['plan_sha256']!=EXPECTED_PLAN or norm['source_split']!='train':raise ValueError('normalization provenance mismatch')
    dtypes={}
    first=entries[0]
    for key,relative in first['arrays'].items():dtypes[key]=str(np.load(safe_source(root,relative),allow_pickle=False,mmap_mode='r').dtype)
    return {'schema_version':'schemagan-v3-csv-export-1','pilot':pilot,'source_dataset':str(root),
            'source_manifest_sha256':manifest_sha,'frozen_plan_sha256':manifest['plan_sha256'],
            'source_config_sha256':manifest['config_sha256'],'source_volume_count':24000,
            'exported_volume_count':len(entries),'source_split_counts':SPLIT_COUNTS,
            'columns':list(COLUMNS),'array_column_mapping':ARRAY_FIELDS,
            'source_array_dtypes':dtypes,'csv_float_read_dtype':'float64; cast to source float32 for exact round-trip',
            'source_axis_order':'(X,Y,Z); x outer, y middle, z inner in CSV',
            'normalization_train_only':norm,'stored_observation':'one observed_mask/profiles set per source volume',
            'evaluation_cpt_counts':list(PROFILE_COUNTS),
            'additional_observation_policy':'nested_sites(Grid(metadata.grid), manifest_entry.evaluation_site_seed, valid_mask, count)',
            'boundary_storage':'sidecar boundaries_m as nested JSON floats; cast back to float32',
            'field_roles':{'Ic':'full-volume ground-truth target; do not use hidden Ic as model input',
                           'observed_mask':'stored CPT observation indicator; sparse Ic input = Ic where observed_mask=1'},
            'checksums_scope':'all CSV/sidecars plus manifest.csv/dataset_metadata.json/README.md; checksums.sha256 excludes itself'}

def readme_text():
    return '''# SchemaGAN V3 synthetic geotechnical 3D dataset — CSV export

This is an exact-value CSV copy of the frozen synthetic V3 dataset, **not field-validated CPT data**. It contains 24,000 volume CSVs: train 16,000; validation ID 4,000; test ID, family holdout, geometry holdout, and response holdout 1,000 each. Keep the split and lineage from `manifest.csv`/sidecars. Do not tune on test or holdout splits.

## Files and grid

Each `<split>/<realization_id>.csv` has one row per voxel in **XYZ** order (`x` outer, `y` middle, `z` inner). Grid shapes are `(12,12,12)`, `(16,16,16)`, `(24,24,12)`; exact origin, spacing, azimuth, and surface elevation are in `sidecars/<split>/<realization_id>.json` under `metadata.grid`. `x_m,y_m` are rotated physical XY voxel centres from `Grid.xy_m`; `z_m=(z+0.5)*dz_m` is positive depth below surface in metres. The sidecar preserves the original manifest entry, complete metadata, CPT profiles, and geological `boundaries_m` with float32 dtype/shape. All other 3D source arrays are in the CSV.

Primary columns: `x,y,z,x_m,y_m,z_m,Ic,qc,fs,unit_id,sbt_id,valid_mask,observed_mask`. `Ic` is continuous physical-scale **full-volume ground truth**. `qc` and `fs` are MPa (source names `qc_mpa`, `fs_mpa`). `unit_id` is the geological unit ID; `sbt_id` is the Soil Behavior Type category. Masks are 0/1. Additional columns preserve `Fr_percent`, `Qt`, `qt_mpa`, `gamma_kn_m3`, `material_group`, `sigma_v0_eff_mpa`, `sigma_v0_mpa`, and `u0_mpa` in their source units/names. Source float arrays are float32; categorical arrays are uint8/int16; masks are bool. CSV readers commonly return float64 for numeric fields; cast to source dtypes to reconstruct exact arrays.

## Deep Learning loading example

```python
import csv, json, numpy as np
from pathlib import Path
root = Path('data/synthetic3d_v2/schemagan_v3_24k_csv')
volume = root/'train'/'<realization_id>.csv'
sidecar = json.loads((root/'sidecars'/'train'/(volume.stem+'.json')).read_text())
shape = tuple(sidecar['manifest_entry']['shape'])
ic = np.empty(shape, dtype=np.float32)
observed = np.empty(shape, dtype=bool)
valid = np.empty(shape, dtype=bool)
with volume.open(newline='') as stream:
    for row in csv.DictReader(stream):
        idx = tuple(int(row[k]) for k in ('x','y','z'))
        ic[idx] = np.float32(row['Ic'])
        observed[idx] = row['observed_mask'] == '1'
        valid[idx] = row['valid_mask'] == '1'
stats = json.loads((root/'dataset_metadata.json').read_text())['normalization_train_only']
normalized = (ic - stats['mean']) / stats['std']
sparse = np.where(observed, normalized, 0).astype(np.float32)  # model input
mask = observed.astype(np.float32)                              # model input
truth = normalized.astype(np.float32)                            # supervised target only
# Existing SchemaGAN adapter transposes XYZ to CZYX before a 3D model call.
```

**Never feed the complete `Ic` column as an input.** It is a target; only the entries where `observed_mask=1` may enter a sparse CPT input. Normalization mean/std come only from the train split; do not refit on validation/test. Hidden-valid metric mask is `valid & ~observed`; RMSE is `sqrt(mean((prediction_ic-target_ic)^2))` and MAE is `mean(abs(prediction_ic-target_ic))` on that mask, in physical Ic units. Aggregate per-volume metrics explicitly when comparing methods.

Each source volume stores **one** observed CPT mask and profile set, with `stored_cpt_count` in `manifest.csv`. For deterministic evaluation with 1,2,3,5,8,12,20 CPT, use the source `evaluation_site_seed` and `nested_sites` policy recorded in `dataset_metadata.json`/sidecar with the volume's valid mask. This creates alternate observation masks for the **same ground-truth volume**; it does not create new independent volumes. Preserve the same sites for every model being compared.

`checksums.sha256` lists the SHA-256 of all CSVs, sidecars, and top-level metadata files. The source frozen plan SHA-256 and source manifest SHA-256 are in `dataset_metadata.json`. CSV float text uses enough decimal precision to cast back to the original float32 values exactly; export validation checks all 3D arrays, masks, coordinates and sidecar boundary values against source files.
'''

def manifest_row(output,entry,sidecar):
    csv_path,sidecar_path=file_paths(output,entry);grid=sidecar['metadata']['grid']
    return {'realization_id':entry['realization_id'],'split':entry['split'],'family':entry['family'],
            'grid_id':entry['grid_id'],'shape_x':grid['nx'],'shape_y':grid['ny'],'shape_z':grid['nz'],
            'dx_m':grid['dx_m'],'dy_m':grid['dy_m'],'dz_m':grid['dz_m'],
            'stored_cpt_count':entry['n_cpt'],'evaluation_site_seed':entry['evaluation_site_seed'],
            'lineage_group':entry['lineage_group'],'csv_path':csv_path.relative_to(output).as_posix(),
            'sidecar_path':sidecar_path.relative_to(output).as_posix(),
            'voxel_rows':sidecar['row_count'],'csv_sha256':sidecar['csv_sha256'],
            'sidecar_sha256':file_sha(sidecar_path)}

def finalize(output,entries,root,manifest,manifest_sha,pilot):
    rows=[]
    for e in entries:
        csv_path,sidecar_path=file_paths(output,e)
        if not valid_receipt(output,e,manifest_sha):raise ValueError(f'missing/invalid CSV receipt: {e["realization_id"]}')
        rows.append(manifest_row(output,e,json.loads(sidecar_path.read_text())))
    temp=output/('manifest.csv.tmp-'+uuid.uuid4().hex)
    with temp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=MANIFEST_COLUMNS);writer.writeheader();writer.writerows(rows)
        f.flush();os.fsync(f.fileno())
    os.replace(temp,output/'manifest.csv')
    atomic_text(output/'dataset_metadata.json',json.dumps(metadata_payload(root,manifest,manifest_sha,entries,pilot),indent=2)+'\n')
    atomic_text(output/'README.md',readme_text())
    temp=output/('checksums.sha256.tmp-'+uuid.uuid4().hex)
    with temp.open('w',encoding='utf-8') as f:
        for row in rows:
            f.write(f"{row['csv_sha256']}  {row['csv_path']}\n")
            f.write(f"{row['sidecar_sha256']}  {row['sidecar_path']}\n")
        for name in ('manifest.csv','dataset_metadata.json','README.md'):
            f.write(f'{file_sha(output/name)}  {name}\n')
        f.flush();os.fsync(f.fileno())
    os.replace(temp,output/'checksums.sha256')
    return rows

def validate_all(root,output,entries,manifest_sha,*,full_roundtrip=True):
    start=time.perf_counter();counts=Counter();total_rows=0;csv_bytes=sidecar_bytes=0
    with (output/'manifest.csv').open(newline='',encoding='utf-8') as f:manifest_rows=list(csv.DictReader(f))
    if len(manifest_rows)!=len(entries):raise ValueError('export manifest row count mismatch')
    index={r['realization_id']:r for r in manifest_rows}
    if len(index)!=len(entries):raise ValueError('duplicate export manifest ID')
    for n,e in enumerate(entries,1):
        rid=e['realization_id'];r=index.get(rid)
        if r is None or r['split']!=e['split']:raise ValueError('split/ID mismatch')
        csv_path,sidecar_path=file_paths(output,e)
        if not valid_receipt(output,e,manifest_sha):raise ValueError(f'checksum/receipt failed: {rid}')
        if r['csv_sha256']!=file_sha(csv_path) or r['sidecar_sha256']!=file_sha(sidecar_path):
            raise ValueError('export manifest checksum mismatch')
        sidecar=json.loads(sidecar_path.read_text())
        if sidecar['manifest_entry']!=e:raise ValueError('manifest entry sidecar mismatch')
        if full_roundtrip:
            arrays,observed,metadata,profiles,grid=source_volume(root,e)
            if sidecar['metadata']!=metadata or sidecar['profiles']!=profiles:raise ValueError('sidecar metadata/profile mismatch')
            boundary=sidecar['boundaries_m']
            if (boundary['dtype']!=str(arrays['boundaries_m'].dtype) or boundary['shape']!=list(arrays['boundaries_m'].shape) or
                    not np.array_equal(np.asarray(boundary['data'],dtype=arrays['boundaries_m'].dtype),arrays['boundaries_m'])):
                raise ValueError('sidecar boundary round-trip mismatch')
            actual=validate_csv(csv_path,arrays,observed,grid)
            if actual!=int(r['voxel_rows']):raise ValueError('voxel row count mismatch')
        counts[e['split']]+=1;total_rows+=int(r['voxel_rows'])
        csv_bytes+=csv_path.stat().st_size;sidecar_bytes+=sidecar_path.stat().st_size
        if n%1000==0:print(f'validated {n}/{len(entries)} volumes',flush=True)
    if len(entries)==24000 and counts!=Counter(SPLIT_COUNTS):raise ValueError('final split counts mismatch')
    with (output/'checksums.sha256').open() as f:
        checks=[line.rstrip('\n').split('  ',1) for line in f]
    if len(checks)!=2*len(entries)+3:raise ValueError('checksums line count mismatch')
    for digest,rel in checks:
        if file_sha(output/rel)!=digest:raise ValueError(f'checksums.sha256 mismatch: {rel}')
    if file_sha(root/'manifest.json')!=manifest_sha:raise ValueError('source manifest changed')
    return {'volumes':len(entries),'split_counts':dict(counts),'voxel_rows':total_rows,
            'csv_bytes':csv_bytes,'sidecar_bytes':sidecar_bytes,
            'round_trip_exact_all_3d_arrays':full_roundtrip,'checksums_verified':len(checks),
            'source_manifest_sha256_unchanged':True,'validation_seconds':time.perf_counter()-start}

def export(root:Path,output:Path,entries,manifest,manifest_sha,*,pilot=False,resume=False):
    if output.exists() and not resume:raise FileExistsError(f'output exists: {output}; use --resume')
    output.mkdir(parents=True,exist_ok=True)
    metadata_path=output/'dataset_metadata.json'
    if metadata_path.exists():
        prior=json.loads(metadata_path.read_text())
        if prior['source_manifest_sha256']!=manifest_sha or prior['exported_volume_count']!=len(entries):
            raise ValueError('existing export identity mismatch')
    else:atomic_text(metadata_path,json.dumps(metadata_payload(root,manifest,manifest_sha,entries,pilot),indent=2)+'\n')
    start=time.perf_counter();new=skipped=0
    for n,e in enumerate(entries,1):
        if valid_receipt(output,e,manifest_sha):skipped+=1
        else:
            export_one(root,output,e,manifest_sha);new+=1
        if n%100==0 or n==len(entries):
            print(f'exported {n}/{len(entries)} volumes; new={new}, validated-skip={skipped}; elapsed={time.perf_counter()-start:.1f}s',flush=True)
    export_seconds=time.perf_counter()-start
    finalize(output,entries,root,manifest,manifest_sha,pilot)
    validation=validate_all(root,output,entries,manifest_sha,full_roundtrip=True)
    result={'pilot':pilot,'new_volumes':new,'validated_skips':skipped,'export_seconds':export_seconds,**validation,
            'total_seconds':time.perf_counter()-start}
    atomic_text(output/'export_summary.json',json.dumps(result,indent=2)+'\n')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('pilot','export','validate'),required=True)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--pilot-count',type=int,default=30)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    root=args.source;output=args.output
    manifest,all_entries,manifest_sha=source_manifest(root)
    if args.mode=='pilot':
        if not 1<=args.pilot_count<=30:parser.error('pilot count must be 1..30')
        result=export(root,output,all_entries[:args.pilot_count],manifest,manifest_sha,pilot=True,resume=args.resume)
    elif args.mode=='export':
        result=export(root,output,all_entries,manifest,manifest_sha,resume=args.resume)
    else:
        if not output.exists():raise FileNotFoundError(output)
        result=validate_all(root,output,all_entries,manifest_sha,full_roundtrip=True)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
