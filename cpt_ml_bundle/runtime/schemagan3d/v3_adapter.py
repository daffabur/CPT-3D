"""Read-only V3 manifest adapter for inference and future training pipelines.

Generator inputs are sparse normalized Ic and observation mask only. Target is kept
separate for supervised use. No checkpoint or model implementation is modified.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterator
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from synthetic3d.v2.core import Grid
from .v3_eval_protocol import PROFILE_COUNTS, nested_sites


@dataclass(frozen=True)
class V3Normalization:
    mean: float
    std: float
    source_split: str = 'train'
    method_version: str = 'v3-valid-voxel-population-v1'
    source_train_ids: tuple[str,...] = ()

    def __post_init__(self):
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std <= 0:
            raise ValueError('finite mean and positive std required')
        if self.source_split != 'train':
            raise ValueError('normalization must come from train split')

    def normalize(self, values):
        return (values-self.mean)/self.std

    def denormalize(self, values):
        return values*self.std+self.mean

    def save(self, path:Path):
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(asdict(self),indent=2)+'\n')

    @classmethod
    def load(cls,path:Path):
        value=json.loads(path.read_text())
        value['source_train_ids']=tuple(value.get('source_train_ids',()))
        return cls(**value)


def _safe(root:Path,relative:str)->Path:
    p=(root/relative).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('manifest path escapes root')
    return p


def load_manifest(root:Path):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('schema_version') not in ('v2-prototype-3','schemagan-v3-pilot-manifest-1'):raise ValueError('unsupported V3 manifest')
    entries=manifest.get('entries')
    if not isinstance(entries,list) or not entries:raise ValueError('manifest entries missing')
    ids=[e['realization_id'] for e in entries]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate realization ID')
    return entries


def _array(root,entry,key):
    return np.load(_safe(root,entry['arrays'][key]),allow_pickle=False)


def compute_train_normalization(root:Path,entries=None)->V3Normalization:
    entries=load_manifest(root) if entries is None else entries
    selected=[e for e in entries if e['split']=='train']
    if not selected:raise ValueError('no train examples')
    total=square=0.0;count=0
    for e in selected:
        ic=_array(root,e,'Ic').astype(np.float64)
        valid=_array(root,e,'valid_mask').astype(bool)
        if ic.shape!=valid.shape or not valid.any() or not np.all(np.isfinite(ic[valid])):raise ValueError('invalid training target/mask')
        x=ic[valid];total+=float(x.sum());square+=float(np.square(x).sum());count+=x.size
    mean=total/count;variance=square/count-mean*mean
    return V3Normalization(mean,float(np.sqrt(max(variance,1e-12))),source_train_ids=tuple(sorted(e['realization_id'] for e in selected)))


def assert_model_shape(shape):
    if len(shape)!=3 or any(not isinstance(n,int) or n<8 or n%4 for n in shape):
        raise ValueError('Generator3D V3 adapter requires X,Y,Z dimensions >=8 and divisible by 4; use a separately reviewed padding/architecture plan')


def xyz_to_czyx(x):
    a=np.asarray(x)
    if a.ndim!=3:raise ValueError('expected XYZ array')
    return np.ascontiguousarray(a.transpose(2,1,0)[None])


def restore_observed(prediction:torch.Tensor,mask:torch.Tensor,stats:V3Normalization,observed_physical:torch.Tensor):
    """Return physical Ic; copy measured values exactly on observed voxels."""
    if prediction.shape!=observed_physical.shape or prediction.shape!=mask.shape:raise ValueError('tensor shapes differ')
    physical=stats.denormalize(prediction)
    return torch.where(mask.bool(),observed_physical,physical)


def masked_hidden_metrics(prediction_physical,target_physical,mask,valid):
    selected=valid.bool()&~mask.bool()
    if not selected.any():raise ValueError('no hidden valid voxels')
    d=(prediction_physical-target_physical)[selected]
    return {'rmse':float(torch.sqrt(torch.mean(d*d)).item()),'mae':float(torch.mean(torch.abs(d)).item()),'n':int(selected.sum().item())}


class V3VolumeDataset(Dataset):
    def __init__(self,root:Path,*,split:str,stats:V3Normalization,require_model_shape=True,
                 evaluation_cpt_count=None,indices=None,cpt_counts_by_index=None):
        self.root=Path(root);all_entries=load_manifest(self.root)
        expected_train_ids=tuple(sorted(e['realization_id'] for e in all_entries if e['split']=='train'))
        if not expected_train_ids or stats.source_train_ids!=expected_train_ids:
            raise ValueError('normalization provenance does not match manifest train entries')
        self.entries=[e for e in all_entries if e['split']==split]
        if indices is not None:
            if len(set(indices))!=len(indices) or any(not isinstance(i,int) or i<0 or i>=len(self.entries) for i in indices):
                raise ValueError('invalid evaluation indices')
            self.entries=[self.entries[i] for i in indices]
        if not self.entries:raise ValueError(f'no examples in {split}')
        if evaluation_cpt_count is not None and evaluation_cpt_count not in PROFILE_COUNTS:
            raise ValueError('unsupported evaluation CPT count')
        if cpt_counts_by_index is not None:
            if evaluation_cpt_count is not None or len(cpt_counts_by_index)!=len(self.entries) or any(c not in PROFILE_COUNTS for c in cpt_counts_by_index):
                raise ValueError('invalid per-sample CPT counts')
        self.stats=stats;self.require_model_shape=require_model_shape
        self.evaluation_cpt_count=evaluation_cpt_count
        self.cpt_counts_by_index=tuple(cpt_counts_by_index) if cpt_counts_by_index is not None else None
        self.shapes=[];self.spacings=[]
        for e in self.entries:
            shape=tuple(int(x) for x in e['shape'])
            if require_model_shape:assert_model_shape(shape)
            self.shapes.append(shape)
            self.spacings.append(tuple(float(x) for x in e['spacing_m']))

    def __len__(self):return len(self.entries)

    def __getitem__(self,index):
        e=self.entries[index];root=self.root
        ic=_array(root,e,'Ic').astype(np.float32)
        valid=_array(root,e,'valid_mask').astype(bool)
        observed=np.load(_safe(root,e['observed_mask']),allow_pickle=False).astype(bool)
        if ic.shape!=self.shapes[index] or valid.shape!=ic.shape or observed.shape!=ic.shape:raise ValueError('array shape mismatch')
        if not np.all(np.isfinite(ic[valid])):raise ValueError('nonfinite valid Ic')
        if np.any(observed&~valid) or not observed.any():raise ValueError('invalid observed mask')
        profiles=json.loads(_safe(root,e['profiles']).read_text())
        if not profiles:raise ValueError('at least one CPT required')
        expected=np.zeros(ic.shape,dtype=bool)
        grid=json.loads(_safe(root,e['metadata']).read_text())['grid']
        g=Grid(**grid)
        seen_sites=set()
        for p in profiles:
            x,y=p['grid_xy']
            if (x,y) in seen_sites or g.index_xy(*p['xy_m'])!=(x,y):raise ValueError('invalid CPT coordinate or duplicate')
            seen_sites.add((x,y));expected[x,y,:]=valid[x,y,:]
            if len(p['Ic'])!=int(valid[x,y,:].sum()) or not np.allclose(p['Ic'],ic[x,y,valid[x,y,:]],rtol=0,atol=1e-6):
                raise ValueError('CPT profile and target disagree')
            if not np.allclose(p['depth_m'],(np.arange(grid['nz'])+.5)[valid[x,y,:]]*grid['dz_m']):
                raise ValueError('CPT depth mismatch')
        if not np.array_equal(expected,observed):raise ValueError('CPT mask/profile mismatch')
        count=(self.cpt_counts_by_index[index] if self.cpt_counts_by_index is not None else self.evaluation_cpt_count)
        if count is not None:
            if 'evaluation_site_seed' not in e:raise ValueError('evaluation site seed missing')
            sites=nested_sites(g,int(e['evaluation_site_seed']),valid,count)
            if len(sites)!=count:raise ValueError('insufficient evaluation CPT sites')
            observed=np.zeros(ic.shape,dtype=bool)
            for x,y in sites:observed[x,y,:]=valid[x,y,:]
        target=np.zeros(ic.shape,dtype=np.float32)
        target[valid]=self.stats.normalize(ic[valid])
        sparse=np.zeros(ic.shape,dtype=np.float32)
        sparse[observed]=target[observed]
        observed_physical=np.zeros(ic.shape,dtype=np.float32)
        observed_physical[observed]=ic[observed]
        # Full truth is returned only as a supervised target, never as an input channel.
        return {'sparse':torch.from_numpy(xyz_to_czyx(sparse)),
                'mask':torch.from_numpy(xyz_to_czyx(observed.astype(np.float32))),
                'target':torch.from_numpy(xyz_to_czyx(target)),
                'observed_ic_physical':torch.from_numpy(xyz_to_czyx(observed_physical)),
                'valid_mask':torch.from_numpy(xyz_to_czyx(valid.astype(np.float32))),
                'spacing_m':torch.tensor(e['spacing_m'],dtype=torch.float32),
                'realization_id':e['realization_id']}


class HomogeneousShapeBatchSampler(Sampler[list[int]]):
    def __init__(self,dataset:V3VolumeDataset,batch_size:int,*,shuffle=False,seed=0,group_spacing=False):
        if batch_size<1:raise ValueError('batch_size must be positive')
        self.dataset=dataset;self.batch_size=batch_size;self.shuffle=shuffle;self.seed=seed;self.group_spacing=group_spacing
        self.groups={}
        for i,shape in enumerate(dataset.shapes):
            key=(shape,dataset.spacings[i]) if group_spacing else shape
            self.groups.setdefault(key,[]).append(i)

    def __iter__(self)->Iterator[list[int]]:
        rng=np.random.default_rng(self.seed);batches=[]
        for indices in self.groups.values():
            ids=indices.copy()
            if self.shuffle:rng.shuffle(ids)
            batches.extend([ids[i:i+self.batch_size] for i in range(0,len(ids),self.batch_size)])
        if self.shuffle:rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return sum((len(v)+self.batch_size-1)//self.batch_size for v in self.groups.values())
