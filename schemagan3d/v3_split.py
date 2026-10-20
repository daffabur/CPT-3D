"""Deterministic group-based split preparation; does not move existing examples."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SPLITS=('train','validation_id','test_id','test_family','test_geometry_parameter','test_response_parameter')

def prepare_split_manifest(records:list[dict],*,heldout_family:str,heldout_geometry_tag:str,heldout_response_tag:str)->dict:
    """Records need realization_id, lineage_group, family, geometry_tag, response_tag.

    A lineage group covers all derivatives and controlled pairs that must stay together.
    OOD tags are mutually exclusive within each group. Remaining groups are assigned
    60/20/20 by stable hash order, with at least one group in each ID split.
    """
    if not records:raise ValueError('no records')
    ids=[r['realization_id'] for r in records]
    if len(set(ids))!=len(ids):raise ValueError('duplicate realization_id')
    groups={}
    for r in records:
        if not all(r.get(k) for k in ('realization_id','lineage_group','family','geometry_tag','response_tag')):
            raise ValueError('incomplete record')
        groups.setdefault(r['lineage_group'],[]).append(r)
    assigned={};ordinary=[]
    for gid,items in groups.items():
        categories=set()
        for r in items:
            if r['family']==heldout_family:categories.add('test_family')
            if r['geometry_tag']==heldout_geometry_tag:categories.add('test_geometry_parameter')
            if r['response_tag']==heldout_response_tag:categories.add('test_response_parameter')
        if len(categories)>1:raise ValueError(f'overlapping holdout definitions in {gid}')
        if categories:
            # All derivatives in a held-out group must share the defining condition.
            category=next(iter(categories))
            for r in items:
                expected={'test_family':r['family']==heldout_family,'test_geometry_parameter':r['geometry_tag']==heldout_geometry_tag,'test_response_parameter':r['response_tag']==heldout_response_tag}[category]
                if not expected:raise ValueError(f'inconsistent holdout group {gid}')
            assigned[gid]=category
        else:ordinary.append(gid)
    by_family={}
    for gid in ordinary:
        families={r['family'] for r in groups[gid]}
        if len(families)!=1:raise ValueError(f'in-distribution lineage crosses families: {gid}')
        by_family.setdefault(next(iter(families)),[]).append(gid)
    for family,family_groups in by_family.items():
        if len(family_groups)<3:raise ValueError(f'at least three ID lineage groups required for {family}')
        ordered=sorted(family_groups,key=lambda gid:hashlib.sha256(gid.encode()).hexdigest())
        n=len(ordered);n_val=max(1,round(n*.2));n_test=max(1,round(n*.2));n_train=n-n_val-n_test
        for i,gid in enumerate(ordered):assigned[gid]='train' if i<n_train else ('validation_id' if i<n_train+n_val else 'test_id')
    entries=[{**r,'split':assigned[r['lineage_group']]} for r in records]
    if any(not any(e['split']==split for e in entries) for split in SPLITS):raise ValueError('complete six-way split requires examples for every holdout')
    if any(r['family']==heldout_family and r['split']!='test_family' for r in entries):raise ValueError('family holdout leakage')
    return {'schema_version':'v3-split-plan-1','heldout_family':heldout_family,
            'heldout_geometry_tag':heldout_geometry_tag,'heldout_response_tag':heldout_response_tag,
            'algorithm':'lineage grouped; OOD tags exclusive; remaining groups SHA-256 ordered 60/20/20 with minimum one val and test group',
            'entries':entries}

def save_split_manifest(plan:dict,path:Path):
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(plan,indent=2)+'\n')
