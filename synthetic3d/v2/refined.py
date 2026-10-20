"""Version 2 of the independent small-scale V2 prototype; previous generator stays intact."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from .core import Grid, METHOD, SBT_NAMES, SBT_BOUNDS, interpret, classify_sbt, extract_cpt, validate as validate_common

SCHEMA_VERSION = 'v2-prototype-2'
SBT_RULES = {'0': 'unclassified: Ic < 1.0 or Ic > 4.0; zones 1,8,9 not inferred from Ic alone',
             '7': '1.0 <= Ic < 1.31', '6': '1.31 <= Ic < 2.05',
             '5': '2.05 <= Ic < 2.60', '4': '2.60 <= Ic < 2.95',
             '3': '2.95 <= Ic < 3.60', '2': '3.60 <= Ic <= 4.0'}


def default_geometry(n_layers: int, family: str, *, mode: str = 'varying') -> dict:
    if mode not in ('common_shift', 'varying'):
        raise ValueError('boundary mode must be common_shift or varying')
    return {'mode': mode, 'thickness_weights': [1.0 + .12*(i % 2) for i in range(n_layers)],
            'common_slope_fraction': .10 if family == 'inclined' else 0.0,
            'common_wave_fraction': .035 if family == 'undulating' else 0.0,
            'slope_direction_rad': .5, 'wave_length_fraction': .9,
            'boundary_slope_fraction': [.018*(-1)**i for i in range(n_layers-1)] if mode == 'varying' else [0.0]*(n_layers-1),
            'boundary_wave_fraction': [.012*(i+1)/max(n_layers-1,1) for i in range(n_layers-1)] if mode == 'varying' else [0.0]*(n_layers-1),
            'boundary_phase_rad': [.4+i*.8 for i in range(n_layers-1)],
            'minimum_thickness_voxels': 2,
            'lens': {'parent_unit': max(0,n_layers//2-1), 'center_x_fraction': .52,
                     'center_y_fraction': .48, 'center_z_fraction_in_parent': .50,
                     'radius_x_fraction_domain': .20, 'radius_y_fraction_domain': .19,
                     'radius_z_fraction_parent': .32, 'azimuth_deg': 20.0,
                     'minimum_voxels': 8} if family == 'lens' else None}


def _boundaries(grid:Grid, family:str, n_layers:int, p:dict) -> np.ndarray:
    if family not in ('horizontal','inclined','undulating','lens'): raise ValueError('unknown family')
    if n_layers < 2 or n_layers > 5: raise ValueError('prototype supports 2–5 background layers')
    if p['mode'] not in ('common_shift','varying'): raise ValueError('unknown boundary mode')
    weights=np.asarray(p['thickness_weights'],float)
    if weights.shape!=(n_layers,) or not np.all(np.isfinite(weights)) or np.any(weights<=0): raise ValueError('invalid thickness weights')
    for key in ('boundary_slope_fraction','boundary_wave_fraction','boundary_phase_rad'):
        if len(p[key])!=n_layers-1: raise ValueError(f'{key}: wrong length')
    minimum=p['minimum_thickness_voxels']*grid.dz_m
    if minimum < grid.dz_m or n_layers*minimum > grid.nz*grid.dz_m: raise ValueError('minimum thickness infeasible')
    xx,yy=np.meshgrid((np.arange(grid.nx)+.5)/grid.nx-.5,(np.arange(grid.ny)+.5)/grid.ny-.5,indexing='ij')
    zmax=grid.nz*grid.dz_m
    common=np.zeros((grid.nx,grid.ny))
    if family=='inclined': common += p['common_slope_fraction']*zmax*(xx*math.cos(p['slope_direction_rad'])+yy*math.sin(p['slope_direction_rad']))
    if family=='undulating': common += p['common_wave_fraction']*zmax*np.sin(2*math.pi*xx/p['wave_length_fraction'])*np.cos(2*math.pi*yy/p['wave_length_fraction'])
    fractions=np.cumsum(weights)[:-1]/weights.sum()
    surfaces=[]
    for i,fraction in enumerate(fractions):
        surface=fraction*zmax+common
        if p['mode']=='varying':
            slope=p['boundary_slope_fraction'][i]*zmax*(xx*math.cos(p['slope_direction_rad']+i*.35)+yy*math.sin(p['slope_direction_rad']+i*.35))
            wave=p['boundary_wave_fraction'][i]*zmax*np.sin(2*math.pi*xx/p['wave_length_fraction']+p['boundary_phase_rad'][i])*np.cos(2*math.pi*yy/p['wave_length_fraction']-.2*i)
            surface=surface+slope+wave
        surfaces.append(surface)
    b=np.stack(surfaces)
    layers=np.concatenate((b[:1],np.diff(b,axis=0),zmax-b[-1:]),axis=0)
    if not np.all(np.isfinite(b)) or np.any(layers < minimum-1e-8): raise ValueError('boundary crossing, domain exit, or minimum thickness violation')
    return b.astype(np.float32)


def _lens_mask(grid:Grid, background:np.ndarray, boundaries:np.ndarray, p:dict) -> tuple[np.ndarray,dict]:
    parent=p['parent_unit']; n_layers=len(boundaries)+1
    if not 0<=parent<n_layers: raise ValueError('lens parent invalid')
    cx=p['center_x_fraction']*grid.nx*grid.dx_m
    cy=p['center_y_fraction']*grid.ny*grid.dy_m
    rx=p['radius_x_fraction_domain']*grid.nx*grid.dx_m
    ry=p['radius_y_fraction_domain']*grid.ny*grid.dy_m
    if min(rx,ry)<=0 or cx-rx<=0 or cx+rx>=grid.nx*grid.dx_m or cy-ry<=0 or cy+ry>=grid.ny*grid.dy_m:
        raise ValueError('lens exits horizontal domain')
    ix=int(np.clip(round(cx/grid.dx_m-.5),0,grid.nx-1)); iy=int(np.clip(round(cy/grid.dy_m-.5),0,grid.ny-1))
    top=0 if parent==0 else float(boundaries[parent-1,ix,iy]); bottom=grid.nz*grid.dz_m if parent==n_layers-1 else float(boundaries[parent,ix,iy])
    cz=top+p['center_z_fraction_in_parent']*(bottom-top)
    rz=p['radius_z_fraction_parent']*(bottom-top)
    if rz < grid.dz_m or cz-rz<top or cz+rz>bottom or cz-rz<0 or cz+rz>grid.nz*grid.dz_m:
        raise ValueError('lens exits parent thickness or lacks vertical resolution')
    theta=math.radians(p['azimuth_deg'])
    x=(np.arange(grid.nx)+.5)*grid.dx_m-cx
    y=(np.arange(grid.ny)+.5)*grid.dy_m-cy
    xr=math.cos(theta)*x[:,None]+math.sin(theta)*y[None,:]
    yr=-math.sin(theta)*x[:,None]+math.cos(theta)*y[None,:]
    z=grid.depth_m()-cz
    ell=(xr[:,:,None]/rx)**2+(yr[:,:,None]/ry)**2+(z[None,None,:]/rz)**2
    geometric=ell<1
    if np.any(geometric & (background!=parent)): raise ValueError('lens intersects another parent unit')
    count=int(geometric.sum())
    if count<p['minimum_voxels']: raise ValueError('lens too small for grid resolution')
    info={'parent_unit':parent,'unit_id':n_layers,'center_local_m':[cx,cy,cz],
          'radii_m':[rx,ry,rz],'azimuth_deg':p['azimuth_deg'],'voxel_count':count,
          'minimum_voxels':p['minimum_voxels'],'shape':'rotated_ellipsoid'}
    return geometric,info


def _field(rng,shape,grid,correlation_length_m):
    sigma=tuple(max(.5,correlation_length_m/d) for d in (grid.dx_m,grid.dy_m,grid.dz_m))
    a=gaussian_filter(rng.normal(size=shape),sigma=sigma,mode='reflect')
    a-=a.mean(); sd=float(a.std())
    if sd<1e-10: raise ValueError('correlated field has negligible variance')
    return a/sd


def default_response(n_units:int,seed:int) -> dict:
    """Illustrative per-realization ranges; independent of numeric unit identifiers."""
    rng=np.random.default_rng(np.random.SeedSequence([seed,102]))
    units=[]
    for _ in range(n_units):
        units.append({'qc_mpa_median':float(np.exp(rng.uniform(math.log(2.5),math.log(11.0)))),
                      'fr_percent_median':float(np.exp(rng.uniform(math.log(.9),math.log(3.0)))),
                      'gamma_kn_m3':float(rng.uniform(17.0,20.0))})
    return {'units':units,'log_qc_field_sd':.24,'log_fr_field_sd':.23,
            'field_correlation_length_m':4.0,'shared_field_correlation':.45,
            'qc_depth_log_slope_per_m':.012,'fr_depth_log_slope_per_m':-.004,
            'parameter_status':'illustrative prototype; not field-calibrated',
            'low_pore_correction_assumed':True,
            'low_pore_correction_basis':'explicit scenario assumption only; water table below domain is not proof of negligible cone correction'}


def generate(grid:Grid,family:str,n_layers:int,seed:int,*,scenario_id:str,parameter_family_id:str,realization_id:str,
             geometry:dict|None=None,response:dict|None=None)->dict:
    geometry=default_geometry(n_layers,family) if geometry is None else geometry
    if response is None:
        response=default_response(n_layers+(family=='lens'),seed)
    if response.get('parameter_status') is None:
        raise ValueError('response parameter status must be explicit')
    if response.get('low_pore_correction_assumed') is not True:
        raise ValueError('normalized Ic without u2 unsupported without explicit low-pore-correction assumption')
    if len(response['units'])!=n_layers+(family=='lens'): raise ValueError('response unit count mismatch')
    if min(response['log_qc_field_sd'],response['log_fr_field_sd'],response['field_correlation_length_m'])<=0: raise ValueError('invalid response variability')
    rho=response['shared_field_correlation']
    if not -1<=rho<=1: raise ValueError('shared field correlation outside [-1,1]')
    boundaries=_boundaries(grid,family,n_layers,geometry)
    depth=grid.depth_m()[None,None,:]
    unit=np.zeros((grid.nx,grid.ny,grid.nz),dtype=np.int16)
    for b in boundaries: unit+=(depth>=b[:,:,None]).astype(np.int16)
    lens_info=None
    if family=='lens':
        mask,lens_info=_lens_mask(grid,unit,boundaries,geometry['lens'])
        unit[mask]=n_layers
    shape=unit.shape
    rng=np.random.default_rng(np.random.SeedSequence([seed,204]))
    shared,ind_q,ind_f=(_field(rng,shape,grid,response['field_correlation_length_m']) for _ in range(3))
    qbase=np.array([u['qc_mpa_median'] for u in response['units']],float)
    frbase=np.array([u['fr_percent_median'] for u in response['units']],float)
    gamma_base=np.array([u['gamma_kn_m3'] for u in response['units']],float)
    if np.any(qbase<=0) or np.any(frbase<=0) or np.any(gamma_base<=0): raise ValueError('response medians and unit weights must be positive')
    depth3=np.broadcast_to(depth,shape)
    qc=qbase[unit]*np.exp(response['log_qc_field_sd']*(rho*shared+math.sqrt(1-rho*rho)*ind_q)+response['qc_depth_log_slope_per_m']*depth3)
    fr=frbase[unit]*np.exp(response['log_fr_field_sd']*(rho*shared+math.sqrt(1-rho*rho)*ind_f)+response['fr_depth_log_slope_per_m']*depth3)
    gamma=gamma_base[unit]
    sigma=np.cumsum(gamma*grid.dz_m/1000,axis=2)-.5*gamma*grid.dz_m/1000
    fs=fr/100*(qc-sigma)
    qc,fs,sigma=(a.astype(np.float32) for a in (qc,fs,sigma))
    qt,Qt,Fr,Ic=interpret(qc,fs,sigma,sigma,qt_approximation=True)
    sbt=classify_sbt(Ic)
    metadata={'schema_version':SCHEMA_VERSION,'scenario_id':scenario_id,'parameter_family_id':parameter_family_id,
              'realization_id':realization_id,'realization_seed':seed,'family':family,'n_background_layers':n_layers,
              'grid':asdict(grid),'domain_m':[grid.nx*grid.dx_m,grid.ny*grid.dy_m,grid.nz*grid.dz_m],
              'coordinate_system':'local rotated Cartesian, voxel centers',
              'depth_convention':'positive_down_from_flat_surface',
              'surface':{'type':'flat','elevation_m':grid.surface_elevation_m},
              'groundwater':{'water_table_depth_m':grid.nz*grid.dz_m+10,'u0_mpa':0.0},
              'method':METHOD,'qt_approximation':True,
              'qt_approximation_reason':response['low_pore_correction_basis'],
              'sbt_method':'Robertson-1990-Ic-zones-2-to-7-prototype-v2',
              'sbt_names':SBT_NAMES,'sbt_bounds':SBT_BOUNDS,'sbt_rules':SBT_RULES,
              'units':{'qc':'MPa','fs':'MPa','qt':'MPa','sigma_v0':'MPa','sigma_v0_eff':'MPa','u0':'MPa','depth':'m'},
              'response_assumptions':response,'geology_params':geometry,'lens':lens_info,
              'minimum_background_thickness_m':geometry['minimum_thickness_voxels']*grid.dz_m,
              'limitations':'Illustrative response ranges; not field-calibrated. No u2 modeled.'}
    return {'unit_id':unit,'boundaries_m':boundaries,'Ic':Ic,'sbt_id':sbt,'qc_mpa':qc,'fs_mpa':fs,
            'qt_mpa':qt,'Qt':Qt,'Fr_percent':Fr,'sigma_v0_mpa':sigma,'sigma_v0_eff_mpa':sigma.copy(),
            'u0_mpa':np.zeros(shape,dtype=np.float32),'valid_mask':np.ones(shape,dtype=bool),'metadata':metadata}


def validate(volume:dict,observation:dict|None=None)->list[str]:
    errors=validate_common(volume,observation)
    m=volume['metadata']; grid=Grid(**m['grid'])
    if m['schema_version']!=SCHEMA_VERSION: errors.append('wrong refined schema version')
    if m.get('method')!=METHOD or m.get('sbt_rules')!=SBT_RULES or tuple(m.get('sbt_bounds',()))!=SBT_BOUNDS:
        errors.append('interpretation metadata inconsistent')
    if m.get('units')!={'qc':'MPa','fs':'MPa','qt':'MPa','sigma_v0':'MPa','sigma_v0_eff':'MPa','u0':'MPa','depth':'m'}:
        errors.append('unit metadata inconsistent')
    if m['response_assumptions'].get('low_pore_correction_assumed') is not True: errors.append('qt approximation assumption missing')
    try:
        expected=_boundaries(grid,m['family'],m['n_background_layers'],m['geology_params'])
        if not np.array_equal(expected,volume['boundaries_m']): errors.append('boundary geometry differs from parameters')
        if m['geology_params']['mode']=='varying' and m['n_background_layers']>1:
            b=np.concatenate((volume['boundaries_m'][:1],np.diff(volume['boundaries_m'],axis=0),grid.nz*grid.dz_m-volume['boundaries_m'][-1:]),axis=0)
            if max(float(np.ptp(layer)) for layer in b)<.02*grid.dz_m: errors.append('lateral thickness variation absent')
        if m['family']=='lens':
            background=np.zeros_like(volume['unit_id'])
            for boundary in volume['boundaries_m']: background+=(grid.depth_m()[None,None,:]>=boundary[:,:,None]).astype(np.int16)
            expected_lens,info=_lens_mask(grid,background,volume['boundaries_m'],m['geology_params']['lens'])
            actual_lens=volume['unit_id']==m['n_background_layers']
            if not np.array_equal(expected_lens,actual_lens) or info!=m['lens']: errors.append('lens geometry/metadata mismatch')
    except (ValueError,KeyError) as exc: errors.append(f'geology: {exc}')
    response=m['response_assumptions']; unit=volume['unit_id']
    if len(response['units'])!=int(unit.max())+1: errors.append('response unit count mismatch')
    for name in ('qc_mpa','fs_mpa'):
        a=volume[name]
        if np.any(a<=0): errors.append(f'{name}: nonpositive response')
        if not np.all(np.isfinite(a)): errors.append(f'{name}: nonfinite response')
        if float(a.std())<=0: errors.append(f'{name}: no variation')
    if np.any(volume['qt_mpa']-volume['sigma_v0_mpa']<=0): errors.append('net cone resistance nonpositive')
    if m['family']=='lens' and len(np.unique(unit))!=m['n_background_layers']+1: errors.append('lens or background unit absent')
    for uid in np.unique(unit):
        selected=unit==uid
        if selected.sum()<2 or volume['qc_mpa'][selected].std()<=0 or volume['fs_mpa'][selected].std()<=0:
            errors.append(f'unit {uid}: qc/fs intraunit variation absent')
    # Overlap and association are diagnostics, not a single-threshold leakage verdict.
    return errors


def validate_reproducibility(volume:dict)->list[str]:
    m=volume['metadata']
    try:
        replica=generate(Grid(**m['grid']),m['family'],m['n_background_layers'],m['realization_seed'],
                         scenario_id=m['scenario_id'],parameter_family_id=m['parameter_family_id'],realization_id=m['realization_id'],
                         geometry=m['geology_params'],response=m['response_assumptions'])
    except (ValueError,KeyError) as exc: return [f'regeneration failed: {exc}']
    return [f'{k}: reproduction mismatch' for k in replica if k!='metadata' and not np.array_equal(replica[k],volume[k])]


def split(metadata:dict)->str:
    if metadata['family']=='lens': return 'test_family'
    if metadata['parameter_family_id']=='high_geometry': return 'test_parameter'
    if metadata['parameter_family_id']=='high_response': return 'test_response_parameter'
    key=f"{SCHEMA_VERSION}|{metadata['scenario_id']}|{metadata['realization_id']}"
    bucket=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%10
    return 'train' if bucket<7 else ('validation' if bucket<9 else 'test_id')


def validate_manifest(entries:list[dict])->list[str]:
    errors=[]; seen={}
    for e in entries:
        rid=e['realization_id']; signature=(e['split'],e['realization_seed'],e['scenario_id'],e['parameter_family_id'])
        if rid in seen: errors.append(f'duplicate realization_id: {rid}')
        seen[rid]=signature
        if e['split']!=split(e): errors.append(f'split mismatch: {rid}')
        if e['family']=='lens' and e['split']!='test_family': errors.append(f'family leakage: {rid}')
        if e['parameter_family_id']=='high_geometry' and e['family']!='lens' and e['split']!='test_parameter': errors.append(f'parameter leakage: {rid}')
    return errors


def response_diagnostics(volume:dict)->dict:
    """Report per-unit variation, cross-response correlation, and range overlap."""
    unit=volume['unit_id']; q=volume['qc_mpa']; f=volume['fs_mpa']
    records=[]
    for uid in np.unique(unit):
        mask=unit==uid; qv=q[mask].astype(float); fv=f[mask].astype(float)
        records.append({'unit_id':int(uid),'voxels':int(mask.sum()),
                        'qc_mpa_mean':float(qv.mean()),'qc_mpa_sd':float(qv.std()),
                        'fs_mpa_mean':float(fv.mean()),'fs_mpa_sd':float(fv.std()),
                        'qc_fs_pearson':float(np.corrcoef(qv,fv)[0,1]) if len(qv)>2 and qv.std()>0 and fv.std()>0 else None,
                        'qc_mpa_p10_p90':[float(x) for x in np.quantile(qv,[.1,.9])]})
    overlaps=0
    for i in range(len(records)):
        for j in range(i+1,len(records)):
            a,b=records[i]['qc_mpa_p10_p90'],records[j]['qc_mpa_p10_p90']
            overlaps+=int(max(a[0],b[0])<=min(a[1],b[1]))
    return {'units':records,'qc_p10_p90_overlap_pairs':overlaps,
            'qc_fs_pearson_all':float(np.corrcoef(q.ravel(),f.ravel())[0,1])}


def diagnostics(entries:list[dict],root, *, near_corr=.995)->dict:
    """Similarity diagnostic only; no single threshold is used as a leakage verdict."""
    items=[]
    for e in entries:
        a=np.load(root/e['arrays']['Ic'],allow_pickle=False).astype(np.float64)
        items.append((e,a))
    pairs=[]
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            ea,a=items[i]; eb,b=items[j]
            same_shape=a.shape==b.shape
            corr=float(np.corrcoef(a.ravel(),b.ravel())[0,1]) if same_shape else None
            rmse=float(np.sqrt(np.mean((a-b)**2))) if same_shape else None
            exact=bool(same_shape and np.array_equal(a,b))
            quantile_distance=float(np.mean(np.abs(np.quantile(a,[.05,.25,.5,.75,.95])-np.quantile(b,[.05,.25,.5,.75,.95]))))
            za=a.mean(axis=(0,1)); zb=b.mean(axis=(0,1))
            common_z=np.linspace(0,1,16)
            pa=np.interp(common_z,np.linspace(0,1,len(za)),za)
            pb=np.interp(common_z,np.linspace(0,1,len(zb)),zb)
            profile_corr=float(np.corrcoef(pa,pb)[0,1]) if pa.std()>0 and pb.std()>0 else None
            pairs.append({'a':ea['realization_id'],'b':eb['realization_id'],'split_a':ea['split'],'split_b':eb['split'],
                          'shape_a':list(a.shape),'shape_b':list(b.shape),'pearson_ic':corr,'rmse_ic':rmse,
                          'ic_quantile_distance':quantile_distance,'normalized_depth_profile_pearson':profile_corr,'exact_duplicate':exact,
                          'near_correlation_flag':bool(corr is not None and corr>=near_corr)})
    cross=[p for p in pairs if p['split_a']!=p['split_b']]
    comparable=[p for p in cross if p['pearson_ic'] is not None]
    return {'all_pairs':len(pairs),'cross_split_pairs':len(cross),'same_shape_cross_split_pairs':len(comparable),
            'exact_cross_split_duplicates':sum(p['exact_duplicate'] for p in cross),
            'near_corr_threshold_diagnostic':near_corr,'near_corr_cross_split_pairs':sum(p['near_correlation_flag'] for p in cross),
            'max_cross_split_pearson':max((p['pearson_ic'] for p in comparable),default=None),
            'min_cross_split_rmse_ic':min((p['rmse_ic'] for p in comparable),default=None),
            'min_cross_split_quantile_distance':min((p['ic_quantile_distance'] for p in cross),default=None),
            'max_cross_split_normalized_depth_profile_pearson':max((p['normalized_depth_profile_pearson'] for p in cross if p['normalized_depth_profile_pearson'] is not None),default=None),
            'pairs':pairs}
