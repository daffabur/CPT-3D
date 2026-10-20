"""Controlled synthetic ML experiment generator. Priors are NOT field calibrated."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from .core import Grid, METHOD, SBT_BOUNDS, classify_sbt, extract_cpt, interpret
from .refined import _boundaries, _lens_mask, default_geometry

VERSION = 'v2-prototype-3'
GROUPS = ('sand_dominant', 'silt_dominant', 'clay_dominant', 'transitional')
# Experimental priors: deliberately overlapping, not material-calibrated ranges.
EXPERIMENTAL_PRIOR = {
    'sand_dominant': {'log_qc_mean': math.log(7.0), 'log_fr_mean': math.log(1.4), 'gamma_kn_m3': 19.0},
    'silt_dominant': {'log_qc_mean': math.log(5.0), 'log_fr_mean': math.log(2.0), 'gamma_kn_m3': 18.5},
    'clay_dominant': {'log_qc_mean': math.log(3.5), 'log_fr_mean': math.log(2.5), 'gamma_kn_m3': 18.0},
    'transitional': {'log_qc_mean': math.log(5.5), 'log_fr_mean': math.log(1.8), 'gamma_kn_m3': 18.5},
}
DEFAULT_RESPONSE = {'prior': EXPERIMENTAL_PRIOR, 'log_covariance': [[0.24**2, -0.012],[-0.012, 0.23**2]],
                    'unit_state_sd': [0.10, 0.08], 'gamma_state_sd_kn_m3': 0.35,
                    'stress_exponent_qc': 0.20, 'stress_exponent_fr': -0.06,
                    'reference_eff_stress_mpa': 0.10,
                    'kernel_sigma_m': [4.0, 4.0, 2.0], 'material_kernel_sigma_m': [5.0,5.0,2.0]}

def _positive_covariance(value):
    cov=np.asarray(value,dtype=float)
    if cov.shape!=(2,2) or not np.all(np.isfinite(cov)) or not np.allclose(cov,cov.T,atol=1e-12) or np.min(np.linalg.eigvalsh(cov))<=0:
        raise ValueError('log covariance must be symmetric positive definite 2x2')
    return cov

def _field(rng, shape, grid, kernel_sigma_m):
    if len(kernel_sigma_m)!=3 or any(not np.isfinite(v) or v<=0 for v in kernel_sigma_m):
        raise ValueError('kernel sigma must contain three positive finite lengths')
    sigma=np.maximum(.5,np.asarray(kernel_sigma_m)/[grid.dx_m,grid.dy_m,grid.dz_m])
    arr=gaussian_filter(rng.normal(size=shape),sigma=sigma,mode='reflect')
    arr-=arr.mean()
    if arr.std()<1e-10: raise ValueError('spatial field has negligible variance')
    return arr/arr.std()

def _unit_mixtures(n_units, seed, options=None):
    rng=np.random.default_rng(np.random.SeedSequence([seed,301]))
    if options is not None:
        p=np.asarray(options,dtype=float)
        if p.shape!=(n_units,len(GROUPS)) or not np.all(np.isfinite(p)) or np.any(p<0) or not np.allclose(p.sum(axis=1),1):
            raise ValueError('invalid unit material mixtures')
    else:
        p=np.stack([rng.dirichlet(np.ones(len(GROUPS))*2) for _ in range(n_units)])
    return p

def _geometry_from_seed(n_layers, family, seed):
    """Small reproducible geometry variation around the refined prototype defaults."""
    p=default_geometry(n_layers,family)
    rng=np.random.default_rng(np.random.SeedSequence([seed,303]))
    p['thickness_weights']=[float(v*rng.uniform(.94,1.06)) for v in p['thickness_weights']]
    p['slope_direction_rad']+=float(rng.uniform(-.15,.15))
    p['boundary_phase_rad']=[float(v+rng.uniform(-.15,.15)) for v in p['boundary_phase_rad']]
    if p['lens'] is not None:
        p['lens']['center_x_fraction']+=float(rng.uniform(-.025,.025))
        p['lens']['center_y_fraction']+=float(rng.uniform(-.025,.025))
    return p

def generate(grid:Grid, family:str, n_layers:int, seed:int, *, scenario_id:str, realization_id:str,
             geometry=None, response=None, mixtures=None, geology_seed=None, material_seed=None):
    if not isinstance(seed,int) or not isinstance(geology_seed if geology_seed is not None else seed,int):
        raise ValueError('seeds must be integers')
    geo_seed=seed if geology_seed is None else geology_seed
    mat_seed=seed if material_seed is None else material_seed
    geometry=_geometry_from_seed(n_layers,family,geo_seed) if geometry is None else geometry
    response=json.loads(json.dumps(DEFAULT_RESPONSE if response is None else response))
    cov=_positive_covariance(response['log_covariance'])
    if set(response['prior'])!=set(GROUPS): raise ValueError('all material priors required')
    if response['reference_eff_stress_mpa']<=0 or any(v<0 for v in response['unit_state_sd']) or response['gamma_state_sd_kn_m3']<0:
        raise ValueError('invalid response state settings')
    bounds=_boundaries(grid,family,n_layers,geometry)
    depth=grid.depth_m()[None,None,:]
    unit=np.zeros((grid.nx,grid.ny,grid.nz),dtype=np.int16)
    for b in bounds: unit+=(depth>=b[:,:,None]).astype(np.int16)
    lens_info=None
    if family=='lens':
        lens,lens_info=_lens_mask(grid,unit,bounds,geometry['lens'])
        unit[lens]=n_layers
    shape=unit.shape; n_units=n_layers+int(family=='lens')
    mix=_unit_mixtures(n_units,mat_seed,mixtures)
    rng=np.random.default_rng(np.random.SeedSequence([mat_seed,302]))
    # A Gaussian score field supplies spatially coherent categorical mixtures within each unit.
    scores=np.stack([_field(rng,shape,grid,response['material_kernel_sigma_m']) for _ in GROUPS])
    scores+=np.log(np.maximum(mix[unit].transpose(3,0,1,2),1e-12))
    material=np.argmax(scores,axis=0).astype(np.uint8)
    state=rng.normal(size=(n_units,2))*np.asarray(response['unit_state_sd'])
    gamma_state=rng.normal(size=n_units)*response['gamma_state_sd_kn_m3']
    priors=[response['prior'][g] for g in GROUPS]
    gamma_base=np.array([p['gamma_kn_m3'] for p in priors])[material]+gamma_state[unit]
    if np.any(gamma_base<=0): raise ValueError('nonpositive unit weight')
    sigma=np.cumsum(gamma_base*grid.dz_m/1000,axis=2)-.5*gamma_base*grid.dz_m/1000
    # G0: u0=0, sigma_eff=sigma, qt≈qc by explicitly declared experimental assumption.
    if np.any(sigma<=0): raise ValueError('nonpositive effective stress')
    stress=np.log(sigma/response['reference_eff_stress_mpa'])
    z=np.stack([_field(rng,shape,grid,response['kernel_sigma_m']) for _ in range(2)])
    correlated=np.einsum('ij,jxyz->ixyz',np.linalg.cholesky(cov),z)
    log_qc=np.array([p['log_qc_mean'] for p in priors])[material]+state[unit,0]+response['stress_exponent_qc']*stress+correlated[0]
    log_fr=np.array([p['log_fr_mean'] for p in priors])[material]+state[unit,1]+response['stress_exponent_fr']*stress+correlated[1]
    qc=np.exp(log_qc); fr_latent=np.exp(log_fr)
    fs=fr_latent*(qc-sigma)/100
    if not np.all(np.isfinite(qc)) or not np.all(np.isfinite(fs)) or np.any(fs<=0):
        raise ValueError('invalid generated CPT response; no clipping applied')
    qc,fs,sigma,gamma=(np.asarray(x,dtype=np.float32) for x in (qc,fs,sigma,gamma_base))
    qt,Qt,Fr,Ic=interpret(qc,fs,sigma,sigma,qt_approximation=True)
    metadata={'schema_version':VERSION,'method_version':METHOD,'sbt_method_version':'Robertson-1990-Ic-zones-2-to-7-prototype-v3',
      'scenario_id':scenario_id,'realization_id':realization_id,'family':family,'n_background_layers':n_layers,
      'seed':seed,'geology_seed':geo_seed,'material_seed':mat_seed,'grid':asdict(grid),
      'geology_params':geometry,'lens':lens_info,'groundwater_scenario':'G0-flat-water-table-below-domain',
      'water_table_depth_m':grid.nz*grid.dz_m+10.0,'qt_approximation':True,
      'qt_approximation_reason':'experimental low cone pore-pressure correction assumption; water table alone is not evidence',
      'parameter_evidence_status':'experimental_prior; not field-calibrated',
      'parameter_provenance':{'CPT_formula':'Robertson 1990/2010 Qt-Fr-Ic n=1','random_field':'experimental Gaussian kernel; no empirical correlation length claimed'},
      'response_settings':response,'unit_mixtures':mix.tolist(),'unit_state':state.tolist(),'gamma_state_kn_m3':gamma_state.tolist(),
      'material_groups':GROUPS,'material_group_id_map':dict(enumerate(GROUPS)),
      'units':{'qc':'MPa','fs':'MPa','qt':'MPa','Fr':'percent','Qt':'dimensionless','Ic':'dimensionless','gamma':'kN/m3','stress':'MPa','u0':'MPa','depth':'m'},
      'sbt_bounds':SBT_BOUNDS,'known_limitations':['not field calibrated','u0=0 and sigma_eff=sigma are G0 assumptions','qt≈qc unverified','fs constructed from qc and latent Fr','full-depth voxel-center CPT only']}
    return {'unit_id':unit,'material_group':material,'boundaries_m':bounds,'valid_mask':np.ones(shape,dtype=bool),
      'qc_mpa':qc,'fs_mpa':fs,'qt_mpa':qt,'Fr_percent':Fr,'Qt':Qt,'Ic':Ic,'sbt_id':classify_sbt(Ic),
      'gamma_kn_m3':gamma,'sigma_v0_mpa':sigma,'sigma_v0_eff_mpa':sigma.copy(),'u0_mpa':np.zeros(shape,dtype=np.float32),'metadata':metadata}

def direction_autocorrelation(arr, axis, lag=1):
    arr=np.asarray(arr,float)
    if arr.ndim!=3 or not 0<=axis<3 or lag<1 or lag>=arr.shape[axis]: raise ValueError('invalid autocorrelation dimensions')
    a=np.take(arr,range(arr.shape[axis]-lag),axis=axis).ravel()
    b=np.take(arr,range(lag,arr.shape[axis]),axis=axis).ravel()
    if a.std()==0 or b.std()==0: raise ValueError('constant directional slice')
    return float(np.corrcoef(a,b)[0,1])

def directional_diagnostics(volume):
    q=np.log(volume['qc_mpa'])
    return {name:{'lag1_autocorrelation':direction_autocorrelation(q,i),
                  'lag1_semivariogram':float(np.mean((np.diff(q,axis=i))**2)/2),
                  'lag_m':float(volume['metadata']['grid'][key])} for i,(name,key) in enumerate((('x','dx_m'),('y','dy_m'),('z','dz_m')))}

def validate(volume, observation=None):
    e=[]; m=volume['metadata']; g=Grid(**m['grid']); shape=(g.nx,g.ny,g.nz)
    required=('unit_id','material_group','valid_mask','qc_mpa','fs_mpa','qt_mpa','Fr_percent','Qt','Ic','sbt_id','gamma_kn_m3','sigma_v0_mpa','sigma_v0_eff_mpa','u0_mpa')
    for k in required:
        if k not in volume or volume[k].shape!=shape: e.append(f'{k}: missing or wrong shape')
    if e:return e
    if m['schema_version']!=VERSION or m['method_version']!=METHOD:e.append('method version mismatch')
    expected_units={'qc':'MPa','fs':'MPa','qt':'MPa','Fr':'percent','Qt':'dimensionless','Ic':'dimensionless','gamma':'kN/m3','stress':'MPa','u0':'MPa','depth':'m'}
    if m.get('units')!=expected_units:e.append('unit metadata mismatch')
    if m['groundwater_scenario']!='G0-flat-water-table-below-domain' or m['water_table_depth_m']<=g.nz*g.dz_m or m['qt_approximation'] is not True:e.append('G0 metadata mismatch')
    if not np.array_equal(volume['u0_mpa'],np.zeros(shape,dtype=np.float32)) or not np.array_equal(volume['sigma_v0_mpa'],volume['sigma_v0_eff_mpa']):e.append('G0 stress mismatch')
    if np.any(volume['material_group']>=len(GROUPS)):e.append('invalid material group')
    mix=np.asarray(m['unit_mixtures'])
    if mix.shape!=(int(volume['unit_id'].max())+1,len(GROUPS)) or np.any(mix<0) or not np.allclose(mix.sum(axis=1),1):e.append('invalid mixture')
    try:_positive_covariance(m['response_settings']['log_covariance'])
    except ValueError as ex:e.append(str(ex))
    for k in required[3:]:
        if not np.all(np.isfinite(volume[k])):e.append(f'{k}: nonfinite')
    if np.any(volume['gamma_kn_m3']<=0) or np.any(volume['qc_mpa']<=0) or np.any(volume['fs_mpa']<=0) or np.any(volume['sigma_v0_eff_mpa']<=0) or np.any(volume['qt_mpa']-volume['sigma_v0_mpa']<=0):e.append('invalid Ic domain')
    expected=np.cumsum(volume['gamma_kn_m3']*g.dz_m/1000,axis=2)-.5*volume['gamma_kn_m3']*g.dz_m/1000
    if not np.allclose(expected,volume['sigma_v0_mpa'],rtol=1e-5,atol=1e-7):e.append('total stress equation mismatch')
    try:
        qt,Qt,Fr,Ic=interpret(volume['qc_mpa'],volume['fs_mpa'],volume['sigma_v0_mpa'],volume['sigma_v0_eff_mpa'],qt_approximation=True)
        for k,v in (('qt_mpa',qt),('Qt',Qt),('Fr_percent',Fr),('Ic',Ic)):
            if not np.allclose(volume[k],v,rtol=1e-5,atol=1e-6):e.append(f'{k}: interpretation mismatch')
        if not np.array_equal(classify_sbt(Ic),volume['sbt_id']):e.append('SBT mismatch')
    except ValueError as ex:e.append(f'interpretation: {ex}')
    if observation is not None:
        profiles=observation.get('profiles',[]); mask=observation.get('observed_mask')
        if len(profiles)<1 or mask is None or mask.shape!=shape:e.append('observation missing')
        else:
            exp=np.zeros(shape,dtype=bool)
            for p in profiles:
                x,y=p['grid_xy'];exp[x,y,:]=volume['valid_mask'][x,y,:]
                valid=volume['valid_mask'][x,y,:]
                if g.index_xy(*p['xy_m'])!=(x,y):e.append('profile coordinate mismatch')
                for field,key in (('qc_mpa','qc_mpa'),('fs_mpa','fs_mpa'),('Ic','Ic'),('sbt_id','sbt_id')):
                    if not np.allclose(p[key],volume[field][x,y,valid]):e.append(f'profile {key} mismatch')
                if not np.allclose(p['depth_m'],g.depth_m()[valid]) or not np.allclose(p['elevation_m'],g.surface_elevation_m-g.depth_m()[valid]):e.append('profile depth mismatch')
                if set(p)-{'grid_xy','xy_m','depth_m','elevation_m','qc_mpa','fs_mpa','Ic','sbt_id'}:e.append('unexpected observation field')
            if not np.array_equal(exp,mask):e.append('observation mask mismatch')
    return e

def fingerprint(volume):
    h=hashlib.sha256()
    for k in ('unit_id','material_group','qc_mpa','fs_mpa','Ic'):
        h.update(np.asarray(volume[k]).tobytes())
    return h.hexdigest()

def save(volume, observation, root:Path, split:str, pair_id:str, pair_role:str):
    rid=volume['metadata']['realization_id']; p=root/rid
    if p.exists():raise FileExistsError(p)
    p.mkdir(parents=True)
    arrays={}
    for k,v in volume.items():
        if k=='metadata':continue
        target=p/f'{k}.npy';np.save(target,v);arrays[k]=str(target.relative_to(root))
    (p/'metadata.json').write_text(json.dumps(volume['metadata'],indent=2)+'\n')
    (p/'profiles.json').write_text(json.dumps(observation['profiles'],indent=2)+'\n')
    np.save(p/'observed_mask.npy',observation['observed_mask'])
    return {'realization_id':rid,'scenario_id':volume['metadata']['scenario_id'],'family':volume['metadata']['family'],
            'split':split,'pair_id':pair_id,'pair_role':pair_role,'shape':list(volume['Ic'].shape),
            'spacing_m':[volume['metadata']['grid'][x] for x in ('dx_m','dy_m','dz_m')],
            'n_cpt':len(observation['profiles']),'fingerprint':fingerprint(volume),'arrays':arrays,
            'metadata':str((p/'metadata.json').relative_to(root)),'profiles':str((p/'profiles.json').relative_to(root)),
            'observed_mask':str((p/'observed_mask.npy').relative_to(root)),
            'Ic_range':[float(volume['Ic'].min()),float(volume['Ic'].max())],
            'sbt_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(volume['sbt_id'],return_counts=True))},
            'material_counts':{GROUPS[int(k)]:int(v) for k,v in zip(*np.unique(volume['material_group'],return_counts=True))},
            'directional_log_qc':directional_diagnostics(volume)}
