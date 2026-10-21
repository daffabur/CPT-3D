"""Independent, small-scale Synthetic Dataset V2 prototype. No model imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from scipy.ndimage import gaussian_filter

METHOD = "Robertson-1990-Qt-Fr-Ic-n1-prototype-v1"
SBT_NAMES = {0: "unclassified", 2: "organic_soil_clay", 3: "clay", 4: "silt_mixture", 5: "sand_mixture", 6: "sand", 7: "dense_sand_gravelly_sand"}
SBT_BOUNDS = (1.31, 2.05, 2.60, 2.95, 3.60)
# Illustrative prototype ranges, not field-calibrated geotechnical parameters.
RESPONSE = {"qc_mpa_by_unit": [2.0, 5.0, 9.0, 14.0, 6.0],
            "fr_percent_by_unit": [3.3, 2.4, 1.4, 0.9, 2.0],
            "gamma_kn_m3_by_unit": [17.0, 18.0, 19.0, 20.0, 18.5],
            "log_qc_field_sd": 0.20, "log_fr_field_sd": 0.18,
            "field_correlation_length_m": 4.0, "shared_field_correlation": 0.55}

@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    nz: int
    dx_m: float
    dy_m: float
    dz_m: float
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0
    azimuth_deg: float = 0.0
    surface_elevation_m: float = 0.0

    def __post_init__(self):
        if any(not isinstance(v, int) or isinstance(v, bool) for v in (self.nx, self.ny, self.nz)) or min(self.nx, self.ny, self.nz) < 4:
            raise ValueError("grid dimensions must be integers >=4")
        if not all(math.isfinite(v) for v in (self.dx_m, self.dy_m, self.dz_m, self.origin_x_m, self.origin_y_m, self.azimuth_deg, self.surface_elevation_m)) or min(self.dx_m, self.dy_m, self.dz_m) <= 0:
            raise ValueError("grid dimensions must be >=4 and spacings positive")

    def xy_m(self, x: int, y: int) -> tuple[float, float]:
        if not (0 <= x < self.nx and 0 <= y < self.ny):
            raise ValueError("grid coordinate outside domain")
        angle = math.radians(self.azimuth_deg)
        east, north = (x + .5) * self.dx_m, (y + .5) * self.dy_m
        return (self.origin_x_m + math.cos(angle)*east - math.sin(angle)*north,
                self.origin_y_m + math.sin(angle)*east + math.cos(angle)*north)

    def index_xy(self, x_m: float, y_m: float) -> tuple[int, int]:
        angle = math.radians(self.azimuth_deg)
        east, north = x_m-self.origin_x_m, y_m-self.origin_y_m
        gx = (math.cos(angle)*east + math.sin(angle)*north)/self.dx_m - .5
        gy = (-math.sin(angle)*east + math.cos(angle)*north)/self.dy_m - .5
        x, y = round(gx), round(gy)
        if not (0 <= x < self.nx and 0 <= y < self.ny) or abs(gx-x) > 1e-7 or abs(gy-y) > 1e-7:
            raise ValueError("CPT XY must be at a valid voxel center")
        return x, y

    def depth_m(self) -> np.ndarray:
        return (np.arange(self.nz) + .5)*self.dz_m


def _field(rng: np.random.Generator, shape: tuple[int, int, int], grid: Grid) -> np.ndarray:
    sigma = tuple(max(.5, RESPONSE['field_correlation_length_m']/d) for d in (grid.dx_m, grid.dy_m, grid.dz_m))
    value = gaussian_filter(rng.normal(size=shape), sigma=sigma, mode='reflect')
    value -= value.mean()
    return value / max(float(value.std()), 1e-9)


def _geology(grid: Grid, family: str, n_layers: int, rng: np.random.Generator, params: dict):
    if family not in ('horizontal', 'inclined', 'undulating', 'lens'):
        raise ValueError(f"unknown geological family: {family}")
    if n_layers not in (2, 3, 4):
        raise ValueError("prototype supports 2–4 background layers")
    zmax = grid.nz * grid.dz_m
    minimum = 2 * grid.dz_m
    if zmax < n_layers * minimum:
        raise ValueError("domain too shallow for minimum two-voxel layers")
    x = (np.arange(grid.nx) + .5)/grid.nx - .5
    y = (np.arange(grid.ny) + .5)/grid.ny - .5
    xx, yy = np.meshgrid(x, y, indexing='ij')
    shift = np.zeros((grid.nx, grid.ny), dtype=float)
    if family == 'inclined':
        shift = params['slope_fraction'] * zmax * (xx*math.cos(params['direction_rad']) + yy*math.sin(params['direction_rad']))
    elif family == 'undulating':
        shift = params['amplitude_fraction'] * zmax * np.sin(2*math.pi*xx/params['wavelength_fraction'] + params['phase_rad']) * np.cos(2*math.pi*yy/params['wavelength_fraction'])
    # Common shift preserves boundary order and thickness.
    fractions = np.cumsum(np.asarray(params['thickness_weights'], float))[:-1]
    fractions /= sum(params['thickness_weights'])
    boundaries = np.stack([np.full_like(shift, f*zmax) + shift for f in fractions], axis=0)
    if np.any(boundaries[0] < minimum) or np.any(boundaries[-1] > zmax-minimum) or (len(boundaries)>1 and np.any(np.diff(boundaries, axis=0) < minimum)):
        raise ValueError("geological boundaries violate minimum thickness or domain")
    depth = grid.depth_m()[None,None,:]
    unit = np.zeros((grid.nx,grid.ny,grid.nz), dtype=np.int16)
    for boundary in boundaries:
        unit += (depth >= boundary[:,:,None]).astype(np.int16)
    lens_info = None
    if family == 'lens':
        parent = n_layers//2
        cx, cy = (grid.nx-.5)*grid.dx_m/2, (grid.ny-.5)*grid.dy_m/2
        rx, ry = .24*grid.nx*grid.dx_m, .24*grid.ny*grid.dy_m
        mid = (boundaries[parent-1] + (boundaries[parent] if parent < n_layers-1 else zmax))/2 if parent > 0 else boundaries[0]/2
        rz = grid.dz_m  # two-voxel vertical diameter before discretization
        gx = (np.arange(grid.nx)+.5)*grid.dx_m
        gy = (np.arange(grid.ny)+.5)*grid.dy_m
        ell = ((gx[:,None,None]-cx)/rx)**2 + ((gy[None,:,None]-cy)/ry)**2 + ((depth-mid[:,:,None])/rz)**2
        lens = (ell < 1) & (unit == parent)
        if not lens.any():
            raise ValueError("lens absent at chosen resolution")
        unit[lens] = n_layers
        lens_info = {'parent_unit': parent, 'unit_id': n_layers, 'center_local_m': [cx,cy,float(mid.mean())], 'radii_m': [rx,ry,rz], 'shape': 'ellipsoid'}
    return unit, boundaries.astype(np.float32), lens_info


def interpret(qc_mpa, fs_mpa, sigma_v0_mpa, sigma_eff_mpa, *, u2_mpa=None, area_ratio=None, qt_approximation=False):
    if u2_mpa is None:
        if not qt_approximation:
            raise ValueError("u2 absent: qt≈qc requires explicit low-pore-correction assumption")
        qt = qc_mpa
    else:
        if area_ratio is None or not 0 < area_ratio <= 1:
            raise ValueError("CPTu requires area ratio in (0,1]")
        qt = qc_mpa + u2_mpa*(1-area_ratio)
    net = qt - sigma_v0_mpa
    if not all(np.all(np.isfinite(v)) for v in (qc_mpa, fs_mpa, sigma_v0_mpa, sigma_eff_mpa, qt)):
        raise ValueError("nonfinite CPT or stress response")
    if np.any(qc_mpa <= 0) or np.any(fs_mpa <= 0) or np.any(sigma_eff_mpa <= 0) or np.any(net <= 0):
        raise ValueError("invalid domain: qc, fs, effective stress, and net cone resistance must be positive")
    qt_norm = net/sigma_eff_mpa
    fr_percent = 100*fs_mpa/net
    ic = np.sqrt((3.47-np.log10(qt_norm))**2 + (np.log10(fr_percent)+1.22)**2)
    return qt.astype(np.float32), qt_norm.astype(np.float32), fr_percent.astype(np.float32), ic.astype(np.float32)


def classify_sbt(ic: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(ic)):
        raise ValueError("Ic contains nonfinite values")
    # Ic-only zone mapping covers zones 2–7; out-of-chart or ambiguous extremes remain 0.
    result = np.zeros(ic.shape, dtype=np.uint8)
    result[(ic >= 1.0) & (ic < 1.31)] = 7
    result[(ic >= 1.31) & (ic < 2.05)] = 6
    result[(ic >= 2.05) & (ic < 2.60)] = 5
    result[(ic >= 2.60) & (ic < 2.95)] = 4
    result[(ic >= 2.95) & (ic < 3.60)] = 3
    result[(ic >= 3.60) & (ic <= 4.0)] = 2
    return result


def generate(grid: Grid, family: str, n_layers: int, seed: int, *, scenario_id: str, parameter_family_id: str, realization_id: str, params: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    if params is None:
        params = {'thickness_weights': [1.0]*n_layers,
                  'slope_fraction': .12, 'direction_rad': .5,
                  'amplitude_fraction': .045, 'wavelength_fraction': .8, 'phase_rad': .3}
    unit, boundaries, lens_info = _geology(grid, family, n_layers, rng, params)
    shape = unit.shape
    shared, extra_q, extra_f = (_field(rng, shape, grid) for _ in range(3))
    # All units may overlap in response; unit identity does not determine SBT.
    if unit.max() >= len(RESPONSE['qc_mpa_by_unit']):
        raise ValueError('unit_id has no configured CPT response distribution')
    response_index = unit
    qc_base = np.take(RESPONSE['qc_mpa_by_unit'], response_index)
    fr_base = np.take(RESPONSE['fr_percent_by_unit'], response_index)
    gamma = np.take(RESPONSE['gamma_kn_m3_by_unit'], response_index)
    qc = qc_base*np.exp(RESPONSE['log_qc_field_sd']*(.75*shared+.25*extra_q))
    fr = fr_base*np.exp(RESPONSE['log_fr_field_sd']*(RESPONSE['shared_field_correlation']*shared + (1-RESPONSE['shared_field_correlation'])*extra_f))
    # Flat surface and water table below domain: u0=0, sigma'=sigma. Stress at voxel centers.
    sigma = np.cumsum(gamma*grid.dz_m/1000, axis=2) - .5*gamma*grid.dz_m/1000
    u0 = np.zeros(shape, dtype=np.float32)
    fs = fr/100*(qc-sigma)
    qc, fs, sigma = (np.asarray(a, dtype=np.float32) for a in (qc, fs, sigma))
    qt, qtn, fr_percent, ic = interpret(qc,fs,sigma,sigma,qt_approximation=True)
    sbt = classify_sbt(ic)
    valid = np.ones(shape, dtype=bool)
    metadata = {'schema_version': 'v2-prototype-1', 'scenario_id': scenario_id, 'parameter_family_id': parameter_family_id,
                'realization_id': realization_id, 'realization_seed': seed, 'family': family, 'n_background_layers': n_layers,
                'grid': asdict(grid), 'domain_m': [grid.nx*grid.dx_m,grid.ny*grid.dy_m,grid.nz*grid.dz_m],
                'coordinate_system': 'local rotated Cartesian, voxel centers', 'depth_convention': 'positive_down_from_flat_surface',
                'surface': {'type':'flat','elevation_m':grid.surface_elevation_m},
                'groundwater': {'water_table_depth_m':grid.nz*grid.dz_m+10, 'u0_mpa':0.0},
                'method': METHOD, 'qt_approximation': True, 'qt_approximation_reason': 'prototype low pore-pressure correction; water table below domain; field validity unverified',
                'sbt_method': 'Robertson-1990-Ic-zones-2-to-7-prototype-v1', 'sbt_names': SBT_NAMES,
                'sbt_bounds': SBT_BOUNDS, 'units': {'qc':'MPa','fs':'MPa','qt':'MPa','sigma_v0':'MPa','sigma_v0_eff':'MPa','u0':'MPa','depth':'m'},
                'response_assumptions': RESPONSE, 'geology_params': params, 'lens': lens_info,
                'minimum_background_thickness_m': 2*grid.dz_m,
                'limitations': 'Illustrative prototype response ranges; not field-calibrated or a soil-strength model.'}
    return {'unit_id':unit,'boundaries_m':boundaries,'Ic':ic,'sbt_id':sbt,'qc_mpa':qc,'fs_mpa':fs,
            'qt_mpa':qt,'Qt':qtn,'Fr_percent':fr_percent,'sigma_v0_mpa':sigma,'sigma_v0_eff_mpa':sigma.copy(),
            'u0_mpa':u0,'valid_mask':valid,'metadata':metadata}


def extract_cpt(volume: dict, sites: list[tuple[int,int]]) -> dict:
    grid=Grid(**volume['metadata']['grid'])
    if not sites or len(set(sites)) != len(sites):
        raise ValueError('at least one unique CPT site required')
    observed=np.zeros(volume['unit_id'].shape,dtype=bool)
    profiles=[]
    for x,y in sites:
        xm,ym=grid.xy_m(x,y)
        assert grid.index_xy(xm,ym)==(x,y)
        valid=volume['valid_mask'][x,y,:]
        observed[x,y,valid]=True
        profiles.append({'grid_xy':[x,y],'xy_m':[xm,ym], 'depth_m':grid.depth_m()[valid].tolist(),
                         'elevation_m':(grid.surface_elevation_m-grid.depth_m()[valid]).tolist(),
                         'qc_mpa':volume['qc_mpa'][x,y,valid].tolist(), 'fs_mpa':volume['fs_mpa'][x,y,valid].tolist(),
                         'Ic':volume['Ic'][x,y,valid].tolist(), 'sbt_id':volume['sbt_id'][x,y,valid].tolist()})
    return {'profiles':profiles,'observed_mask':observed}


def validate(volume:dict, observation:dict|None=None) -> list[str]:
    errors=[]
    m=volume['metadata']; grid=Grid(**m['grid']); shape=(grid.nx,grid.ny,grid.nz)
    fields=('unit_id','Ic','sbt_id','qc_mpa','fs_mpa','qt_mpa','Qt','Fr_percent','sigma_v0_mpa','sigma_v0_eff_mpa','u0_mpa','valid_mask')
    for name in fields:
        if name not in volume or volume[name].shape!=shape: errors.append(f'{name}: shape mismatch or missing')
    if errors: return errors
    valid=volume['valid_mask']
    if not valid.any(): errors.append('valid_mask empty')
    if volume['boundaries_m'].shape!=(m['n_background_layers']-1,grid.nx,grid.ny): errors.append('boundary shape mismatch')
    else:
        b=volume['boundaries_m']; minimum=m['minimum_background_thickness_m']; zmax=grid.nz*grid.dz_m
        if np.any(b[0]<minimum-1e-5) or np.any(b[-1]>zmax-minimum+1e-5) or (len(b)>1 and np.any(np.diff(b,axis=0)<minimum-1e-5)):
            errors.append('geological boundaries or minimum thickness invalid')
        expected=np.zeros(shape,dtype=np.int16)
        for boundary in b: expected+=(grid.depth_m()[None,None,:]>=boundary[:,:,None])
        if m['lens'] is None:
            if not np.array_equal(volume['unit_id'],expected): errors.append('unit_id disagrees with geological boundaries')
        else:
            changed=volume['unit_id']!=expected
            if not changed.any() or np.any(volume['unit_id'][changed]!=m['lens']['unit_id']) or np.any(expected[changed]!=m['lens']['parent_unit']):
                errors.append('lens absent or overrides wrong parent unit')
    for name in fields:
        if name not in ('unit_id','sbt_id','valid_mask') and not np.all(np.isfinite(volume[name][valid])): errors.append(f'{name}: NaN/Inf on valid voxels')
    try:
        qt, qtn, fr, ic=interpret(volume['qc_mpa'],volume['fs_mpa'],volume['sigma_v0_mpa'],volume['sigma_v0_eff_mpa'],qt_approximation=m['qt_approximation'])
        for name,calculated in (('qt_mpa',qt),('Qt',qtn),('Fr_percent',fr),('Ic',ic)):
            if not np.allclose(volume[name][valid],calculated[valid],rtol=2e-5,atol=2e-5): errors.append(f'{name}: formula mismatch')
    except ValueError as exc: errors.append(f'CPT domain: {exc}')
    if np.all(np.isfinite(volume['Ic'][valid])):
        if not np.array_equal(volume['sbt_id'][valid],classify_sbt(volume['Ic'])[valid]): errors.append('SBT does not match Ic')
    else:
        errors.append('SBT cannot be verified: Ic nonfinite')
    if np.any(volume['sigma_v0_mpa'][valid]<volume['u0_mpa'][valid]) or not np.allclose(volume['sigma_v0_eff_mpa'][valid],volume['sigma_v0_mpa'][valid]-volume['u0_mpa'][valid],rtol=1e-5): errors.append('effective stress inconsistent')
    if observation is not None:
        mask=np.zeros(shape,dtype=bool)
        for p in observation['profiles']:
            try: x,y=grid.index_xy(*p['xy_m'])
            except ValueError as exc: errors.append(f'profile coordinate: {exc}'); continue
            if [x,y]!=p['grid_xy']: errors.append('profile grid/physical coordinate mismatch')
            mask[x,y,valid[x,y,:]]=True
            for name,source in (('qc_mpa','qc_mpa'),('fs_mpa','fs_mpa'),('Ic','Ic'),('sbt_id','sbt_id')):
                if not np.allclose(p[name],volume[source][x,y,valid[x,y,:]],rtol=1e-5): errors.append(f'profile {name} mismatch')
            if not np.allclose(p['depth_m'],grid.depth_m()[valid[x,y,:]]): errors.append('profile depth mismatch')
        if not np.array_equal(mask,observation['observed_mask']): errors.append('observed_mask mismatch')
    return errors


def assign_split(metadata:dict, *, heldout_family='lens', heldout_parameter_family='high_geometry') -> str:
    if metadata['family']==heldout_family: return 'test_family'
    if metadata['parameter_family_id']==heldout_parameter_family: return 'test_parameter'
    key=f"v2-prototype-4|{metadata['scenario_id']}|{metadata['realization_id']}"
    bucket=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%10
    return 'train' if bucket<7 else ('validation' if bucket<9 else 'test_id')


def validate_manifest(manifest:list[dict]) -> list[str]:
    errors=[]; seen={}
    for entry in manifest:
        rid=entry['realization_id']; split=entry['split']
        identity=(split,entry.get('realization_seed'),entry.get('scenario_id'),entry.get('parameter_family_id'))
        if rid in seen and seen[rid]!=identity: errors.append(f'realization {rid} conflicts or leaks across splits')
        seen[rid]=identity
        if entry['family']=='lens' and split!='test_family': errors.append(f'lens family leaked: {rid}')
        if entry['parameter_family_id']=='high_geometry' and entry['family']!='lens' and split!='test_parameter': errors.append(f'parameter family leaked: {rid}')
    return errors


def validate_reproducibility(volume: dict) -> list[str]:
    """Regenerate from saved seed and parameters; compare every ground-truth array."""
    m = volume['metadata']
    regenerated = generate(Grid(**m['grid']), m['family'], m['n_background_layers'], m['realization_seed'],
                           scenario_id=m['scenario_id'], parameter_family_id=m['parameter_family_id'],
                           realization_id=m['realization_id'], params=m['geology_params'])
    errors = []
    for name in volume:
        if name != 'metadata' and not np.array_equal(volume[name], regenerated[name]):
            errors.append(f'{name}: seed/config reproduction mismatch')
    return errors


def save_volume(volume:dict, observation:dict, root:Path, split:str) -> dict:
    errors=validate(volume,observation)
    if errors: raise ValueError('; '.join(errors))
    rid=volume['metadata']['realization_id']; target=root/rid
    target.mkdir(parents=True,exist_ok=False)
    paths={}
    for name,array in volume.items():
        if name=='metadata': continue
        path=target/f'{name}.npy'; np.save(path,array,allow_pickle=False)
        paths[name]=str(path.relative_to(root))
    path=target/'observed_mask.npy'; np.save(path,observation['observed_mask'],allow_pickle=False)
    paths['observed_mask']=str(path.relative_to(root))
    (target/'profiles.json').write_text(json.dumps(observation['profiles'],indent=2)+'\n')
    (target/'metadata.json').write_text(json.dumps(volume['metadata'],indent=2)+'\n')
    return {'realization_id':rid,'scenario_id':volume['metadata']['scenario_id'], 'parameter_family_id':volume['metadata']['parameter_family_id'],
            'realization_seed':volume['metadata']['realization_seed'],'family':volume['metadata']['family'],
            'split':split,'shape':list(volume['unit_id'].shape),'spacing_m':[volume['metadata']['grid'][k] for k in ('dx_m','dy_m','dz_m')],
            'n_units':int(len(np.unique(volume['unit_id']))),'n_cpt':len(observation['profiles']),
            'ic_range':[float(volume['Ic'].min()),float(volume['Ic'].max())],
            'sbt_counts':{str(int(i)):int(n) for i,n in zip(*np.unique(volume['sbt_id'],return_counts=True))},
            'arrays':paths,'profiles':str((target/'profiles.json').relative_to(root)),'metadata':str((target/'metadata.json').relative_to(root))}
