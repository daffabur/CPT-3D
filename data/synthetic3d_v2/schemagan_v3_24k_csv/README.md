# SchemaGAN V3 synthetic geotechnical 3D dataset — CSV export

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
