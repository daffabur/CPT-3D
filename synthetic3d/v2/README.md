# Synthetic Dataset V2 prototype

This module is independent of Dataset V1 and the trained SchemaGAN/DQN models. It creates **eight functional examples only**, never a training corpus. Use the repository's existing `./cpt/bin/python`; no new dependency is needed.

```bash
PYTHONDONTWRITEBYTECODE=1 ./cpt/bin/python -m synthetic3d.v2.generate_examples
PYTHONDONTWRITEBYTECODE=1 ./cpt/bin/python -m synthetic3d.v2.validate_examples
PYTHONDONTWRITEBYTECODE=1 ./cpt/bin/python -m unittest synthetic3d.v2.test_v2 -v
```

`generate_examples` refuses to overwrite an existing output directory. The generated prototype is in `data/synthetic3d_v2/prototype/`. Each realization owns separate `.npy` arrays for `unit_id`, `boundaries_m`, `Ic`, `sbt_id`, CPT responses, stresses, and `valid_mask`; `profiles.json` and `observed_mask.npy` contain observations. `metadata.json` records grid shape and spacing, local origin/azimuth, flat surface elevation, seed, scenario and parameter family, method versions, assumptions, units, and geology parameters. `manifest.json` indexes all artifacts and splits. Arrays use `(X,Y,Z)`; boundaries use `(boundary,X,Y)`. Depth is positive down from the flat surface and is measured at voxel centers. The grid transform rotates local East/North by `azimuth_deg`.

## Geological families

Horizontal, inclined, undulating, and ellipsoidal lens cases have two to four background layers. Background boundaries share the same deformation, maintaining a minimum thickness of two vertical voxels and preventing intersections. A lens replaces part of a named parent unit. Small correlated 3D fields modulate response within units. Faults and erosion are outside the prototype.

## CPT interpretation

The target is a **normalized** Robertson-style index with stress exponent `n=1`:

- `qt = qc` only under the explicit low pore-pressure correction assumption; `qt_approximation=true` is stored. Water table is 10 m below the modeled domain; `u0=0` and `sigma_v0_eff=sigma_v0` here. This is a prototype assumption, not a field correction.
- CPTu interface: `interpret(..., u2_mpa=array, area_ratio=a)` uses `qt=qc+u2(1-a)`. The generator does **not** fabricate u2.
- `Qt=(qt-sigma_v0)/sigma_v0_eff`; `Fr=100 fs/(qt-sigma_v0)` as percent; `Ic=sqrt((3.47-log10(Qt))²+(log10(Fr)+1.22)²)`.
- `sigma_v0` is the cumulative unit weight times vertical thickness at voxel centers. `qc`, `fs`, `qt`, `u0`, and stresses use MPa; unit weight uses kN/m³; distances use m.
- SBT zones 2–7 use Ic boundaries 1.31, 2.05, 2.60, 2.95, 3.60. Values outside the prototype chart range 1.0–4.0 are `unclassified=0`; zones 1, 8, and 9 cannot be inferred reliably from Ic alone. SBT is behavior classification, **not** `unit_id`.

Method references: [Robertson CPT GeoEnviron Guide (2010)](https://www.cpt-robertson.com/PublicationsPDF/CPT%20GeoEnvironGuide%202010.pdf) and [Robertson, updated CPT-based SBT classification (2016)](https://www.cpt-robertson.com/PublicationsPDF/Robertson%20Updated%20SBT%20CGJ%202016.pdf). The response distributions are **illustrative prototype assumptions, not literature-calibrated field ranges**: base `qc` by unit `[2,5,9,14,6]` MPa; base `Fr` `[3.3,2.4,1.4,0.9,2.0]` percent; unit weight `[17,18,19,20,18.5]` kN/m³; log-field SDs `0.20` for qc and `0.18` for Fr; correlation length 4 m; shared-field coefficient 0.55. The full values are stored under `response_assumptions` in every metadata file. Invalid formula domains raise an error; they are not silently clipped.

## Splits and limits

`scenario_id`, `parameter_family_id`, `realization_id`, and `realization_seed` are recorded. The lens family is held out as `test_family`; `high_geometry` non-lens cases are `test_parameter`; remaining realization IDs are assigned by a stable SHA-256 bucket to train/validation/in-distribution test. The tiny eight-example set demonstrates the mechanism and must not be used for model training or tuning. If normalization is added later, it must use the train split only. `validate_manifest` rejects conflicting realization IDs across splits.

The simulator is deliberately simple: no DEM variation, capillary/groundwater effects, pore-pressure response model, lab-calibrated material parameters, irregular CPT depth sampling, or field noise model. Real geological and CPT validity needs domain review before production or claims about performance on real sites. V1 data and model checkpoints are unchanged.

## Refined prototype (schema `v2-prototype-2`)

The original `core.py` generator and `data/synthetic3d_v2/prototype/` remain available. The refined implementation is in `refined.py` and writes **ten new validation-only examples** to `data/synthetic3d_v2/prototype_refined/`:

```bash
PYTHONDONTWRITEBYTECODE=1 ./cpt/bin/python -m synthetic3d.v2.generate_refined_examples
PYTHONDONTWRITEBYTECODE=1 ./cpt/bin/python -m synthetic3d.v2.validate_refined
PYTHONDONTWRITEBYTECODE=1 ./cpt/bin/python -m unittest synthetic3d.v2.test_v2 synthetic3d.v2.test_refined -v
```

The generator refuses to overwrite an existing output directory. It keeps a `common_shift` geometry mode and adds a `varying` mode: each boundary has bounded slope/wave terms, so layer thickness changes laterally. Invalid crossings, domain exits, and thickness below the configured minimum are rejected. The same background strata remain ordered; no fault or erosion is implied. Lens center, three radii, parent unit, orientation, and minimum voxel count are parameters saved in metadata. An ellipsoid that leaves its parent unit or domain is rejected.

Response distributions are generated per realization and written to `response_assumptions`, with one record per geological unit. Numeric unit IDs are labels, not a fixed response lookup. Overlapping unit responses are allowed. Correlated 3D fields use a length in metres and a shared component for qc and friction ratio; depth trends are explicit. `fs` is constructed from a simulated friction ratio and positive net cone resistance, then Qt/Fr/Ic are recomputed from qc, fs, and stress. All numerical response ranges and trends remain **illustrative prototype assumptions**, not geotechnically validated field parameters. The low pore-pressure *cone correction* assumption is separately recorded and required; a water table below the domain does not prove it. No u2 values are invented.

SBT rules in metadata specify zones 2–7 and their exact half-open bounds, outer chart range `[1.0,4.0]`, and `unclassified=0` outside it. The methods are the same Robertson-style normalized Qt–Fr index described above. Tests include a hand-calculated numerical example, exact class-boundary cases, and invalid-domain rejection.

`validate_refined` checks saved arrays, geometry from parameters, lens occupancy, units, response variation, CPT profiles and masks, Ic/SBT recomputation, split/derivative paths, and seed/config reproduction. `similarity_diagnostics.json` reports exact duplicates, IC quantile distances across all pairs, and normalized depth-profile correlation even when shapes differ. Voxelwise correlation/RMSE is reported only for same-shape pairs; a high correlation alone is **not** treated as leakage proof. Field CPT coordinates that are not voxel centers are not yet supported; a future adapter should map physical point/depth measurements with an explicit interpolation or support-volume policy and uncertainty metadata rather than silently rounding.
