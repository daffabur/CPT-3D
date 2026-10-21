# CPT ML V3 inference bundle

Portable Python inference for the frozen SchemaGAN V3 reconstruction model and scaled DQN V3 candidate scorer. The bundle contains two neural networks plus their frozen preprocessing, postprocessing, Uniform policy, and hybrid routing. It requires Python with NumPy, PyTorch, and SciPy. CPU inference only. No dataset folder is needed.

## API

```python
from pathlib import Path
from pipeline import CPTModel

model = CPTModel.load(Path("/path/to/cpt_ml_bundle"))
result = model.predict(
    grid={"nx": 12, "ny": 12, "nz": 12,
          "dx_m": 2.0, "dy_m": 2.0, "dz_m": 1.0,
          "origin_x_m": 0.0, "origin_y_m": 0.0,
          "azimuth_deg": 0.0},
    cpt_data={"observed_ic": observed_ic, "observed_mask": observed_mask},
    valid_mask=valid_mask,
    access_mask=access_mask,
)
```

`grid` can be a dictionary of `synthetic3d.v2.core.Grid` fields or a bundled `Grid` object. `observed_ic` is physical Ic `float32 [X,Y,Z]`, zero outside observed voxels. `observed_mask` and optional `valid_mask` are boolean `[X,Y,Z]`. Each observed XY site must contain every valid depth there. If omitted, `valid_mask` is all true. Optional `access_mask` is boolean `[X,Y]`; false means inaccessible. At least one complete CPT profile is required. Neither ground truth nor a fixed sampling budget appears in this API.

`result['ic_3d']` is physical Ic `[X,Y,Z]` with measured values restored exactly. `recommended_index` is `(x_index,y_index)` and `recommended_xy` is the physical `(x_m,y_m)` cell center; both are `None` when no recommendation is available. `policy_used`, `status`, `candidate_count`, and `q_score` are also returned. `q_score` is `None` for Uniform, unsupported counts, and no candidate. The result includes equivalent scalar XY fields and `observed_cpt_count` for compatibility with the existing policy output.

## Numerical contract

The generator receives sparse normalized Ic, observed mask, and normalized IDW, each `float32 [1,1,Z,Y,X]`. Normalization uses the train-only mean `2.314278231778639` and standard deviation `0.27927494076233944`. Sparse values outside observations remain zero. IDW uses measured CPT profiles at physical XY coordinates, power 2, independently at each depth. The output is denormalized, then observed values are restored. The 17 candidate and 11 global feature names and order are in `configs/feature_schema.json`. Legal candidates satisfy `valid.any(z) & ~observed.any(z) & access_mask`; `np.argwhere` gives X then Y order. Physical cell centers incorporate spacing, origin, and azimuth. Uniform maximizes distance to the nearest CPT site with lexicographic tie break.

The frozen hybrid route counts unique observed XY sites: 1, 2, 3, 5, 8 use DQN; 12, 20 use Uniform. Other counts return `unsupported_cpt_count` after reconstruction. When no legal candidate exists, status is `no_candidate`. The implementation supports g12 `[12,12,12]`, g16 `[16,16,16]`, and g24 `[24,24,12]`; model dimensions must be at least 8 and divisible by 4. Routing validation does not extend to every possible CPT count.

## Integrity and portability

`manifest.json` records source checkpoint SHA-256, extracted inference checkpoint SHA-256, tensor fingerprints, and a hash of each bundled file. `CPTModel.load` verifies bundled file hashes and strict model state loading. Inference checkpoint files contain weights and minimal metadata; optimizer, replay buffer, history, and synthetic volumes are excluded. The inference checkpoint SHA differs from its training checkpoint because of extraction; tensor values are identical. The bundle is relocatable and uses no repository or dataset relative path at inference. Run it in an isolated Python process so its bundled `synthetic3d`, `schemagan3d`, and `dqn3d` modules take precedence over any preloaded repository packages.

This bundle has not been converted to Core ML. iOS operator support, dynamic candidate counts, precision, memory, and latency require a separate feasibility study.
