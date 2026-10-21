# CPT ML V3 bundle report

## Decision

**READY_FOR_COREML_FEASIBILITY** for the Python inference bundle copied into `/Users/daffaburanenugraha/Developer/cpt_super_learn/cpt_ml_bundle`. This confirms preparation for a Core ML feasibility study, not a completed Core ML conversion or iOS validation. The repository checkout is read-only to this task, so this updated report copy remains at `/Users/daffaburanenugraha/Documents/ChatGPT/Challange 67/CPT_ML_BUNDLE_REPORT.md`; the repository report was not modified.

## Repository revalidation — 2026-10-21

- Verified all 24 manifest-listed bundle files by byte size and SHA-256 in the repository. Both source checkpoint SHA-256 values matched the frozen expected values. Installed dependencies in the repository venv: NumPy 1.23.5, PyTorch 2.5.1, SciPy 1.14.1.
- Copied the repository bundle to an external writable directory and loaded it in an isolated Python process. The process asserted that hybrid runtime modules loaded from that relocated copy. A separate isolated process loaded the bundle directly from the repository while its working directory was `/private/tmp`. Both worked without a dataset path.
- Recomputed reference outputs from the existing repository checkpoints. Exact array parity (`np.array_equal`) passed for SchemaGAN reconstruction and complete DQN Q-score vectors on g12/1, g16/8, g24/12, g12/20, and g12/4 cases. Hybrid route, location, and status matched. The g12/4 case correctly returned `unsupported_cpt_count`.
- All frozen route counts (1, 2, 3, 5, 8, 12, 20) passed. Edge tests passed for determinism, finite values, observed restoration, physical XY, legal location, no-candidate status, and rejection of zero CPT.
- Relevant existing regression tests: **47 passed**, with 14 installed Matplotlib/Pyparsing deprecation warnings.
- No training, checkpoint modification, dataset modification, commit, or push was performed.

## Bundle contents and identity

- `models/schemagan_v3.pt` (196,814 bytes): generator weights, base channels 4. Source epoch 26, IDW-guided residual, MSE-only. Source SHA-256 `5694bb7b43efd7054a99fac8b74dadc5d74ebe6d291f42a8393b57fe9da52b64`.
- `models/dqn_v3.pt` (26,878 bytes): online Q weights, hidden width 64. Source SHA-256 `687539d5ad2606b198f2aadfaa6761bc9d9968ec8ce1fd2c915350413530345a`.
- `configs/normalization.json`, `feature_schema.json`, `hybrid_policy.json` are byte-for-byte copies of frozen files; `model.json` records minimal architecture and checkpoint metadata.
- `pipeline.py` is the unified API. `runtime/` holds the exact existing model, IDW, feature, Uniform, and hybrid logic, except the feature module imports frozen normalization constants from a bundled module. The Q scorer class is extracted unchanged from the existing agent. `manifest.json` hashes every runtime file.

The source checkpoints were SHA-verified before extraction. Every tensor in each inference checkpoint was compared exactly with its source tensor after serialization. Extracted SHA-256: SchemaGAN `e39232d92f31614d8bce81ca7b939720e65e4b1866c0482bd1bfc411cdf670ea`; DQN `8869fc8bff0cfe2447112b9259708fa35f7c3795cca4fb5655d2c02916449599`. Different file hashes reflect the removal of training state. No training, tuning, or source checkpoint overwrite occurred.

## Inference API and dependencies

`CPTModel.load(bundle_path)` verifies the manifest and loads both frozen CPU models. `CPTModel.predict(grid,cpt_data,valid_mask=None,access_mask=None)` returns physical `ic_3d [X,Y,Z]`, next CPT index and physical XY, policy, status, candidate count, and optional Q score. Input format, axes, normalization, IDW, feature dimensions, legal mask, coordinate conversion, Uniform tie breaking, and routes are documented in `cpt_ml_bundle/README.md`. Runtime dependencies: Python, NumPy, PyTorch, SciPy. No dataset or ground truth is read at inference.

## Verification

- Relocated bundle loaded in an isolated Python process from `work/relocated_bundle`, with no repository package import. Exact `np.array_equal` reconstruction parity and exact full-vector DQN Q-score parity passed for g12/1 CPT and g16/8 CPT. g24/12 CPT and g12/20 CPT matched the Uniform route and location. g12/4 CPT returned `unsupported_cpt_count`. Five reference cases used identical synthetic input arrays; no dataset samples were bundled.
- All seven supported CPT counts (1, 2, 3, 5, 8, 12, 20) passed frozen routing checks.
- Edge checks passed: repeat inference was bitwise deterministic; physical XY matched grid mapping; all predicted valid voxels were finite; observed values were exact; no duplicate recommendation; inaccessible candidates yielded `no_candidate`; zero CPT was rejected.
- Existing relevant regression suite: **47 passed** (`dqn3d/v3/test_v3.py`, `schemagan3d/test_v3_adapter.py`, `schemagan3d/test_v3_idw_guided.py`, `synthetic3d/v2/test_v2.py`). The 14 warnings came from installed Matplotlib/Pyparsing deprecations.
- Numerical tolerance: zero tolerance (`np.array_equal`) for reconstruction and full Q-score vectors; `1e-7` absolute tolerance for scalar Q score in the output dictionary.

## Core ML feasibility limits

This is a portable Python bundle, not a Core ML model. Core ML conversion, 3D operator support, dynamic candidate count, iOS precision, latency, memory, and app integration are untested. Frozen routing was selected on validation ID and is only supported for the listed counts. It does not establish performance on new geological families. The inference runtime still carries some unused repository data-loader definitions inside copied modules; it carries no dataset or training state. The Python module names can collide with an already imported repository package, so load in a fresh process or package under a dedicated namespace before embedding in a larger Python app.

## Commands run

```sh
PYTHONPYCACHEPREFIX="$PWD/work/pycache" MPLCONFIGDIR="$PWD/work/matplotlib" /Users/daffaburanenugraha/Developer/cpt_super_learn/cpt/bin/python work/build_bundle.py
PYTHONPYCACHEPREFIX="$PWD/work/pycache" MPLCONFIGDIR="$PWD/work/matplotlib" /Users/daffaburanenugraha/Developer/cpt_super_learn/cpt/bin/python work/reference_parity.py
PYTHONPYCACHEPREFIX="$PWD/work/pycache" MPLCONFIGDIR="$PWD/work/matplotlib" /Users/daffaburanenugraha/Developer/cpt_super_learn/cpt/bin/python -I work/check_parity.py "$PWD/work/relocated_bundle" "$PWD/work/parity"
PYTHONPYCACHEPREFIX="$PWD/work/pycache" MPLCONFIGDIR="$PWD/work/matplotlib" /Users/daffaburanenugraha/Developer/cpt_super_learn/cpt/bin/python -I work/check_edge_cases.py "$PWD/work/relocated_bundle"
PYTHONPYCACHEPREFIX="$PWD/work/pycache" MPLCONFIGDIR="$PWD/work/matplotlib" /Users/daffaburanenugraha/Developer/cpt_super_learn/cpt/bin/python -I work/check_routes.py "$PWD/cpt_ml_bundle"
PYTHONPYCACHEPREFIX='/Users/daffaburanenugraha/Documents/ChatGPT/Challange 67/work/pycache' MPLCONFIGDIR='/Users/daffaburanenugraha/Documents/ChatGPT/Challange 67/work/matplotlib' cpt/bin/python -m pytest -q -p no:cacheprovider dqn3d/v3/test_v3.py schemagan3d/test_v3_adapter.py schemagan3d/test_v3_idw_guided.py synthetic3d/v2/test_v2.py
```

## Repository revalidation commands

```sh
# In /Users/daffaburanenugraha/Developer/cpt_super_learn
cpt/bin/python -m pytest -q -p no:cacheprovider dqn3d/v3/test_v3.py schemagan3d/test_v3_adapter.py schemagan3d/test_v3_idw_guided.py synthetic3d/v2/test_v2.py
cpt/bin/python -I /Users/daffaburanenugraha/Documents/ChatGPT/Challange\ 67/work/check_parity.py /Users/daffaburanenugraha/Documents/ChatGPT/Challange\ 67/work/repo_validation_bundle /Users/daffaburanenugraha/Documents/ChatGPT/Challange\ 67/work/parity
cpt/bin/python -I /Users/daffaburanenugraha/Documents/ChatGPT/Challange\ 67/work/check_routes.py /Users/daffaburanenugraha/Documents/ChatGPT/Challange\ 67/work/repo_validation_bundle
cpt/bin/python -I /Users/daffaburanenugraha/Documents/ChatGPT/Challange\ 67/work/check_edge_cases.py /Users/daffaburanenugraha/Developer/cpt_super_learn/cpt_ml_bundle
```

`PYTHONPYCACHEPREFIX` and `MPLCONFIGDIR` were directed to the writable workspace during these runs. The SHA-256 audit used a read-only inline Python command. Reference cases were regenerated by `work/reference_parity.py` before parity checks.
