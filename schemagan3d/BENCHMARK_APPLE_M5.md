# Apple M5 benchmark (2026-10-13)

Host: MacBook Pro `Mac17,2`, Apple M5 (10 CPU / 10 GPU cores), 24 GB RAM,
macOS 26.6.2, Python 3.11.16, PyTorch 2.5.1.

The installed wheel reports `mps_built=True`. MPS is unavailable inside the
Codex sandbox, while a normal VS Code terminal reaches MPS and reproduces
`RuntimeError: ConvTranspose 3D is not supported on MPS` for the existing
generator.

Ten measured optimization steps followed two warm-up steps, used batch size 8,
FP32, base channels 16, Adam, the same discriminator and loss path, and excluded
CSV loading.

| Generator | Device | Mean step | Min/max | Parameters | Peak process RSS |
|---|---|---:|---:|---:|---:|
| Existing ConvTranspose3D | CPU | 0.0891 s | 0.0790/0.1227 s | 716,737 | 493.25 MiB |
| Experimental resize+Conv3D | CPU | 0.1030 s | 0.0940/0.1190 s | 622,017 | 477.00 MiB |

The experimental generator is isolated in `experimental_models.py`; the
default generator and training selection remain unchanged.

Reproduce CPU:

```bash
python -m schemagan3d.benchmark_apple --device cpu --model both --steps 10 --batch-size 8
```

Run native-MPS experimental model from a normal macOS terminal:

```bash
python -m schemagan3d.benchmark_apple --device mps --model upsample --steps 10 --batch-size 8
```

Diagnose the existing model with documented CPU fallback enabled:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m schemagan3d.benchmark_apple \
  --device mps --model transpose --steps 10 --batch-size 8
```

MPS timing and allocator memory must only be recorded when the JSON report
contains no model error. CPU fallback is not equivalent to native GPU support.
