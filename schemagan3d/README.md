# SchemaGAN 3D prototype

This folder is independent of the existing 2D SchemaGAN. It reconstructs a
`16 x 16 x 8` IC volume from complete depth profiles observed at sparse `(X,Y)`
locations.

## Data flow

CSV axes `(X,Y,Z)` are loaded lazily and converted to PyTorch layout
`(batch, channel, Z, Y, X)`. The generator receives exactly two channels:
normalized sparse IC and a binary observed mask. The hidden ground truth is
used only for reconstruction loss and the discriminator's real sample.

The generator is a small two-level 3D U-Net. Its final output uses a hard
data-consistency merge, so values at measured CPT profiles are copied exactly.
The conditional 3D PatchGAN discriminator receives candidate volume, sparse
volume, and mask. Training combines hidden-voxel L1 loss and adversarial loss.

## Commands

Smoke test (no training run):

```bash
python -m schemagan3d.smoke_test --data-root data/synthetic3d
```

Train with only the first 500 training files:

```bash
python -m schemagan3d.train \
  --data-root data/synthetic3d \
  --train-limit 500 --val-limit 100 \
  --epochs 5 --batch-size 8 --device auto
```

Evaluate the saved generator on a held-out test subset:

```bash
python -m schemagan3d.evaluate \
  --data-root data/synthetic3d --split test --limit 100 \
  --checkpoint results/schemagan3d/checkpoint.pt
```

Evaluate horizontal IDW without a neural checkpoint:

```bash
python -m schemagan3d.evaluate \
  --data-root data/synthetic3d --split test --limit 100
```

`--device auto` prefers Apple Metal (MPS), then CUDA, and otherwise uses CPU.
MPS availability depends on the installed PyTorch build and macOS runtime.

## Apple GPU compatibility benchmark

The default architecture is intentionally unchanged. An isolated resize plus
Conv3D generator and a repeatable CPU/MPS benchmark are available in
`experimental_models.py` and `benchmark_apple.py`. See `BENCHMARK_APPLE_M5.md`
for measured results and commands.

## Visual evaluation of a trained checkpoint

Run inference on a seeded sample of 12 held-out test volumes. This compares
the same CPT sites for IDW and SchemaGAN at 2, 4, 8, and 16 sites using random,
spatially dispersed, and clustered placement. Site sets are nested within each
volume and placement pattern.

```bash
python -m schemagan3d.visual_evaluation \
  --checkpoint results/schemagan3d/checkpoint.pt \
  --data-root data/synthetic3d --volumes 12 --device cpu
```

Outputs go to a new timestamped `results/schemagan3d_visual_*` folder:
per-volume and summary CSV metrics, three-plane slices, 3D cutaway views,
an error-versus-CPT plot, and a manifest with selected files and checkpoint
hash. The checkpoint is never written. The full numerical results from the
2026-10-13 run are in `results/schemagan3d_visual_20261013/`.
