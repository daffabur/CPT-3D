# SchemaGAN 3D readiness notes

## Existing implementation (verified)

- `CPTSuperLearn/interpolator.py` hard-codes a `512 x 32` X-Z section.
- Sparse profiles are assembled as a 2D array and reshaped to
  `(batch, 32, 512, 1)`.
- The bundled `schemaGAN/schemaGAN.h5` input and output are both
  `(batch, 32, 512, 1)`.
- The bundled generator uses `Conv2D`, `Conv2DTranspose`, 2D skip connections,
  and contains 67,011,201 parameters.
- The repository contains the pretrained inference model and wrapper, not a
  complete 3D SchemaGAN training pipeline.

Therefore the existing model cannot consume `16 x 16 x 8` CSV volumes directly.

## Changes required before 3D training

1. Add a loader that sorts `x,y,z`, checks full coverage, and reshapes each CSV
   to a consistent tensor such as `(batch, 16, 16, 8, 1)`.
2. Define sparse-conditioning tensors and masks in three spatial dimensions.
3. Replace 2D generator/discriminator operations with a deliberately designed
   3D architecture (`Conv3D`, `Conv3DTranspose`, and compatible 3D skip paths),
   or select another volumetric architecture.
4. Revisit downsampling depth: `16 x 16 x 8` cannot follow the existing
   `32 x 512` downsampling schedule unchanged.
5. Refit IC normalization using training split statistics only and persist the
   transform for validation/test inference.
6. Implement volume-aware losses and metrics, including masked-data fidelity
   and held-out volumetric error.
7. Keep train/validation/test file lists and seeds isolated to prevent leakage.
8. Benchmark batch size, activation memory, and throughput on the target GPU
   before selecting channel widths. A direct filter-for-filter Conv3D port is
   likely unnecessarily expensive.

No architecture or training code is changed by the dataset generator work.
