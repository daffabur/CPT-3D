# 3D soil reconstruction proof of concept

This standalone module does not modify or depend on the existing 2D DRL
environment. It demonstrates the revised sampling model:

1. Generate a `16 x 16 x 8` soil volume with three smoothly varying layers.
2. Select reproducible, nested CPT locations on the horizontal X-Y grid.
3. Observe the complete Z profile at each selected location.
4. Reconstruct all profiles using horizontal X-Y inverse-distance weighting.
5. Compare 3, 5, 10, and 20 CPT profiles.
6. Report volumetric RMSE and MAE.
7. Save X-Y, X-Z, and Y-Z ground-truth/reconstruction/error slices.

The interpolator API accepts only sampled X-Y positions and their observed Z
profiles. The complete synthetic volume is kept in the runner for evaluation
and is never passed to the IDW function.

Run it from the repository root with the project environment:

```sh
source cpt/bin/activate
python -m prototype_3d.run_poc
```

The default outputs are written to `prototype_3d/output/`.

Generated files include `metrics.csv`, a metric comparison plot, one slice
comparison per CPT count, and a compressed NPZ containing the ground truth,
observations, and reconstructions.
