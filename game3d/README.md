# CPT Super Learn - Digital Twin 3D

This is a standalone 3D implementation. It does not import or modify the
existing `game/`, `CPTSuperLearn/`, or DQN training pipeline.

## Modes

### Synthetic 3D Mode

- Hidden ground-truth IC volume: `16 x 16 x 8`.
- Three layers with smooth boundaries varying in X and Y.
- Controlled IC variation in X, Y, and Z.
- Space reveals the complete Z profile at the vehicle X/Y location.
- Horizontal IDW is refreshed from observed profiles only.
- The scene shows ground truth and prediction side-by-side plus RMSE/MAE.

### Real CPT Data Mode

- Extracted measurements from S1, S2, and S3 use `qc` in `kg/cm2`.
- Geographic coordinates are converted to local east/north metres relative to
  S1 with a local equirectangular approximation.
- Vertical reduced level is consistently `surface elevation - depth` in metres.
- Measurements at depth 0.00 are missing in the source and remain missing.
- Red lines are measured CPTs; an orange line is only a prediction preview.
- Space never creates a real observation in this mode.
- No RMSE is reported because no real 3D ground truth exists.
- A high-uncertainty warning is always displayed because only three CPTs exist.

## Install on macOS

From the repository root:

```bash
source cpt/bin/activate
python -m pip install -r game3d/requirements-game3d.txt
```

## Run

Interactive mode selector:

```bash
python -m game3d.main
```

Or select a mode directly:

```bash
python -m game3d.main --mode synthetic
python -m game3d.main --mode real
```

Controls:

- `W/A/S/D` or arrow keys: move on X/Y.
- `Space`: sample in synthetic mode; preview prediction in real mode.
- Mouse drag: rotate.
- Scroll: zoom.
- Shift/control plus mouse interaction: pan according to PyVista defaults.

## Re-extract the real PDF

```bash
python -m game3d.main --extract-only
```

The normalized data and audit report are written to:

- `data/real_cpt/sondir_ngc_2025.csv`
- `data/real_cpt/extraction_report.json`

## Tests and headless smoke rendering

```bash
python -m pytest -q game3d/test_game3d.py
python -m game3d.main --mode synthetic --off-screen
python -m game3d.main --mode real --off-screen
```

`--off-screen` validates PyVista volume, vehicle, and profile geometry without
creating an OpenGL window. Screenshots should be taken from the interactive
window because sandbox/headless macOS sessions do not provide a Cocoa OpenGL
context.

The uncertainty value in the UI is a distance-to-nearest-CPT proxy, not a
calibrated geostatistical confidence interval.
