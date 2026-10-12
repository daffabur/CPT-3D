# CPT-3D - Digital Twin for CPT Soil Reconstruction

CPT-3D is an experimental three-dimensional digital twin for visualizing and
reconstructing subsurface conditions from Cone Penetration Test (CPT) profiles.
The latest game lets a simple CPT vehicle move over an X-Y surface while the
subsurface volume is displayed interactively with PyVista.

The repository contains two independent paths:

- `game3d/`: the latest standalone 3D game and the recommended entry point.
- `CPTSuperLearn/` and `game/`: the original 2D Deep Reinforcement Learning
  implementation, preserved without modification.

No DRL training is required to run the 3D game.

## 3D game modes

### 1. Synthetic 3D Mode

- Generates a reproducible `16 x 16 x 8` hidden soil volume.
- Contains three soil layers with smooth X-Y boundaries.
- IC varies along X, Y, and Z.
- Moving the vehicle and pressing Space samples a complete IC depth profile.
- Horizontal IDW reconstructs the volume using sampled profiles only.
- Ground truth and prediction are displayed side-by-side.
- Volumetric RMSE and MAE are updated after sampling.

### 2. Real CPT Data Mode

- Uses the S1, S2, and S3 profiles extracted from the NGC 2025 sondir PDF.
- Visualizes `qc` in `kg/cm2`; it is never mixed with synthetic IC.
- Converts geographic coordinates to local east/north metres relative to S1.
- Uses a consistent vertical reduced level: `RL = surface elevation - depth`.
- Red lines represent the three measured CPT locations.
- Pressing Space away from S1-S3 previews an IDW prediction only; it does not
  create a fictitious measurement.
- RMSE is unavailable because a measured 3D ground-truth volume does not exist.
- The UI warns that interpolation uncertainty is high with only three CPTs.

Real project data is intentionally excluded from Git. Place an authorized copy
of the PDF locally at:

```text
data/real_cpt/SONDIR SONDIR NGC 2025(1).pdf
```

## Requirements

- macOS
- Python 3.11
- A local graphical session for the interactive PyVista window

## Installation

Clone the repository and enter it:

```bash
git clone https://github.com/daffabur/CPT-3D.git
cd CPT-3D
```

Create and activate an isolated environment. With Conda:

```bash
conda create -n cpt3d python=3.11 -y
conda activate cpt3d
```

Or with the standard library:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install only the dependencies needed by the 3D game:

```bash
python -m pip install -r game3d/requirements-game3d.txt
```

## Running the latest game

Start the mode selector:

```bash
python -m game3d.main
```

Choose `1` for Synthetic 3D Mode or `2` for Real CPT Data Mode.

You can also start a mode directly:

```bash
python -m game3d.main --mode synthetic
python -m game3d.main --mode real
```

Controls:

| Control | Action |
|---|---|
| `W/A/S/D` | Move on the X-Y surface |
| Arrow keys | Move on the X-Y surface |
| `Space` | Sample synthetic profile or preview real-data prediction |
| Mouse drag | Rotate camera |
| Scroll | Zoom |
| `Control+C` in Terminal | Stop the game |

Close the PyVista window before starting another game instance.

## Preparing the real CPT data

After placing the authorized PDF in `data/real_cpt/`, extract and validate it:

```bash
python -m game3d.main --extract-only
```

This creates local, Git-ignored files:

```text
data/real_cpt/sondir_ngc_2025.csv
data/real_cpt/extraction_report.json
```

The extractor verifies the three tabular pages, depth interval, row count, and
units before writing the normalized CSV.

## Tests

Run the 3D tests:

```bash
python -m pip install pytest
python -m pytest -q game3d/test_game3d.py
```

Real-data extraction tests are skipped when the authorized PDF/CSV is not
present in the local `data/real_cpt/` directory.

Run a headless PyVista geometry smoke test:

```bash
python -m game3d.main --mode synthetic --off-screen
python -m game3d.main --mode real --off-screen
```

Run all repository tests:

```bash
python -m pytest -q
```

## Project structure

```text
CPT-3D/
├── game3d/
│   ├── main.py
│   ├── visualization.py
│   ├── environment3d.py
│   ├── interpolator3d.py
│   ├── data_loader.py
│   ├── synthetic_generator.py
│   └── requirements-game3d.txt
├── prototype_3d/          # Standalone reconstruction proof of concept
├── CPTSuperLearn/         # Original 2D DRL package
├── game/                  # Original 2D game
└── test/
```

## Limitations

- Real mode currently has only three CPT locations, so the interpolated volume
  must not be treated as verified ground truth.
- The displayed uncertainty is a nearest-CPT distance proxy, not a calibrated
  geostatistical confidence interval.
- Horizontal IDW smooths between profiles and does not explicitly model
  geological discontinuities or anisotropy.
- The real-data surface elevation is preserved, which is why S1 appears above
  S2 and S3.

## Original project and citation

The 2D DRL implementation is based on
[brunozc/cpt_super_learn](https://github.com/brunozc/cpt_super_learn) and the
following publication:

```bibtex
@article{DRL_2025,
  author={Zuada Coelho, B. and Smyrniou, E. and Campos Montero, F.A.},
  title={Optimizing geotechnical in-situ site investigations using Deep Reinforcement Learning},
  ISSN={3050483X},
  DOI={10.1016/j.geoai.2025.100038},
  journal={Geodata and AI},
  year={2025},
  pages={100038}
}
```

## License

This repository retains the existing MIT license. See [LICENSE](LICENSE).
