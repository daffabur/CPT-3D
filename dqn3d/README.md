# DQN 3D CPT placement

This package prepares a placement policy for a **16×16×8 synthetic IC volume**.
It loads the frozen SchemaGAN 3D checkpoint, simulates one complete Z profile
per CPT at `(X,Y)`, and records reconstruction quality after each CPT. The
training command below fits a separate masked DQN; the SchemaGAN checkpoint,
existing 2D code, and the 3D game are unchanged.

## Existing 2D implementation and what carries over

`CPTSuperLearn/agent.py` implements replay memory, epsilon-greedy choice,
Q-learning targets, and a target network. Those concepts are reusable. Its
current network accepts six scalar state features; the exploration and target
max operations do not mask invalid actions. They cannot be connected to this
environment as-is. `CPTSuperLearn/environment.py` samples full depth profiles
on an X–Z section, uses positive X offsets as actions, interpolates, and
scores against the complete true section. `CPTSuperLearn/runner.py` connects
the 2D agent, environment, dataset, and validation loop.

The 3D environment changes the decision to a horizontal `(X,Y)` CPT site,
returns a 3D reconstruction, and exposes a spatial state and legal-action
mask. A future Q-network will need a small 3D encoder plus a 16-output head.

## State and actions

`reset(volume, initial_xy)` accepts one CPT or a list of CPT locations. DQN
training uses **three warm-start CPTs** (within the SchemaGAN training range).
The total budget remains 8, so the policy selects five more sites.
The returned state has:

| Field | Shape | Meaning |
|---|---|---|
| `volume[0]` | 8×16×16 | observed IC; zero at unknown voxels |
| `volume[1]` | 8×16×16 | binary observed mask |
| `volume[2]` | 8×16×16 | current SchemaGAN IC prediction |
| `remaining_budget` | scalar | fraction of CPT budget remaining |
| `valid_actions` | 16 booleans | legal action regions |

The hidden true volume, true error, reward diagnostics, and future CPT profile
are absent from the state. Oracle metrics are returned separately in `info`
for simulation evaluation; a policy must not use them as state features.

Action `a` is one of 16 cells on a 4×4 horizontal grid. Each cell covers
4×4 potential `(X,Y)` sites. An action selects the unused site nearest that
cell's center. The mapping is deterministic; if one site is already used,
the next closest unused site is selected. An exhausted cell is masked out.
Sampling a duplicate coordinate is prohibited. The action mask is applied
during epsilon exploration, greedy selection, and Bellman target computation.

This compact action scheme is a prototype trade-off. A four-direction local
movement space would require a vehicle-position state and travel/sampling
semantics; reaching distant high-value sites might take many decisions.
Absolute selection over every `(X,Y)` is expressive but yields 256 Q outputs,
more sparse action coverage in replay, and heavier exploration. The 16-cell
scheme keeps the Q head small while preserving full-area access. It restricts
within-cell choice to a fixed rule, which should be revisited after baselines.

## Reward

The default simulated reward is

`RMSE_all_before − RMSE_all_after − sampling_cost`.

Both RMSE terms cover **the same 2,048 voxels**. Hidden-voxel RMSE is recorded
for evaluation but not used for reward because the hidden set shrinks after
each CPT. RMSE reward requires synthetic ground truth and is not available in
the field. It is returned only as scalar reward, never as an observation.
Negative rewards are valid: a CPT may fail to improve the reconstruction
enough to repay its cost. The alternative `coverage_proxy` reward uses the
reduction in mean horizontal distance to the nearest CPT. It needs no hidden
truth but measures geometry, not calibrated model uncertainty. An ensemble or
calibrated posterior would be needed for a genuine uncertainty reward.

The checkpoint was trained with 3–20 randomly placed profiles. The DQN run
starts with three pseudorandom, distinct locations derived from the volume
index and seed. During validation, DQN, Random, and Uniform receive identical
three locations for each volume. The sites are sampled before the first agent
decision and count against the budget.

The Q-network encodes the three 3D channels with two Conv3D layers, concatenates
remaining budget, and outputs 16 values. Replay stores observations and masks;
the target network is synchronized every 100 optimization steps. The 2D agent
is used as a conceptual reference only; no 2D source is modified.

Exploration and replay sampling now have independent RNG streams. Greedy
validation does not advance either stream, so changing validation frequency
does not alter subsequent training exploration. The corresponding three-seed
revalidation and metrics are in `REPRO_VALIDATION.md`.

## Baseline and smoke benchmark

```bash
cd ~/Developer/cpt_super_learn
source cpt/bin/activate
python -m pytest -q dqn3d/test_environment.py
python -m dqn3d.baselines --volumes 6 --budget 8 --device cpu
```

Random and uniform policies use the **same 16 legal actions**, the same six
test volumes, the same initial CPT per volume, and the same total CPT budget.
Uniform chooses the action candidate farthest from already observed sites.
The output directory contains `per_step.csv` and `summary.json`, including
hidden-voxel RMSE/MAE, full-volume RMSE, reward, selected location, and
SchemaGAN inference time at each step. This is a small smoke benchmark, not a
trained DQN comparison.

## Initial DQN training

```bash
cd ~/Developer/cpt_super_learn
./cpt/bin/python -m pytest -q dqn3d
./cpt/bin/python -m dqn3d.train --train-limit 500 --val-limit 100 \
  --episodes 500 --eval-every 250 --budget 8 --output results/dqn3d
```

This CPU experiment uses the first 500 train and first 100 validation CSVs.
The test split is not read. It writes `best.pt`, `config.json`,
`training_metrics.csv`, `validation_per_volume.csv`, and
`validation_summary.json`. The command refuses to overwrite an existing output
directory. The three policies use the same validation volumes, warm-start
sites, budget, and 16-action space. Final metrics report both hidden-voxel
RMSE/MAE and fixed-domain full-volume RMSE.
