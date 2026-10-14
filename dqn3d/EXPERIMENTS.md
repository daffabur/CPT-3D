# DQN 3D optimization audit and validation experiments

All runs use the frozen SchemaGAN checkpoint, CPU, synthetic train/validation
only, three identical warm-start CPTs per validation volume, total budget eight,
and five learned/baseline decisions. The validation seed is fixed at 1000014.
Each run has a separate configuration, best checkpoint, training history, and
per-volume validation table in `results/dqn3d_opt/`. The test split was not
read. Run `python -m dqn3d.analyze_experiments` to reproduce `aggregate.csv`
and `aggregate.json` from the saved per-run files.

## Code audit

- The 3D environment returns observed IC, observed mask, SchemaGAN prediction,
  normalized remaining budget, and valid-action mask. Ground truth is private
  to sampling and oracle reward; the reconstructor receives only sparse IC and
  mask. Diagnostics in `info` are logged, never passed to the policy.
- A 4×4 action region chooses the unused `(x,y)` nearest its geometric center;
  ties resolve by ascending `x`, then `y`. Direct 256-action mode uses one
  exact coordinate per action. Already sampled exact coordinates cannot recur.
  With only eight CPTs, a 4×4 region of 16 coordinates is never exhausted, so
  all 16 region actions normally stay valid. Direct mode actively masks sampled
  coordinates.
- Replay stores observation, action, reward, next observation, and `done`.
  Epsilon exploration and greedy selection use legal actions only. The Bellman
  max masks invalid next actions; terminal targets have zero bootstrap. One
  optimization occurs per subsequent environment step once 32 transitions
  exist. The target network copies weights every 100 updates.
- Validation performs no replay insertion or gradient update. It selects
  checkpoints by hidden-voxel RMSE. Since the same validation set is also
  reported, those figures have checkpoint-selection optimism and are not a
  held-out estimate. The test split remains untouched as requested.
- Every episode has exactly five post-warm-start samples. The fixed cost
  `0.005` therefore contributes exactly `-0.025` to cumulative reward for
  every policy. Full-volume RMSE reduction telescopes:
  `sum(reward) = initial_RMSE_all - final_RMSE_all - 0.025`.
  Cost cannot distinguish actions under the fixed budget. Full-volume MAE
  reduction is the tested alternative, also over a fixed 2,048-voxel domain.
- The validation calls consume agent RNG draws even at epsilon zero. Runs are
  reproducible with the saved seed and validation schedule, but changing the
  validation frequency can change subsequent exploration. Decouple these RNG
  streams before a larger study.

## Results

Values are mean final hidden-voxel RMSE across 100 validation volumes, then
mean ± sample standard deviation **across training seeds** where three seeds
were run. Volume-wise median, volume-wise standard deviation, hidden MAE,
full-volume RMSE, runtime, win counts, and checkpoint scores are in the
aggregate and per-run result files.

| One-variable configuration | Seeds | DQN hidden RMSE | Action-matched Uniform | Action-matched Random | Mean DQN wins vs Uniform / 100 |
|---|---:|---:|---:|---:|---:|
| 500 train volumes, 500 episodes, 16 actions, RMSE reward | 3 | 0.09889 ± 0.00379 | 0.09499 | 0.09979 | 46.3 |
| 500 train volumes, 2,000 episodes, 16 actions, RMSE reward | 3 | 0.09354 ± 0.00077 | 0.09499 | 0.09979 | 53.3 |
| 2,000 train volumes, 2,000 episodes, 16 actions, RMSE reward | 3 | **0.09159 ± 0.00164** | 0.09499 | 0.09979 | 54.7 |
| 500 volumes, 500 episodes, direct 256 actions | 3 | 0.09420 ± 0.00246 | 0.10565 | 0.09779 | 63.0 |
| 500 volumes, 500 episodes, extra coverage channel | 1 | 0.10120 | 0.09499 | 0.09979 | 42 |
| 500 volumes, 500 episodes, full-volume MAE reward | 1 | 0.10030 | 0.09499 | 0.09979 | 43 |

The direct-256 baselines differ from 16-region baselines because their action
space differs. Compare policies **within** a row/action space, not a 256-action
DQN against a 16-region Uniform baseline. Additional training data gives a
smaller gain than the initial 500-vs-2,000-episode comparison suggested:
at a fixed 2,000 episodes, DQN improves from 0.09354 to 0.09159. This is
consistent across the three tested seeds, but the difference is small and
still measured on validation data used for checkpoint selection.

The 2,000-volume 16-region configuration has the best mean RMSE and is the
current candidate for further validation. It does **not** decisively dominate
Uniform volume by volume (54–55 wins out of 100 per seed). Do not claim a
general DQN advantage yet. The 256-action variant beats its weaker matched
Uniform baseline, but its improvement over matched Random is also small.

On seed 14, DQN's five added sites overlap Uniform's by about 1.28 sites in
the 2,000-volume 16-action run and 0.03 sites in direct-256 mode. DQN also
selects sites closer to prior CPTs: mean nearest-previous-site distance
3.80 cells versus Uniform's 6.56 in the 16-action run. The policies choose
meaningfully different locations; different is not automatically better.

Training took about 18–20 seconds for 500 episodes and 60–70 seconds for
2,000 episodes on this Mac CPU. Per-run JSON records both policy-decision
time and frozen SchemaGAN reconstruction time. These short local timings do
not predict 16,000-volume training cost linearly without profiling.

## Reproduce a run

```bash
cd ~/Developer/cpt_super_learn
./cpt/bin/python -m pytest -q dqn3d
./cpt/bin/python -m dqn3d.train --train-limit 2000 --val-limit 100 \
  --episodes 2000 --eval-every 500 --seed 14 \
  --validation-seed 1000014 --output results/dqn3d_opt/reproduction_seed14
./cpt/bin/python -m dqn3d.analyze_experiments
```

The training command refuses to overwrite an output directory. Before any
full-scale run, separate validation RNG from exploration RNG, assess more
seeds and validation volumes, and reserve the untouched test split for a
single final assessment after configuration choice.
