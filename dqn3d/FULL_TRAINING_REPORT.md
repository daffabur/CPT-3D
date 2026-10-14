# Full DQN 3D training and locked final test

Three independent CPU runs used 16,000 episodes each, visiting every one of
the 16,000 train CSVs exactly once per seed. The common configuration was
16 region actions (4×4 regions), three warm-start CPT profiles, five agent
choices, total budget eight, full-volume RMSE-reduction reward with cost
0.005 per new CPT, and a frozen SchemaGAN 3D checkpoint. No 2D model, game,
synthetic dataset, or SchemaGAN checkpoint was modified.

Exploration and replay have independent RNG streams. Validation uses neither
for greedy DQN decisions; baselines have separate RNGs. Validation used the
same first 100 `vali` volumes and warm-start seed 1000014 for every run.
Checkpoints were considered every 2,000 episodes. The test split was not read
until all three trainings finished and a checkpoint was locked.

## Training and validation

| Training seed | Best episode | Mean training reward | Validation hidden RMSE | Hidden MAE | Full-volume RMSE | DQN wins vs Uniform | CPU training time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 14 | 14,000 | 0.01768 | **0.09023** | 0.02831 | 0.08881 | 57/100 | 479.4 s |
| 37 | 10,000 | 0.01789 | 0.09308 | 0.02909 | 0.09161 | 50/100 | 490.1 s |
| 73 | 16,000 | 0.01755 | 0.09228 | 0.02887 | 0.09082 | 56/100 | 483.2 s |

The minimum-validation-RMSE rule selected **seed 14, episode 14,000**.
`results/dqn3d_full/locked_model.json` records all candidates and the
selected checkpoint SHA-256 (`788ebcc2d6a42339462ce3c83d5a841d92c926818b3d11a00a08aa95f9004087`).
The checkpoint and configuration were not changed after locking. Validation
scores are checkpoint-selection scores, not an unbiased final estimate.

## One final test on all 4,000 test volumes

All three methods received the same three starting `(X,Y)` locations for
each volume, chose five more CPTs under the same 16-region action space,
and ended with eight unique locations. The locked DQN was evaluated once;
no architecture, hyperparameter, checkpoint, or seed was selected using test.

| Method | Hidden RMSE mean ± volume SD | Hidden MAE mean | Full-volume RMSE mean |
|---|---:|---:|---:|
| Locked DQN | 0.09491 ± 0.03034 | 0.02964 | 0.09341 |
| Uniform | **0.09386 ± 0.03037** | **0.02871** | **0.09239** |
| Random | 0.09855 ± 0.03337 | 0.03062 | 0.09700 |

Paired differences are **DQN minus baseline**, so negative favors DQN.
The intervals are 95% paired percentile-bootstrap confidence intervals for
the mean difference, with 5,000 resamples and fixed seed 9901.

| Comparison | Mean paired hidden-RMSE difference | 95% CI | DQN wins |
|---|---:|---:|---:|
| DQN − Uniform | +0.00104 | [−0.00008, +0.00216] | 1,940/4,000 (48.5%) |
| DQN − Random | −0.00364 | [−0.00484, −0.00246] | 2,121/4,000 (53.0%) |

**Conclusion:** DQN is better than Random on mean hidden RMSE in this final
test, but it is **not shown to outperform Uniform**. Its mean is about 0.00104
IC units worse than Uniform and the paired 95% interval includes zero. This
small difference is compatible with either a near tie or a modest Uniform
advantage. Validation ranking did not generalize to a clear Uniform win.

`results/dqn3d_full/final_test/per_volume.csv` contains all 12,000 policy-
volume rows; `summary.json` contains distributions and paired intervals.
An integrity check found exactly 4,000 matched volume triplets, identical
warm starts within each triplet, and eight distinct CPT locations per policy.
All 15 `dqn3d` unit tests passed.

## Reproduce the workflow

Each full training run uses a distinct output directory and refuses overwrite:

```bash
cd ~/Developer/cpt_super_learn
./cpt/bin/python -m dqn3d.train --train-limit 16000 --val-limit 100 \
  --episodes 16000 --eval-every 2000 --progress-every 1000 \
  --seed 14 --validation-seed 1000014 --action-cells 4 --budget 8 \
  --output results/dqn3d_full/seed14
```

Repeat with seeds 37 and 73 in matching output directories. After all
training runs are complete, `python -m dqn3d.final_test lock` selects only
from validation and writes the lock manifest. `python -m dqn3d.final_test
evaluate` performs the one-shot 4,000-volume test, refuses to overwrite its
output, and checks both DQN and SchemaGAN hashes. The test should **not** be
rerun to choose another configuration.
