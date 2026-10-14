# Revalidation after RNG separation

The DQN architecture, frozen SchemaGAN checkpoint, train/validation CSVs,
16 region actions, three warm-start CPTs, eight-CPT total budget, and CPU
device were unchanged. `MaskedDQNAgent` now uses independent exploration and
replay RNG streams. Greedy validation consumes neither. Random and Uniform
baseline RNGs are separate from training and each other; all validation runs
use warm-start seed `1000014`.

Each run uses 2,000 train volumes, 2,000 episodes, and 100 identical validation
volumes. Checkpoints are selected at episodes 500, 1000, 1500, and 2000 by
validation hidden RMSE. Validation never inserts replay transitions or updates
weights. No test volume was read. Full per-volume metrics and separate best
checkpoints are in `results/dqn3d_repro/seed14`, `seed37`, and `seed73`.

| Seed | DQN hidden RMSE | DQN hidden MAE | DQN full RMSE | Uniform hidden RMSE | Random hidden RMSE | DQN wins vs Uniform | Training seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| 14 | 0.09058 | 0.02810 | 0.08916 | 0.09499 | 0.09979 | 58/100 | 65.93 |
| 37 | 0.08802 | 0.02749 | 0.08663 | 0.09499 | 0.09979 | 55/100 | 65.21 |
| 73 | 0.09467 | 0.02936 | 0.09318 | 0.09499 | 0.09979 | 51/100 | 64.73 |

Across training seeds, DQN hidden RMSE is **0.09109 ± 0.00335** (mean ±
sample SD), compared with 0.09159 ± 0.00164 before the RNG fix. The mean
change is −0.00050 IC units, small relative to seed variation. The stronger
seed variation and near tie on seed 73 argue against claiming a robust
policy advantage from this validation set.

`test_validation_frequency_does_not_change_training_actions_or_weights`
confirms identical training sampled-site sequences and final network weights
when validation runs every episode versus every three episodes under the same
training seed. All 13 `dqn3d` tests pass.

To reproduce one run:

```bash
cd ~/Developer/cpt_super_learn
./cpt/bin/python -m dqn3d.train --train-limit 2000 --val-limit 100 \
  --episodes 2000 --eval-every 500 --seed 14 \
  --validation-seed 1000014 --output results/dqn3d_repro/repeat_seed14
```

The run refuses to overwrite existing output. The first 2,000 train CSVs are
used once each. Before full training, choose an episode count explicitly:
16,000 episodes would process each of the 16,000 train volumes once, while
2,000 episodes would not use the full set under this training loop.
