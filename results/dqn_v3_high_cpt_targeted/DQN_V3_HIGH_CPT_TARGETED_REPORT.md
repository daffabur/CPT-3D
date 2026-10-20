# DQN V3 — Targeted Policy Improvement (High-CPT)

## 1. Executive summary dan keputusan

**Keputusan: `RETAIN_SCALED_DQN_AND_UNIFORM_HYBRID`.** Satu eksperimen baru 2.000 episode selesai dengan training CPT awal `[12,20]` dan seed 28. Best checkpoint dipilih pada episode **1800** dari rata-rata validation RMSE 12 dan 20 CPT yang berbobot sama. Model baru mengungguli checkpoint scaled DQN pada kedua target, tetapi masih memiliki RMSE mean lebih tinggi daripada Uniform pada 12 dan 20 CPT; pada 20 CPT CI paired sepenuhnya di atas nol. Model baru juga merusak guardrail 1–3 CPT terhadap scaled DQN. Karena itu satu checkpoint high-CPT tidak direkomendasikan untuk seluruh rentang dan Uniform tetap pilihan validation untuk 12/20. Tidak ada hybrid yang diimplementasikan pada aplikasi.

## 2. Identitas dataset, checkpoint, dan pemisahan split

- Dataset `data/synthetic3d_v2/schemagan_v3_24k/`; 2.000 train dan 300 validation ID pertama sesuai manifest. Realization ID train/validation tidak overlap. Kedua pool identik dengan scaled pilot. Test ID dan geological holdout tidak diakses.
- SHA-256 manifest `2d7b378199c5c21ddf449fde99b87838b12188af62c4116779787293bb21d563`; feature schema `440d9e750d609c32daecd013af9b86cc4c92845f387a4854ec0637a81bbccb07`; konfigurasi high-CPT `e698ab41bf50f5842bb7d74b70aa8115b59d46b590b5a20301d77409fa2fcb0f`.
- SchemaGAN `results/schemagan3d_v3_training_continuation/mse_continuation/best.pt`: SHA-256 sebelum/sesudah `5694bb7b43efd7054a99fac8b74dadc5d74ebe6d291f42a8393b57fe9da52b64`, `idw_guided_residual`, objective `mse_only`, epoch 26, mode `eval()` dan seluruh parameter frozen. Normalisasi Ic train-only serta observed-value restoration existing dipertahankan. Policy hanya melihat observasi, prediksi, kandidat legal, dan grid; truth tetap privat pada simulator sampai reveal/scoring.
- Pembanding scaled DQN `results/dqn_v3_scaled_pilot_2000e/run/best.pt`: SHA-256 sebelum/sesudah `687539d5ad2606b198f2aadfaa6761bc9d9968ec8ce1fd2c915350413530345a`; tidak ditimpa. Checkpoint ini episode 1800 dari eksperimen sebelumnya.
- Perubahan eksperimen **dua hal sekaligus**: distribusi CPT training `[12,20]` dan seed inisialisasi/sampling 28 (sebelumnya 14). Efek tidak boleh diklaim kausal hanya berasal dari salah satunya.

## 3. Perubahan kode minimum dan preflight

`dqn3d/v3/runner.py` diperluas untuk membaca daftar CPT validation terpisah dari training, memuat pembanding scaled DQN SHA-terverifikasi sebagai policy keempat, memakai common hidden-valid mask empat-policy, dan memilih best dari mean RMSE 12/20 berbobot sama. Logging Q min/median/max/selected dan aksi tepi ditambahkan per episode. Default runner lama tetap memakai satu daftar CPT, tiga policy, dan aturan selection sebelumnya. Arsitektur DQN, feature schema, reward, data, SchemaGAN, dan hyperparameter tidak diubah.
Preflight `passed`: checkpoint/dataset/schema cocok; 2.000/300 volume dan ID split valid; tiga grid; sampling deterministik; reward common-mask dan legal action; batch 32 dengan kandidat bervariasi; checkpoint roundtrip; SchemaGAN tetap frozen; perkiraan konservatif replay kapasitas 5.000 sekitar 457 MiB (fitur replay saja, bukan peak RSS). Dua tes regresi baru memeriksa train `[12,20]`, validation tujuh CPT, metadata checkpoint, dan mask empat-policy. **65 tests passed, 14 warnings** pada suite relevan. Tidak ada optimizer step pada volume 24K selama preflight.

## 4. Konfigurasi eksperimen yang dibekukan

| Parameter | Nilai |
|---|---:|
| Train / validation ID | 2.000 / 300 volume |
| Episode / interval validation | 2.000 / 200 |
| CPT train | 12, 20 |
| CPT validation | 1, 2, 3, 5, 8, 12, 20 |
| Selection CPT | 12 dan 20, bobot sama |
| Horizon | 1 |
| Train seed / validation seed | 28 / 1000014 |
| Learning rate / gamma | 0,0001 / 0,99 |
| Batch / replay capacity | 32 / 5.000 |
| Target sync | 200 optimizer steps |
| Epsilon start/end/decay | 1,0 / 0,05 / 1.500 episode |
| Feature / Q scorer | 17 kandidat + 11 global; MLP 28→64→64→1 |

Learning rate, batch, replay, target sync, epsilon, Adam, SmoothL1, clipping, reward, dan baseline sama dengan scaled pilot. Model baru dimulai dari random initialization, tidak me-resume checkpoint lama. Horizon satu langkah tidak membuktikan strategi CPT multi-step.

## 5. Training dan monitoring

2.000 episode selesai, tepat 1.000 episode dengan 12 CPT dan 1.000 dengan 20 CPT; 1.969 optimizer update. Semua reward, TD loss, gradient norm, dan Q values finite; seluruh aksi train legal dan tidak ada CPT duplikat. Online Q-network berubah dari inisialisasi (norma L2 0.500357, selisih absolut maksimum 0.053005). Mean reward train 0.003377; ini bukan ukuran utama kualitas greedy policy.

| Episode | Mean reward train | Mean TD loss | Fraksi aksi tepi train |
|---|---:|---:|---:|
| 1–200 | 0.001538 | 0.00002877 | 32.0% |
| 201–400 | 0.001664 | 0.00000891 | 35.0% |
| 401–600 | 0.002155 | 0.00000653 | 36.5% |
| 601–800 | 0.002888 | 0.00000624 | 50.5% |
| 801–1000 | 0.003371 | 0.00000723 | 53.0% |
| 1001–1200 | 0.003842 | 0.00000737 | 64.5% |
| 1201–1400 | 0.003851 | 0.00000750 | 74.5% |
| 1401–1600 | 0.004501 | 0.00000782 | 84.0% |
| 1601–1800 | 0.004936 | 0.00000802 | 81.5% |
| 1801–2000 | 0.005024 | 0.00000865 | 80.0% |

## 6. Validation berkala dan best checkpoint

Setiap checkpoint dievaluasi pada 300 volume validation ID yang sama × tujuh jumlah CPT awal = 2.100 kasus per policy. DQN baru, scaled DQN, Uniform, dan Random menerima initial CPT/kandidat legal identik. RMSE dihitung pada `valid & ~(observed_after_high | observed_after_scaled | observed_after_uniform | observed_after_random)` yang sama bagi keempat policy. Ground truth tidak dipakai untuk memilih aksi. Score selection adalah **(mean RMSE DQN pada 12 CPT + mean RMSE DQN pada 20 CPT)/2**; lower better. Full-range mean dicatat terpisah sebagai guardrail.

| Episode | Target score 12/20 | Full-range DQN | Scaled DQN | Uniform | Random |
|---:|---:|---:|---:|---:|---:|
| 200 | 0.116660 | 0.141577 | 0.139100 | 0.140020 | 0.141265 |
| 400 | 0.115379 | 0.141076 | 0.139029 | 0.139946 | 0.141191 |
| 600 | 0.114296 | 0.140285 | 0.139019 | 0.139944 | 0.141170 |
| 800 | 0.113891 | 0.139984 | 0.139016 | 0.139969 | 0.141158 |
| 1000 | 0.113623 | 0.139821 | 0.139030 | 0.139971 | 0.141160 |
| 1200 | 0.113628 | 0.139662 | 0.139045 | 0.139987 | 0.141166 |
| 1400 | 0.113404 | 0.139540 | 0.139060 | 0.140016 | 0.141183 |
| 1600 | 0.113352 | 0.139646 | 0.139045 | 0.139997 | 0.141171 |
| 1800 | 0.113171 | 0.139532 | 0.139068 | 0.140022 | 0.141193 |
| 2000 | 0.113282 | 0.139589 | 0.139070 | 0.140023 | 0.141189 |

Best checkpoint episode 1800 mempunyai target score 0.113171. Last checkpoint episode 2000 terpisah; checkpoint best/last menyimpan online/target Q-network, optimizer, replay, RNG, episode, learn steps, config, feature schema SHA, manifest SHA, dan SchemaGAN SHA. Resume identity dicek oleh runner. Reevaluasi best checkpoint secara terpisah menghasilkan CSV per-volume yang identik SHA-256 dengan validasi episode 1800.

## 7. Perbandingan akhir empat policy

RMSE hidden-valid Ic pada skala fisik, lebih rendah lebih baik. Improvement = RMSE before−after pada common mask; reward lingkungan own-mask dicatat terpisah.

| Policy | Mean RMSE after | Mean improvement | Valid | Duplicate |
|---|---:|---:|---:|---:|
| dqn | 0.139532 | 0.006783 | 100% | 0% |
| scaled_dqn | 0.139068 | 0.007247 | 100% | 0% |
| uniform | 0.140022 | 0.006293 | 100% | 0% |
| random | 0.141193 | 0.005122 | 100% | 0% |

Selisih berpasangan high-CPT DQN−pembanding: negatif menguntungkan model baru. Tujuh kondisi CPT dirata-ratakan dahulu per realization, lalu SD dan bootstrap CI 95% dihitung atas **300 volume**, bukan 2.100 kasus. CI ini adalah development validation setelah checkpoint selection dan dapat optimistis.

| Pembanding | Mean Δ | SD volume | CI 95% | Win rate high-CPT DQN |
|---|---:|---:|---:|---:|
| scaled_dqn | +0.000464 | 0.003934 | [0.000007, 0.000897] | 42.3% |
| uniform | -0.000490 | 0.002534 | [-0.000770, -0.000208] | 59.7% |
| random | -0.001661 | 0.004699 | [-0.002200, -0.001151] | 68.0% |

## 8. Target 12/20 dan guardrail 1–8 CPT

| CPT awal | High-CPT DQN | Scaled DQN | Uniform | Random | Δ vs Scaled (CI 95%) | Δ vs Uniform (CI 95%) | Win vs Scaled | Win vs Uniform |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.168750 | 0.166208 | 0.170441 | 0.166605 | +0.002542 [0.001160, 0.003819] | -0.001691 [-0.002652, -0.000778] | 40.7% | 48.3% |
| 2 | 0.159348 | 0.157047 | 0.160061 | 0.159107 | +0.002302 [0.001280, 0.003314] | -0.000713 [-0.001523, 0.000078] | 36.0% | 37.7% |
| 3 | 0.152012 | 0.150973 | 0.153148 | 0.154076 | +0.001039 [0.000134, 0.001899] | -0.001136 [-0.001824, -0.000528] | 37.7% | 43.3% |
| 5 | 0.141590 | 0.140825 | 0.141952 | 0.143571 | +0.000765 [-0.000107, 0.001734] | -0.000362 [-0.000867, 0.000130] | 40.0% | 38.0% |
| 8 | 0.128681 | 0.129663 | 0.129238 | 0.132623 | -0.000982 [-0.001729, -0.000261] | -0.000557 [-0.001079, -0.000084] | 51.3% | 33.0% |
| 12 | 0.118858 | 0.120047 | 0.118505 | 0.122135 | -0.001189 [-0.001780, -0.000634] | +0.000353 [-0.000059, 0.000762] | 53.0% | 31.7% |
| 20 | 0.107485 | 0.108716 | 0.106809 | 0.110233 | -0.001232 [-0.001717, -0.000724] | +0.000675 [0.000256, 0.001117] | 56.3% | 26.3% |

Pada 12 dan 20 CPT model baru secara berpasangan lebih baik dari scaled DQN (kedua CI di bawah nol). Terhadap Uniform: 12 CPT mean model baru sedikit lebih buruk, CI melintasi nol; 20 CPT lebih buruk dengan CI sepenuhnya di atas nol. Guardrail 1–3 CPT lebih buruk daripada scaled DQN dengan CI positif; pada 5 CPT belum jelas, pada 8 CPT model baru lebih baik. Hasil ini tidak mendukung satu checkpoint baru untuk seluruh rentang.

## 9. Grid, Q-value, dan distribusi lokasi

| Grid | n kasus | High-CPT DQN | Scaled DQN | Uniform | Random |
|---|---:|---:|---:|---:|---:|
| g12 | 700 | 0.136732 | 0.136537 | 0.137252 | 0.138899 |
| g16 | 700 | 0.148833 | 0.148179 | 0.149383 | 0.149191 |
| g24 | 700 | 0.133031 | 0.132489 | 0.133431 | 0.135489 |

Aksi greedy validation model baru mencakup 300 lokasi `(grid,x,y)` unik dari 2.100 kasus; fraksi tepi 88.5%. Pembanding scaled 57.3%, Uniform 97.2%, Random 24.3%. Tidak ada aksi ilegal/duplikat. Q terpilih selama training: min -0.046155, median 0.004564, max 0.020933; Q seluruh kandidat per episode tersimpan pada histori. Perubahan Q/lokasi tidak membuktikan fitur rekonstruksi dipakai secara kausal tanpa ablation.

## 10. Rekomendasi policy selection

`recommended_policy_selection.json` merekomendasikan aturan **validation-only, belum diimplementasikan**: untuk jumlah CPT awal yang telah diuji **1, 2, 3, 5, 8**, gunakan checkpoint scaled DQN episode 1800; untuk **12 atau 20**, gunakan Uniform spatial coverage. Keputusan routing hanya bergantung pada jumlah CPT awal dan observasi/lokasi kandidat legal, bukan hidden truth. Jumlah CPT selain tujuh nilai tersebut belum divalidasi dan tidak diberi aturan. Ambang ini mengikuti batas target eksperimen, bukan hasil pencarian banyak ambang. Jika suatu saat diuji pada final test, aturan harus dibekukan sebelum test dibuka.

## 11. Runtime, batasan, dan keputusan

Selisih timestamp config dan histori akhir sekitar 70.9 menit termasuk sepuluh validation, checkpoint, dan I/O; total durasi episode yang dicatat 105.2 detik, tidak mencakup biaya validation/I/O. Peak RSS training tidak diinstrumentasi. Estimasi preflight replay sekitar 457.1 MiB hanya untuk fitur replay konservatif, bukan memori total.
Validation ID dipakai untuk memilih checkpoint dan menilai routing; CI tidak setara estimasi test independen. Tidak ada klaim geologi baru, penggunaan fitur secara kausal, atau strategi multi-step. Perubahan seed dan distribusi CPT terjadi bersamaan. Tidak ada final training, test/holdout evaluation, deployment hybrid, atau eksperimen tambahan setelah run ini.
**Keputusan akhir: `RETAIN_SCALED_DQN_AND_UNIFORM_HYBRID`.** Pipeline valid; model high-CPT memperbaiki scaled DQN pada target, tetapi masih kalah dari Uniform pada 20 CPT dan tidak melampauinya secara meyakinkan pada 12 CPT, sementara low-CPT guardrail memburuk.

## 12. Artefak dan perintah yang benar-benar dijalankan

Folder ini memuat `high_cpt_config.json`, `high_cpt_training_history.csv`, `high_cpt_validation_history.csv`, `high_cpt_vs_baselines.json`, `high_cpt_per_cpt_comparison.json`, `high_cpt_learning_curves.png`, `recommended_policy_selection.json`, `best.pt`, `last.pt`, `preflight.json`, `high_cpt_integrity.json`, `high_cpt_policy_behavior.json`, `run_status.json`, `run/` (10 validation checkpoint), dan `best_validation/` (reevaluasi). Field `status` dalam `high_cpt_config.json` adalah snapshot pra-run yang tidak diubah agar SHA checkpoint/resume tetap valid; status aktual selesai tercatat dalam `run_status.json`. Tidak ada Git commit/push/reset/checkout atau instalasi dependency.

Perintah inti dari root repository, dengan `PYTHONPYCACHEPREFIX` dan `MPLCONFIGDIR` diarahkan ke workspace sementara:

```bash
cpt/bin/python /Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/dqn_v3_high_cpt_preflight.py
cpt/bin/python -m pytest -q -p no:cacheprovider --disable-warnings dqn3d schemagan3d/test_v3_adapter.py schemagan3d/test_v3_idw_guided.py schemagan3d/test_v3_training_monitoring.py
cpt/bin/python -m dqn3d.v3.runner train --config results/dqn_v3_high_cpt_targeted/high_cpt_config.json --output results/dqn_v3_high_cpt_targeted/run
cpt/bin/python -m dqn3d.v3.runner validate --config results/dqn_v3_high_cpt_targeted/high_cpt_config.json --checkpoint results/dqn_v3_high_cpt_targeted/run/best.pt --output results/dqn_v3_high_cpt_targeted/best_validation
cpt/bin/python /Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/analyze_dqn_v3_high_cpt.py
```
