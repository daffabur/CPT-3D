# DQN V3 — Scaled Pilot 2.000 Episode

## 1. Executive summary dan keputusan

**Keputusan: `NEEDS_TARGETED_POLICY_IMPROVEMENT`.** Training 2.000 episode selesai dengan validitas teknis penuh. Best checkpoint episode 1800 menghasilkan mean hidden-valid Ic RMSE 0,139181, lebih rendah daripada Uniform 0,140069 dan Random 0,141354 pada 300 validation ID. Selisih DQN−Uniform −0,000888 (CI bootstrap 95% tingkat volume [−0,001306, −0,000451]); DQN menang pada 63,7% volume. Namun keuntungan terutama pada 1–5 CPT awal: pada 12 dan 20 CPT DQN lebih buruk daripada Uniform dengan CI yang tidak melintasi nol. Maka policy belum layak diangkat sebagai baseline final tunggal untuk seluruh kondisi CPT. Tidak ada full training, test, atau holdout yang dijalankan.

## 2. Identitas dataset, checkpoint, dan pembatasan split

- Dataset: `data/synthetic3d_v2/schemagan_v3_24k/`; 2.000 realization train dan 300 validation ID pertama dari manifest, dipilih deterministik. Tidak ada overlap realization ID dan tidak ada akses test ID/geological holdout.
- SHA-256 manifest: `2d7b378199c5c21ddf449fde99b87838b12188af62c4116779787293bb21d563`. SHA-256 konfigurasi scaled: `b282ef67271a3194de588d143fcc24a510f978e1adfc63499f0bc28abdc70f0e`. SHA-256 feature schema: `440d9e750d609c32daecd013af9b86cc4c92845f387a4854ec0637a81bbccb07`.
- SchemaGAN V3: `results/schemagan3d_v3_training_continuation/mse_continuation/best.pt`, `idw_guided_residual`, objective `mse_only`, epoch 26. SHA-256 sebelum dan sesudah: `5694bb7b43efd7054a99fac8b74dadc5d74ebe6d291f42a8393b57fe9da52b64`; model mode `eval()`, seluruh parameter frozen, forward inference-only. Normalisasi Ic berasal dari train (mean 2,3142782318; std 0,2792749408), dan adapter memulihkan observed values. API rekonstruksi hanya menerima observed Ic/mask, valid mask, dan grid; hidden truth berada di simulator untuk reveal dan scoring, tidak pada policy.
- Training diinisialisasi baru dengan seed 14; **bukan resume** checkpoint 64 episode. File dataset, SchemaGAN, feature schema, reward, arsitektur, serta hyperparameter tidak diubah selama run.

## 3. Preflight dan automated tests

Preflight `passed`: identitas dataset/checkpoint/schema, split dan disjoint IDs, ketiga grid, sampling awal deterministik, baseline legal, reward = RMSE before−after pada common hidden-valid mask, batch 32 dengan jumlah kandidat bervariasi, optimizer DQN pada fixture, serta checkpoint roundtrip/replay/RNG identity. Tidak ada optimizer step pada volume produksi selama preflight. Estimasi batas konservatif memori fitur replay kapasitas 5.000 adalah 466 MiB (hanya replay; bukan peak RSS). Test suite relevan: **63 passed, 14 warnings**. Runner existing memiliki aturan best yang dibekukan sebelum training: minimum mean validation DQN RMSE after pada common mask.

## 4. Konfigurasi eksperimen

| Parameter | Nilai |
|---|---:|
| Train / validation ID volumes | 2.000 / 300 |
| Episode / validation interval | 2.000 / 200 |
| Initial CPT | 1, 2, 3, 5, 8, 12, 20 |
| Horizon | 1 |
| Seed / validation seed | 14 / 1000014 |
| Learning rate / gamma | 0,0001 / 0,99 |
| Batch / replay capacity | 32 / 5.000 |
| Target sync | 200 optimizer steps |
| Epsilon start/end/decay | 1,0 / 0,05 / 1.500 episode |
| MLP | 28→64→64→1 |

17 fitur kandidat + 11 fitur global digunakan sesuai schema existing. Loss `SmoothL1`, optimizer Adam, clipping gradient, dan baseline Uniform/Random tidak diubah. Perbedaan replay, batch, target sync, dan epsilon decay dibanding pilot 64 adalah konfigurasi skala baru secara bersamaan; eksperimen ini tidak mengisolasi pengaruh masing-masing parameter.

## 5. Validitas training dan monitoring

2.000 episode selesai, termasuk 1.969 optimizer update pada data train; replay terakhir 2.000 transisi. Semua reward, TD loss, dan gradient norm finite. Reward persis RMSE before−after pada common hidden-valid mask per transisi. Mean reward train 0,006233 (rentang −0,026781 sampai 0,117914); TD loss saat update 0,00000853–0,00128920; gradient norm pasca clip 0,000451–0,093109. Online Q-network berubah dari bobot inisialisasi (norma L2 0,594492; selisih absolut maksimum 0,054841). Semua 2.000 aksi train legal, duplikat 0%; validation best juga valid 100%, duplikat 0%. Epsilon mencapai 0,05 setelah jadwal 1.500 episode. Training reward dan TD loss tidak dipakai sebagai ukuran utama kualitas greedy policy.

| Episode | Mean reward train | Mean TD loss |
|---|---:|---:|
| 1–200 | 0.004762 | 0.00011383 |
| 201–400 | 0.005338 | 0.00004790 |
| 401–600 | 0.005656 | 0.00003785 |
| 601–800 | 0.005569 | 0.00003118 |
| 801–1000 | 0.004808 | 0.00003122 |
| 1001–1200 | 0.006709 | 0.00003225 |
| 1201–1400 | 0.006377 | 0.00003056 |
| 1401–1600 | 0.007516 | 0.00003244 |
| 1601–1800 | 0.008252 | 0.00003687 |
| 1801–2000 | 0.007341 | 0.00003926 |

## 6. Validation berkala dan best checkpoint

Setiap titik memakai 300 volume validation ID yang sama × tujuh jumlah CPT awal = 2.100 kasus per policy. DQN greedy, Uniform, dan Random menerima CPT awal dan kandidat legal identik; masing-masing memilih tanpa truth. RMSE dinilai pada mask `valid & ~(observed_after_DQN | observed_after_Uniform | observed_after_Random)` yang sama bagi ketiganya. Best ditentukan dari RMSE DQN terendah, bukan TD loss atau melihat test.

| Episode | DQN | Uniform | Random | DQN−Uniform | Win DQN vs Uniform |
|---:|---:|---:|---:|---:|---:|
| 200 | 0.142150 | 0.140117 | 0.141422 | +0.002033 | 28.3% |
| 400 | 0.141502 | 0.140121 | 0.141426 | +0.001380 | 33.7% |
| 600 | 0.140949 | 0.140094 | 0.141377 | +0.000855 | 37.0% |
| 800 | 0.140875 | 0.140092 | 0.141384 | +0.000783 | 39.3% |
| 1000 | 0.140350 | 0.140086 | 0.141382 | +0.000265 | 47.3% |
| 1200 | 0.139920 | 0.140073 | 0.141374 | -0.000154 | 53.3% |
| 1400 | 0.140067 | 0.140074 | 0.141370 | -0.000007 | 51.3% |
| 1600 | 0.139650 | 0.140077 | 0.141375 | -0.000427 | 55.0% |
| 1800 | 0.139181 | 0.140069 | 0.141354 | -0.000888 | 63.7% |
| 2000 | 0.139654 | 0.140097 | 0.141384 | -0.000443 | 54.3% |

Best checkpoint **episode 1800**, score 0,139181. Last checkpoint episode 2000, DQN RMSE 0,139654; best dan last disimpan terpisah. `best.pt` memuat online/target network, optimizer, replay, RNG, episode, learning step, config, feature schema SHA, manifest SHA, dan SchemaGAN SHA; resume loader memeriksa identitas. Evaluasi ulang `best.pt` secara terpisah menghasilkan CSV per-volume dengan SHA-256 sama persis dengan validasi episode 1800. Checkpoint roundtrip dan determinisme update RNG pada fixture lulus; kelanjutan training dari checkpoint produksi tidak dijalankan karena akan menambah eksperimen di luar protokol.

## 7. Perbandingan best DQN, Uniform, Random

Mean RMSE before pada common mask 0,146481. Improvement = before−after, positif berarti rekonstruksi membaik. Reward own-mask environment dibedakan dari improvement cross-policy common-mask.

| Policy | Mean RMSE after | Mean improvement | Valid | Duplicate |
|---|---:|---:|---:|---:|
| Dqn | 0.139181 | 0.007300 | 100% | 0% |
| Uniform | 0.140069 | 0.006412 | 100% | 0% |
| Random | 0.141354 | 0.005128 | 100% | 0% |

Selisih berpasangan DQN−baseline: negatif menguntungkan DQN. Tujuh kondisi CPT diagregasi dahulu per realization ID; SD dan 95% CI bootstrap memakai **300 volume** sebagai unit resampling, bukan 2.100 kasus.

| Baseline | Mean Δ | SD volume | 95% CI | Win rate DQN |
|---|---:|---:|---:|---:|
| Uniform | -0.000888 | 0.003770 | [-0.001306, -0.000451] | 63.7% |
| Random | -0.002172 | 0.004418 | [-0.002658, -0.001680] | 68.7% |

Karena best dipilih berdasarkan pool validation yang sama, CI ini bersifat evaluasi development setelah selection dan dapat optimistis. Ini bukan estimasi pada test independen.

## 8. Rincian grid dan jumlah CPT awal

| Grid | Kasus | DQN | Uniform | Random | DQN−Uniform |
|---|---:|---:|---:|---:|---:|
| g12 | 700 | 0.136731 | 0.137363 | 0.139152 | -0.000632 |
| g16 | 700 | 0.148271 | 0.149407 | 0.149322 | -0.001136 |
| g24 | 700 | 0.132541 | 0.133438 | 0.135587 | -0.000896 |

| CPT awal | Kasus | DQN | Uniform | Random | Δ vs Uniform (95% CI) |
|---:|---:|---:|---:|---:|---:|
| 1 | 300 | 0.166312 | 0.170430 | 0.166771 | -0.004117 [-0.005210, -0.002953] |
| 2 | 300 | 0.157098 | 0.160059 | 0.159208 | -0.002960 [-0.004017, -0.001939] |
| 3 | 300 | 0.151035 | 0.153127 | 0.154179 | -0.002092 [-0.002921, -0.001298] |
| 5 | 300 | 0.140911 | 0.141962 | 0.143697 | -0.001052 [-0.001883, -0.000238] |
| 8 | 300 | 0.129786 | 0.129315 | 0.132821 | +0.000471 [-0.000273, 0.001242] |
| 12 | 300 | 0.120219 | 0.118615 | 0.122325 | +0.001605 [0.000928, 0.002287] |
| 20 | 300 | 0.108907 | 0.106978 | 0.110475 | +0.001929 [0.001354, 0.002523] |

DQN unggul atas Uniform pada 1, 2, 3, 5 CPT, praktis belum jelas pada 8 CPT, dan kalah pada 12 serta 20 CPT. Grid g12/g16/g24 seluruhnya menunjukkan mean DQN lebih rendah, tetapi ini tidak menghapus heterogenitas menurut jumlah CPT. Keluarga horizontal (n=100), inclined (n=99), undulating (n=101) tercatat dalam JSON; subgroup family tetap analisis development, bukan klaim generalisasi ke keluarga baru.

## 9. Perilaku policy, Q-value, dan distribusi aksi

Greedy action best berubah dari inisialisasi pada 2025/2.100 kasus dan dari episode 200 pada 1990/2.100 kasus. DQN memilih 571 lokasi `(grid,x,y)` unik. Aksi tepi: DQN 57.3%, Uniform 97.2%, Random 24.3%. Lokasi DQN paling sering muncul 47 dari 2.100 kasus. Tidak ada collapse ke satu XY, meski preferensi tepi tetap perlu dipantau.
Q seluruh kandidat: min -0.037858, median 0.003068, max 0.024858, SD 0.004453; Q aksi terpilih: min 0.000689, median 0.008318, max 0.024858. Tidak tampak ledakan numerik. Distribusi train seimbang: grid g12/g16/g24 667/666/667 episode dan tiap jumlah CPT sekitar 285–286 episode. Hasil ini menunjukkan policy berubah dan menghasilkan manfaat pada beberapa strata, tetapi tidak membuktikan secara kausal apakah ia memakai fitur rekonstruksi atau terutama jarak CPT; itu memerlukan ablation terkontrol.

## 10. Pembanding pilot 64 episode pada kasus yang sama

Checkpoint 64 episode dan scaled best dievaluasi ulang pada **300 volume validation ID dan tujuh jumlah CPT awal yang sama**. Untuk menjaga fairness antarcheckpoint, empat policy (scaled DQN, pilot64 DQN, Uniform, Random) dinilai pada satu mask bersama yang mengeluarkan seluruh lokasi setelah aksi keempat policy. Karena mask empat-policy ini berbeda dari mask tiga-policy pada tabel utama, angka mentah di dua tabel tidak boleh dicampur.

| Policy | Mean RMSE after pada joint mask |
|---|---:|
| scaled_dqn | 0.139136 |
| pilot64_dqn | 0.142710 |
| uniform | 0.140043 |
| random | 0.141313 |

| Paired scaled−pembanding | Mean Δ | SD volume | 95% CI | Scaled menang |
|---|---:|---:|---:|---:|
| pilot64_dqn | -0.003574 | 0.004904 | [-0.004133, -0.003011] | 79.0% |
| uniform | -0.000907 | 0.003773 | [-0.001325, -0.000471] | 63.3% |
| random | -0.002177 | 0.004424 | [-0.002667, -0.001681] | 68.3% |

Pembandingan ini memperlihatkan perubahan checkpoint pada kasus identik, tetapi perbedaan eksperimen mencakup jumlah episode, ukuran pool train, batch/replay/target sync, dan jadwal epsilon sekaligus. Tidak ada atribusi kausal ke satu faktor.

## 11. Runtime, memori, dan integritas output

Selisih timestamp config dan `training_history.csv` sekitar 55.1 menit untuk training beserta sepuluh validation dan I/O. Jumlah durasi episode yang dicatat runner 102.0 detik; ini mengecualikan validation dan checkpoint I/O. Peak RSS training tidak diinstrumentasi sehingga tidak diklaim. Preflight memberi batas konservatif fitur replay 465.9 MiB pada kapasitas 5.000, bukan total memori proses. File `last.pt` terakhir dan `best.pt` tersedia; resume state tercakup dan identitas dicek saat load. SHA SchemaGAN tetap sama sebelum/sesudah.

## 12. Keterbatasan dan interpretasi

- Validation ID digunakan untuk memilih best; interval bootstrap pasca-selection tidak menggantikan test independen. Test/holdout sengaja tidak dibuka.
- Horizon satu langkah berarti return hampir seluruhnya reward langsung; ini tidak membuktikan strategi multi-step walaupun gamma dan target sync ada.
- Kemenangan overall atas Uniform kecil secara absolut dan tidak berlaku pada CPT awal tinggi. Performa menurut grid/family/CPT adalah diagnosis development, dengan risiko multiple comparisons.
- Q dan distribusi aksi tidak membuktikan penggunaan fitur rekonstruksi secara kausal. No field deployment validation.
- Runner menulis checkpoint tiap episode dan mengevaluasi 300 volume × 7 CPT × 3 policy tiap 200 episode; biaya waktu/I/O besar. Tidak ada peak RSS yang terekam.
- Perbandingan dengan pilot 64 pada joint mask tetap bersifat diagnostik karena konfigurasi training berbeda dan keduanya ditinjau pada validation ID.

## 13. Satu eksperimen terarah berikutnya (belum dilakukan)

**Lakukan satu eksperimen policy yang ditargetkan pada kondisi 12–20 CPT awal**, dengan arsitektur/reward/feature schema tetap dan protokol validation berpasangan yang dibekukan sebelum training. Contoh yang terdefinisi jelas: latih satu model dari seed baru dengan sampling train hanya kondisi 12 dan 20 CPT; ukur perubahan pada 12/20 sebagai sasaran utama dan 1–8 sebagai guardrail, lalu bandingkan dengan best scaled checkpoint pada kasus identik. Ini menguji apakah kekurangan pada CPT tinggi dapat diperbaiki tanpa mengubah model; tidak dijalankan pada tugas ini. Jika baseline Uniform tetap lebih baik pada strata sasaran, pertahankan Uniform untuk kondisi tersebut. Jangan memakai test/holdout untuk tuning.

## 14. Artefak dan perintah yang benar-benar dijalankan

Output: `scaled_training_history.csv`, `scaled_validation_history.csv`, `scaled_vs_baselines.json`, `scaled_policy_behavior.json`, `scaled_learning_curves.png`, `scaled_config.json`, `best.pt`, `last.pt`, `scaled_integrity.json`, `preflight.json`, serta `joint_checkpoint_comparison.json` dan CSV. Folder `run/` berisi validation summary dan per-volume untuk kesepuluh checkpoint; `best_validation/` berisi reevaluasi best. Tidak ada Git commit/push/reset/checkout, instalasi dependency, atau perubahan source code.

Perintah inti dijalankan dari root repository dengan `PYTHONPYCACHEPREFIX` dan `MPLCONFIGDIR` diarahkan ke workspace sementara:

```bash
cpt/bin/python /Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/dqn_v3_scaled_preflight.py
cpt/bin/python -m pytest -q -p no:cacheprovider --disable-warnings dqn3d schemagan3d/test_v3_adapter.py schemagan3d/test_v3_idw_guided.py schemagan3d/test_v3_training_monitoring.py
cpt/bin/python -m dqn3d.v3.runner train --config results/dqn_v3_scaled_pilot_2000e/scaled_config.json --output results/dqn_v3_scaled_pilot_2000e/run
cpt/bin/python -m dqn3d.v3.runner validate --config results/dqn_v3_scaled_pilot_2000e/scaled_config.json --checkpoint results/dqn_v3_scaled_pilot_2000e/run/best.pt --output results/dqn_v3_scaled_pilot_2000e/best_validation
cpt/bin/python /Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/analyze_dqn_v3_scaled.py
cpt/bin/python /Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/dqn_v3_joint_compare.py
```

## 15. Keputusan akhir

**`NEEDS_TARGETED_POLICY_IMPROVEMENT`**. Pipeline valid dan scaled DQN unggul atas Uniform serta Random secara keseluruhan pada validation ID, dengan CI overall di bawah nol. Namun kegagalan yang jelas pada 12/20 CPT awal bertentangan dengan tujuan policy tunggal untuk seluruh rentang CPT. Satu eksperimen terarah di atas direkomendasikan sebelum final DQN training. Tidak ada full training yang dimulai.
