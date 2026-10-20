# DQN V3 — implementation and SchemaGAN V3 integration

## Executive summary

**Keputusan: `READY_FOR_DQN_V3_PILOT`** untuk pilot kecil berikutnya pada split **train/validation ID saja**. Implementasi terpisah di `dqn3d/v3/` berhasil memuat checkpoint SchemaGAN V3 MSE-only epoch 26, menerima satu atau lebih CPT awal, memproses g12/g16/g24, memberi skor untuk setiap kandidat XY legal, dan mengeluarkan satu rekomendasi fisik tanpa truth. Simulator sintetis menghitung reward pada common hidden-valid mask. Replay, satu optimizer step fixture, save/load, dan determinisme resume lulus tes. **Tidak ada pilot training atau optimizer step pada dataset 24K pada tahap ini.**

Status READY berarti komponen siap **diuji** dalam pilot, bukan bukti bahwa policy terlatih akan mengungguli Uniform. Dry run memakai Q-network berbobot acak dan hanya memeriksa fungsi integrasi.

## 1. File yang dibuat dan batas perubahan

| File | Peran |
|---|---|
| `dqn3d/v3/adapter.py` | Loader checkpoint V3 beku, validasi SHA/identitas/normalisasi, IDW fisik, rekonstruksi dan restoration. |
| `dqn3d/v3/features.py` | Kontrak state dan 17 fitur kandidat + 11 fitur global. |
| `dqn3d/v3/agent.py` | Shared MLP Q scorer, epsilon-greedy legal, replay kandidat variabel, target network, checkpoint/resume. |
| `dqn3d/v3/service.py` | API inferensi tanpa ground truth, satu rekomendasi XY atau `no_candidate`. |
| `dqn3d/v3/simulator.py` | Truth privat, reveal CPT sintetis, reward common mask. |
| `dqn3d/v3/baselines.py` | Random legal XY dan Uniform berdasarkan meter fisik. |
| `dqn3d/v3/runner.py` | Runner training dan evaluasi validation ID, protokol perbandingan berpasangan. |
| `dqn3d/v3/dry_run.py`, `dqn3d/v3/test_v3.py` | Dry run ringan dan 27 tes V3. |
| `results/dqn_v3_implementation/*` | Skema fitur, konfigurasi pilot, keluaran dry run/benchmark, laporan ini. |

Modul DQN lama, SchemaGAN, dataset 24K, checkpoint, game, dan visualisasi tidak diubah. Tidak ada Git commit/push/reset/checkout atau dependency baru.

## 2. Adapter SchemaGAN V3

Adapter memverifikasi checkpoint SHA-256 `5694bb7b43efd7054a99fac8b74dadc5d74ebe6d291f42a8393b57fe9da52b64`, epoch 26, `idw_guided_residual`, `mse_only`, digest konfigurasi, SHA manifest dan frozen plan, serta normalisasi train-only mean `2.314278231778639` dan std `0.27927494076233944`. Ia memakai `_models` existing, `idw_normalized_from_observed` existing (power 2, jarak XY fisik, per depth), `V3Normalization`, dan `restore_observed`. Input yang diizinkan hanya observed Ic, observed mask, valid mask, dan `Grid`; tidak ada argumen target/truth. Mask CPT harus mencakup seluruh kedalaman valid di tiap lokasi yang teramati. Model `eval()`, seluruh parameter `requires_grad=False`, dan forward memakai `torch.inference_mode()` CPU. Nilai teramati dicek sama persis setelah restoration.

Checkpoint ini dibuat dengan NumPy 2; virtualenv `cpt` memakai NumPy 1.23. Setelah SHA terpercaya diverifikasi, adapter memasang alias path pickle `numpy._core` ke `numpy.core` untuk membaca metadata checkpoint. Tidak ada instalasi atau perubahan checkpoint. Uji pada g12/g16/g24 menghasilkan shape benar, finite, dan output deterministik. Grid harus tetap memenuhi kontrak V3: dimensi ≥8 dan kelipatan empat; tidak ada resampling tersembunyi.

## 3. State, fitur, dan aksi

`dqn_v3_feature_schema.json` mengunci nama, urutan, dimensi, dan normalisasi. `CandidateState` berisi `candidate_xy[N,2]` yang diurut leksikografis, `features[N,17]`, dan `global_context[11]`. Fitur kandidat: XY lokal yang setara koordinat fisik ter-normalisasi setelah invers rotasi grid; jarak minimum CPT dalam meter dibagi diagonal domain fisik; densitas CPT radius 4 m; fraksi Z valid; mean/std Ic rekonstruksi ter-normalisasi; empat mean bin kedalaman beserta empat flag kosong; mean Ic lokal dan fraksi valid lokal radius 4 m. Konteks global: log jumlah CPT relatif kapasitas lokasi valid, bentang X/Y/Z dan spacing fisik terskala dengan konstanta katalog train V3, mean/std Ic rekonstruksi, fraksi valid, fraksi observed. Semua fitur Ic memakai statistik train-only di atas; tidak ada truth/error/unit geologi/profil mendatang/fixed remaining budget.

Legal XY memerlukan sedikitnya satu voxel Z valid, belum diobservasi, dan berada pada optional survey access mask. CPT teramati yang sudah ada boleh berada di luar access mask saat ini; mask tersebut membatasi rekomendasi **baru**. G12/g16/g24 mempunyai paling banyak 144/256/576 kandidat sebelum observasi. Shared MLP `28→64→64→1` memberi Q setiap kandidat dengan bobot sama. `argmax` pada urutan leksikografis menjadi tie break deterministik. `Grid.xy_m()` menghasilkan koordinat meter dengan origin dan azimuth; `Grid.index_xy()` dipakai dalam tes roundtrip. Jika kosong, API mengembalikan `status=no_candidate`. `q_score` tidak dipresentasikan sebagai probabilitas, confidence, atau penjelasan kausal; output opsional menyertakan ranking dan fitur jarak deskriptif.

## 4. Simulator dan reward

`SyntheticCPTSimulator` menyimpan `_truth` privat; state hanya berisi fitur dari observasi dan rekonstruksi. Setelah aksi, simulator membuka profil valid pada XY terpilih, merekonstruksi ulang, dan membekukan `H = valid & ~observed_after`. Reward adalah `RMSE_before(H) − RMSE_after(H)` pada Ic fisik. Kedua prediksi dinilai pada H yang identik; kolom yang baru dibuka tidak memberikan keuntungan mekanis. Jika H kosong, reward serta dua RMSE didefinisikan `0.0`, `done=True`, dan tidak ada NaN. API inferensi `recommend()` tidak mempunyai parameter truth, tidak membuka profil, dan hanya memberi satu rekomendasi per panggilan. Horizon training default 1 adalah batas simulasi pilot, bukan batas CPT aplikasi; kode runner mengizinkan horizon lebih panjang.

## 5. Replay, Q-learning, dan resume

Replay menyimpan fitur kandidat/global, mapping XY legal, action index **dan** XY terpilih, reward, next state, done; tidak menyimpan volume 3D penuh. Mapping divalidasi sebelum update. Batch dapat berisi jumlah kandidat berbeda: forward dan max dilakukan per state, kemudian TD values di-stack. Epsilon exploration memilih hanya kandidat legal. Terminal tidak melakukan bootstrap; nonterminal memaksimalkan Q target hanya pada kandidat legal. Loss `SmoothL1`, optimisasi Adam, gradient clipping, target sync periodik. Target model tidak menerima gradient. Dua RNG terpisah untuk eksplorasi/replay; checkpoint menyimpan kedua state RNG, torch RNG, online/target weights, optimizer, replay, learn step, dan metadata episode/identitas. Save memakai temporary file dan atomic replace. Tes fixture membuktikan satu step mengubah online weights tetapi tidak target, dan load/resume menghasilkan aksi RNG serta update berikutnya identik.

## 6. Random, Uniform, training runner, dan evaluasi

Random memilih indeks legal secara acak dengan seed tetap. Uniform memilih kandidat yang memaksimalkan jarak fisik minimum ke CPT teramati; tie break leksikografis. Tidak ada baseline 16 region. `runner.py` hanya mengizinkan split `train` dan `validation_id`; ia menolak nama split test/holdout. Training runner mendukung jumlah volume train/validation, episode, daftar CPT awal, horizon, kapasitas replay, batch, LR, target sync, epsilon, seed, validation interval, best/last checkpoint, resume identitas, CSV/JSON history. **Runner disiapkan tetapi tidak dijalankan untuk training dataset.** Konfigurasi kecil tersimpan dalam `dqn_v3_training_config.json`: 32 train, 12 validation ID, 64 episode, evaluasi tiap 16 episode, CPT awal `[1,2,3,5,8,12,20]`, horizon 1, replay 512, batch 8, LR 1e-4, target sync 50, seed 14. Angka ini default pilot awal, bukan hasil tuning.

Validation runner menjalankan DQN, Random, dan Uniform pada volume, CPT awal, dan kandidat legal yang sama. Tiap metode memilih satu aksi tanpa truth sebagai input policy. Untuk perbandingan antar-metode, RMSE sebelum/sesudah diukur pada **mask bersama lintas-kebijakan** `valid & ~(observed_after_DQN | observed_after_Uniform | observed_after_Random)`. Reward training tetap memakai common mask pre/post masing-masing transisi. Paired CI bootstrap memakai realization ID sebagai unit; jika satu volume diuji pada beberapa jumlah CPT awal, selisih dirata-ratakan dahulu per volume. Summary melaporkan mean after RMSE, improvement, win rate terhadap Uniform/Random, valid/duplicate action rate, hasil per grid dan CPT awal. Test/holdout tidak diakses oleh runner development dan tidak menjadi dasar selection.

## 7. Unit tests dan regresi

Perintah yang dijalankan:

```bash
PYTHONPYCACHEPREFIX=/Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/pycache MPLCONFIGDIR=/Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/matplotlib cpt/bin/python -m pytest -q -p no:cacheprovider --disable-warnings dqn3d schemagan3d/test_v3_adapter.py schemagan3d/test_v3_idw_guided.py schemagan3d/test_v3_training_monitoring.py
```

**63 passed**, termasuk **27 tes DQN V3**. Tes mencakup load/determinisme checkpoint, tiga grid, koordinat fisik, 1/multi CPT, candidate/no candidate, normalisasi dan schema, leakage, reward/reveal, Q-output variabel, epsilon masking, tie break, replay index, target gradient, fixture optimizer step, save/load dan resume RNG/update, baseline, inference tanpa truth, penolakan test split, dan kontrak DQN lama. Empat belas warning berasal dari dependency/test environment; tidak ada kegagalan. Sistem `python3` memiliki NumPy/PyTorch yang memuat checkpoint tetapi tidak memiliki pytest; suite dijalankan dengan virtualenv repository `cpt` tanpa menginstal paket.

## 8. Integration dry run dan benchmark ringan

Perintah yang dijalankan:

```bash
PYTHONPYCACHEPREFIX=/Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/pycache MPLCONFIGDIR=/Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/matplotlib cpt/bin/python -m dqn3d.v3.dry_run
```

Dry run membaca **tiga volume train** (satu per g12/g16/g24) dan **tiga volume validation ID**, membentuk state dari 1/2 CPT awal, menjalankan rekomendasi, reveal sintetis, reward common-mask, serta validation comparison 18 baris (3 volume × 2 CPT count × 3 policy). Semua aksi valid dan unik; jumlah kandidat sebelum aksi adalah 143 (g12), 255 (g16), 575 (g24). **Satu-satunya optimizer step** dilakukan pada tensor fitur fixture kecil tanpa volume dataset; save/load fixture dan perubahan bobot online valid. `dqn_v3_integration_dry_run.json` memuat identitas kasus dan pemeriksaan. Angka RMSE kebijakan acak pada dry run **bukan evaluasi performa DQN terlatih**.

`dqn_v3_benchmark.json` memuat timing CPU end-to-end rekomendasi hangat (termasuk rekonstruksi dan ekstraksi fitur) median tiga ulangan: g12 sekitar **0,007 s**, g16 **0,012 s**, g24 **0,030 s**; satu simulation step masing-masing sekitar 0,007/0,012/0,030 s. Angka bervariasi menurut cache/host dan berasal dari hanya tiga volume train, sehingga tidak diekstrapolasi ke throughput training penuh. Peak RSS proses dry run sekitar 850 MB, termasuk import PyTorch, manifest, dan cache; bukan memori replay pilot maksimum. CPU digunakan karena `ConvTranspose3D` existing belum didukung native MPS pada host seperti tercatat dalam benchmark SchemaGAN lama.

## 9. Remaining limitations dan langkah berikutnya

- Pilot train/validation belum dijalankan; kemampuan DQN mengungguli **Uniform** belum diketahui. Dry run random-weight hanya bukti fungsional.
- Validitas lapangan belum diuji. SchemaGAN V3 mendukung ukuran kelipatan empat ≥8; bentuk/resolusi lapangan lain memerlukan desain eksplisit, bukan resampling tersembunyi.
- IDW per kedalaman yang dipakai sebagai conditioning mensyaratkan sedikitnya satu nilai CPT teramati pada setiap level Z yang direkonstruksi. Jika valid mask proyek memungkinkan seluruh CPT awal tidak mencapai suatu level, adapter menolak input; kebijakan untuk profil parsial perlu ditetapkan dan divalidasi terpisah.
- Fitur kandidat ringkas mungkin melewatkan struktur geologi kompleks. Jangan ubah schema dari hasil test/holdout; evaluasi pilot pada validation ID dahulu.
- Cache rekonstruksi dan batch variable-candidate yang lebih efisien belum diimplementasikan. Runner memproses state per transisi dan menyimpan last checkpoint tiap episode, aman untuk pilot kecil tetapi belum benchmark skala penuh.
- Test/holdout SchemaGAN sudah pernah dianalisis di penelitian sebelumnya. Protokol final DQN harus dibekukan setelah pilot validation; hasil test DQN tidak boleh dipakai untuk tuning. Set final baru diperlukan bila ingin estimasi yang sepenuhnya independen dari seluruh proses riset.
- Untuk proyek lapangan, definisi valid mask, survey access mask, dan koordinat CPT harus disepakati dan divalidasi sebelum layanan rekomendasi digunakan. Biaya CPT tidak dimasukkan pada varian awal.

**Keputusan akhir: `READY_FOR_DQN_V3_PILOT`.** Tahap berikutnya adalah meninjau konfigurasi pilot, menjalankan pilot kecil train/validation ID dengan checkpoint SchemaGAN beku, memeriksa valid/duplicate action rate dan perbandingan terhadap Uniform, kemudian membekukan protokol sebelum evaluasi test DQN. Tidak ada pilot training dalam pekerjaan ini.
