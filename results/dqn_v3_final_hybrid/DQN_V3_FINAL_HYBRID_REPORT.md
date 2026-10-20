# DQN V3 — Final Hybrid Policy & Independent Evaluation

## Keputusan dan ruang lingkup

**`READY_FOR_COREML_CONVERSION_FEASIBILITY`**. Hybrid yang dibekukan telah dievaluasi sekali pada keempat split final, seluruh hasil disimpan, 91 tes regresi lulus, dan kontrak inference terdokumentasi. Status ini hanya mengizinkan studi kelayakan konversi; belum membuktikan kompatibilitas Core ML/iPhone atau generalisasi lapangan. Tidak ada training, konversi, perubahan checkpoint/dataset/schema/reward, atau integrasi aplikasi.

Hybrid menang secara agregat melawan Uniform pada empat split, tetapi **pada 8 CPT awal Hybrid kalah dari Uniform pada keempat split** dengan CI berpasangan di atas nol. Hasil ini tidak mengubah routing final yang dibekukan sebelum test. Untuk rencana aplikasi, Uniform adalah baseline konservatif pada 8 CPT sampai aturan baru diuji pada data independen lain; jangan mengklaim rute test-informed itu telah tervalidasi secara independen.

## Identitas dan pembekuan protokol

- Dataset: `data/synthetic3d_v2/schemagan_v3_24k/manifest.json`, SHA-256 `2d7b378199c5c21ddf449fde99b87838b12188af62c4116779787293bb21d563`. Train 16.000, validation ID 4.000, masing-masing test 1.000; realization ID unik dan lineage tidak melintasi split. Hash 12.000 berkas input Ic/valid/metadata yang dipakai cocok dengan manifest.
- SchemaGAN: `results/schemagan3d_v3_training_continuation/mse_continuation/best.pt`, SHA-256 `5694bb7b43efd7054a99fac8b74dadc5d74ebe6d291f42a8393b57fe9da52b64`; MSE-only epoch 26, varian IDW-guided residual, normalisasi Ic train-only mean 2.314278231778639/std 0.27927494076233944.
- DQN scaled: `results/dqn_v3_scaled_pilot_2000e/run/best.pt`, SHA-256 `687539d5ad2606b198f2aadfaa6761bc9d9968ec8ce1fd2c915350413530345a`; best episode 1800. Feature schema SHA-256 `440d9e750d609c32daecd013af9b86cc4c92845f387a4854ec0637a81bbccb07` (17 kandidat, 11 global).
- Aturan: CPT XY unik 1,2,3,5,8 → scaled DQN; 12,20 → Uniform; jumlah lain → `unsupported_cpt_count`. `no_candidate` eksplisit. Aturan bersumber dari validation dan tersimpan di `frozen_hybrid_policy.json`.
- `frozen_final_evaluation_protocol.json` telah ditulis **sebelum membuka array test**, SHA-256 `527ac6485f12152faa13c4131d07a3a2f3fcb3d24a65e96feab4e92d6637fb97`. Tidak diubah sesudah evaluasi.

## Metode

Untuk tiap realization dan tujuh jumlah CPT awal, lokasi awal ditentukan oleh `nested_sites` dengan seed `evaluation_site_seed + 1000014`, sama untuk Hybrid, Uniform, dan Random. Kandidat memerlukan profil valid, belum teramati, serta lolos access mask; evaluasi sintetis memakai akses seluruh domain. Uniform memaksimumkan jarak XY fisik ke CPT terdekat; Random memakai seed SHA-256 kasus tetap. Pemilihan aksi hanya menerima CPT teramati, prediksi SchemaGAN, mask, grid, dan (untuk DQN) fitur kandidat. Truth hanya dibaca oleh evaluator untuk mengungkap profil CPT terpilih dan menghitung skor.

Setelah ketiga aksi, mask skor = `valid & ~(observed_after_hybrid | observed_after_uniform | observed_after_random)`. RMSE before/after dan improvement = before−after dihitung pada voxel Ic fisik yang sama. Selisih adalah Hybrid−baseline, sehingga negatif menguntungkan Hybrid. SD dan CI 95% berasal dari selisih berpasangan yang dirata-rata per realization ID atas tujuh CPT, lalu bootstrap persentil 2.000 resample pada volume (seed 1000014). Win rate memakai selisih volume < 0. Tabel per CPT memakai 1.000 volume per split; ties saat Hybrid=Uniform pada 12/20 tidak dihitung menang.

## Hasil per split

| Split | Hybrid RMSE | Uniform RMSE | Random RMSE | Improvement H/U/R | Δ H−U (CI 95%) | Δ H−R (CI 95%) | Win H vs U |
|---|---:|---:|---:|---:|---|---|---:|
| `test_id` | 0.139084 | 0.139973 | 0.141590 | 0.007897/0.007009/0.005392 | -0.000888 [-0.001111, -0.000669] | -0.002505 [-0.002762, -0.002247] | 60.5% |
| `test_family` | 0.140142 | 0.140994 | 0.142557 | 0.007871/0.007019/0.005456 | -0.000852 [-0.001094, -0.000616] | -0.002415 [-0.002684, -0.002130] | 61.8% |
| `test_geometry_parameter` | 0.140238 | 0.141252 | 0.142716 | 0.007978/0.006963/0.005499 | -0.001014 [-0.001229, -0.000790] | -0.002479 [-0.002764, -0.002203] | 64.2% |
| `test_response_parameter` | 0.153611 | 0.154652 | 0.156628 | 0.009259/0.008218/0.006242 | -0.001041 [-0.001309, -0.000784] | -0.003017 [-0.003341, -0.002698] | 61.5% |

Setiap split memuat 1.000 volume × 7 CPT awal × 3 policy = 21.000 baris, total 84.000 baris. Valid action 100%, duplicate 0%, no-candidate 0% pada semua split dan policy.

## Rincian jumlah CPT awal

| Split | CPT | Hybrid RMSE | Uniform RMSE | Δ H−U | CI 95% Δ H−U |
|---|---:|---:|---:|---:|---|
| `test_id` | 1 | 0.168331 | 0.171602 | -0.003271 | [-0.003935, -0.002626] |
| `test_id` | 2 | 0.158812 | 0.161249 | -0.002437 | [-0.003035, -0.001834] |
| `test_id` | 3 | 0.151702 | 0.153265 | -0.001563 | [-0.002107, -0.001035] |
| `test_id` | 5 | 0.141324 | 0.141099 | 0.000226 | [-0.000228, 0.000701] |
| `test_id` | 8 | 0.129344 | 0.128517 | 0.000827 | [0.000390, 0.001256] |
| `test_id` | 12 | 0.118398 | 0.118398 | 0.000000 | [0.000000, 0.000000] |
| `test_id` | 20 | 0.105679 | 0.105679 | 0.000000 | [0.000000, 0.000000] |
| `test_family` | 1 | 0.169201 | 0.172354 | -0.003153 | [-0.003836, -0.002453] |
| `test_family` | 2 | 0.159512 | 0.161603 | -0.002091 | [-0.002654, -0.001517] |
| `test_family` | 3 | 0.152753 | 0.154228 | -0.001475 | [-0.002060, -0.000915] |
| `test_family` | 5 | 0.142389 | 0.142449 | -0.000060 | [-0.000521, 0.000445] |
| `test_family` | 8 | 0.131001 | 0.130185 | 0.000816 | [0.000434, 0.001205] |
| `test_family` | 12 | 0.119309 | 0.119309 | 0.000000 | [0.000000, 0.000000] |
| `test_family` | 20 | 0.106831 | 0.106831 | 0.000000 | [0.000000, 0.000000] |
| `test_geometry_parameter` | 1 | 0.169750 | 0.173456 | -0.003706 | [-0.004380, -0.003014] |
| `test_geometry_parameter` | 2 | 0.159840 | 0.162082 | -0.002242 | [-0.002780, -0.001691] |
| `test_geometry_parameter` | 3 | 0.152646 | 0.154300 | -0.001654 | [-0.002142, -0.001181] |
| `test_geometry_parameter` | 5 | 0.141827 | 0.142313 | -0.000487 | [-0.000948, -0.000039] |
| `test_geometry_parameter` | 8 | 0.131109 | 0.130121 | 0.000988 | [0.000539, 0.001453] |
| `test_geometry_parameter` | 12 | 0.119512 | 0.119512 | 0.000000 | [0.000000, 0.000000] |
| `test_geometry_parameter` | 20 | 0.106980 | 0.106980 | 0.000000 | [0.000000, 0.000000] |
| `test_response_parameter` | 1 | 0.187215 | 0.191031 | -0.003815 | [-0.004607, -0.003019] |
| `test_response_parameter` | 2 | 0.176253 | 0.178806 | -0.002552 | [-0.003234, -0.001892] |
| `test_response_parameter` | 3 | 0.167955 | 0.169804 | -0.001849 | [-0.002481, -0.001213] |
| `test_response_parameter` | 5 | 0.155713 | 0.155904 | -0.000191 | [-0.000764, 0.000366] |
| `test_response_parameter` | 8 | 0.143126 | 0.142007 | 0.001120 | [0.000610, 0.001612] |
| `test_response_parameter` | 12 | 0.129874 | 0.129874 | 0.000000 | [0.000000, 0.000000] |
| `test_response_parameter` | 20 | 0.115138 | 0.115138 | 0.000000 | [0.000000, 0.000000] |

Pada 12 dan 20 CPT Hybrid **identik** dengan Uniform karena routing; selisih dan CI nol adalah konsekuensi desain. Pada 8 CPT, Δ H−U positif dengan CI di atas nol pada semua split. Pada 5 CPT hasil bervariasi dan umumnya CI melintasi nol. Pada 1–3 CPT Hybrid menang konsisten.

## Rincian grid dan keluarga

| Split | Stratum | n volume | Hybrid RMSE | Uniform RMSE | Δ H−U (CI 95%) |
|---|---|---:|---:|---:|---|
| `test_id` | grid: `g12` | 333 | 0.135285 | 0.135877 | -0.000592 [-0.001036, -0.000125] |
| `test_id` | grid: `g16` | 334 | 0.147768 | 0.148894 | -0.001126 [-0.001465, -0.000781] |
| `test_id` | grid: `g24` | 333 | 0.134175 | 0.135121 | -0.000946 [-0.001317, -0.000565] |
| `test_id` | family: `horizontal` | 333 | 0.138067 | 0.138866 | -0.000799 [-0.001214, -0.000376] |
| `test_id` | family: `inclined` | 334 | 0.138749 | 0.139972 | -0.001223 [-0.001645, -0.000849] |
| `test_id` | family: `undulating` | 333 | 0.140438 | 0.141080 | -0.000642 [-0.001006, -0.000259] |
| `test_family` | grid: `g12` | 334 | 0.135194 | 0.135260 | -0.000067 [-0.000461, 0.000329] |
| `test_family` | grid: `g16` | 333 | 0.149187 | 0.150681 | -0.001494 [-0.001877, -0.001133] |
| `test_family` | grid: `g24` | 333 | 0.136061 | 0.137058 | -0.000997 [-0.001413, -0.000589] |
| `test_family` | family: `lens` | 1000 | 0.140142 | 0.140994 | -0.000852 [-0.001094, -0.000616] |
| `test_geometry_parameter` | grid: `g12` | 334 | 0.134765 | 0.135445 | -0.000680 [-0.001118, -0.000268] |
| `test_geometry_parameter` | grid: `g16` | 333 | 0.147870 | 0.149213 | -0.001342 [-0.001667, -0.001038] |
| `test_geometry_parameter` | grid: `g24` | 333 | 0.138094 | 0.139116 | -0.001022 [-0.001410, -0.000644] |
| `test_geometry_parameter` | family: `inclined` | 1000 | 0.140238 | 0.141252 | -0.001014 [-0.001229, -0.000790] |
| `test_response_parameter` | grid: `g12` | 334 | 0.147748 | 0.148062 | -0.000315 [-0.000800, 0.000157] |
| `test_response_parameter` | grid: `g16` | 333 | 0.164203 | 0.165932 | -0.001728 [-0.002171, -0.001280] |
| `test_response_parameter` | grid: `g24` | 333 | 0.148898 | 0.149981 | -0.001083 [-0.001526, -0.000658] |
| `test_response_parameter` | family: `horizontal` | 1000 | 0.153611 | 0.154652 | -0.001041 [-0.001309, -0.000784] |

Rincian lengkap per grid, CPT, keluarga, kebijakan, dan interval berada di tiga berkas JSON hasil; CSV per kasus menyimpan lokasi dan mask count untuk audit. Analisis banyak strata bersifat deskriptif, tanpa koreksi multiple comparisons.

## Validation, generalization gap, dan overfitting

Validation ID development (300 realization) direkonstruksi dari baris tiga policy existing tanpa inferensi baru: mean RMSE Hybrid 0.138676, Uniform 0.140069, Δ -0.001393. Test ID memberi Hybrid 0.139084, Δ -0.000888. Keuntungan relatif Hybrid tetap negatif di seluruh split; test response memiliki RMSE absolut lebih tinggi, 0.153611, yang konsisten dengan pergeseran parameter respons tetapi tidak mengisolasi penyebabnya ke policy DQN atau SchemaGAN.

**Batas perbandingan:** validation memakai common mask tiga policy DQN/Uniform/Random asli. Pada CPT 12/20, Hybrid hasil rekombinasi memakai Uniform tetapi mask validation masih mengeluarkan aksi DQN; final mask memakai Hybrid/Uniform/Random. Karena itu gap angka RMSE mentah validation−test bukan estimasi terkontrol sempurna. Checkpoint/routing dipilih dengan validation, sehingga angka validation cenderung optimistis. Tidak ada evaluasi train DQN dengan protokol final pada tugas ini. Keempat test split berasal dari generator sintetis V3; family lens, geometry, dan response holdout menguji variasi dalam rancangan sintetis, bukan geologi lapangan. Split ini pernah dipakai untuk evaluasi SchemaGAN, sehingga independensi berlaku relatif terhadap **pemilihan DQN**, bukan seluruh riset. Tidak ada bukti cukup untuk menyimpulkan model bebas overfitting atau pasti overfitting.

## Pemeriksaan inference dan kesiapan port

`dqn3d/v3/hybrid.py` adalah API satu langkah tanpa argumen ground truth. Ia menghitung jumlah lokasi XY unik, mempertahankan fitur DQN training, membatasi valid/access mask, menolak duplikat, memberi koordinat fisik, dan memakai tie-break deterministik. Kontrak tensor, normalisasi, fitur, operasi non-neural, dan asumsi perangkat ada di `coreml_inference_contract.md`. Operator 3D, dynamic candidate count, numerik Swift, latensi, memori, versi iOS, dan kesetaraan output masih harus diuji sebelum klaim kompatibilitas Core ML.

## Pengujian dan integritas

- 91 automated tests lulus; 14 warnings existing. Kasus baru mencakup routing seluruh tujuh CPT pada g12/g16/g24, unsupported/no-candidate, API tanpa truth, minimum satu CPT, koordinat fisik, mask akses, determinisme, SHA checkpoint, urutan feature schema, common mask, dan isolasi lineage.
- Audit 12.000 input Ic/valid/metadata cocok checksum manifest. Keempat CSV masing-masing memiliki 21.000 baris dan 7.000 triplet berpasangan dengan initial CPT serta common mask count identik; seluruh RMSE finite. `test_results.json` menyimpan rincian.
- Semua output JSON/CSV evaluasi memuat SHA protokol atau merujuk ke berkas yang memilikinya. `artifact_checksums.json` memuat hash output final.

## Perintah yang benar-benar dijalankan

```bash
cpt/bin/python -m pytest -q -p no:cacheprovider --disable-warnings dqn3d schemagan3d/test_v3_adapter.py schemagan3d/test_v3_idw_guided.py schemagan3d/test_v3_training_monitoring.py
cpt/bin/python -m dqn3d.v3.final_eval --output results/dqn_v3_final_hybrid --protocol results/dqn_v3_final_hybrid/frozen_final_evaluation_protocol.json
```

`PYTHONPYCACHEPREFIX` dan `MPLCONFIGDIR` diarahkan ke workspace sementara. Skrip audit lokal memeriksa checksum input, pasangan CSV, dan menyusun laporan dari JSON hasil; tidak ada optimizer step.

## Rekomendasi tahap berikutnya

1. Lakukan studi kelayakan Core ML/Swift dengan pengujian kesetaraan numerik pada g12/g16/g24 dan pengukuran latensi/memori iPhone, tanpa mengubah hasil evaluasi ini.
2. Untuk kebijakan produk, pertimbangkan Uniform pada 8 CPT sebagai baseline konservatif karena empat split test menunjukkan kerugian Hybrid di kondisi itu. Rute revisi tersebut memerlukan validation/prospective dataset baru sebelum dinyatakan tervalidasi independen; jangan gunakan test yang sama untuk mengklaim keberhasilannya.
3. Uji pada geologi dan CPT lapangan yang berbeda sebelum klaim generalisasi praktis.

## Artefak

`frozen_hybrid_policy.json`, `frozen_final_evaluation_protocol.json`, `final_evaluation_by_split.json`, `final_evaluation_by_cpt.json`, `final_paired_confidence_intervals.json`, `final_policy_comparison.csv`, empat CSV per kasus, `development_validation_reference.json`, `input_integrity.json`, `test_results.json`, `coreml_inference_contract.md`, `artifact_checksums.json`.
