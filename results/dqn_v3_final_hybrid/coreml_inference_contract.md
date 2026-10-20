# Kontrak inference untuk studi kelayakan Core ML

Status: spesifikasi porting, **belum dikonversi atau diuji di iPhone**. Identitas checkpoint dan skema dibekukan di `frozen_final_evaluation_protocol.json`.

## Komponen neural

1. SchemaGAN V3 `idw_guided_residual`, objective MSE-only, epoch 26. Input PyTorch `sparse_normalized`, `observed_mask`, `idw_normalized` masing-masing float32 `[B,1,Z,Y,X]`; output prediction normalized Ic dengan urutan axis sama. Pada pipeline saat ini `B=1`. Bentuk grid XYZ yang diuji: g12 = `[12,12,12]`, g16 = `[16,16,16]`, g24 = `[24,24,12]`; metadata tiap realization tetap menjadi sumber bentuk final. Dimensi XYZ minimal 8 dan kelipatan 4. Valid mask/observed CPT pada domain fisik `[X,Y,Z]`.
2. DQN V3 shared candidate Q scorer: input `features` float32 `[N,17]` dan `global_context` float32 `[11]`, keluaran skor float32 `[N]`. MLP `28→64→64→1` dengan ReLU setelah dua layer pertama. `N` berubah sesuai kandidat legal; argmax memilih indeks pertama pada urutan kandidat leksikografis `(x,y)`.

## Operasi non-neural yang perlu direplikasi

- Validasi setidaknya satu lokasi XY CPT unik. Setiap profil teramati harus berisi semua kedalaman valid pada XY tersebut; tidak boleh ada titik observasi di luar valid mask. Input observasi fisik Ic float32 `[X,Y,Z]`, observed/valid mask boolean bentuk sama. Tidak ada ground truth tersembunyi pada API.
- Koordinat pusat sel: `east=(x+0.5)*dx_m`, `north=(y+0.5)*dy_m`; putar dengan `azimuth_deg`, lalu tambahkan origin XY. Mapping balik memakai invers rotasi dan toleransi `1e-7` sel. Uniform memaksimumkan jarak Euclidean fisik ke CPT terdekat, tie pada kandidat `(x,y)` pertama.
- Normalisasi Ic train-only: `mean=2.314278231778639`, `std=0.27927494076233944`; `(Ic-mean)/std`. Nilai sparse di luar observasi tetap nol, mask terpisah. IDW normalized dihitung dari profil teramati, grid fisik, dan normalisasi yang sama. Keluaran generator di-denormalisasi dan nilai observed dikembalikan tepat.
- Kandidat legal: `valid.any(z) & ~observed.any(z) & access_mask` bila access mask diberikan. Urutan `np.argwhere` leksikografis X kemudian Y. `no_candidate` bila kosong.
- Fitur kandidat 17 dalam urutan: `x_local_fraction`, `y_local_fraction`, `nearest_cpt_distance_fraction`, `local_observed_density_4m`, `valid_depth_fraction`, `reconstructed_ic_mean_z`, `reconstructed_ic_std_z`, `depth_bin_0_ic_mean_z`, `depth_bin_1_ic_mean_z`, `depth_bin_2_ic_mean_z`, `depth_bin_3_ic_mean_z`, `depth_bin_0_empty`, `depth_bin_1_empty`, `depth_bin_2_empty`, `depth_bin_3_empty`, `local_4m_reconstructed_ic_mean_z`, `local_4m_valid_fraction`.
- Fitur global 11 dalam urutan: `log_observed_cpt_fraction`, `extent_x_over_32m`, `extent_y_over_32m`, `extent_z_over_16m`, `dx_over_2m`, `dy_over_2m`, `dz_over_2m`, `global_reconstructed_ic_mean_z`, `global_reconstructed_ic_std_z`, `valid_voxel_fraction`, `observed_valid_voxel_fraction`. Implementasi dan rincian skala berada di `results/dqn_v3_implementation/dqn_v3_feature_schema.json` SHA `440d9e750d609c32daecd013af9b86cc4c92845f387a4854ec0637a81bbccb07`.
- Routing berdasar jumlah XY CPT unik: 1,2,3,5,8 → DQN; 12,20 → Uniform; lainnya → `unsupported_cpt_count`. Keluaran `status`, `policy_used`, indeks XY, koordinat fisik XY, `candidate_count`, dan `q_score` hanya untuk DQN.

## Determinisme dan perangkat

Implementasi referensi menggunakan CPU float32 dan PyTorch `eval()`/`inference_mode()`. Tie argmax DQN dan Uniform memakai kandidat `(x,y)` pertama; Random hanya baseline evaluasi, bukan jalur hybrid deployment. Port Swift harus menjaga axis transpose, urutan kandidat, fitur, IDW, normalisasi, valid/access mask, dan observed-value restoration. Uji kesetaraan numerik pada g12/g16/g24 dan toleransi yang ditetapkan sebelum porting. Kapasitas memori, dukungan operator 3D, dynamic N, versi iOS, Neural Engine/CPU/GPU, latensi, dan ukuran paket **belum diverifikasi**; tidak ada klaim kompatibilitas Core ML saat ini.
