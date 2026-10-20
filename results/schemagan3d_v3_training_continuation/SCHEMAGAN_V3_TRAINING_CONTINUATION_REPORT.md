# SchemaGAN V3 — Training Continuation & GAN Control

## Ringkasan eksekutif

**Keputusan: `READY_FOR_INDEPENDENT_EVALUATION`.** MSE-only dilanjutkan dari checkpoint `last.pt` epoch 20 dengan state generator, optimizer, RNG, dan early stopping utuh. GAN control dimulai dari seed 14 dengan loss L1+GAN dan kebijakan early stopping sama seperti L1-only. Seluruh perbandingan di bawah masih memakai validation ID; tidak ada test/holdout yang dievaluasi.

## Validasi resume dan konfigurasi

`resume_preflight.json` mencatat SHA-256 checkpoint `last.pt` asal, hash history/subset asal, serta metadata salinan yang diubah. Hanya `epochs` 20→30, `output_dir`, hash config terkait, dan penanda stop `max_epochs` pada **salinan** yang dibuka ulang. Seluruh tensor bobot, optimizer state dan RNG dalam salinan sama dengan asal; guard `load_training_checkpoint` dan `validate_resume_history` membuktikan mulai epoch 21. Original checkpoint/history tidak diubah. Tidak ada scheduler pada runner sebelumnya; LR G tetap 0,0002. GAN control baru berbeda dari L1-only hanya pada aktivasi objective GAN, update D, dan output path. Keduanya memakai 8.000 train, 1.000 validation, seed 14, batch 2, base channels 4, weight 100, mixed CPT dan normalisasi train-only yang sama.

| Run | Objective | Best epoch | Stop epoch | Best val RMSE 5 CPT | G steps at best | D steps at best | Stop reason | Best SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|---|
| mse_epoch20 | mse_only | 20 | 20 | 0.148686 | 80020 | none | max_epochs | `39844dc021b0ae1b81098195060efcda61ff2812455d0f1a0f674e4035fbf5ba` |
| mse_continuation | mse_only | 26 | 29 | 0.148269 | 104026 | none | patience_exhausted | `5694bb7b43efd7054a99fac8b74dadc5d74ebe6d291f42a8393b57fe9da52b64` |
| l1_only | l1_only | 14 | 20 | 0.149063 | 56014 | none | patience_exhausted | `eb548d9a2ccb79860d143d960ee4ebf8af2e1c5cae22fc1df815741d8da1e7cc` |
| gan_control | gan_l1 | 7 | 13 | 0.158258 | 28007 | 28007 | patience_exhausted | `8e661725f8e877d0ac84b013ba4956b08c4276c3ff11163e9567c24016d6a68e` |
| historical_gan | gan_l1 | 1 | 5 | 0.158402 | 4001 | 4001 | patience_exhausted | `5a7e08f09564f45c4224e46c04c974fc80c0b976e4e2ab6b3c5c42d5e66be69b` |

## History, early stopping, dan generalization gap

MSE continuation mempertahankan seluruh 20 baris metrik asli, kemudian menambah epoch 21 sampai stop/batas 30. Baris ke-20 pada salinan mengosongkan `stop_reason` agar resume berjalan, dan alasan historis `max_epochs` tercatat dalam provenance. Histori GAN control baru berada di CSV tersendiri. Reconstruction loss menggunakan normalized Ic; RMSE/MAE/gap menggunakan physical Ic. `gradient norm = NOT_LOGGED` karena pipeline existing tidak menyimpannya.

| Run | Epoch | Train recon | G loss | D loss | Train RMSE | Val RMSE | Train MAE | Val MAE | Gap | LR G/D | Bad | Detik |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| mse_continuation | 21 | 0.29345 | 29.345 | NOT_LOGGED | 0.148515 | 0.148956 | 0.113007 | 0.113410 | +0.000440 | 0.0002/none | 1 | 62.1 |
| mse_continuation | 22 | 0.29182 | 29.182 | NOT_LOGGED | 0.148420 | 0.148921 | 0.112625 | 0.113070 | +0.000501 | 0.0002/none | 2 | 63.3 |
| mse_continuation | 23 | 0.29225 | 29.225 | NOT_LOGGED | 0.147982 | 0.148353 | 0.111956 | 0.112325 | +0.000372 | 0.0002/none | 0 | 64.5 |
| mse_continuation | 24 | 0.29190 | 29.190 | NOT_LOGGED | 0.148119 | 0.148351 | 0.112465 | 0.112715 | +0.000232 | 0.0002/none | 1 | 63.6 |
| mse_continuation | 25 | 0.29065 | 29.065 | NOT_LOGGED | 0.148398 | 0.148670 | 0.112740 | 0.113019 | +0.000272 | 0.0002/none | 2 | 65.0 |
| mse_continuation | 26 | 0.28993 | 28.993 | NOT_LOGGED | 0.147898 | 0.148269 | 0.112173 | 0.112510 | +0.000370 | 0.0002/none | 3 | 65.8 |
| mse_continuation | 27 | 0.29000 | 29.000 | NOT_LOGGED | 0.148149 | 0.148391 | 0.112244 | 0.112521 | +0.000242 | 0.0002/none | 4 | 63.8 |
| mse_continuation | 28 | 0.28882 | 28.882 | NOT_LOGGED | 0.148125 | 0.148590 | 0.111927 | 0.112389 | +0.000465 | 0.0002/none | 5 | 63.3 |
| mse_continuation | 29 | 0.28858 | 28.858 | NOT_LOGGED | 0.147913 | 0.148491 | 0.111908 | 0.112432 | +0.000578 | 0.0002/none | 6 | 64.3 |
| gan_control | 1 | 0.46058 | 47.854 | 0.4063 | 0.158225 | 0.158402 | 0.119799 | 0.120048 | +0.000177 | 0.0002/0.0002 | 0 | 79.5 |
| gan_control | 2 | 0.43886 | 45.906 | 0.3805 | 0.158403 | 0.158548 | 0.119277 | 0.119599 | +0.000145 | 0.0002/0.0002 | 1 | 81.2 |
| gan_control | 3 | 0.43763 | 45.837 | 0.3758 | 0.160299 | 0.160383 | 0.120552 | 0.120742 | +0.000085 | 0.0002/0.0002 | 2 | 81.7 |
| gan_control | 4 | 0.43622 | 45.702 | 0.3760 | 0.158930 | 0.158965 | 0.118881 | 0.119058 | +0.000035 | 0.0002/0.0002 | 3 | 83.0 |
| gan_control | 5 | 0.43556 | 45.626 | 0.3765 | 0.159023 | 0.158930 | 0.119017 | 0.119019 | -0.000093 | 0.0002/0.0002 | 4 | 83.8 |
| gan_control | 6 | 0.43327 | 45.393 | 0.3732 | 0.162219 | 0.161922 | 0.121667 | 0.121498 | -0.000297 | 0.0002/0.0002 | 5 | 83.4 |
| gan_control | 7 | 0.43331 | 45.389 | 0.3732 | 0.158074 | 0.158258 | 0.118034 | 0.118198 | +0.000184 | 0.0002/0.0002 | 0 | 83.9 |
| gan_control | 8 | 0.43201 | 45.259 | 0.3745 | 0.158673 | 0.158933 | 0.118749 | 0.118981 | +0.000260 | 0.0002/0.0002 | 1 | 84.1 |
| gan_control | 9 | 0.43175 | 45.228 | 0.3711 | 0.158556 | 0.158793 | 0.118400 | 0.118698 | +0.000237 | 0.0002/0.0002 | 2 | 83.0 |
| gan_control | 10 | 0.43109 | 45.142 | 0.3811 | 0.158344 | 0.158567 | 0.118389 | 0.118578 | +0.000222 | 0.0002/0.0002 | 3 | 83.5 |
| gan_control | 11 | 0.42916 | 44.951 | 0.3756 | 0.159650 | 0.160218 | 0.119166 | 0.119700 | +0.000568 | 0.0002/0.0002 | 4 | 82.9 |
| gan_control | 12 | 0.42867 | 44.887 | 0.3794 | 0.158947 | 0.159379 | 0.118408 | 0.118909 | +0.000432 | 0.0002/0.0002 | 5 | 83.0 |
| gan_control | 13 | 0.42775 | 44.808 | 0.3732 | 0.159890 | 0.160272 | 0.119366 | 0.119648 | +0.000382 | 0.0002/0.0002 | 6 | 83.3 |

Mixed CPT histogram, best numerical RMSE, dan stopping reference untuk setiap epoch berada di CSV. MSE continuation dan L1-only/GAN control memakai mean per-volume hidden-valid RMSE 5 CPT pada subset train/validation tetap. Plot menggabungkan epoch 1–30 MSE dan arm L1/GAN.

## Evaluasi berpasangan: 400 validation ID

Enam metode dihitung ulang pada mask dan koordinat CPT identik. Empat baseline lama diverifikasi ulang per volume dengan toleransi 5e-6. CI 95% adalah bootstrap 2.000 resample **volume**, bukan voxel; selisih negatif menguntungkan metode pertama. Semua angka bersifat deskriptif validation ID.

| CPT | IDW | Pilot 3 | L1-only | MSE ep20 | MSE lanjut | GAN control | Δ lanjut−ep20 [CI95] | Δ GAN−L1 [CI95] |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.21843 | 0.18654 | 0.18424 | 0.18252 | 0.17925 | 0.20288 | -0.00326 [-0.00397,-0.00257] | +0.01864 [+0.01715,+0.02008] |
| 2 | 0.18959 | 0.17689 | 0.16976 | 0.16875 | 0.16734 | 0.18495 | -0.00141 [-0.00196,-0.00087] | +0.01520 [+0.01401,+0.01652] |
| 3 | 0.17372 | 0.16903 | 0.16096 | 0.16003 | 0.15935 | 0.17330 | -0.00068 [-0.00110,-0.00028] | +0.01234 [+0.01138,+0.01330] |
| 5 | 0.15686 | 0.15922 | 0.15003 | 0.14951 | 0.14912 | 0.15912 | -0.00040 [-0.00068,-0.00010] | +0.00909 [+0.00837,+0.00984] |
| 8 | 0.14148 | 0.14726 | 0.13684 | 0.13714 | 0.13668 | 0.14371 | -0.00046 [-0.00069,-0.00022] | +0.00687 [+0.00623,+0.00751] |
| 12 | 0.12791 | 0.13617 | 0.12380 | 0.12506 | 0.12440 | 0.12866 | -0.00066 [-0.00086,-0.00046] | +0.00486 [+0.00428,+0.00545] |
| 20 | 0.11473 | 0.12385 | 0.10848 | 0.11108 | 0.11054 | 0.11233 | -0.00054 [-0.00071,-0.00038] | +0.00385 [+0.00334,+0.00434] |

Mean MAE, win rates, SD pasangan, semua CI dan sembilan sel family×grid per CPT tersedia di `paired_evaluation.json`; 2.800 baris metrik individu tersedia di `paired_per_volume.csv`.

## Analisis hasil dan MAE

Continuation MSE memperbaiki minimum validation 5 CPT dari **0,148686 pada epoch 20** menjadi **0,148269 pada epoch 26** (selisih −0,000417). Epoch 29 berhenti karena patience, bukan karena mencapai batas 30. Pada 400 volume paired, continuation juga memperbaiki mean RMSE terhadap MSE epoch 20 pada **7/7 CPT count**; CI bootstrap selisih berada di bawah nol untuk semuanya. Family×grid: 9/9 sel membaik pada 1, 2, 3, 12, dan 20 CPT; 8/9 pada 5 dan 8 CPT. Perbaikan paling besar pada 1 CPT (−0,003262), sedangkan 5–20 CPT berkisar sekitar −0,0004 sampai −0,0007. Jadi manfaat tambahan epoch nyata dalam validasi ini tetapi kecil pada banyak jumlah CPT.

GAN control mempunyai minimum epoch 7 **0,158258**, sedikit lebih baik daripada GAN historis epoch 1 **0,158402** (−0,000144), tetapi masih lebih buruk daripada L1-only **0,149063** pada 5 CPT. Pada paired 400, GAN control lebih buruk daripada L1-only pada **7/7 CPT count** dengan CI selisih seluruhnya di atas nol, dan pada **9/9 sel family×grid** untuk setiap CPT. Ini bukti kuat pada seed/subset/validation ini bahwa mengaktifkan objective GAN dan update D pada protokol yang disamakan menurunkan akurasi dibanding L1-only. Ini tidak membuktikan setiap GAN atau semua seed merugikan. D loss training berada 0,371–0,406; train dan validation RMSE GAN bergerak naik bersama, sehingga **tidak ada bukti spesifik bahwa discriminator divergen atau overfitting klasik**. Gradient norm dan logits per batch tidak tersedia (`NOT_LOGGED`).

Gap MSE continuation meningkat dari +0,000245 (epoch 1) menjadi +0,000578 (epoch 29), tetapi tetap kecil dibanding RMSE sekitar 0,148. Train RMSE turun dari 0,152773 menjadi 0,147913 dan validation dari 0,153018 menjadi 0,148491; tidak ada pemisahan besar. GAN control gap akhir +0,000382, juga kecil. Tidak ada NaN/Inf yang tercatat.

Mean hidden-valid physical Ic MAE pada 400 volume:

| CPT | IDW | Pilot 3 | L1-only | MSE ep20 | MSE lanjut | GAN control |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.17158 | 0.14841 | 0.14621 | 0.14558 | 0.14275 | 0.15995 |
| 2 | 0.14580 | 0.13907 | 0.13263 | 0.13276 | 0.13138 | 0.14322 |
| 3 | 0.13158 | 0.13147 | 0.12385 | 0.12421 | 0.12341 | 0.13237 |
| 5 | 0.11668 | 0.12185 | 0.11300 | 0.11368 | 0.11318 | 0.11893 |
| 8 | 0.10384 | 0.11083 | 0.10054 | 0.10228 | 0.10177 | 0.10526 |
| 12 | 0.09356 | 0.10162 | 0.08947 | 0.09208 | 0.09145 | 0.09305 |
| 20 | 0.08371 | 0.09198 | 0.07698 | 0.08067 | 0.08023 | 0.08004 |

MSE continuation mempunyai mean RMSE terendah pada 1–8 CPT; L1-only tetap lebih baik pada 12 dan 20 CPT. Pada 5 CPT, MSE continuation MAE sedikit lebih tinggi daripada L1-only walaupun RMSE lebih rendah. Perbedaan kecil harus dinilai bersama CI dan tujuan metrik, bukan dibulatkan menjadi keunggulan universal.

## Interpretasi dan limitasi

GAN control versus L1-only mengisolasi aktivasi adversarial objective bersama update discriminator **dengan aturan training sama**; satu seed dan validation ID membatasi generalisasi inferensi. Perbandingan GAN control dengan GAN historis terutama mengubah aturan/durasi stop; kenaikan kecil tidak membuktikan adversarial loss membantu. CI bootstrap bersifat deskriptif pada validation ID yang juga dipakai untuk checkpoint selection. Tidak ada klaim generalisasi ke test independen atau keluarga geologi baru dari dataset sintetis ini.

## Rekomendasi berikutnya

Bekukan **MSE continuation epoch 26** sebagai kandidat utama untuk independent evaluation, dan **L1-only epoch 14** sebagai pembanding reconstruction-only yang lebih kuat pada 12/20 CPT. Pertahankan IDW dan Pilot 3 sebagai baseline, serta GAN control untuk uji objective. Tetapkan protokol dan keputusan sebelum membuka test/holdout; jangan memakai hasil test untuk tuning. Keputusan `READY_FOR_INDEPENDENT_EVALUATION` berarti artefak dan validation evidence cukup untuk tahap uji independen, **bukan klaim performa independen**. Jangan otomatis menjalankan full 16K training, perubahan arsitektur, atau LR sweep.

## Perintah dan berkas

Perintah utama yang dijalankan dari root repository (dengan `PYTHONPYCACHEPREFIX` dan `MPLCONFIGDIR` diarahkan ke task `work/`):

```text
python3 /Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/prepare_continuation.py
python3 -m schemagan3d.train_v3 --config results/schemagan3d_v3_training_continuation/gan_control_config.json --mode monitoring-dry-run --report results/schemagan3d_v3_training_continuation/gan_control_preflight.json
python3 -m unittest schemagan3d.test_v3_objective_ablation schemagan3d.test_v3_idw_training_integration schemagan3d.test_v3_training_monitoring -q
python3 -m schemagan3d.train_v3 --config results/schemagan3d_v3_training_continuation/mse_continuation_config.json --mode train --resume
python3 -m schemagan3d.train_v3 --config results/schemagan3d_v3_training_continuation/gan_control_config.json --mode train
python3 /Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/evaluate_continuation.py
python3 /Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/report_continuation.py
```

Satu pemanggilan awal `prepare_continuation.py` dari task cwd gagal karena path dataset dalam config relatif terhadap root repository; pemanggilan ulang dari root repository berhasil sebelum training dimulai. Tidak ada source, dataset, atau checkpoint lama yang diubah. Tidak ada operasi Git maupun instalasi dependency.
