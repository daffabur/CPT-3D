# SchemaGAN V3 — Full 24K CSV export

**Status: `READY_FOR_DATASET_SHARING`.** Konversi membuat tepat **24.000 CSV 3D** dari dataset frozen V3 tanpa menjalankan generator, training, atau mengubah data sumber. Validasi ulang seluruh volume menemukan **101.880.000 baris voxel**, seluruh split sesuai manifest, dan round-trip **exact** untuk setiap array 3D, mask, koordinat, serta boundary geologi pada sidecar. Seluruh checksum inti diverifikasi.

## 1. Sumber dan identitas

- Sumber: `/Users/daffaburanenugraha/Developer/cpt_super_learn/data/synthetic3d_v2/schemagan_v3_24k`; frozen plan SHA-256 `ff2884e0370e9597cb3afa3f8c0b613f19670835fc96ff2dc26cdebf0990d663`; source manifest SHA-256 `2d7b378199c5c21ddf449fde99b87838b12188af62c4116779787293bb21d563`. Manifest tidak berubah sebelum/sesudah ekspor, dan hash source per file dibandingkan dengan manifest saat ekspor serta validasi penuh.
- Source terdiri atas `.npy` voxel XYZ, `metadata.json`, `profiles.json`, `observed_mask.npy`, dan `valid_mask.npy` per volume. Bentuk grid: `(12,12,12)` **8.001**, `(16,16,16)` **8.001**, `(24,24,12)` **7.998** volume. Origin, spacing, azimuth, dan elevasi permukaan dibaca dari `metadata.grid`; koordinat CSV memakai pusat voxel fisik.
- Setiap volume mempunyai **satu** observed mask/CPT profile tersimpan; jumlah CPT stored pada dataset asli adalah 1,2,3,5,8,12,20. Konfigurasi evaluasi tambahan memakai `evaluation_site_seed` dan `nested_sites` dari observation policy existing, tanpa menggandakan ground truth sebagai volume independen.

## 2. Skema CSV dan sidecar

Satu file `<split>/<realization_id>.csv` per volume, satu baris per voxel, urutan **x outer, y middle, z inner**. Kolom utama:

```text
x,y,z,x_m,y_m,z_m,Ic,qc,fs,unit_id,sbt_id,valid_mask,observed_mask
```

Kolom tambahan yang mempertahankan seluruh array 3D sumber: `Fr_percent`, `Qt`, `qt_mpa`, `gamma_kn_m3`, `material_group`, `sigma_v0_eff_mpa`, `sigma_v0_mpa`, `u0_mpa`. `qc` dan `fs` dalam MPa sesuai `qc_mpa`/`fs_mpa` sumber. `Ic` adalah **ground truth penuh pada skala asli**, bukan input sparse. Model hanya boleh menerima Ic pada `observed_mask=1`; voxel tersembunyi harus dikeluarkan dari input dan dievaluasi pada `valid_mask & ~observed_mask`.

Tipe sumber: float32 untuk field kontinu; `unit_id` int16; `sbt_id` dan `material_group` uint8; mask bool. Pembaca CSV umum menghasilkan float64; cast kembali ke tipe sumber untuk merekonstruksi array. Float ditulis dengan representasi desimal dari nilai float32 asli tanpa pembulatan dua/tiga digit. `x_m,y_m` mengikuti `Grid.xy_m` (meter, termasuk origin/azimuth); `z_m=(z+0,5)×dz_m` adalah kedalaman positif dari permukaan.

Sidecar `sidecars/<split>/<realization_id>.json` memuat entri manifest asli, metadata penuh, profil CPT penuh, boundary lapisan `boundaries_m` beserta dtype/shape, checksum CSV, dan sumber file hash. Data 2D boundary tidak dipaksakan ke tabel voxel. `dataset_metadata.json` menyimpan mapping kolom, dtype, normalisasi Ic train-only, plan hash, dan observation policy. `README.md` memuat contoh membaca CSV menjadi tensor XYZ/CZYX dan membentuk input sparse CPT.

## 3. Hasil ekspor dan validasi

| Split | CSV/volume |
|---|---:|
| Train | 16.000 |
| Validation ID | 4.000 |
| Test ID | 1.000 |
| Test family | 1.000 |
| Test geometry parameter | 1.000 |
| Test response parameter | 1.000 |
| **Total** | **24.000** |

- Jumlah baris voxel: **101,880,000**. ID unik **24.000**; tidak ada filename collision, file hilang, atau duplikasi.
- **Seluruh 24.000 CSV** dibaca kembali. Untuk tiap volume, indeks/urutan XYZ, koordinat fisik, Ic, qc/fs, semua field 3D tambahan, unit/SBT ID, valid/observed mask, serta jumlah baris dibandingkan dengan sumber. Semua array 3D lulus `np.array_equal` setelah cast ke dtype sumber; boundary sidecar juga exact setelah cast ke float32. Metadata dan profil CPT sidecar sama nilainya dengan sumber.
- **48,003 entri checksum** terverifikasi (`24.000 CSV + 24.000 sidecar + manifest.csv + dataset_metadata.json + README.md`). `checksums.sha256` tidak memuat hash dirinya sendiri; `export_summary.json` dan laporan adalah artefak audit di luar cakupan checksum dataset inti.
- Source manifest SHA-256 tidak berubah; seluruh source files dicocokkan terhadap hash di source manifest selama dua lintasan. Pembagian split akhir sama persis dengan sumber.

## 4. Uji terbatas, kapasitas, waktu, dan resource

Uji **30 volume** representatif (11 g12, 10 g16, 9 g24) menghasilkan **122,176 baris**, **25,010,027 byte CSV** dan **885,423 byte sidecar**. Round-trip dan checksum lulus dalam **2.86 detik**. Estimasi berbobot grid sebelum ekspor **20.09 GiB**, ruang bebas saat itu **533.00 GiB**; bahkan setelah cadangan setara dua salinan output dan buffer 10 GiB, kapasitas cukup.

Ukuran aktual: CSV **19.43 GiB** (20,858,544,532 byte), sidecar **0.65 GiB** (699,206,594 byte); total termasuk file indeks/dokumentasi sekitar **20.09 GiB**. Waktu ekspor+validasi **37.85 menit**: penulisan/round-trip awal **24.44 menit**, validasi menyeluruh kedua **12.90 menit**. Peak RAM tidak dicatat oleh environment; konverter memproses satu volume pada satu waktu.

## 5. Penggunaan Deep Learning dan kebocoran data

Lihat [`README.md`](/Users/daffaburanenugraha/Developer/cpt_super_learn/data/synthetic3d_v2/schemagan_v3_24k_csv/README.md) untuk contoh kode lengkap. Ringkasnya: baca CSV, rekonstruksi array `Ic[x,y,z]`, buat `sparse = where(observed_mask, (Ic−mean_train)/std_train, 0)`, dan berikan `sparse` bersama observed mask sebagai input; ground truth ternormalisasi dipakai **hanya sebagai target**. Mean/std berasal dari 16.000 volume train saja: **2,314278231778639 / 0,27927494076233944**. Jangan fit ulang pada validation/test. Ukur hidden-valid RMSE/MAE pada skala Ic asli dan pisahkan split serta lineage. Dataset sintetis ini **belum divalidasi terhadap CPT lapangan nyata**.

## 6. Implementasi, tes, dan keterbatasan

File baru: `schemagan3d/export_v3_csv.py` dan `schemagan3d/test_export_v3_csv.py`. Konverter memakai file `.tmp` dan atomic replace; bila terinterupsi, `--resume` memverifikasi hash CSV/sidecar sebelum skip. CSV yang sudah valid tidak ditimpa. Checksum dan `manifest.csv` dibangun setelah semua volume selesai. **65 automated tests lulus** pada suite repository, termasuk tiga tes ekspor baru (round-trip, resume tanpa overwrite, dan deteksi kerusakan axis). Tidak ada dependency baru.

Batasan: CSV berukuran jauh lebih besar daripada `.npy`; pembacaan CSV untuk training masif bisa lambat. `Ic` penuh di CSV perlu diperlakukan sebagai target agar tidak bocor ke input. Mask tambahan 1/2/3/5/8/12/20 tidak diekspor sebagai volume baru; reproduksinya didokumentasikan dari seed/policy. `export_summary.json` dan laporan dibuat setelah checksum dataset inti. Tidak ada klaim bahwa dataset sintetis mewakili geologi lapangan.

## 7. Lokasi output dan perintah

Dataset CSV: `/Users/daffaburanenugraha/Developer/cpt_super_learn/data/synthetic3d_v2/schemagan_v3_24k_csv`. Struktur mencakup enam folder split berisi **24.000 CSV**, `sidecars/` berisi **24.000 JSON**, `manifest.csv`, `dataset_metadata.json`, `README.md`, `checksums.sha256`, dan `export_summary.json`. Laporan ini juga disimpan di direktori output. Tidak ada Git push.

Perintah utama yang benar-benar dijalankan dari root repository:

```bash
PYTHONPYCACHEPREFIX=/Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/pycache /usr/bin/time -p python3 -m schemagan3d.export_v3_csv --mode pilot --pilot-count 30 --output /Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/v3_csv_pilot
MPLCONFIGDIR=/Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/matplotlib python3 -m unittest discover -s schemagan3d -t . -p 'test_*.py'
PYTHONPYCACHEPREFIX=/Users/daffaburanenugraha/Documents/Codex/2026-10-19/files-pasted-by-the-user-audit/work/pycache /usr/bin/time -p python3 -m schemagan3d.export_v3_csv --mode export
```

Full validation was invoked automatically by `--mode export` after the final CSV was written. Command stdout/stderr is saved under the task's `work/` directory. The frozen source dataset and previous experiment artifacts remain unchanged.
