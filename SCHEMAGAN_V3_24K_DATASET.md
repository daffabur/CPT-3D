# SchemaGAN V3 24K synthetic geotechnical dataset

The CSV dataset is distributed as GitHub Release assets under the proposed tag `schemagan-v3-24k-csv-v1`. The release contains 24 lossless `.tar.gz` archives, each holding 1,000 volume CSVs and their matching JSON sidecars. Download **all** archives and the small metadata files. The repository does not track the 24,000 CSVs directly.

Dataset split: train 16,000; validation ID 4,000; test ID 1,000; test family holdout 1,000; test geometry parameter holdout 1,000; test response parameter holdout 1,000. Total: 24,000 volumes and 101,880,000 voxel rows. These are synthetic volumes, **not calibrated field CPT data**.

The release's `RELEASE_INSTRUCTIONS.md` explains download, SHA-256 verification, extraction, CSV-to-tensor conversion, sparse CPT input, split use, and leakage prevention. `README.md` contains the full loading example. Each CSV includes `x,y,z`, physical coordinates, full ground-truth `Ic`, `qc`, `fs`, `unit_id`, `sbt_id`, and valid/observed masks. Matching `sidecars/<split>/<realization_id>.json` files contain grid geometry, geological boundaries, CPT observations, and lineage. Never pass the full `Ic` column to a model as sparse input: use only voxels with `observed_mask=1`. Apply the train-only Ic normalization (`mean=2.314278231778639`, `std=0.27927494076233944`) without refitting on validation or test data.

Release package: `PACKAGE_MANIFEST.json` lists all 24 archives and their SHA-256 hashes; `ARCHIVE_SHA256SUMS.txt` verifies downloaded assets; `checksums.sha256` verifies every restored CSV and sidecar. Preserve the split labels in `manifest.csv` and use validation ID for model selection; all test/holdout splits are for final evaluation.
