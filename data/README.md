This directory contains the compact inputs needed for public reproduction:

- `20-epochs/benchmark_records.json` and
  `200-epochs/benchmark_records.json` contain the profile-specific per-seed
  metrics, validation-selected epoch metadata, HLS C-synthesis estimates, and
  seven representative training histories. The public CSV tables and PNG plots
  are generated only from these files.
- `splits/` contains fixed train, validation, and test indices for each of the
  three classification tasks.

The OpenML dataset and processed caches are intentionally not committed. They
are downloaded or regenerated on demand by the training workflow.
