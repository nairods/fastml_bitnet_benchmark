This directory contains the compact inputs needed for public reproduction:

- `benchmark_records.json` contains the curated per-seed metrics, selected HLS
  C-synthesis estimates, and the seven retained training histories. The public
  CSV tables and PNG plots are generated from this file.
- `splits/` contains fixed train, validation, and test indices for each of the
  three classification tasks.

The OpenML dataset and processed caches are intentionally not committed. They
are downloaded or regenerated on demand by the training workflow.
