This repository keeps the small synthesis metadata and fixed split archives needed
for reproducibility.

Large raw OpenML caches and regenerated benchmark caches are intentionally not
committed. They are recreated by the benchmark scripts on first run.

The public train/validation/test index arrays are stored here as
`train_idx.npy`, `val_idx.npy`, and `test_idx.npy`. Task-specific split archives
are stored separately under `data/splits/`.
