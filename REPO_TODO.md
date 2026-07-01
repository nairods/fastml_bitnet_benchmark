# Repository TODO Before Public Release

## Critical

- Decide whether this repository is an artifact package or a full raw benchmark release. At present it is an artifact package.
- If claiming full end-to-end reproducibility, add a documented way to obtain or regenerate:
  - trained model checkpoints
  - ONNX exports
  - HLS projects
  - Conifer projects
  - raw C-synthesis reports
  - raw prediction scores for fixed-FPR signal efficiency
- Add a small smoke-test workflow that trains/evaluates one lightweight model on a small OpenML subset.
- Pin exact dependency versions for the full training and synthesis environments, especially TensorFlow, QKeras, HGQ, hls4ml, conifer, torch, xgboost, and scikit-learn.

## Important

- Improve plot label placement further for dense high-AUC clusters.
- Add `CITATION.cff` once author and abstract metadata are final.
- Add explicit file-size policy for generated artifacts that are intentionally excluded.
- Add checksums for key result CSVs and plots.
- Add a `docs/` page explaining the BitNet custom synthesis variants and which variant is used in the final table.
- Remove or clearly mark exploratory scripts that are not part of the public benchmark path.
- Add a CI job that runs:
  - `python -m py_compile ...`
  - `python scripts/reproduce_paper.py`
  - a CSV schema check for the abstract tables.

## Optional

- Add notebooks for table inspection and figure reproduction.
- Add SVG or PDF versions of the main figures.
- Add a compact HTML report for browsing results.
- Add a minimal Dockerfile for the lightweight plotting/reporting path.
- Add a separate FPGA-toolchain setup note for institutional clusters.
