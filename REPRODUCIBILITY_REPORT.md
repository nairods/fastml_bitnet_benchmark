# Reproducibility Report

Audit date: 2026-07-01

Repository audited: `fastml_bitnet_benchmark`

## Summary

This repository is a usable public artifact package for the benchmark tables and plots, but it is not a fully self-contained raw benchmark workspace. The committed CSVs and PNGs support the paper claims, and the lightweight reproduction command regenerates the main public plots from committed result tables. Full retraining and HLS synthesis still require external dependencies and uncommitted generated artifacts.

Readiness score: 7/10.

## Commands Rerun

Passed:

```bash
python -m py_compile scripts/reproduce_public_artifacts.py scripts/reproduce_paper.py scripts/generate_benchmark_artifacts.py
python scripts/reproduce_paper.py
python scripts/generate_benchmark_artifacts.py
```

Re-verified on 2026-07-01 with:

```bash
python -m py_compile scripts/reproduce_public_artifacts.py scripts/reproduce_paper.py scripts/generate_benchmark_artifacts.py scripts/run_binary_benchmark_workflow.py hardware_benchmark/prepare.py
python scripts/reproduce_paper.py
python scripts/generate_benchmark_artifacts.py
```

These commands regenerated:

- `results/benchmark_main_binary_table.csv`
- `results/benchmark_secondary_top_table.csv`
- `results/benchmark_multiclass_summary.csv`
- `plots/benchmark_pareto_auc_vs_lut_qg_vs_wzt.png`
- `plots/benchmark_pareto_auc_vs_latency_qg_vs_wzt.png`
- `plots/benchmark_pareto_auc_vs_lut_qg_vs_top.png`
- `plots/benchmark_pareto_auc_vs_latency_qg_vs_top.png`
- `plots/benchmark_pareto_auc_vs_lut_multiclass.png`
- `plots/benchmark_pareto_auc_vs_latency_multiclass.png`

Current lightweight artifact path:

- Added `scripts/reproduce_public_artifacts.py`.
- Updated `scripts/reproduce_paper.py` to use the public-safe regeneration path.
- Updated `scripts/generate_benchmark_artifacts.py` to fall back to the public-safe path when raw per-run JSON files are absent.

## Environment Used For Audit

Lightweight table and plot audit environment:

- Python: 3.13.9
- numpy: 2.4.4
- pandas: 3.0.2
- matplotlib: 3.10.8

Unavailable in the audit shell:

- scikit-learn
- torch
- xgboost
- onnx
- onnxruntime
- openml
- tensorflow
- qkeras
- hls4ml
- conifer

The declared `environment.yml` uses Python 3.10 and includes the base scientific stack needed for benchmark execution. QKeras, HGQ, hls4ml, conifer, and Vivado/Vitis should be installed in compatible dedicated environments for full reruns.

## What Was Validated

Validated from committed files:

- Primary and secondary benchmark tables exist and are complete.
- The multiclass summary exists and is marked partial where seed or synthesis coverage is incomplete.
- Primary `q/g vs W/Z/top` results match expected benchmark numbers within rounding.
- Means and standard deviations in `results/benchmark_main_binary_table.csv` are reported over seeds 42, 43, 44.
- The class mapping in `benchmark.py` is correct:
  - `g/q -> 0` background
  - `w/z/t -> 1` signal
  - for top-vs-QCD, W and Z are dropped.
- Split archives exist under `data/splits/`.
- Each task fits its own `StandardScaler` on training data only; no committed scaler pickle is used.
- Public plot regeneration works from committed benchmark tables.
- HLS values are labelled as C-synthesis estimates, not place-and-route.

Validated but with limitation:

- `signal_eff_at_1pct_fpr` is present in benchmark tables and matches expected values, but cannot be recomputed from the shipped artifact package because raw prediction scores are not included.
- Hardware metrics are available in summary CSVs but raw HLS project/report directories are not included.

Not rerun:

- Full neural training.
- QKeras/HGQ training.
- XGBoost training.
- ONNX export.
- hls4ml conversion.
- Conifer conversion.
- Vivado/Vitis HLS C-synthesis.

Reason: those steps require optional environments, OpenML download/cache rebuild, and licensed FPGA tooling. The repository currently ships summaries rather than the raw generated model and synthesis artifacts.

## Artifact Completeness

Present:

- `configs/hardware_benchmark.json`
- `data/train_idx.npy`, `data/val_idx.npy`, `data/test_idx.npy`
- `data/splits/`
- `results/benchmark_main_binary_table.csv`, `results/benchmark_secondary_top_table.csv`, `results/benchmark_multiclass_summary.csv`
- `plots/*.png`
- benchmark and synthesis scripts
- README and environment files

Missing as committed folders:

- `models/`
- `onnx/`
- `hls/`
- `conifer/`
- `notebooks/`

No trained checkpoints, ONNX exports, generated HLS projects, Conifer projects, or raw C-synthesis reports are committed.

## Code Changes Made During Audit

- Added `scripts/reproduce_public_artifacts.py`.
- Updated `scripts/reproduce_paper.py`.
- Updated `scripts/generate_benchmark_artifacts.py`.
- Updated plotting style and public plot aliases.
- Added environment-variable overrides:
  - `FASTML_HLSENV_PYTHON`
  - `FASTML_KERAS_PYTHON`
- Updated `README.md`.

## Current Reproduction Contract

Fresh-clone artifact reproduction:

```bash
conda env create -f environment.yml
conda activate fastml-bitnet-benchmark
python scripts/reproduce_paper.py
```

Full benchmark reproduction:

```bash
python scripts/run_binary_benchmark_workflow.py --class-mode binary_qg_vs_wzt --namespace binary --log-subdir binary_benchmark --seeds 42 43 44
python scripts/run_binary_benchmark_workflow.py --class-mode binary_top_vs_qg --namespace binary_topqg --log-subdir binary_topqg_benchmark --seeds 42 43 44
python scripts/generate_benchmark_artifacts.py
```

The full workflow requires dependencies and FPGA tools not provided by this repository.
