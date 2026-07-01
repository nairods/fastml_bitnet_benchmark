# BitNet FastML Benchmark

This repository contains the public artifact package for the FastML for Science 2026 extended abstract:

**Do BitNet Gains Survive Synthesis? An Implementation-Aware Benchmark of Low-Precision FPGA Inference for Jet Classification**

Last reproducibility audit: 2026-07-01.

The benchmark tests whether low-bit arithmetic advantages survive realistic FPGA-oriented implementation. The main result is that operation-count reduction alone is not enough to predict synthesized latency or resource use. Scaling-factor realization, sparsity lowering, compiler scheduling, and implementation path can dominate the final tradeoff.

## Dataset

Primary dataset:

- OpenML dataset ID: `42468`
- Name: `hls4ml_lhc_jets_hlf`
- Inputs: 16 high-level jet observables
- Classes: gluon, light quark, W, Z, top
- Size: about 830k jets

The code downloads the dataset through `sklearn.datasets.fetch_openml` when raw caches are absent.

## Tasks

Primary paper task:

- `q/g vs W/Z/top`
- background: gluon + light quark
- signal: W + Z + top

Secondary robustness task:

- `q/g vs top`
- background: gluon + light quark
- signal: top
- W and Z jets are dropped

The class mapping is implemented in [benchmark.py](/home/dsloot/agentic_optimisation/opendata_benchmark/fastml_bitnet_benchmark/benchmark.py:110).

## Splits And Preprocessing

All models use a fixed stratified split:

- 64% train
- 16% validation
- 20% test
- split seed 42

Canonical split indices are stored in:

- `splits/train_idx.npy`
- `splits/val_idx.npy`
- `splits/test_idx.npy`

Task-specific split archives are stored under `data/splits/`.

The committed scaler is `artifacts/scaler.pkl`. It was fit on the training split only. The raw OpenML cache arrays are not committed, so full retraining from scratch will download OpenML 42468 and rebuild local caches.

## Models

Core model families:

- Dense MLP
- QKeras 7-bit fixed-point
- HGQ
- QKeras binary
- QKeras ternary
- BitNet binary
- BitNet-1.58 sparse ternary
- XGBoost BDT, 100 trees, max depth 4

Main neural architecture:

- `16 -> 64 -> 32 -> 32 -> 1`

Secondary robustness architecture:

- `16 -> 128 -> 32 -> 1`

The main abstract table uses the primary architecture plus the unrolled BDT. The `128-32` architecture is included in robustness tables and Pareto plots.

## Hardware Target

Hardware numbers are HLS C-synthesis estimates only:

- AMD/Xilinx VU13P
- part `xcvu13p-flga2577-2-e`
- 5 ns clock period
- initiation interval target `II = 1`

No place-and-route numbers are used in the paper tables.

## Installation

Base environment:

```bash
conda env create -f environment.yml
conda activate fastml-bitnet-benchmark
```

Optional packages for full retraining and synthesis:

```bash
python -m pip install -r requirements-qkeras.txt
python -m pip install -r requirements-hgq.txt
```

The local audit was run with Python 3.13 for table and plot regeneration only. The declared environment uses Python 3.10 because TensorFlow/QKeras/HGQ/hls4ml compatibility is stricter than the lightweight plotting path.

For hardware workflows, Vivado/Vitis HLS, hls4ml, and conifer must be installed separately. Public helper scripts accept these environment variables when non-default Python environments are needed:

```bash
export FASTML_HLSENV_PYTHON=/path/to/hlsenv/bin/python
export FASTML_KERAS_PYTHON=/path/to/keras-env/bin/python
```

## How To Reproduce The Extended Abstract

Artifact-level reproduction from the committed result CSVs:

```bash
python scripts/reproduce_paper.py
```

This regenerates:

- `results/abstract_seed_statistics.csv`
- `results/abstract_lowbit_comparison.csv`
- `results/abstract_pareto_candidates.csv`
- `plots/abstract_pareto_auc_vs_lut_qg_vs_wzt.png`
- `plots/abstract_pareto_auc_vs_latency_qg_vs_wzt.png`
- `plots/AUC_vs_LUT.png`
- `plots/AUC_vs_latency.png`
- `plots/abstract_lowbit_comparison.png`

Important limitation: fixed-FPR signal efficiency is read from the committed abstract tables. Raw per-event prediction scores are not included, so `signal_eff_at_1pct_fpr` cannot be recomputed from first principles in this artifact package.

The lower-level plot/table script is also safe in the public artifact package:

```bash
python scripts/prepare_abstract_artifacts.py
```

When raw per-run JSON files are absent, it delegates to the public artifact regenerator instead of creating empty tables.

## Full Rerun Workflow

Full retraining and HLS synthesis require the optional ML and FPGA toolchain environments. The primary and secondary workflows are:

```bash
python scripts/run_binary_benchmark_workflow.py --class-mode binary_qg_vs_wzt --namespace binary --log-subdir binary_benchmark --seeds 42 43 44
python scripts/run_binary_benchmark_workflow.py --class-mode binary_top_vs_qg --namespace binary_topqg --log-subdir binary_topqg_benchmark --seeds 42 43 44
python scripts/prepare_abstract_artifacts.py
```

These commands are not expected to run in a minimal clone unless the OpenML download path, model dependencies, hls4ml, conifer, and Vivado/Vitis HLS are available.

## Main Result Files

Primary paper table:

- `results/abstract_main_binary_table.csv`

Secondary robustness table:

- `results/abstract_secondary_top_table.csv`

Seed statistics:

- `results/abstract_seed_statistics.csv`

Pareto candidates:

- `results/abstract_pareto_candidates.csv`

Main figures:

- `plots/AUC_vs_LUT.png`
- `plots/AUC_vs_latency.png`
- `plots/abstract_lowbit_comparison.png`

## Expected Main Results

Primary `q/g vs W/Z/top`, `64-32-32` architecture unless noted:

| Model | Accuracy | AUC | Signal eff. @ 1% FPR | Latency | LUT | DSP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense MLP | 0.85954 +/- 0.00026 | 0.93247 +/- 0.00012 | 0.58573 +/- 0.00213 | 12.00 | 183.5k | 3455 |
| QKeras fixed b7 | 0.85951 +/- 0.00029 | 0.93239 +/- 0.00010 | 0.58132 +/- 0.00302 | 9.00 | 135.4k | 665 |
| HGQ | 0.85380 +/- 0.00118 | 0.92761 +/- 0.00024 | 0.57113 +/- 0.00123 | 6.33 | 8.0k | 0 |
| QKeras binary | 0.82944 +/- 0.00120 | 0.88860 +/- 0.00633 | 0.43192 +/- 0.14802 | 20.00 | 62.0k | 0 |
| QKeras ternary | 0.83545 +/- 0.00230 | 0.90246 +/- 0.00124 | 0.41383 +/- 0.01268 | 11.67 | 28.3k | 0 |
| BitNet binary | 0.84538 +/- 0.00032 | 0.91758 +/- 0.00063 | 0.49753 +/- 0.01025 | 12.00 | 88.3k | 0 |
| Bit158 sparse ternary | 0.84894 +/- 0.00194 | 0.92324 +/- 0.00125 | 0.53454 +/- 0.00762 | 9.00 | 91.9k | 0 |
| BDT unrolled | 0.84719 +/- 0.00025 | 0.92075 +/- 0.00003 | 0.56123 +/- 0.00061 | 4.00 | 74.0k | 0 |

## Interpretation Supported By The Results

- BitNet and Bit158 recover predictive performance relative to naive QKeras binary and ternary quantization.
- BitNet variants do not universally dominate the FPGA implementation tradeoff.
- HGQ is the strongest neural resource-efficiency point by LUT use.
- The tested unrolled BDT is the lowest-latency implementation.
- Scaling factors, sparse lowering, and synthesis implementation details materially affect hardware results.

## Limitations

- The repository is an artifact package, not a complete raw-workspace dump.
- Trained checkpoint files, ONNX exports, generated HLS projects, Conifer projects, and raw C-synthesis reports are not committed.
- HLS values are taken from committed summary CSVs unless the full toolchain is installed and the full workflow is rerun.
- Fixed-FPR signal efficiency cannot be recomputed without raw prediction scores.
- Some exploratory scripts remain from the development workspace and are not part of the main abstract benchmark.

See also:

- `REPRODUCIBILITY_REPORT.md`
- `RESULTS_CHECK.md`
- `EXTENDED_ABSTRACT_CONSISTENCY.md`
- `REPO_TODO.md`
