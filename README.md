# BitNet FPGA Benchmark

This repository contains the public artifact and reproducibility package for the study:

**Do BitNet Gains Survive Synthesis? An Implementation-Aware Benchmark of Low-Precision FPGA Inference for Jet Classification**

The benchmark investigates whether the theoretical advantages of low-bit arithmetic translate into efficient FPGA implementations. The results show that reductions in numerical precision and operation count alone do not reliably predict synthesised latency or resource usage. Scaling-factor implementation, structural sparsity, compiler scheduling and the chosen implementation path can substantially influence the final accuracy–hardware trade-off.

## Benchmark Overview

### Dataset

The benchmark uses the public OpenML [`hls4ml_lhc_jets_hlf`](https://www.openml.org/d/42468) dataset (ID `42468`), containing approximately 830,000 jets described by 16 high-level observables and labelled as:

* gluon
* light quark
* W
* Z
* top

The dataset is downloaded automatically through `sklearn.datasets.fetch_openml` when no local cache is available.

### Classification Tasks

**Primary task — q/g vs W/Z/top**

* background: gluon + light quark
* signal: W + Z + top

**Robustness task — q/g vs top**

* background: gluon + light quark
* signal: top
* W and Z jets are excluded

The task definitions and class mappings are implemented in [`fastml_bitnet_benchmark/benchmark.py`](fastml_bitnet_benchmark/benchmark.py).

### Data Splits and Preprocessing

All models use the same stratified split:

* 64% training
* 16% validation
* 20% testing
* split seed: `42`

Canonical split indices are stored in `splits/`, with task-specific split information under `data/splits/`.

Input standardisation uses the committed scaler in `artifacts/scaler.pkl`, fitted on the training data only.

### Models

The benchmark compares:

* Dense MLP
* QKeras 7-bit fixed-point
* HGQ
* QKeras binary
* QKeras ternary
* BitNet binary
* BitNet-1.58 sparse ternary
* XGBoost BDT with fully unrolled FPGA implementation

The primary neural architecture is:

```text
16 → 64 → 32 → 32 → 1
```

A wider architecture is included as a robustness check:

```text
16 → 128 → 32 → 1
```

### Hardware Target

FPGA results are based on **HLS C-synthesis estimates** for:

* AMD/Xilinx VU13P
* part: `xcvu13p-flga2577-2-e`
* target clock period: **5 ns**
* target initiation interval: **II = 1**

The reported hardware results do not include place-and-route estimates.

## Installation

Create the base environment with:

```bash
conda env create -f environment.yml
conda activate fastml-bitnet-benchmark
```

Optional dependencies for model retraining and FPGA workflows are available through:

```bash
python -m pip install -r requirements-qkeras.txt
python -m pip install -r requirements-hgq.txt
```

Full hardware reproduction additionally requires **hls4ml, Conifer and Vivado/Vitis HLS**.

See [`REPRODUCIBILITY_REPORT.md`](REPRODUCIBILITY_REPORT.md) for detailed environment configuration, toolchain requirements and reproducibility notes.

## Reproducing the Results

The figures and summary tables used in the benchmark can be regenerated directly from the committed result files:

```bash
python scripts/reproduce_paper.py
```

This recreates the main benchmark tables and plots in `results/` and `plots/` without requiring the full training or FPGA toolchain.

> **Reproducibility scope:** fixed-FPR signal efficiency is taken from the committed benchmark tables because raw per-event prediction scores are not included in the public artifact.

For additional reproducibility details, see [`REPRODUCIBILITY_REPORT.md`](REPRODUCIBILITY_REPORT.md).

## Full Benchmark Rerun

Retraining and HLS synthesis require the optional ML and FPGA toolchains, including the model dependencies, hls4ml, Conifer and Vivado/Vitis HLS.

The primary and robustness benchmarks can be rerun with:

```bash
python scripts/run_binary_benchmark_workflow.py \
    --class-mode binary_qg_vs_wzt \
    --namespace binary \
    --log-subdir binary_benchmark \
    --seeds 42 43 44

python scripts/run_binary_benchmark_workflow.py \
    --class-mode binary_top_vs_qg \
    --namespace binary_topqg \
    --log-subdir binary_topqg_benchmark \
    --seeds 42 43 44

python scripts/prepare_abstract_artifacts.py
```

These commands are not expected to run in a minimal clone without the required datasets and FPGA synthesis environment.

## Key Results

Primary task: **q/g vs W/Z/top**, using the 64–32–32 architecture unless noted.

| Model          |          Accuracy |               AUC | Signal eff. @ 1% FPR | Latency [cycles] |      LUT |  DSP |
| -------------- | ----------------: | ----------------: | -------------------: | ---------------: | -------: | ---: |
| Dense MLP      | 0.85954 ± 0.00026 | 0.93247 ± 0.00012 |    0.58573 ± 0.00213 |            12.00 |   183.5k | 3455 |
| QKeras 7-bit   | 0.85951 ± 0.00029 | 0.93239 ± 0.00010 |    0.58132 ± 0.00302 |             9.00 |   135.4k |  665 |
| HGQ            | 0.85380 ± 0.00118 | 0.92761 ± 0.00024 |    0.57113 ± 0.00123 |             6.33 | **8.0k** |    0 |
| QKeras binary  | 0.82944 ± 0.00120 | 0.88860 ± 0.00633 |    0.43192 ± 0.14802 |            20.00 |    62.0k |    0 |
| QKeras ternary | 0.83545 ± 0.00230 | 0.90246 ± 0.00124 |    0.41383 ± 0.01268 |            11.67 |    28.3k |    0 |
| BitNet binary  | 0.84538 ± 0.00032 | 0.91758 ± 0.00063 |    0.49753 ± 0.01025 |            12.00 |    88.3k |    0 |
| BitNet-1.58    | 0.84894 ± 0.00194 | 0.92324 ± 0.00125 |    0.53454 ± 0.00762 |             9.00 |    91.9k |    0 |
| BDT (unrolled) | 0.84719 ± 0.00025 | 0.92075 ± 0.00003 |    0.56123 ± 0.00061 |         **4.00** |    74.0k |    0 |

The benchmark shows that BitNet-style binary and ternary networks recover substantial predictive performance relative to conventional low-bit quantisation, but do not universally dominate the FPGA accuracy–hardware trade-off. HGQ provides the lowest neural LUT usage, while the tested fully unrolled BDT achieves the lowest synthesised latency.

## Artifact Scope

The repository contains the code, summary results and reproducibility artifacts needed to inspect and regenerate the reported benchmark results. To keep the public repository lightweight, it does not include trained checkpoints, ONNX exports, generated HLS projects, raw synthesis reports or per-event prediction outputs.

Key result files include:

* [`results/abstract_main_binary_table.csv`](results/abstract_main_binary_table.csv) — primary benchmark
* [`results/abstract_secondary_top_table.csv`](results/abstract_secondary_top_table.csv) — robustness task
* [`results/abstract_seed_statistics.csv`](results/abstract_seed_statistics.csv) — seed-level statistics
* [`results/abstract_pareto_candidates.csv`](results/abstract_pareto_candidates.csv) — Pareto comparison

Additional validation and development notes are available in:

* [`REPRODUCIBILITY_REPORT.md`](REPRODUCIBILITY_REPORT.md)
* [`RESULTS_CHECK.md`](RESULTS_CHECK.md)
* [`EXTENDED_ABSTRACT_CONSISTENCY.md`](EXTENDED_ABSTRACT_CONSISTENCY.md)
