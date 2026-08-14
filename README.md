# BitNet FPGA Benchmark

Public artifact for **Do BitNet Gains Survive Synthesis? An
Implementation-Aware Benchmark of Low-Precision FPGA Inference for Jet
Classification**.

This repository compares predictive performance and FPGA HLS C-synthesis
estimates for dense, quantized, BitNet, and boosted-tree classifiers on the
OpenML `hls4ml_lhc_jets_hlf` dataset. It is organized around two reproducibility
levels:

1. A lightweight, deterministic path regenerates every committed result table,
   Pareto plot, and training plot from compact per-seed records.
2. An optional research path downloads OpenML 42468 and retrains selected models.
   Generic hls4ml and Conifer synthesis helpers are included for extension work;
   vendor HLS tools and compatible backend environments are required.

## Reproduce the public artifacts

Create the small plotting environment and regenerate all public outputs:

```bash
conda env create -f environment.yml
conda activate fastml-bitnet-benchmark
make reproduce
make check
```

`make reproduce` writes the three CSV tables under `results/`, the six Pareto
plots, and the seven seed-42 training plots under `plots/`. `make check`
regenerates everything in a temporary directory and verifies that the committed
files match byte for byte.

The source for this command is
[`data/benchmark_records.json`](data/benchmark_records.json). It contains only
the per-seed metrics, selected C-synthesis estimates, and training histories
needed for the public artifacts. It does not contain checkpoints, per-event
predictions, generated HLS projects, or raw OpenML data.

## Repository guide

- [`configs/benchmark.json`](configs/benchmark.json) defines the dataset, tasks,
  model families, architectures, seeds, hardware target, and public
  implementation labels.
- [`data/splits/`](data/splits/) contains fixed task-specific train,
  validation, and test indices.
- [`scripts/generate_benchmark_artifacts.py`](scripts/generate_benchmark_artifacts.py)
  is the single table and plot generator.
- [`scripts/run_benchmark.py`](scripts/run_benchmark.py) is the configurable
  training and evaluation entry point.
- [`benchmark.py`](benchmark.py) implements dataset loading, preprocessing,
  training utilities, and metrics. [`model_registry.py`](model_registry.py)
  contains only the published model families.
- The root `train_*` and `test_*` files, plus `torchDNN.py`, are backend
  adapters invoked by `scripts/run_benchmark.py`; they are not separate user
  workflows.
- `hardware_benchmark/` and the two synthesis scripts contain small reusable
  helpers for stock hls4ml projects and Conifer BDT projects.

## Benchmark protocol

### Dataset and tasks

The benchmark uses [OpenML dataset 42468](https://www.openml.org/d/42468):
approximately 830,000 jets, 16 high-level observables, and five labels (`g`,
`q`, `W`, `Z`, `top`). The training workflow downloads it on demand with
`sklearn.datasets.fetch_openml`.

Three tasks are defined:

| Task | Background | Signal/classes | Coverage |
| --- | --- | --- | --- |
| q/g vs W/Z/top | gluon + light quark | W + Z + top | complete |
| q/g vs top | gluon + light quark | top; W/Z excluded | complete except one partial hardware row |
| multiclass | n/a | g, q, W, Z, top | partial seed/synthesis coverage |

Each task has its own fixed stratified 64%/16%/20% split with split seed 42.
The loader uses the matching archive in `data/splits/`. A separate
`StandardScaler` is fitted on that task's training subset only and then applied
to validation and test data. No committed scaler pickle is used.

### Models

The neural models use either `16-64-32-32-output` (standard) or
`16-128-32-output` (wide). The BDT uses 100 trees with maximum depth 4. Model
seeds are 42, 43, and 44.

The compared families are:

- Dense MLP
- QKeras 7-bit fixed point
- HGQ
- QKeras binary
- QKeras ternary
- BitNet binary
- BitNet-1.58 sparse ternary
- XGBoost BDT

Accuracy, macro ROC AUC, and signal efficiency at 1% false-positive rate are
computed on the fixed test subset. The fixed-FPR values in the compact records
were computed from the original per-event predictions; those large predictions
are not committed.

### Hardware measurements

All hardware values are **HLS C-synthesis estimates**, not place-and-route
results:

- FPGA: AMD/Xilinx VU13P, `xcvu13p-flga2577-2-e`
- target clock: 5 ns
- target initiation interval: 1
- I/O: parallel

The implementation path is part of the result and is not uniform across model
families:

| Public label | Meaning |
| --- | --- |
| `hls4ml_latency_rf1` | Stock hls4ml, latency strategy, reuse factor 1 |
| `hls4ml_patched_bitnet_latency_rf1` | hls4ml-generated BitNet project with architecture-specific patched kernels |
| `custom_hls_bitnet_sigmoid_rf1` | Custom BitNet HLS path including the binary sigmoid endpoint |
| `custom_hls_bitnet_logits_rf1` | Custom BitNet HLS path with multiclass logits output |
| `conifer_unrolled` | Fully unrolled Conifer BDT |

This distinction is important: most published BitNet rows are not stock
hls4ml conversions. The compact record keeps the original internal variant name
for provenance while exposing stable, descriptive public labels in the CSVs.
For q/g vs top, the standard BitNet-1.58 row has three metric seeds but only one
valid sigmoid-inclusive synthesis record and is therefore marked `partial`.

## Results

The generated tables are:

- [Primary q/g vs W/Z/top table](results/benchmark_main_binary_table.csv)
- [Secondary q/g vs top table](results/benchmark_secondary_top_table.csv)
- [Partial multiclass table](results/benchmark_multiclass_summary.csv)

The generated Pareto plots are:

- [Primary AUC vs LUT](plots/benchmark_pareto_auc_vs_lut_qg_vs_wzt.png)
- [Primary AUC vs latency](plots/benchmark_pareto_auc_vs_latency_qg_vs_wzt.png)
- [Secondary AUC vs LUT](plots/benchmark_pareto_auc_vs_lut_qg_vs_top.png)
- [Secondary AUC vs latency](plots/benchmark_pareto_auc_vs_latency_qg_vs_top.png)
- [Multiclass macro AUC vs LUT](plots/benchmark_pareto_auc_vs_lut_multiclass.png)
- [Multiclass macro AUC vs latency](plots/benchmark_pareto_auc_vs_latency_multiclass.png)

Colors identify model families and marker shapes identify architectures. Error
bars show the sample standard deviation over available seeds.

The retained training plots cover the standard architecture, primary task, and
seed 42:

- [Dense MLP](plots/dense_baseline_64_32_32__seed42_training.png)
- [QKeras 7-bit](plots/qkeras_b7_64_32_32__seed42_training.png)
- [HGQ](plots/hgq_64_32_32__seed42_training.png)
- [QKeras binary](plots/qkeras_binary_64_32_32__seed42_training.png)
- [QKeras ternary](plots/qkeras_ternary_64_32_32__seed42_training.png)
- [BitNet binary](plots/bitnet_64_32_32__seed42_training.png)
- [BitNet-1.58](plots/bit158_64_32_32__seed42_training.png)

For the primary standard architecture, Dense and QKeras 7-bit reach an AUC of
about 0.9325. BitNet-1.58 reaches 0.9232, ahead of the naive binary and ternary
QKeras models. HGQ is the lowest-LUT neural implementation, while the unrolled
BDT is the lowest-latency point. The exact values and seed counts are in the
CSV tables.

## Retraining

The TensorFlow requirements for QKeras and HGQ are incompatible, so they must
be installed in separate environments. Do not install both optional requirement
files into the same environment.

For PyTorch, BitNet, and XGBoost models:

```bash
python -m venv .venv-training
. .venv-training/bin/activate
python -m pip install -r requirements-training.txt
python scripts/run_benchmark.py --task qg_vs_wzt \
  --models dense bitnet_binary bitnet_158 xgboost_bdt
```

For QKeras models, use a separate environment with
`requirements-qkeras.txt`; for HGQ use another environment with
`requirements-hgq.txt`. Examples:

```bash
python scripts/run_benchmark.py --task qg_vs_wzt \
  --models qkeras_b7 qkeras_binary qkeras_ternary

python scripts/run_benchmark.py --task qg_vs_wzt --models hgq
```

Replace the task with `qg_vs_top` or `multiclass` as needed. Use
`--architectures`, `--seeds`, or `--stages` to select a subset. Run configs are
written under ignored `logs/run_configs/`; checkpoints and raw per-seed outputs
are also ignored.

Hardware reruns additionally require `requirements-hardware.txt`, a compatible
hls4ml/Conifer stack, and licensed Vivado or Vitis HLS. For example, after
training a stock hls4ml-compatible run:

```bash
RUN=qkeras_b7_64_32_32__seed42
python export_reference_predictions.py \
  --config logs/run_configs/${RUN}.json
python scripts/prepare_hls4ml_project.py \
  --config logs/run_configs/${RUN}.json
python scripts/synthesize_hls4ml_project.py \
  --run-name ${RUN} --project-dir hls_projects/${RUN}/native
```

Reference prediction export is required by the QKeras and HGQ conversion
worker. The stock dense conversion does not require that export. For a trained
BDT, reproduce the public unrolled implementation with:

```bash
RUN=xgboost_bdt_d4_100__seed42
python scripts/synthesize_xgboost_conifer.py \
  --config logs/run_configs/${RUN}.json --unroll
```

The repository does not claim that these generic helpers reproduce the archived
custom BitNet kernels, and `prepare_hls4ml_project.py` deliberately rejects
BitNet runs rather than substituting a non-equivalent implementation. Those
measurements remain inspectable in
`data/benchmark_records.json`, including implementation boundary, tool version,
part, clock, and original source-variant identifier.

## Extending the benchmark

To add a model or quantization method:

1. Add its public metadata, backend, architecture names, and quantization
   settings to `configs/benchmark.json`.
2. Register a PyTorch model in `model_registry.py`, or add a backend adapter
   following the QKeras/HGQ/XGBoost train and evaluation scripts.
3. Run `scripts/run_benchmark.py` on the same task splits and seeds.
4. Add the reviewed per-seed metrics and synthesis records to
   `data/benchmark_records.json`, preserving the output boundary and synthesis
   implementation label.
5. Run `make reproduce` and `make check`.

The generator validates model names, architectures, base run names, seeds, and
hardware implementation labels against the benchmark configuration before
writing any public artifact.

## License

MIT; see [LICENSE](LICENSE).
