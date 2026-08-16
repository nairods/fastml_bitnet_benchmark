# BitNet FPGA Benchmark

Public artifact for **Do BitNet Gains Survive Synthesis? An
Implementation-Aware Benchmark of Low-Precision FPGA Inference for Jet
Classification**.

This repository compares predictive performance and FPGA HLS C-synthesis
estimates for dense, quantized, BitNet, and boosted-tree classifiers on OpenML
42468. Two training profiles are kept independently:

| Profile | Neural training budget | Checkpoint used for evaluation and synthesis |
| --- | ---: | --- |
| `20-epochs` | exactly 20 epochs | lowest validation loss among epochs 1-20 |
| `200-epochs` | exactly 200 epochs | lowest validation loss among epochs 1-200 |

Training never stops early. The test split is evaluated only after checkpoint
selection and is never used to select an epoch or tune a model.

## Reproduce tables and plots

Create the lightweight artifact environment:

```bash
conda env create -f environment.yml
conda activate fastml-bitnet-benchmark
```

Regenerate and validate either committed profile:

```bash
make reproduce PROFILE=20-epochs
make check PROFILE=20-epochs

make reproduce PROFILE=200-epochs
make check PROFILE=200-epochs
```

`make check` compares generated CSV tables byte-for-byte and verifies the full
generated PNG inventory. PNG byte streams can vary across valid Matplotlib,
FreeType, and PNG renderer builds, so `make check-byte-exact` is reserved for
the local rendering environment that produced the committed images.

The generator reads one compact source record per profile:

- [`data/20-epochs/benchmark_records.json`](data/20-epochs/benchmark_records.json)
- [`data/200-epochs/benchmark_records.json`](data/200-epochs/benchmark_records.json)

It writes three summary tables under `results/<profile>/` and the six Pareto
plots plus seven representative seed-42 learning curves under
`plots/<profile>/`. The compact records contain per-seed test metrics,
checkpoint-selection metadata, synthesis estimates, and the histories needed
for the public plots. Raw OpenML arrays and generated HLS projects are not
committed.

## Results

### 20 epochs

- [q/g vs W/Z/top table](results/20-epochs/benchmark_main_binary_table.csv)
- [q/g vs top table](results/20-epochs/benchmark_secondary_top_table.csv)
- [multiclass table](results/20-epochs/benchmark_multiclass_summary.csv)
- [q/g vs W/Z/top AUC vs LUT](plots/20-epochs/benchmark_pareto_auc_vs_lut_qg_vs_wzt.png)
- [q/g vs W/Z/top AUC vs latency](plots/20-epochs/benchmark_pareto_auc_vs_latency_qg_vs_wzt.png)
- [q/g vs top AUC vs LUT](plots/20-epochs/benchmark_pareto_auc_vs_lut_qg_vs_top.png)
- [q/g vs top AUC vs latency](plots/20-epochs/benchmark_pareto_auc_vs_latency_qg_vs_top.png)
- [multiclass macro AUC vs LUT](plots/20-epochs/benchmark_pareto_auc_vs_lut_multiclass.png)
- [multiclass macro AUC vs latency](plots/20-epochs/benchmark_pareto_auc_vs_latency_multiclass.png)

### 200 epochs

- [q/g vs W/Z/top table](results/200-epochs/benchmark_main_binary_table.csv)
- [q/g vs top table](results/200-epochs/benchmark_secondary_top_table.csv)
- [multiclass table](results/200-epochs/benchmark_multiclass_summary.csv)
- [q/g vs W/Z/top AUC vs LUT](plots/200-epochs/benchmark_pareto_auc_vs_lut_qg_vs_wzt.png)
- [q/g vs W/Z/top AUC vs latency](plots/200-epochs/benchmark_pareto_auc_vs_latency_qg_vs_wzt.png)
- [q/g vs top AUC vs LUT](plots/200-epochs/benchmark_pareto_auc_vs_lut_qg_vs_top.png)
- [q/g vs top AUC vs latency](plots/200-epochs/benchmark_pareto_auc_vs_latency_qg_vs_top.png)
- [multiclass macro AUC vs LUT](plots/200-epochs/benchmark_pareto_auc_vs_lut_multiclass.png)
- [multiclass macro AUC vs latency](plots/200-epochs/benchmark_pareto_auc_vs_latency_multiclass.png)

Plot colors identify model families and markers identify architectures. Error
bars are sample standard deviations over the available model seeds.

## Benchmark protocol

### Dataset and tasks

The workflow downloads the OpenML `hls4ml_lhc_jets_hlf` dataset (ID 42468) on
demand. It contains approximately 830,000 jets, 16 high-level observables, and
five labels: gluon, light quark, W, Z, and top.

| Task | Background | Signal/classes |
| --- | --- | --- |
| `qg_vs_wzt` | gluon + light quark | W + Z + top |
| `qg_vs_top` | gluon + light quark | top; W/Z excluded |
| `multiclass` | n/a | gluon, quark, W, Z, top |

Each task has fixed stratified 64%/16%/20% train/validation/test indices under
[`data/splits/`](data/splits/), generated with split seed 42. Each task fits
its own `StandardScaler` on its training subset only, then applies that scaler
to validation and test data. No committed `scaler.pkl` is loaded.

Model seeds are 42, 43, and 44. Neural models use either a `64-32-32`
standard hidden architecture or a `128-32` wide architecture. The multiclass
QKeras configurations retain only the published standard architecture. The BDT
uses exactly 100 trees with maximum depth 4.

Fairness is enforced through identical data, split indices, preprocessing,
architectures, seed set, loss definitions, full-budget training, and
validation-loss checkpoint selection. Optimizer settings are fixed per model
family and are identical between profiles except for the maximum epoch count
and HGQ schedule length:

| Models | Optimizer | Batch | Initial LR | Schedule |
| --- | --- | ---: | ---: | --- |
| Dense, QKeras, BitNet | Adam | 1024 | 0.001 | constant |
| HGQ | Adam | 16384 | 0.02 | cosine decay over the full profile |

PyTorch training and evaluation use 16 CPU intra-op threads. All neural models
shuffle the training data deterministically from their model seed.
Losses are binary cross entropy from logits for the binary tasks and sparse
categorical cross entropy from logits for multiclass. HGQ keeps `beta=3e-6`;
QKeras uses 7-bit ReLU activations and either 7-bit, binary, or ternary weights
and biases. BitNet uses binary weights and BitNet-1.58 ternary weights; both use
8-bit activations with 7 fractional bits. All quantizer widths and model
definitions are frozen in [`configs/benchmark.json`](configs/benchmark.json).

The BDT does not have a neural epoch budget. It is independently trained for
each task and seed with 100 boosting rounds, depth 4, learning rate 0.1,
subsample 0.8, column subsample 0.8, histogram tree construction, and no early
stopping. These settings are identical in both profile directories; this keeps
the non-neural reference fixed while the neural training budget changes. The
resulting 20- and 200-profile XGBoost boosters are byte-identical for every
task/seed pair, so the 200-profile compact record explicitly reuses the matching
20-profile Conifer synthesis estimate instead of synthesizing identical logic
twice. Binary BDT synthesis covers all three seeds. Multiclass BDT synthesis is
reported as partial with seed 42 only because fully unrolling the five-class
ensemble is substantially more expensive; the software metrics still cover all
three seeds.

The compared model families are Dense MLP, QKeras 7-bit, HGQ, QKeras binary,
QKeras ternary, BitNet binary, BitNet-1.58, and XGBoost BDT. Accuracy, ROC AUC,
and signal efficiency at 1% false-positive rate are measured on the fixed test
subset. Multiclass uses macro one-vs-rest ROC AUC.

### Hardware estimates

All hardware numbers are **HLS C-synthesis estimates**, not place-and-route
results:

- FPGA: AMD/Xilinx VU13P, `xcvu13p-flga2577-2-e`
- target clock: 5 ns
- target initiation interval: 1
- I/O: parallel
- synthesis tool for neural models: Vitis HLS 2023.1

The `synth_variant` column identifies the implementation used for each row.
Stock hls4ml uses latency strategy and reuse factor 1. BitNet and BitNet-1.58
use one architecture- and output-independent hls4ml patch: binary weights map
to add/subtract operations, ternary zero weights are skipped, binary outputs
include the sigmoid lookup, and multiclass outputs expose logits without
softmax. The compact record preserves the implementation boundary, tool
version, target, and source variant for every synthesis result.

## Retrain a profile

TensorFlow/QKeras and HGQ dependency constraints require separate environments;
see `requirements-training.txt`, `requirements-qkeras.txt`, and
`requirements-hgq.txt`. These files also pin hls4ml 1.2.0 and, where needed,
Conifer 1.8 and ONNX 1.17.0. The runner writes profile-local configs,
checkpoints, histories, and raw results, so profiles cannot overwrite each
other.

```bash
# PyTorch, BitNet, and XGBoost environment
python scripts/run_benchmark.py --profile 200-epochs --task qg_vs_wzt \
  --models dense bitnet_binary bitnet_158 xgboost_bdt

# QKeras environment
python scripts/run_benchmark.py --profile 200-epochs --task qg_vs_wzt \
  --models qkeras_b7 qkeras_binary qkeras_ternary

# HGQ environment
python scripts/run_benchmark.py --profile 200-epochs --task qg_vs_wzt \
  --models hgq
```

Repeat with `qg_vs_top` and `multiclass`. Use `--architectures`, `--seeds`, or
`--stages` for a resumable subset; existing stage outputs are skipped unless
`--force` is passed. Checkpoints are written to `models/<profile>/`, histories
and generated run configs to ignored `logs/<profile>/`, and raw metrics to
ignored `results/<profile>/raw/`.

Synthesize each selected checkpoint from the environment matching its backend:

```bash
python scripts/run_synthesis.py \
  --config logs/200-epochs/run_configs/bit158_64_32_32__seed42.json
```

The wrapper chooses stock hls4ml, the common patched BitNet route, or unrolled
Conifer from the model config. It validates BitNet numerical output before
synthesis, keeps only reports and compact results, and puts disposable project
trees under `/tmp`. A licensed Vitis HLS 2023.1 installation is required;
`--allow-unverified-license` only bypasses the preflight check, not licensing.

After evaluation and synthesis, collect the raw outputs and regenerate the
public artifacts:

```bash
python scripts/collect_benchmark_records.py --profile 200-epochs
make reproduce PROFILE=200-epochs
make check PROFILE=200-epochs
```

The collector fails on missing per-seed metrics or expected synthesis reports.
`--allow-missing-synthesis` is available only for explicitly partial work in
progress.

## Repository guide

- [`scripts/run_benchmark.py`](scripts/run_benchmark.py) expands the frozen
  protocol into backend run configs and executes training/evaluation.
- [`scripts/collect_benchmark_records.py`](scripts/collect_benchmark_records.py)
  validates and collects profile-local raw outputs.
- [`scripts/run_synthesis.py`](scripts/run_synthesis.py) dispatches one selected
  checkpoint through its declared hardware implementation.
- [`scripts/generate_benchmark_artifacts.py`](scripts/generate_benchmark_artifacts.py)
  is the public table and plotting implementation.
- [`benchmark.py`](benchmark.py) owns task loading, split/scaler handling,
  PyTorch training, checkpoint selection, and metrics.
- [`model_registry.py`](model_registry.py) contains the published PyTorch model
  definitions; the root train/test files are backend adapters used by the
  runner.
- `hardware_benchmark/` and the synthesis scripts provide stock hls4ml,
  patched BitNet, and Conifer conversion/report handling.

To extend the benchmark, add the model metadata and fixed hyperparameters to
`configs/benchmark.json`, register or add its backend adapter, train it on the
same profile/tasks/seeds, synthesize the selected checkpoint, then run the
collector and generator. Their validation prevents an undeclared model,
architecture, seed set, implementation label, or training profile from entering
the public tables silently. The complete contribution and artifact-update
contract is in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Apache 2.0; see [LICENSE](LICENSE).
