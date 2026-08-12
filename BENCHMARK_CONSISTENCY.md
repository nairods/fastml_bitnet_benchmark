# Benchmark Consistency

This file maps the main benchmark claims to repository evidence.

## Dataset Claim

Claim: The benchmark uses OpenML 42468, `hls4ml_lhc_jets_hlf`, with 16 high-level features and five jet classes.

Evidence:

- Dataset loading: `benchmark.py`
- Dataset metadata and split provenance: `data/splits/`

Status: supported.

## Primary Task Claim

Claim: The main task is `q/g vs W/Z/top`, with q/g as background and W/Z/top as signal.

Evidence:

- Class mapping: `benchmark.py`, `_classification_spec`, mode `binary_qg_vs_wzt`
- Primary table: `results/benchmark_main_binary_table.csv`
- Primary plots: `plots/benchmark_pareto_auc_vs_lut_qg_vs_wzt.png`, `plots/benchmark_pareto_auc_vs_latency_qg_vs_wzt.png`

Status: supported.

## Secondary Robustness Task Claim

Claim: The secondary task is `q/g vs top`, with W/Z removed.

Evidence:

- Class mapping: `benchmark.py`, mode `binary_top_vs_qg`, `drop_labels={"w", "z"}`
- Secondary table: `results/benchmark_secondary_top_table.csv`

Status: supported.

## Multiclass Task Claim

Claim: The supporting multiclass task is five-way `q/g/W/Z/top` classification.

Evidence:

- Class mapping: `benchmark.py`, mode `multiclass`
- Multiclass summary: `results/benchmark_multiclass_summary.csv`
- Multiclass plots: `plots/benchmark_pareto_auc_vs_lut_multiclass.png`, `plots/benchmark_pareto_auc_vs_latency_multiclass.png`

Status: supported as a partial table where synthesis or seed coverage is incomplete.

## Split Claim

Claim: All models use a fixed stratified 64/16/20 split with split seed 42.

Evidence:

- Split code: `benchmark.py`
- Split archives: `data/splits/openml_42468_nall_splitseed42_*_train0p64_val0p16_test0p2.npz`
- Public split indices: `splits/train_idx.npy`, `splits/val_idx.npy`, `splits/test_idx.npy`

Status: supported for committed split indices. Fresh recomputation requires OpenML access.

## Seed Claim

Claim: Reported means and standard deviations use seeds 42, 43, 44.

Evidence:

- Workflow script: `scripts/run_binary_benchmark_workflow.py`
- Benchmark tables: `results/benchmark_main_binary_table.csv`, `results/benchmark_secondary_top_table.csv`
- Seed summary: `results/benchmark_seed_statistics.csv`

Status: supported. Main rows report `seeds_metrics=3` and `seeds_synth=3`.

## Model Family Claim

Claim: The core comparison includes Dense MLP, QKeras fixed b7, HGQ, QKeras binary, QKeras ternary, BitNet binary, Bit158 sparse ternary, and XGBoost BDT.

Evidence:

- Model construction/workflow: `scripts/run_binary_benchmark_workflow.py`
- Model registry/configs: `configs/`
- Result tables: `results/benchmark_main_binary_table.csv`

Status: supported in summary artifacts. Trained checkpoints are not committed.

## Hardware Target Claim

Claim: Hardware target is VU13P, 5 ns, II=1, HLS C-synthesis only.

Evidence:

- Hardware configs: `configs/hardware_benchmark.json`, `configs/hls4ml_hardware.json`
- Synthesis scripts: `scripts/synthesize_hls4ml_project.py`, `scripts/synthesize_xgboost_conifer.py`, BitNet synthesis scripts
- Artifact coverage: `results/benchmark_status_matrix.csv`
- README hardware section

Status: supported in summaries. Raw HLS reports are not committed.

## Result Claim: BitNet Improves Over Naive Binary/Ternary QKeras

Claim: BitNet-style scaling improves predictive performance over naive binary and ternary QKeras.

Evidence from `results/benchmark_main_binary_table.csv`, primary `64-32-32`:

- QKeras binary AUC: 0.88860
- QKeras ternary AUC: 0.90246
- BitNet binary AUC: 0.91758
- Bit158 sparse ternary AUC: 0.92324

Status: supported.

## Result Claim: BitNet Does Not Universally Dominate FPGA Tradeoffs

Claim: BitNet improves accuracy over naive low-bit baselines but is not always best in latency/resource tradeoff.

Evidence:

- BitNet binary: AUC 0.91758, latency 12 cycles, LUT 88.3k
- HGQ: AUC 0.92761, latency 6.33 cycles, LUT 8.0k
- BDT unrolled: AUC 0.92075, latency 4 cycles, LUT 74.0k

Status: supported.

## Result Claim: HGQ Is Strongest Neural Resource-Efficiency Point

Claim: HGQ gives the strongest neural LUT efficiency among core neural models.

Evidence:

- `results/benchmark_main_binary_table.csv`
- HGQ `64-32-32`: AUC 0.92761, LUT 8.0k, DSP 0

Status: supported.

## Result Claim: Unrolled BDT Is Lowest Latency

Claim: The tested unrolled BDT gives the lowest HLS C-synthesis latency.

Evidence:

- `results/benchmark_main_binary_table.csv`
- BDT unrolled latency: 4 cycles
- Next-best neural latency in main rows: HGQ at 6.33 cycles

Status: supported.

## Plot Claim

Claim: Pareto plots show AUC vs LUT and AUC vs latency for the primary task.

Evidence:

- `plots/benchmark_pareto_auc_vs_lut_qg_vs_wzt.png`
- `plots/benchmark_pareto_auc_vs_latency_qg_vs_wzt.png`
- Source tables: `results/benchmark_main_binary_table.csv`
- Regeneration script: `scripts/reproduce_public_artifacts.py`

Status: supported. The plots include both `64-32-32` and `128-32` architectures plus the unrolled BDT.

## Limitation Claim

Claim: C-synthesis estimates should not be interpreted as place-and-route results.

Evidence:

- README hardware section
- `results/benchmark_readiness_report.md`

Status: supported.

## Known Consistency Gaps

- Raw trained checkpoints are not committed.
- ONNX exports are not committed.
- Generated HLS and Conifer projects are not committed.
- Raw C-synthesis reports are not committed.
- Raw prediction scores are not committed, so fixed-FPR signal efficiency is preserved in tables but not recomputable from first principles.
