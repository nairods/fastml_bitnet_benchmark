# Benchmark Readiness Report

Scope: public benchmark artifact generated from committed summary tables.
Hardware numbers are VU13P, 5 ns HLS C-synthesis estimates, not place-and-route.

## Regenerated Files
- `results/benchmark_seed_statistics.csv`
- `results/benchmark_lowbit_comparison.csv`
- `results/benchmark_pareto_candidates.csv`
- `results/benchmark_status_matrix.csv`
- `plots/benchmark_pareto_auc_vs_lut_qg_vs_wzt.png`
- `plots/benchmark_pareto_auc_vs_latency_qg_vs_wzt.png`
- `plots/benchmark_pareto_auc_vs_lut_qg_vs_top.png`
- `plots/benchmark_pareto_auc_vs_latency_qg_vs_top.png`
- `plots/benchmark_pareto_auc_vs_lut_multiclass.png`
- `plots/benchmark_pareto_auc_vs_latency_multiclass.png`
- `plots/benchmark_lowbit_comparison.png`

## Coverage
- Primary q/g vs W/Z/top rows: 16
- Secondary q/g vs top rows: 16
- Multiclass rows: 12; rows remain marked partial when seed metrics or synthesis estimates are incomplete.

## Notes
- Fixed-FPR signal efficiency is preserved in committed tables because raw per-event prediction scores are not included.
- The public artifact does not include trained checkpoints, ONNX exports, generated HLS projects, or raw C-synthesis reports.
