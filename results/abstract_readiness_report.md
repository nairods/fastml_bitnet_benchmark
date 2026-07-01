# Abstract Readiness Report

Scope: implementation-aware benchmark artifacts for the FastML extended abstract.
Hardware numbers are VU13P, 5 ns HLS C-synthesis estimates, not place-and-route.
Fixed-background signal efficiency is reported at FPR/background acceptance = 0.01.

## Completeness
- Complete status rows: 94 / 126
- Missing or partial status rows: 32 / 126
- Primary binary q/g vs W/Z/top has full seed metrics and full synthesis coverage for the core rows used in the main table.
- Secondary q/g vs top has full seed metrics and full synthesis coverage for the core rows used in the secondary table.
- Multiclass is included as compact supporting material only.

## Best Rows
- Best accuracy: binary_topqg QKeras fixed b7 128-32 acc=0.91051 AUC=0.96450
- Best latency: binary_qg_vs_wzt XGBoost BDT d4x100 (unrolled) 100 trees depth 4 latency=4.00 cycles LUT=74042.7
- Best LUT: binary_qg_vs_wzt HGQ 64-32-32 LUT=8043.0 latency=6.33 cycles
- Best DSP: binary_qg_vs_wzt HGQ 64-32-32 DSP=0.00 LUT=8043.0

## Pareto Candidates
| task | model | architecture | auc_mean | latency_cycles_mean | lut_mean | dsp_mean |
| --- | --- | --- | --- | --- | --- | --- |
| binary_qg_vs_wzt | HGQ | 64-32-32 | 0.92761 | 6.33 | 8043.0 | 0.00 |
| binary_qg_vs_wzt | XGBoost BDT d4x100 (unrolled) | 100 trees depth 4 | 0.92075 | 4.00 | 74042.7 | 0.00 |
| binary_topqg | QKeras fixed b7 | 64-32-32 | 0.96419 | 9.33 | 134430.3 | 627.00 |
| binary_topqg | QKeras fixed b7 | 128-32 | 0.96450 | 8.00 | 186231.7 | 680.00 |
| binary_topqg | HGQ | 64-32-32 | 0.96192 | 7.33 | 9170.0 | 0.00 |
| binary_topqg | HGQ | 128-32 | 0.96159 | 6.67 | 8059.3 | 0.00 |
| binary_topqg | XGBoost BDT d4x100 (unrolled) | 100 trees depth 4 | 0.95842 | 4.00 | 74494.0 | 0.00 |

## Failed Jobs
- None recorded in the benchmark workflow status files.

## Missing Or Partial Rows
- multiclass mlp_baseline__seed42: config=False weights=True metrics=True export=False hls=True synth=True parsed=True
- multiclass mlp_baseline__seed43: config=False weights=True metrics=True export=False hls=False synth=False parsed=False
- multiclass mlp_baseline__seed44: config=False weights=True metrics=True export=False hls=False synth=False parsed=False
- multiclass mlp_topo__seed42: config=False weights=True metrics=True export=False hls=True synth=True parsed=True
- multiclass mlp_topo__seed43: config=False weights=True metrics=True export=False hls=False synth=False parsed=False
- multiclass mlp_topo__seed44: config=False weights=True metrics=True export=False hls=False synth=False parsed=False
- multiclass qkeras_mlp_b7__seed42: config=False weights=True metrics=True export=True hls=True synth=True parsed=True
- multiclass qkeras_mlp_b7__seed43: config=False weights=True metrics=True export=True hls=False synth=False parsed=False
- multiclass qkeras_mlp_b7__seed44: config=False weights=True metrics=True export=True hls=False synth=False parsed=False
- multiclass qkeras_mlp_binary__seed43: config=True weights=False metrics=False export=True hls=False synth=False parsed=False
- multiclass qkeras_mlp_binary__seed44: config=True weights=False metrics=False export=True hls=False synth=False parsed=False
- multiclass qkeras_mlp_ternary__seed43: config=True weights=False metrics=False export=True hls=False synth=False parsed=False
- multiclass qkeras_mlp_ternary__seed44: config=True weights=False metrics=False export=True hls=False synth=False parsed=False
- multiclass hgq_mlp__seed43: config=True weights=True metrics=True export=True hls=False synth=False parsed=False
- multiclass hgq_mlp__seed44: config=True weights=True metrics=True export=True hls=False synth=False parsed=False
- multiclass hgq_mlp_topo__seed43: config=True weights=False metrics=False export=True hls=False synth=False parsed=False
- multiclass hgq_mlp_topo__seed44: config=True weights=False metrics=False export=True hls=False synth=False parsed=False
- multiclass bitnet_mlp_f7_fixed__seed42: config=False weights=True metrics=True export=True hls=True synth=True parsed=True
- multiclass bitnet_mlp_f7_fixed__seed43: config=False weights=True metrics=True export=True hls=True synth=True parsed=True
- multiclass bitnet_mlp_f7_fixed__seed44: config=False weights=True metrics=True export=True hls=True synth=True parsed=True
- multiclass bitnet_topo_f7_fixed__seed42: config=False weights=True metrics=True export=True hls=True synth=True parsed=True
- multiclass bitnet_topo_f7_fixed__seed43: config=False weights=True metrics=True export=True hls=True synth=True parsed=True
- multiclass bitnet_topo_f7_fixed__seed44: config=False weights=True metrics=True export=True hls=True synth=True parsed=True
- multiclass bit158_mlp_f7_fixed__seed42: config=False weights=True metrics=True export=True hls=True synth=True parsed=True
- multiclass bit158_mlp_f7_fixed__seed43: config=False weights=True metrics=True export=True hls=True synth=True parsed=True
- multiclass bit158_mlp_f7_fixed__seed44: config=False weights=True metrics=True export=True hls=True synth=True parsed=True
- multiclass bit158_topo_f7_fixed__seed42: config=False weights=True metrics=True export=True hls=True synth=True parsed=True
- multiclass bit158_topo_f7_fixed__seed43: config=False weights=True metrics=True export=True hls=True synth=True parsed=True
- multiclass bit158_topo_f7_fixed__seed44: config=False weights=True metrics=True export=True hls=True synth=True parsed=True
- multiclass xgboost_bdt__seed42: config=True weights=True metrics=True export=True hls=True synth=False parsed=False
- multiclass xgboost_bdt__seed43: config=True weights=True metrics=True export=True hls=False synth=False parsed=False
- multiclass xgboost_bdt__seed44: config=True weights=True metrics=True export=True hls=False synth=False parsed=False

## Rerun Commands
- No binary workflow reruns required by the status matrix.

## Recommended 4-page Abstract Material
- Main table: results/abstract_main_binary_table.csv
- Secondary table: results/abstract_secondary_top_table.csv
- Low-bit comparison: results/abstract_lowbit_comparison.csv and plots/abstract_lowbit_comparison.png
- Pareto figures: plots/abstract_pareto_auc_vs_lut.png and plots/abstract_pareto_auc_vs_latency.png
- Task-filtered Pareto figures: plots/abstract_pareto_auc_vs_lut_qg_vs_wzt.png and plots/abstract_pareto_auc_vs_latency_qg_vs_wzt.png
- Use multiclass only as a compact supporting table: results/abstract_multiclass_summary.csv

## Interpretation Notes
- BitNet improves over plain QKeras binary/ternary in the primary binary task when comparing AUC at similar 8-11 cycle latencies.
- BitNet does not dominate HGQ or unrolled BDT in this artifact set.
- HGQ is the strongest neural resource-efficiency point by LUT for both binary tasks.
- Unrolled BDT is the fastest overall in the secondary task and should be presented separately from tree-mode BDT.
- Bit158 custom_v9 uses the refreshed sparse-pruned path; the sigmoid variant removes the DSP penalty at 8-9 cycles but costs high LUT.
- Scaling-factor implementation is a first-order hardware variable; use plots/abstract_bitnet_scaling_sweep.png for this point.
