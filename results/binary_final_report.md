# Binary OpenML 42468 Benchmark Report

Task: binary `quark/gluon` versus `W/Z/top`. Hardware target VU13P, 5 ns. Values are HLS C-synthesis, not place-and-route. Neural benchmark rows use a binary sigmoid output boundary unless explicitly labelled `custom_v9_logits`; no softmax is included. BDT rows are Conifer ensemble top-level reports.

## Generated Files
- `results/binary_final_benchmark_summary.csv`
- `results/binary_final_seed_statistics.csv`
- `results/binary_final_pareto_candidates.csv`
- `plots/binary_accuracy_vs_luts.png`
- `plots/binary_accuracy_vs_latency.png`

## Best Rows
- Best by accuracy: `binary_mlp_baseline_64_32_32__seed42` `hls4ml_latency_rf1` acc=0.85984 AUC=0.93250 latency=12 LUT=183159
- Best by AUC: `binary_qkeras_topo_128_32_b7__seed43` `hls4ml_latency_rf1` acc=0.85978 AUC=0.93265 latency=8 LUT=189386
- Best by latency: `binary_xgboost_bdt_d4_100__seed42` `conifer_unrolled` acc=0.84748 AUC=0.92074 latency=4 LUT=73977
- Best by LUT: `binary_hgq_mlp_64_32_32__seed44` `hls4ml_latency_rf1` acc=0.85473 AUC=0.92761 latency=6 LUT=7425

## Aggregate C-Synth Means
- `binary_bit158_sigmoid_f7_fixed` `custom_v9_logits`: acc=0.84894 AUC=0.92324, latency=9.00 cycles, LUT=91810, DSP=2, valid_hw_n=3
- `binary_bit158_sigmoid_f7_fixed` `custom_v9_sigmoid`: acc=0.84894 AUC=0.92324, latency=9.00 cycles, LUT=91932, DSP=0, valid_hw_n=3
- `binary_bit158_topo_sigmoid_f7_fixed` `custom_v9_logits`: acc=0.85000 AUC=0.92334, latency=9.00 cycles, LUT=100132, DSP=2, valid_hw_n=3
- `binary_bit158_topo_sigmoid_f7_fixed` `custom_v9_sigmoid`: acc=0.85000 AUC=0.92334, latency=8.00 cycles, LUT=99954, DSP=0, valid_hw_n=3
- `binary_bitnet_sigmoid_f7_fixed` `custom_v9_logits`: acc=0.84538 AUC=0.91758, latency=7.00 cycles, LUT=88879, DSP=602, valid_hw_n=3
- `binary_bitnet_sigmoid_f7_fixed` `custom_v9_sigmoid`: acc=0.84538 AUC=0.91758, latency=8.00 cycles, LUT=88744, DSP=596, valid_hw_n=2
- `binary_bitnet_sigmoid_f7_fixed` `hls4ml_parent_v26_patch_sigmoid`: acc=0.84538 AUC=0.91758, latency=12.00 cycles, LUT=88349, DSP=0, valid_hw_n=3
- `binary_bitnet_topo_sigmoid_f7_fixed` `custom_v9_logits`: acc=0.84610 AUC=0.91905, latency=6.00 cycles, LUT=122389, DSP=767, valid_hw_n=3
- `binary_bitnet_topo_sigmoid_f7_fixed` `custom_v9_sigmoid`: acc=0.84610 AUC=0.91905, latency=6.00 cycles, LUT=122540, DSP=765, valid_hw_n=3
- `binary_hgq_mlp_64_32_32` `hls4ml_latency_rf1`: acc=0.85380 AUC=0.92761, latency=6.33 cycles, LUT=8043, DSP=0, valid_hw_n=3
- `binary_hgq_topo_128_32` `hls4ml_latency_rf1`: acc=0.85343 AUC=0.92754, latency=6.33 cycles, LUT=8165, DSP=0, valid_hw_n=3
- `binary_mlp_baseline_64_32_32` `hls4ml_latency_rf1`: acc=0.85954 AUC=0.93247, latency=12.00 cycles, LUT=183516, DSP=3455, valid_hw_n=3
- `binary_mlp_topo_128_32` `hls4ml_latency_rf1`: acc=0.85933 AUC=0.93213, latency=10.33 cycles, LUT=269745, DSP=4981, valid_hw_n=3
- `binary_qkeras_mlp_64_32_32_b7` `hls4ml_latency_rf1`: acc=0.85951 AUC=0.93239, latency=9.00 cycles, LUT=135428, DSP=665, valid_hw_n=3
- `binary_qkeras_mlp_binary_64_32_32` `hls4ml_latency_rf1`: acc=0.82944 AUC=0.88860, latency=20.00 cycles, LUT=62026, DSP=0, valid_hw_n=3
- `binary_qkeras_mlp_ternary_64_32_32` `hls4ml_latency_rf1`: acc=0.83545 AUC=0.90246, latency=11.67 cycles, LUT=28287, DSP=0, valid_hw_n=3
- `binary_qkeras_topo_128_32_b7` `hls4ml_latency_rf1`: acc=0.85966 AUC=0.93248, latency=8.00 cycles, LUT=188545, DSP=745, valid_hw_n=3
- `binary_qkeras_topo_binary_128_32` `hls4ml_latency_rf1`: acc=0.83234 AUC=0.89635, latency=21.67 cycles, LUT=91422, DSP=0, valid_hw_n=3
- `binary_qkeras_topo_ternary_128_32` `hls4ml_latency_rf1`: acc=0.83947 AUC=0.90779, latency=12.00 cycles, LUT=33086, DSP=0, valid_hw_n=3
- `binary_xgboost_bdt_d4_100` `conifer_tree`: acc=0.84719 AUC=0.92075, latency=227.00 cycles, LUT=15443, DSP=207, valid_hw_n=3
- `binary_xgboost_bdt_d4_100` `conifer_unrolled`: acc=0.84719 AUC=0.92075, latency=4.00 cycles, LUT=74043, DSP=0, valid_hw_n=3

## Notes
- Bit158 rows above are refreshed from fresh reruns with real sparse pruning in the handwritten custom path, so zero ternary weights are no longer carried through the datapath.
- These are C-synthesis estimates only; no place-and-route was run in this pass.
