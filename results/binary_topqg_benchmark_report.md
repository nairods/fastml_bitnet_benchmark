# Binary Benchmark Report

Task: top signal versus quark/gluon background on OpenML 42468 HLF.
All generated configs use split seed 42; model seeds are 42, 43 and 44.
Hardware target: VU13P, 5 ns clock, C-synthesis only unless noted.

## Best Current Rows
- Best by accuracy: binary_topqg_qkeras_mlp_64_32_32_b7__seed43 hls4ml_latency_rf1 acc=0.9109657389954541 auc=0.964356478492389 latency=10 cycles LUT=134806
- Best by AUC: binary_topqg_qkeras_topo_128_32_b7__seed43 hls4ml_latency_rf1 acc=0.9107943835740709 auc=0.964451142820897 latency=8 cycles LUT=186107
- Best by latency: binary_topqg_bitnet_sigmoid_f7_fixed__seed43 custom_v9_sigmoid acc=0.897025471479402 auc=0.9535961087618655 latency=0 cycles LUT=0
- Best by LUT: binary_topqg_bitnet_sigmoid_f7_fixed__seed43 custom_v9_sigmoid acc=0.897025471479402 auc=0.9535961087618655 latency=0 cycles LUT=0

## Hardware Rows
- binary_topqg_mlp_baseline_64_32_32__seed43 hls4ml_latency_rf1: acc=0.9088993942081868 auc=0.9633057550918842 latency=12 cycles II=1 LUT=184821 FF=33286 DSP=3439 BRAM18=1
- binary_topqg_mlp_topo_128_32__seed43 hls4ml_latency_rf1: acc=0.9085264441734117 auc=0.9632543223542429 latency=10 cycles II=1 LUT=269854 FF=45630 DSP=4966 BRAM18=1
- binary_topqg_qkeras_mlp_64_32_32_b7__seed43 hls4ml_latency_rf1: acc=0.9109657389954541 auc=0.964356478492389 latency=10 cycles II=1 LUT=134806 FF=11376 DSP=597 BRAM18=1
- binary_topqg_qkeras_topo_128_32_b7__seed43 hls4ml_latency_rf1: acc=0.9107943835740709 auc=0.964451142820897 latency=8 cycles II=1 LUT=186107 FF=15863 DSP=676 BRAM18=1
- binary_topqg_qkeras_mlp_binary_64_32_32__seed43 hls4ml_latency_rf1: acc=0.8776119102097591 auc=0.9368676386119217 latency=20 cycles II=1 LUT=61680 FF=16877 DSP=0 BRAM18=1
- binary_topqg_qkeras_topo_binary_128_32__seed43 hls4ml_latency_rf1: acc=0.8794262617302866 auc=0.9423471353415417 latency=23 cycles II=1 LUT=91296 FF=26207 DSP=0 BRAM18=1
- binary_topqg_qkeras_mlp_ternary_64_32_32__seed43 hls4ml_latency_rf1: acc=0.8844560473344152 auc=0.9428247317095908 latency=12 cycles II=1 LUT=26254 FF=7332 DSP=0 BRAM18=1
- binary_topqg_qkeras_topo_ternary_128_32__seed43 hls4ml_latency_rf1: acc=0.886683667812396 auc=0.9465143875488647 latency=10 cycles II=1 LUT=22408 FF=11040 DSP=0 BRAM18=1
- binary_topqg_hgq_mlp_64_32_32__seed43 hls4ml_latency_rf1: acc=0.907871261679888 auc=0.9621061911903996 latency=7 cycles II=1 LUT=10116 FF=1019 DSP=0 BRAM18=1
- binary_topqg_hgq_topo_128_32__seed43 hls4ml_latency_rf1: acc=0.9068532088822586 auc=0.9610403763456086 latency=7 cycles II=1 LUT=7257 FF=637 DSP=0 BRAM18=1
- binary_topqg_bitnet_sigmoid_f7_fixed__seed43 custom_v9_logits: acc=0.897025471479402 auc=0.9535961087618655 latency=18 cycles II=1 LUT=145788 FF=27791 DSP=24 BRAM18=0
- binary_topqg_bitnet_sigmoid_f7_fixed__seed43 custom_v9_sigmoid: acc=0.897025471479402 auc=0.9535961087618655 latency=0 cycles II=1 LUT=0 FF=1 DSP=0 BRAM18=0
- binary_topqg_bitnet_sigmoid_f7_fixed__seed43 hls4ml_parent_v26_patch_sigmoid: acc=0.897025471479402 auc=0.9535961087618655 latency=12 cycles II=1 LUT=91799 FF=16961 DSP=0 BRAM18=1
- binary_topqg_bitnet_topo_sigmoid_f7_fixed__seed43 custom_v9_logits: acc=0.8951909604975355 auc=0.952894557189487 latency=19 cycles II=1 LUT=197431 FF=38509 DSP=26 BRAM18=0
- binary_topqg_bitnet_topo_sigmoid_f7_fixed__seed43 custom_v9_sigmoid: acc=0.8951909604975355 auc=0.952894557189487 latency=20 cycles II=1 LUT=197582 FF=38438 DSP=24 BRAM18=1
- binary_topqg_bit158_sigmoid_f7_fixed__seed43 custom_v9_logits: acc=0.8986886270398855 auc=0.9548659907520943 latency=9 cycles II=1 LUT=101735 FF=17602 DSP=2 BRAM18=0
- binary_topqg_bit158_sigmoid_f7_fixed__seed43 custom_v9_sigmoid: acc=0.8986886270398855 auc=0.9548659907520943 latency=0 cycles II=1 LUT=0 FF=1 DSP=0 BRAM18=0
- binary_topqg_bit158_topo_sigmoid_f7_fixed__seed43 custom_v9_logits: acc=0.9002308258323337 auc=0.9575502485780907 latency=9 cycles II=1 LUT=113875 FF=23385 DSP=2 BRAM18=0
- binary_topqg_bit158_topo_sigmoid_f7_fixed__seed43 custom_v9_sigmoid: acc=0.9002308258323337 auc=0.9575502485780907 latency=8 cycles II=1 LUT=113700 FF=22653 DSP=0 BRAM18=1
- binary_topqg_xgboost_bdt_d4_100__seed43 conifer_tree: acc=0.9027910774224113 auc=0.9583052636039631 latency=227 cycles II=228 LUT=17309 FF=25561 DSP=216 BRAM18=0
- binary_topqg_xgboost_bdt_d4_100__seed43 conifer_unrolled: acc=0.9027910774224113 auc=0.9583052636039631 latency=4 cycles II=1 LUT=74553 FF=3319 DSP=0 BRAM18=0
- binary_topqg_mlp_baseline_64_32_32__seed44 hls4ml_latency_rf1: acc=0.909967845659164 auc=0.9640204696148866 latency=13 cycles II=1 LUT=183374 FF=35654 DSP=3454 BRAM18=1
- binary_topqg_mlp_topo_128_32__seed44 hls4ml_latency_rf1: acc=0.9091816266669355 auc=0.9636450825033097 latency=10 cycles II=1 LUT=271305 FF=45756 DSP=4973 BRAM18=1
- binary_topqg_qkeras_mlp_64_32_32_b7__seed44 hls4ml_latency_rf1: acc=0.9099073672751464 auc=0.9639956938377068 latency=9 cycles II=1 LUT=134104 FF=10609 DSP=633 BRAM18=1
- binary_topqg_qkeras_topo_128_32_b7__seed44 hls4ml_latency_rf1: acc=0.9102399983872431 auc=0.96437792518441 latency=8 cycles II=1 LUT=185543 FF=16163 DSP=654 BRAM18=1
- binary_topqg_qkeras_mlp_binary_64_32_32__seed44 hls4ml_latency_rf1: acc=0.8674313822334667 auc=0.933522114853923 latency=22 cycles II=1 LUT=62357 FF=17039 DSP=0 BRAM18=1
- binary_topqg_qkeras_topo_binary_128_32__seed44 hls4ml_latency_rf1: acc=0.8721990948401859 auc=0.9377801376432842 latency=22 cycles II=1 LUT=90388 FF=26014 DSP=0 BRAM18=1
- binary_topqg_qkeras_mlp_ternary_64_32_32__seed44 hls4ml_latency_rf1: acc=0.8868953421564576 auc=0.9408153479603102 latency=12 cycles II=1 LUT=25486 FF=7891 DSP=0 BRAM18=1
- binary_topqg_qkeras_topo_ternary_128_32__seed44 hls4ml_latency_rf1: acc=0.8870465381165016 auc=0.9443739483713489 latency=10 cycles II=1 LUT=25990 FF=11472 DSP=0 BRAM18=1
- binary_topqg_hgq_mlp_64_32_32__seed44 hls4ml_latency_rf1: acc=0.9061476277353869 auc=0.9615278001122359 latency=7 cycles II=1 LUT=8269 FF=664 DSP=0 BRAM18=1
- binary_topqg_hgq_topo_128_32__seed44 hls4ml_latency_rf1: acc=0.906792730498241 auc=0.9617301308418894 latency=6 cycles II=1 LUT=8469 FF=664 DSP=0 BRAM18=1
- binary_topqg_bitnet_sigmoid_f7_fixed__seed44 custom_v9_logits: acc=0.8979628864316745 auc=0.9541716427241391 latency=19 cycles II=1 LUT=146538 FF=27861 DSP=26 BRAM18=0
- binary_topqg_bitnet_sigmoid_f7_fixed__seed44 custom_v9_sigmoid: acc=0.8979628864316745 auc=0.9541716427241391 latency=0 cycles II=1 LUT=0 FF=1 DSP=0 BRAM18=0
- binary_topqg_bitnet_sigmoid_f7_fixed__seed44 hls4ml_parent_v26_patch_sigmoid: acc=0.8979628864316745 auc=0.9541716427241391 latency=12 cycles II=1 LUT=86457 FF=15911 DSP=0 BRAM18=1
- binary_topqg_bitnet_topo_sigmoid_f7_fixed__seed44 custom_v9_logits: acc=0.8980636837383705 auc=0.9544133156094472 latency=20 cycles II=1 LUT=202256 FF=39285 DSP=26 BRAM18=0
- binary_topqg_bitnet_topo_sigmoid_f7_fixed__seed44 custom_v9_sigmoid: acc=0.8980636837383705 auc=0.9544133156094472 latency=21 cycles II=1 LUT=202407 FF=39220 DSP=24 BRAM18=1
- binary_topqg_bit158_sigmoid_f7_fixed__seed44 custom_v9_logits: acc=0.8986079891945287 auc=0.9554914984316248 latency=9 cycles II=1 LUT=100816 FF=17561 DSP=2 BRAM18=0
- binary_topqg_bit158_sigmoid_f7_fixed__seed44 custom_v9_sigmoid: acc=0.8986079891945287 auc=0.9554914984316248 latency=0 cycles II=1 LUT=0 FF=1 DSP=0 BRAM18=0
- binary_topqg_bit158_topo_sigmoid_f7_fixed__seed44 custom_v9_logits: acc=0.9012085597072846 auc=0.9568518704317761 latency=9 cycles II=1 LUT=111629 FF=23458 DSP=2 BRAM18=0
- binary_topqg_bit158_topo_sigmoid_f7_fixed__seed44 custom_v9_sigmoid: acc=0.9012085597072846 auc=0.9568518704317761 latency=8 cycles II=1 LUT=111454 FF=22772 DSP=0 BRAM18=1
- binary_topqg_xgboost_bdt_d4_100__seed44 conifer_tree: acc=0.903174107187856 auc=0.9583113593820537 latency=233 cycles II=234 LUT=24839 FF=45305 DSP=214 BRAM18=0
- binary_topqg_xgboost_bdt_d4_100__seed44 conifer_unrolled: acc=0.903174107187856 auc=0.9583113593820537 latency=4 cycles II=1 LUT=74418 FF=3213 DSP=0 BRAM18=0

## Failures
- binary_topqg_bitnet_sigmoid_f7_fixed__seed42:synth_bitnet_custom_v9_logits: log=logs/binary_topqg_benchmark/binary_topqg_bitnet_sigmoid_f7_fixed__seed42_synth_bitnet_custom_v9_logits.log returncode=1
- binary_topqg_bitnet_sigmoid_f7_fixed__seed42:synth_bitnet_custom_v9_sigmoid: log=logs/binary_topqg_benchmark/binary_topqg_bitnet_sigmoid_f7_fixed__seed42_synth_bitnet_custom_v9_sigmoid.log returncode=1
- binary_topqg_bitnet_topo_sigmoid_f7_fixed__seed42:synth_bitnet_custom_v9_logits: log=logs/binary_topqg_benchmark/binary_topqg_bitnet_topo_sigmoid_f7_fixed__seed42_synth_bitnet_custom_v9_logits.log returncode=1
- binary_topqg_bitnet_topo_sigmoid_f7_fixed__seed42:synth_bitnet_custom_v9_sigmoid: log=logs/binary_topqg_benchmark/binary_topqg_bitnet_topo_sigmoid_f7_fixed__seed42_synth_bitnet_custom_v9_sigmoid.log returncode=1

## Preliminary BitNet Interpretation
The binary track is designed to test the strongest BitNet case: a one-logit sigmoid model where cumulative alpha scaling can be folded into the final sigmoid/table implementation. Compare `bitnet_hls4ml_v26_sigmoid` and `bitnet_custom_v9_sigmoid` rows against dense, QKeras, HGQ and Conifer rows at similar accuracy. If BitNet is not Pareto competitive here, the benchmark conclusion should be that alpha folding alone is not enough for this HLF task and toolchain; if it is competitive only in the sigmoid endpoint, the advantage is endpoint-specific rather than a general multiclass/logits advantage.
