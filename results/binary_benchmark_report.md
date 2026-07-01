# Binary Benchmark Report

Task: quark/gluon background versus W/Z/top signal on OpenML 42468 HLF.
All generated configs use split seed 42; model seeds are 42, 43 and 44.
Hardware target: VU13P, 5 ns clock, C-synthesis only unless noted.

## Best Current Rows
- Best by accuracy: binary_mlp_baseline_64_32_32__seed42 hls4ml_latency_rf1 acc=0.8598433734939759 auc=0.9324996431198108 latency=12 cycles LUT=183159
- Best by AUC: binary_qkeras_topo_128_32_b7__seed43 hls4ml_latency_rf1 acc=0.8597771084337349 auc=0.932652113663417 latency=8 cycles LUT=189386
- Best by latency: binary_bitnet_sigmoid_f7_fixed__seed42 custom_v9_sigmoid acc=0.845210843373494 auc=0.9168716834413151 latency=0 cycles LUT=0
- Best by LUT: binary_bitnet_sigmoid_f7_fixed__seed42 custom_v9_sigmoid acc=0.845210843373494 auc=0.9168716834413151 latency=0 cycles LUT=0

## Hardware Rows
- binary_mlp_baseline_64_32_32__seed42 hls4ml_latency_rf1: acc=0.8598433734939759 auc=0.9324996431198108 latency=12 cycles II=1 LUT=183159 FF=31903 DSP=3479 BRAM18=1
- binary_mlp_topo_128_32__seed42 hls4ml_latency_rf1: acc=0.8593734939759036 auc=0.9320562738637983 latency=10 cycles II=1 LUT=269538 FF=47533 DSP=4980 BRAM18=1
- binary_qkeras_mlp_64_32_32_b7__seed42 hls4ml_latency_rf1: acc=0.8595903614457832 auc=0.9322736903719058 latency=9 cycles II=1 LUT=134907 FF=10846 DSP=636 BRAM18=1
- binary_qkeras_topo_128_32_b7__seed42 hls4ml_latency_rf1: acc=0.8594397590361446 auc=0.9325468456431845 latency=8 cycles II=1 LUT=188249 FF=16393 DSP=728 BRAM18=1
- binary_qkeras_mlp_binary_64_32_32__seed42 hls4ml_latency_rf1: acc=0.8298855421686747 auc=0.889254866122259 latency=18 cycles II=1 LUT=62162 FF=16203 DSP=0 BRAM18=1
- binary_qkeras_topo_binary_128_32__seed42 hls4ml_latency_rf1: acc=0.8309457831325301 auc=0.8946455014783776 latency=22 cycles II=1 LUT=91435 FF=25822 DSP=0 BRAM18=1
- binary_qkeras_mlp_ternary_64_32_32__seed42 hls4ml_latency_rf1: acc=0.8374216867469879 auc=0.903684653586073 latency=13 cycles II=1 LUT=31965 FF=9829 DSP=0 BRAM18=1
- binary_qkeras_topo_ternary_128_32__seed42 hls4ml_latency_rf1: acc=0.840789156626506 auc=0.9080946709952538 latency=12 cycles II=1 LUT=36539 FF=15300 DSP=0 BRAM18=1
- binary_hgq_mlp_64_32_32__seed42 hls4ml_latency_rf1: acc=0.8524698795180723 auc=0.9273719515165576 latency=6 cycles II=1 LUT=7765 FF=579 DSP=0 BRAM18=1
- binary_hgq_topo_128_32__seed42 hls4ml_latency_rf1: acc=0.853578313253012 auc=0.9273320884319116 latency=7 cycles II=1 LUT=8135 FF=685 DSP=0 BRAM18=2
- binary_bitnet_sigmoid_f7_fixed__seed42 custom_v9_logits: acc=0.845210843373494 auc=0.9168716834413151 latency=7 cycles II=1 LUT=89460 FF=15461 DSP=608 BRAM18=0
- binary_bitnet_sigmoid_f7_fixed__seed42 custom_v9_sigmoid: acc=0.845210843373494 auc=0.9168716834413151 latency=0 cycles II=1 LUT=0 FF=1 DSP=0 BRAM18=0
- binary_bitnet_sigmoid_f7_fixed__seed42 hls4ml_parent_v26_patch_sigmoid: acc=0.845210843373494 auc=0.9168716834413151 latency=12 cycles II=1 LUT=88635 FF=16528 DSP=0 BRAM18=1
- binary_bitnet_topo_sigmoid_f7_fixed__seed42 custom_v9_logits: acc=0.8475963855421687 auc=0.9201585557594041 latency=6 cycles II=1 LUT=121369 FF=15075 DSP=749 BRAM18=0
- binary_bitnet_topo_sigmoid_f7_fixed__seed42 custom_v9_sigmoid: acc=0.8475963855421687 auc=0.9201585557594041 latency=6 cycles II=1 LUT=121520 FF=18709 DSP=747 BRAM18=1
- binary_bit158_sigmoid_f7_fixed__seed42 custom_v9_logits: acc=0.8496325301204819 auc=0.9242838920077538 latency=9 cycles II=1 LUT=92250 FF=16446 DSP=2 BRAM18=0
- binary_bit158_sigmoid_f7_fixed__seed42 custom_v9_sigmoid: acc=0.8496325301204819 auc=0.9242838920077538 latency=9 cycles II=1 LUT=92371 FF=15799 DSP=0 BRAM18=1
- binary_bit158_topo_sigmoid_f7_fixed__seed42 custom_v9_logits: acc=0.8488132530120482 auc=0.9225514387786714 latency=9 cycles II=1 LUT=92183 FF=19367 DSP=2 BRAM18=0
- binary_bit158_topo_sigmoid_f7_fixed__seed42 custom_v9_sigmoid: acc=0.8488132530120482 auc=0.9225514387786714 latency=8 cycles II=1 LUT=92003 FF=18837 DSP=0 BRAM18=1
- binary_xgboost_bdt_d4_100__seed42 conifer_tree: acc=0.8474819277108434 auc=0.9207411789830275 latency=227 cycles II=228 LUT=15526 FF=23715 DSP=207 BRAM18=0
- binary_xgboost_bdt_d4_100__seed42 conifer_unrolled: acc=0.8474819277108434 auc=0.9207411789830275 latency=4 cycles II=1 LUT=73977 FF=2806 DSP=0 BRAM18=0
- binary_mlp_baseline_64_32_32__seed43 hls4ml_latency_rf1: acc=0.8594156626506024 auc=0.9323458154730129 latency=12 cycles II=1 LUT=183747 FF=32245 DSP=3424 BRAM18=1
- binary_mlp_topo_128_32__seed43 hls4ml_latency_rf1: acc=0.8593072289156627 auc=0.9325106022973398 latency=10 cycles II=1 LUT=269261 FF=45440 DSP=5000 BRAM18=1
- binary_qkeras_mlp_64_32_32_b7__seed43 hls4ml_latency_rf1: acc=0.8597469879518073 auc=0.9324038079534698 latency=9 cycles II=1 LUT=134848 FF=10578 DSP=676 BRAM18=1
- binary_qkeras_topo_128_32_b7__seed43 hls4ml_latency_rf1: acc=0.8597771084337349 auc=0.932652113663417 latency=8 cycles II=1 LUT=189386 FF=16182 DSP=781 BRAM18=1
- binary_qkeras_mlp_binary_64_32_32__seed43 hls4ml_latency_rf1: acc=0.8280783132530121 auc=0.8945885098994748 latency=20 cycles II=1 LUT=61590 FF=16717 DSP=0 BRAM18=1
- binary_qkeras_topo_binary_128_32__seed43 hls4ml_latency_rf1: acc=0.8292710843373494 auc=0.893559272212676 latency=22 cycles II=1 LUT=91439 FF=25934 DSP=0 BRAM18=1
- binary_qkeras_mlp_ternary_64_32_32__seed43 hls4ml_latency_rf1: acc=0.8329277108433735 auc=0.9012144136735029 latency=10 cycles II=1 LUT=25888 FF=6961 DSP=0 BRAM18=1
- binary_qkeras_topo_ternary_128_32__seed43 hls4ml_latency_rf1: acc=0.8401987951807229 auc=0.9072378286012781 latency=12 cycles II=1 LUT=32009 FF=15364 DSP=0 BRAM18=1
- binary_hgq_mlp_64_32_32__seed43 hls4ml_latency_rf1: acc=0.8541867469879518 auc=0.9278454792031832 latency=7 cycles II=1 LUT=8939 FF=719 DSP=0 BRAM18=1
- binary_hgq_topo_128_32__seed43 hls4ml_latency_rf1: acc=0.8540602409638555 auc=0.9281584322086689 latency=6 cycles II=1 LUT=8498 FF=661 DSP=0 BRAM18=1
- binary_bitnet_sigmoid_f7_fixed__seed43 custom_v9_logits: acc=0.8457530120481928 auc=0.9180566692505001 latency=7 cycles II=1 LUT=89633 FF=13125 DSP=617 BRAM18=0
- binary_bitnet_sigmoid_f7_fixed__seed43 custom_v9_sigmoid: acc=0.8457530120481928 auc=0.9180566692505001 latency=8 cycles II=1 LUT=89787 FF=13061 DSP=615 BRAM18=1
- binary_bitnet_sigmoid_f7_fixed__seed43 hls4ml_parent_v26_patch_sigmoid: acc=0.8457530120481928 auc=0.9180566692505001 latency=12 cycles II=1 LUT=88120 FF=16082 DSP=0 BRAM18=1
- binary_bitnet_topo_sigmoid_f7_fixed__seed43 custom_v9_logits: acc=0.845855421686747 auc=0.9191553739984853 latency=6 cycles II=1 LUT=122024 FF=15161 DSP=763 BRAM18=0
- binary_bitnet_topo_sigmoid_f7_fixed__seed43 custom_v9_sigmoid: acc=0.845855421686747 auc=0.9191553739984853 latency=6 cycles II=1 LUT=122175 FF=18736 DSP=761 BRAM18=1
- binary_bit158_sigmoid_f7_fixed__seed43 custom_v9_logits: acc=0.8467409638554216 auc=0.9218601438091953 latency=9 cycles II=1 LUT=90920 FF=15666 DSP=2 BRAM18=0
- binary_bit158_sigmoid_f7_fixed__seed43 custom_v9_sigmoid: acc=0.8467409638554216 auc=0.9218601438091953 latency=9 cycles II=1 LUT=91044 FF=15047 DSP=0 BRAM18=1
- binary_bit158_topo_sigmoid_f7_fixed__seed43 custom_v9_logits: acc=0.8504638554216868 auc=0.9235433439835938 latency=9 cycles II=1 LUT=100326 FF=20787 DSP=2 BRAM18=0
- binary_bit158_topo_sigmoid_f7_fixed__seed43 custom_v9_sigmoid: acc=0.8504638554216868 auc=0.9235433439835938 latency=8 cycles II=1 LUT=100149 FF=20179 DSP=0 BRAM18=1
- binary_xgboost_bdt_d4_100__seed43 conifer_tree: acc=0.8470722891566265 auc=0.9207803852236567 latency=227 cycles II=228 LUT=15654 FF=24171 DSP=212 BRAM18=0
- binary_xgboost_bdt_d4_100__seed43 conifer_unrolled: acc=0.8470722891566265 auc=0.9207803852236567 latency=4 cycles II=1 LUT=74181 FF=2826 DSP=0 BRAM18=0
- binary_mlp_baseline_64_32_32__seed44 hls4ml_latency_rf1: acc=0.8593614457831326 auc=0.9325757144462619 latency=12 cycles II=1 LUT=183643 FF=32089 DSP=3463 BRAM18=1
- binary_mlp_topo_128_32__seed44 hls4ml_latency_rf1: acc=0.8593132530120482 auc=0.931833084695634 latency=11 cycles II=1 LUT=270436 FF=48971 DSP=4962 BRAM18=1
- binary_qkeras_mlp_64_32_32_b7__seed44 hls4ml_latency_rf1: acc=0.8591927710843373 auc=0.9324786238076672 latency=9 cycles II=1 LUT=136529 FF=10721 DSP=682 BRAM18=1
- binary_qkeras_topo_128_32_b7__seed44 hls4ml_latency_rf1: acc=0.8597530120481928 auc=0.9322459758655477 latency=8 cycles II=1 LUT=188000 FF=15772 DSP=727 BRAM18=1
- binary_qkeras_mlp_binary_64_32_32__seed44 hls4ml_latency_rf1: acc=0.8303614457831325 auc=0.8819696778225912 latency=22 cycles II=1 LUT=62327 FF=17871 DSP=0 BRAM18=1
- binary_qkeras_topo_binary_128_32__seed44 hls4ml_latency_rf1: acc=0.8368072289156626 auc=0.9008506205312448 latency=21 cycles II=1 LUT=91392 FF=25517 DSP=0 BRAM18=1
- binary_qkeras_mlp_ternary_64_32_32__seed44 hls4ml_latency_rf1: acc=0.8360060240963856 auc=0.9024909446317996 latency=12 cycles II=1 LUT=27009 FF=8051 DSP=0 BRAM18=1
- binary_qkeras_topo_ternary_128_32__seed44 hls4ml_latency_rf1: acc=0.8374156626506024 auc=0.9080463169075836 latency=12 cycles II=1 LUT=30710 FF=14516 DSP=0 BRAM18=1
- binary_hgq_mlp_64_32_32__seed44 hls4ml_latency_rf1: acc=0.8547349397590361 auc=0.9276077925221098 latency=6 cycles II=1 LUT=7425 FF=551 DSP=0 BRAM18=1
- binary_hgq_topo_128_32__seed44 hls4ml_latency_rf1: acc=0.8526506024096385 auc=0.9271416155834051 latency=6 cycles II=1 LUT=7863 FF=637 DSP=0 BRAM18=1
- binary_bitnet_sigmoid_f7_fixed__seed44 custom_v9_logits: acc=0.8451867469879518 auc=0.9178251329489482 latency=7 cycles II=1 LUT=87544 FF=15798 DSP=580 BRAM18=0
- binary_bitnet_sigmoid_f7_fixed__seed44 custom_v9_sigmoid: acc=0.8451867469879518 auc=0.9178251329489482 latency=8 cycles II=1 LUT=87701 FF=13543 DSP=578 BRAM18=1
- binary_bitnet_sigmoid_f7_fixed__seed44 hls4ml_parent_v26_patch_sigmoid: acc=0.8451867469879518 auc=0.9178251329489482 latency=12 cycles II=1 LUT=88291 FF=16938 DSP=0 BRAM18=1
- binary_bitnet_topo_sigmoid_f7_fixed__seed44 custom_v9_logits: acc=0.8448493975903615 auc=0.9178488895412658 latency=6 cycles II=1 LUT=123775 FF=15295 DSP=788 BRAM18=0
- binary_bitnet_topo_sigmoid_f7_fixed__seed44 custom_v9_sigmoid: acc=0.8448493975903615 auc=0.9178488895412658 latency=6 cycles II=1 LUT=123926 FF=18657 DSP=786 BRAM18=1
- binary_bit158_sigmoid_f7_fixed__seed44 custom_v9_logits: acc=0.8504397590361445 auc=0.9235886700916927 latency=9 cycles II=1 LUT=92261 FF=16152 DSP=2 BRAM18=0
- binary_bit158_sigmoid_f7_fixed__seed44 custom_v9_sigmoid: acc=0.8504397590361445 auc=0.9235886700916927 latency=9 cycles II=1 LUT=92382 FF=15505 DSP=0 BRAM18=1
- binary_bit158_topo_sigmoid_f7_fixed__seed44 custom_v9_logits: acc=0.8507289156626506 auc=0.92393848693064 latency=9 cycles II=1 LUT=107886 FF=21723 DSP=2 BRAM18=0
- binary_bit158_topo_sigmoid_f7_fixed__seed44 custom_v9_sigmoid: acc=0.8507289156626506 auc=0.92393848693064 latency=8 cycles II=1 LUT=107709 FF=20965 DSP=0 BRAM18=1
- binary_xgboost_bdt_d4_100__seed44 conifer_tree: acc=0.8470301204819277 auc=0.9207318808906974 latency=227 cycles II=228 LUT=15148 FF=23356 DSP=202 BRAM18=0
- binary_xgboost_bdt_d4_100__seed44 conifer_unrolled: acc=0.8470301204819277 auc=0.9207318808906974 latency=4 cycles II=1 LUT=73970 FF=2780 DSP=0 BRAM18=0

## Failures
- None recorded.

## Preliminary BitNet Interpretation
The binary track is designed to test the strongest BitNet case: a one-logit sigmoid model where cumulative alpha scaling can be folded into the final sigmoid/table implementation. Compare `bitnet_hls4ml_v26_sigmoid` and `bitnet_custom_v9_sigmoid` rows against dense, QKeras, HGQ and Conifer rows at similar accuracy. If BitNet is not Pareto competitive here, the benchmark conclusion should be that alpha folding alone is not enough for this HLF task and toolchain; if it is competitive only in the sigmoid endpoint, the advantage is endpoint-specific rather than a general multiclass/logits advantage.
