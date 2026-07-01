# Stage 1: Predictive Performance

## Scope

- Dataset: OpenML 42468 (`hls4ml_lhc_jets_hlf`), 830,000 jets.
- Inputs: 16 scalar high-level features.
- Classes: gluon, quark, W, Z, top.
- Fixed stratified split: 531,200 train, 132,800 validation, 166,000 test.
- Split seed: 42. StandardScaler fitted on training indices only.
- Primary evidence: three training seeds (42, 43, 44).
- Accuracy, macro one-vs-rest AUC, macro F1, per-class AUC/precision/recall/F1, and confusion matrices are recorded.
- Extended evaluation records average precision, cross-entropy, Brier score, calibration error, confidence coverage, and q/g-background proxy trigger rates.

Confidence intervals use a two-sided 95% Student t interval across training seeds. With only three seeds they are descriptive, not strong significance claims.

## Repeated-Seed Results

| Model | Backend | Accuracy mean +/- SD (%) | Macro AUC mean +/- SD | Macro F1 | Delta accuracy vs MLP (pp) | Parameters |
|---|---|---:|---:|---:|---:|---:|
| jetformer_hlf | pytorch | 76.953 +/- 0.022 | 0.94427 +/- 0.00030 | 0.77241 | 0.311 | 59621 |
| multihead_attention_hlf | pytorch | 76.866 +/- 0.070 | 0.94387 +/- 0.00034 | 0.77135 | 0.223 | 18917 |
| linformer_hlf | pytorch | 76.836 +/- 0.025 | 0.94357 +/- 0.00014 | 0.77083 | 0.194 | 18765 |
| mlp_mixer_hlf | pytorch | 76.714 +/- 0.011 | 0.94298 +/- 0.00015 | 0.76975 | 0.072 | 12037 |
| qkeras_mlp_b12 | qkeras | 76.631 +/- 0.043 | 0.94232 +/- 0.00013 | 0.76900 | -0.011 | 4389 |
| qkeras_mlp_b10 | qkeras | 76.628 +/- 0.010 | 0.94226 +/- 0.00007 | 0.76895 | -0.015 | 4389 |
| qkeras_mlp_b8 | qkeras | 76.638 +/- 0.051 | 0.94221 +/- 0.00014 | 0.76900 | -0.004 | 4389 |
| qkeras_mlp_b7 | qkeras | 76.634 +/- 0.032 | 0.94218 +/- 0.00024 | 0.76928 | -0.008 | 4389 |
| mlp_topo | pytorch | 76.634 +/- 0.020 | 0.94215 +/- 0.00019 | 0.76952 | -0.008 | 6469 |
| mlp_baseline | pytorch | 76.642 +/- 0.026 | 0.94208 +/- 0.00030 | 0.76942 | 0.000 | 4389 |
| qkeras_mlp | qkeras | 76.579 +/- 0.048 | 0.94176 +/- 0.00020 | 0.76838 | -0.063 | 4389 |
| qkeras_mlp_b5 | qkeras | 76.400 +/- 0.068 | 0.94081 +/- 0.00012 | 0.76704 | -0.242 | 4389 |
| hgq_mlp | hgq | 76.143 +/- 0.014 | 0.93924 +/- 0.00016 | 0.76460 | -0.499 | 8804 |
| binary_448_224_224 | pytorch | 75.651 +/- 0.034 | 0.93662 +/- 0.00041 | 0.75929 | -0.991 | 159717 |
| ternary_128_64_64_64 | pytorch | 75.386 +/- 0.060 | 0.93578 +/- 0.00036 | 0.75715 | -1.257 | 19077 |
| deepsets_hlf | pytorch | 75.240 +/- 0.171 | 0.93412 +/- 0.00081 | 0.75575 | -1.402 | 5349 |
| bit158_mlp_f7_fixed | pytorch | 74.207 +/- 0.202 | 0.92915 +/- 0.00039 | 0.74489 | -2.436 | 4389 |
| bit158_topo_f7_fixed | pytorch | 74.125 +/- 0.099 | 0.92821 +/- 0.00053 | 0.74433 | -2.517 | 6469 |
| bitnet_mlp_f5_fixed | pytorch | 72.154 +/- 0.089 | 0.91613 +/- 0.00074 | 0.72397 | -4.488 | 4389 |
| bitnet_mlp_f10_fixed | pytorch | 72.117 +/- 0.134 | 0.91581 +/- 0.00159 | 0.72386 | -4.526 | 4389 |
| bitnet_mlp_f12_fixed | pytorch | 72.045 +/- 0.444 | 0.91556 +/- 0.00158 | 0.72289 | -4.597 | 4389 |
| bitnet_mlp_f8_fixed | pytorch | 72.284 +/- 0.041 | 0.91509 +/- 0.00020 | 0.72490 | -4.359 | 4389 |
| bitnet_mlp_f7_power2 | pytorch | 72.249 +/- 0.438 | 0.91482 +/- 0.00144 | 0.72448 | -4.394 | 4389 |
| bitnet_mlp_f7_fixed | pytorch | 71.979 +/- 0.336 | 0.91389 +/- 0.00022 | 0.72240 | -4.663 | 4389 |
| bitnet_topo_f7_fixed | pytorch | 72.080 +/- 0.197 | 0.91354 +/- 0.00204 | 0.72276 | -4.563 | 6469 |

## Trigger And Calibration Summary

The trigger columns use the combined `W+Z+top` score against q/g background.

| Model | ECE | Macro AP | Proxy rate at 80% signal efficiency (kHz) | Signal efficiency at 100 kHz |
|---|---:|---:|---:|---:|
| jetformer_hlf | 0.00575 | 0.83710 | 2091.3 | 0.4640 |
| multihead_attention_hlf | 0.00592 | 0.83603 | 2112.3 | 0.4645 |
| linformer_hlf | 0.00479 | 0.83577 | 2137.7 | 0.4566 |
| mlp_mixer_hlf | 0.00634 | 0.83432 | 2154.2 | 0.4623 |
| qkeras_mlp_b12 | 0.00808 | 0.83302 | 2175.2 | 0.4511 |
| qkeras_mlp_b10 | 0.00795 | 0.83282 | 2177.9 | 0.4505 |
| qkeras_mlp_b8 | 0.00713 | 0.83289 | 2187.3 | 0.4511 |
| qkeras_mlp_b7 | 0.00709 | 0.83271 | 2176.3 | 0.4518 |
| mlp_topo | 0.00533 | 0.83294 | 2196.8 | 0.4537 |
| mlp_baseline | 0.00527 | 0.83261 | 2190.6 | 0.4509 |
| qkeras_mlp | 0.00764 | 0.83161 | 2180.8 | 0.4507 |
| qkeras_mlp_b5 | 0.00769 | 0.82941 | 2225.1 | 0.4360 |
| hgq_mlp | 0.00536 | 0.82636 | 2374.8 | 0.4305 |
| binary_448_224_224 | 0.01319 | 0.82013 | 2434.6 | 0.4123 |
| ternary_128_64_64_64 | 0.01176 | 0.81840 | 2424.1 | 0.4105 |
| deepsets_hlf | 0.00507 | 0.81559 | 2641.7 | 0.4247 |
| bit158_mlp_f7_fixed | 0.01067 | 0.80364 | 2785.5 | 0.3591 |
| bit158_topo_f7_fixed | 0.01800 | 0.80154 | 2863.8 | 0.3495 |
| bitnet_mlp_f5_fixed | 0.01038 | 0.77456 | 3192.8 | 0.2967 |
| bitnet_mlp_f10_fixed | 0.01393 | 0.77494 | 3149.3 | 0.3328 |
| bitnet_mlp_f12_fixed | 0.00939 | 0.77482 | 3231.4 | 0.3115 |
| bitnet_mlp_f8_fixed | 0.02174 | 0.77405 | 3273.3 | 0.3076 |
| bitnet_mlp_f7_power2 | 0.01025 | 0.77167 | 3135.6 | 0.3196 |
| bitnet_mlp_f7_fixed | 0.01736 | 0.77145 | 3414.9 | 0.2997 |
| bitnet_topo_f7_fixed | 0.02703 | 0.77196 | 3278.1 | 0.2966 |

## Exploratory Single-Seed Results

These runs are useful for selecting configurations but are not equivalent to the repeated-seed comparison.

| Model | Accuracy (%) | Macro AUC | Macro F1 |
|---|---:|---:|---:|

## Interpretation Rules

- The primary BitNet comparisons are topology-matched float MLP versus BitNet/BitNet-1.58, plus QKeras, HGQ, binary, and ternary dense models.
- Transformer, Linformer, MLP-Mixer, and Deep Sets entries are HLF adaptations, not particle-constituent reproductions.
- HGQ follows its 200-epoch reference schedule; the common suite uses 20 epochs.
- Framework checkpoint bytes do not represent packed low-bit storage.
- CPU latency is retained only as a software sanity measurement. Resource use, throughput, power, and synthesized latency belong to stage two.
- Proxy trigger rate is `q/g background efficiency * 31,037.856 kHz` for 2760 colliding bunches. It is not a physical minimum-bias rate.
- Physical trigger rates require an unbiased minimum-bias sample with the intended online preselection and pileup conditions.
- Confidence coverage measures selective-classification coverage, not detector or kinematic phase-space coverage.
- Physics robustness to pileup, detector response, calibration shifts, and changing run conditions cannot be established from this dataset alone.
- A publication-grade paired bootstrap of AUC differences would require retaining per-event predictions; current uncertainty is across training seeds.

## Artifact Coverage

- Canonical non-ONNX model runs: 76.
- ONNX validation records: 7.
- Every legacy run includes accuracy, macro and per-class AUC, confusion matrix, parameter count, model artifact size, and CPU latency.
- Re-evaluated runs additionally include trigger operating points, average precision, proper scoring rules, calibration, and confidence coverage.
- Per-class precision, recall, F1, macro F1, balanced accuracy, seed variation, confidence intervals, and paired MLP deltas are derived in `stage1_performance.csv`.

## Stage-One Readiness

- Complete: discrimination, class-wise behavior, calibration, confidence coverage, seed stability, topology-matched comparisons, and q/g proxy trigger curves.
- Not available from OpenML 42468: physical minimum-bias trigger rate, pileup robustness, detector-systematic robustness, and kinematic coverage versus jet pT/eta.
- These unavailable items require additional representative datasets, not further processing of the existing HLF table.

Detailed machine-readable output: `results/stage1_performance.csv` and `results/stage1_performance.json`.
