# Results Check

This file compares the expected extended-abstract values against the committed repository values in `results/abstract_main_binary_table.csv`.

All values below are for the primary `q/g vs W/Z/top` task. Neural rows use the `64-32-32` architecture unless otherwise stated. Values are mean +/- standard deviation over seeds 42, 43, 44.

## Primary Table Check

| Model | Metric | Expected | Repository | Status |
| --- | --- | ---: | ---: | --- |
| Dense MLP | accuracy | 0.8595 +/- 0.0003 | 0.85954 +/- 0.00026 | pass |
| Dense MLP | AUC | 0.9325 +/- 0.0001 | 0.93247 +/- 0.00012 | pass |
| Dense MLP | signal eff. @ 1% FPR | 0.5857 +/- 0.0021 | 0.58573 +/- 0.00213 | pass |
| Dense MLP | latency | 12.0 +/- 0.0 | 12.00 +/- 0.00 | pass |
| Dense MLP | LUT | 183.5k +/- 0.3k | 183.5k +/- 0.3k | pass |
| Dense MLP | DSP | 3455 | 3455.33 | pass |
| QKeras fixed b7 | accuracy | 0.8595 +/- 0.0003 | 0.85951 +/- 0.00029 | pass |
| QKeras fixed b7 | AUC | 0.9324 +/- 0.0001 | 0.93239 +/- 0.00010 | pass |
| QKeras fixed b7 | signal eff. @ 1% FPR | 0.5813 +/- 0.0030 | 0.58132 +/- 0.00302 | pass |
| QKeras fixed b7 | latency | 9.0 +/- 0.0 | 9.00 +/- 0.00 | pass |
| QKeras fixed b7 | LUT | 135.4k +/- 1.0k | 135.4k +/- 1.0k | pass |
| QKeras fixed b7 | DSP | 665 | 664.67 | pass |
| HGQ | accuracy | 0.8538 +/- 0.0012 | 0.85380 +/- 0.00118 | pass |
| HGQ | AUC | 0.9276 +/- 0.0002 | 0.92761 +/- 0.00024 | pass |
| HGQ | signal eff. @ 1% FPR | 0.5711 +/- 0.0012 | 0.57113 +/- 0.00123 | pass |
| HGQ | latency | 6.3 +/- 0.6 | 6.33 +/- 0.58 | pass |
| HGQ | LUT | 8.0k +/- 0.8k | 8.0k +/- 0.8k | pass |
| HGQ | DSP | 0 | 0.00 | pass |
| QKeras binary | accuracy | 0.8294 +/- 0.0012 | 0.82944 +/- 0.00120 | pass |
| QKeras binary | AUC | 0.8886 +/- 0.0063 | 0.88860 +/- 0.00633 | pass |
| QKeras binary | signal eff. @ 1% FPR | 0.4319 +/- 0.1480 | 0.43192 +/- 0.14802 | pass |
| QKeras binary | latency | 20.0 +/- 2.0 | 20.00 +/- 2.00 | pass |
| QKeras binary | LUT | 62.0k +/- 0.4k | 62.0k +/- 0.4k | pass |
| QKeras binary | DSP | 0 | 0.00 | pass |
| QKeras ternary | accuracy | 0.8355 +/- 0.0023 | 0.83545 +/- 0.00230 | pass |
| QKeras ternary | AUC | 0.9025 +/- 0.0012 | 0.90246 +/- 0.00124 | pass |
| QKeras ternary | signal eff. @ 1% FPR | 0.4138 +/- 0.0127 | 0.41383 +/- 0.01268 | pass |
| QKeras ternary | latency | 11.7 +/- 1.5 | 11.67 +/- 1.53 | pass |
| QKeras ternary | LUT | 28.3k +/- 3.2k | 28.3k +/- 3.2k | pass |
| QKeras ternary | DSP | 0 | 0.00 | pass |
| BitNet binary | accuracy | 0.8454 +/- 0.0003 | 0.84538 +/- 0.00032 | pass |
| BitNet binary | AUC | 0.9176 +/- 0.0006 | 0.91758 +/- 0.00063 | pass |
| BitNet binary | signal eff. @ 1% FPR | 0.4975 +/- 0.0103 | 0.49753 +/- 0.01025 | pass |
| BitNet binary | latency | 12.0 +/- 0.0 | 12.00 +/- 0.00 | pass |
| BitNet binary | LUT | 88.3k +/- 0.3k | 88.3k +/- 0.3k | pass |
| BitNet binary | DSP | 0 | 0.00 | pass |
| Bit158 sparse ternary | accuracy | 0.8489 +/- 0.0019 | 0.84894 +/- 0.00194 | pass |
| Bit158 sparse ternary | AUC | 0.9232 +/- 0.0013 | 0.92324 +/- 0.00125 | pass |
| Bit158 sparse ternary | signal eff. @ 1% FPR | 0.5345 +/- 0.0076 | 0.53454 +/- 0.00762 | pass |
| Bit158 sparse ternary | latency | 9.0 +/- 0.0 | 9.00 +/- 0.00 | pass |
| Bit158 sparse ternary | LUT | 91.9k +/- 0.8k | 91.9k +/- 0.8k | pass |
| Bit158 sparse ternary | DSP | 0 | 0.00 | pass |
| BDT unrolled | accuracy | 0.8472 +/- 0.0003 | 0.84719 +/- 0.00025 | pass |
| BDT unrolled | AUC | 0.92075 +/- 0.00003 | 0.92075 +/- 0.00003 | pass |
| BDT unrolled | signal eff. @ 1% FPR | 0.5612 +/- 0.0006 | 0.56123 +/- 0.00061 | pass |
| BDT unrolled | latency | 4.0 +/- 0.0 | 4.00 +/- 0.00 | pass |
| BDT unrolled | LUT | 74.0k +/- 0.1k | 74.0k +/- 0.1k | pass |
| BDT unrolled | DSP | 0 | 0.00 | pass |

## Consistency Notes

- The expected values match the committed repository values within ordinary rounding.
- `signal_eff_at_1pct_fpr` is present and consistent in the abstract tables, but cannot be recomputed from the shipped repository because raw prediction scores are not committed.
- Hardware numbers are HLS C-synthesis estimates only.
- The `BitNet binary` primary table row uses `hls4ml_parent_v26_patch_sigmoid` for the stable DSP-free sigmoid-boundary result.
- The `Bit158 sparse ternary` primary table row uses `custom_v9_sigmoid`.

## Robustness Snapshot

The secondary `q/g vs top` table is present in `results/abstract_secondary_top_table.csv`. It supports the same qualitative ordering:

- QKeras fixed b7 is dense-like in AUC.
- HGQ remains the lowest-LUT neural point.
- BitNet and Bit158 outperform naive QKeras binary and ternary in AUC.
- The unrolled BDT remains the lowest-latency implementation at 4 cycles.
