# Overnight Benchmark Report

All FPGA numbers in this report are C-synthesis estimates unless a row explicitly states otherwise. Current BitNet custom-native results report `place_and_route_status=not_run`.

## Completed Models
- None fully complete across weights, metrics, ONNX, HLS project, synthesis report, and parsed report.

## Failed Or Blocked Models
- deepsets_hlf and mlp_mixer_hlf have lowering packages only; custom operator lowering is still required before synthesis.
- Transformer-style HLF models are ignored for this benchmark unless explicitly revived.

## Missing Or Incomplete Rows
- mlp_baseline seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- mlp_baseline seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- mlp_baseline seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- qkeras_mlp seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- qkeras_mlp seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- qkeras_mlp seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- hgq_mlp seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- hgq_mlp seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- hgq_mlp seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_topo_f7_fixed seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_topo_f7_fixed seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_topo_f7_fixed seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bit158_topo_f7_fixed seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bit158_topo_f7_fixed seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bit158_topo_f7_fixed seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bit158_mlp_f7_fixed seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bit158_mlp_f7_fixed seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bit158_mlp_f7_fixed seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- xgboost_bdt seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- xgboost_bdt seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- xgboost_bdt seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f5_fixed seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f5_fixed seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f5_fixed seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f7_fixed seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f7_fixed seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f7_fixed seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f8_fixed seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f8_fixed seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f8_fixed seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f10_fixed seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f10_fixed seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f10_fixed seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f12_fixed seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f12_fixed seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f12_fixed seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f7_power2 seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f7_power2 seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- bitnet_mlp_f7_power2 seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- deepsets_hlf seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- deepsets_hlf seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- deepsets_hlf seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- mlp_mixer_hlf seed 42: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- mlp_mixer_hlf seed 43: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False
- mlp_mixer_hlf seed 44: weights=False, metrics=False, onnx=False, hls=False, synth=False, parsed=False

## Best Current Rows
These best rows are restricted to the requested Tier 1-3 OpenML multi-class benchmark models.
- Best software accuracy: n/a accuracy=n/a
- Best C-synth latency: n/a latency_cycles=n/a
- Best LUT efficiency: n/a accuracy_per_lut=n/a

## Pareto Frontier Candidates
- No rows have accuracy, latency, and LUT simultaneously.

## Preliminary Paper Observations
- The current trustworthy FPGA numbers are C-synthesis estimates, not place-and-route timing/resource numbers.
- BitNet custom-native logits-only kernels are the most complete synthesized family in the current repository.
- QKeras and HGQ have trained weights, test metrics, and representative generated HLS projects, but no completed synthesis reports in the current artifacts.
- The XGBoost BDT reaches MLP-like software accuracy for seed 42, but Conifer/Vitis synthesis is not yet producing a usable report for the large tuned ensemble.
- Seed statistics should be interpreted by implementation mode; mixing tree and unrolled BitNet runs can hide implementation effects.
