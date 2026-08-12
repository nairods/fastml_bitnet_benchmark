PYTHON ?= python

.PHONY: reproduce paper main secondary figures

reproduce: paper

paper:
	$(PYTHON) scripts/reproduce_paper.py

main:
	$(PYTHON) scripts/run_binary_benchmark_workflow.py --class-mode binary_qg_vs_wzt --namespace binary --log-subdir binary_benchmark --seeds 42 43 44

secondary:
	$(PYTHON) scripts/run_binary_benchmark_workflow.py --class-mode binary_top_vs_qg --namespace binary_topqg --log-subdir binary_topqg_benchmark --seeds 42 43 44

figures:
	$(PYTHON) scripts/generate_benchmark_artifacts.py
