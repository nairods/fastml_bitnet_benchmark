PYTHON ?= python

.PHONY: reproduce figures check train-primary train-secondary train-multiclass

reproduce: figures

figures:
	$(PYTHON) scripts/generate_benchmark_artifacts.py

check:
	$(PYTHON) scripts/generate_benchmark_artifacts.py --check

train-primary:
	$(PYTHON) scripts/run_benchmark.py --task qg_vs_wzt

train-secondary:
	$(PYTHON) scripts/run_benchmark.py --task qg_vs_top

train-multiclass:
	$(PYTHON) scripts/run_benchmark.py --task multiclass
