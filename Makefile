PYTHON ?= python
PROFILE ?= 20-epochs

.PHONY: reproduce reproduce-all figures check check-all collect train-primary train-secondary train-multiclass

reproduce: figures

reproduce-all:
	$(MAKE) reproduce PROFILE=20-epochs
	$(MAKE) reproduce PROFILE=200-epochs

figures:
	$(PYTHON) scripts/generate_benchmark_artifacts.py --profile $(PROFILE)

check:
	$(PYTHON) scripts/generate_benchmark_artifacts.py --profile $(PROFILE) --check

check-all:
	$(MAKE) check PROFILE=20-epochs
	$(MAKE) check PROFILE=200-epochs

collect:
	$(PYTHON) scripts/collect_benchmark_records.py --profile $(PROFILE)

train-primary:
	$(PYTHON) scripts/run_benchmark.py --profile $(PROFILE) --task qg_vs_wzt

train-secondary:
	$(PYTHON) scripts/run_benchmark.py --profile $(PROFILE) --task qg_vs_top

train-multiclass:
	$(PYTHON) scripts/run_benchmark.py --profile $(PROFILE) --task multiclass
