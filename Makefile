SHELL := /bin/bash
PY ?= python
RESEARCH ?= /home/jokubasb/protein_protein
VORONOTA ?= /home/jokubasb/voronota_1.29.4781/expansion_js
SIF ?= jbbind.sif
PORT ?= 8000

export PATH := $(PATH):$(VORONOTA)

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------ develop

.PHONY: serve
serve: ## run the web app locally on $(PORT)
	$(PY) -m uvicorn jbbind.main:app --host 127.0.0.1 --port $(PORT) --reload

.PHONY: test
test: ## run the test suite (no research repo needed)
	$(PY) -m pytest tests/ -p no:warnings

.PHONY: info
info: ## show device, checkpoints and cache state
	$(PY) -m jbbind.cli info

# ------------------------------------------------------------------ models

.PHONY: export-models
export-models: ## copy checkpoints + build MANIFEST.json / METRICS.json from the research repo
	$(PY) scripts/export_models.py --repo $(RESEARCH)

.PHONY: fixtures
fixtures: ## regenerate the golden parity fixtures (needs the research environment)
	cd $(RESEARCH) && $(PY) $(CURDIR)/scripts/make_parity_fixtures.py

# ------------------------------------------------------------- verification

.PHONY: verify
verify: verify-tool verify-renumbering ## run every verification against the research data

.PHONY: verify-tool
verify-tool: ## Tier 3a: forked voronota script reproduces the training features exactly
	$(PY) scripts/verify_tool_reproduction.py --n 40

.PHONY: verify-renumbering
verify-renumbering: ## Tier 3c: SEQRES renumbering matches the training numbering
	$(PY) scripts/check_renumbering.py --n 200 --workers 8

.PHONY: verify-provenance
verify-provenance: ## Tier 3b: how much RCSB-vs-PPI3D provenance shifts the predictions
	$(PY) scripts/verify_provenance_shift.py --n 50

.PHONY: diff-tool
diff-tool: ## audit the delta between the forked voronota script and the original
	@diff -u tools/reference/extract-and-describe-receptor-protein.orig \
	         tools/describe-receptor-chain || true

.PHONY: palette
palette: ## re-validate the chart palette (colour is computable — compute it)
	@echo "--- architecture series, light (adjacent pairlist: bars, lines)"
	@$(PY) scripts/validate_palette.py "#2a78d6,#eb6834,#1baf7a,#eda100" --mode light
	@echo "--- architecture series, dark"
	@$(PY) scripts/validate_palette.py "#3987e5,#d95926,#199e70,#c98500" --mode dark

# ------------------------------------------------------------- containers

.PHONY: docker
docker: ## build the CPU docker image
	docker build -t jbbind:cpu .

.PHONY: docker-cuda
docker-cuda: ## build the CUDA docker image
	docker build -t jbbind:cuda \
	  --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
	  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 .

.PHONY: sif
sif: ## build the Apptainer image (works without a docker daemon)
	apptainer build --fakeroot --force $(SIF) apptainer.def

.PHONY: sif-run
sif-run: ## run the web app from the Apptainer image
	mkdir -p .apptainer-data
	apptainer run --cleanenv --bind $(CURDIR)/.apptainer-data:/data $(SIF)

.PHONY: sif-test
sif-test: ## run the test suite inside the Apptainer image
	apptainer test --cleanenv $(SIF)

.PHONY: clean
clean: ## remove build and cache artefacts
	rm -rf .apptainer-data $(SIF) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
