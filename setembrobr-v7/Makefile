PYTHON ?= python3
CONFIG ?= config.json
SOURCE_PKL ?= /Users/aluisioamorim/Downloads/corpus/SetembroBR-v7-min.pkl
ARCHIVE_ROOT ?= /run/media/aluisioamorim/SETEMBROBR/setembrobr-v6-restricted-archive
OUTPUT_DIR ?= .work/output
TEMPORARY_DIR ?= .work/tmp

FEDORA_HOST ?= fedora
FEDORA_PROJECT_DIR ?= /home/aluisioamorim/experiments/setembrobr-v7-qwen-logreg
FEDORA_SOURCE_PKL ?= /home/aluisioamorim/setembro_br_v7/SetembroBR-v7-min.pkl
FEDORA_ARCHIVE_ROOT ?= /run/media/aluisioamorim/SETEMBROBR/setembrobr-v6-restricted-archive
FEDORA_OUTPUT_DIR ?= $(FEDORA_PROJECT_DIR)/.work/output
FEDORA_TEMPORARY_DIR ?= /home/aluisioamorim/tmp/setembrobr-v7-qwen-logreg
FEDORA_UNIT ?= setembrobr-v7-qwen-logreg

ENSEMBLE_CONFIG ?= ensemble-config.json
ENSEMBLE_OUTPUT_DIR ?= .work/ensemble
FEATURE_HELPER ?= ../scripts/raw_ternary_prepare_setembrobr.py
FEDORA_ENSEMBLE_PYTHON ?= /home/aluisioamorim/codex-runs/depression-prediction-setembrobr-raw-embeddings/.venv/bin/python
FEDORA_ENSEMBLE_HOST ?= fedora.local
FEDORA_ENSEMBLE_SSH_OPTS ?= -o HostKeyAlias=192.168.15.136
FEDORA_ENSEMBLE_OUTPUT_DIR ?= $(FEDORA_PROJECT_DIR)/.work/ensemble
FEDORA_ENSEMBLE_TEMPORARY_DIR ?= /home/aluisioamorim/tmp/setembrobr-v7-ensemble
FEDORA_ENSEMBLE_UNIT ?= setembrobr-v7-ensemble

PIPELINE = $(PYTHON) scripts/setembrobr_v7_qwen_logreg.py --config $(CONFIG) --output-dir $(OUTPUT_DIR)
ENSEMBLE_PIPELINE = $(PYTHON) scripts/setembrobr_v7_ensemble.py --config $(ENSEMBLE_CONFIG) \
	--baseline-config $(CONFIG) --baseline-output $(OUTPUT_DIR) --output-dir $(ENSEMBLE_OUTPUT_DIR) \
	--feature-helper $(FEATURE_HELPER)

LABEL_COMPARISON_CONFIG ?= label-comparison-config.json
LABEL_COMPARISON_OUTPUT_DIR ?= .work/label-comparison
SPECIALIST_LABELS ?= ../specialist-student-corpus/.work/fedora-results-v3/relabels/corpus.csv
FEDORA_SPECIALIST_LABELS ?= /home/aluisioamorim/codex-runs/depression-prediction-setembrobr-specialist-student-corpus-v3/output/relabels/corpus.csv
FEDORA_LABEL_COMPARISON_OUTPUT_DIR ?= $(FEDORA_PROJECT_DIR)/.work/label-comparison
LABEL_COMPARISON = $(PYTHON) scripts/setembrobr_v7_label_comparison.py \
	--config $(LABEL_COMPARISON_CONFIG) --baseline-output $(OUTPUT_DIR) \
	--output-dir $(LABEL_COMPARISON_OUTPUT_DIR)

.PHONY: help setup lint test validate seal pool-embeddings train-oof audit-oof lock-threshold \
	fit-full score-test audit-test evaluate-test materialize-corpus audit-final run \
	promote-reports fedora-sync fedora-setup fedora-run fedora-start fedora-status fedora-fetch
.PHONY: ensemble-prepare-train ensemble-train-tabular-oof ensemble-train-sequence-oof \
	ensemble-train-stack-oof ensemble-audit-oof ensemble-lock ensemble-prepare-test \
	ensemble-score-test ensemble-audit-test ensemble-evaluate ensemble-run-oof ensemble-run-test \
	ensemble-audit-feature-support \
	fedora-ensemble-sync fedora-ensemble-setup fedora-ensemble-start fedora-ensemble-status \
	fedora-ensemble-fetch
.PHONY: label-comparison-prepare label-comparison-train-oof label-comparison-audit-oof \
	label-comparison-lock label-comparison-fit-full label-comparison-score-test \
	label-comparison-audit-test label-comparison-evaluate label-comparison-audit-final \
	label-comparison-run promote-label-comparison-reports fedora-label-comparison-run \
	fedora-label-comparison-fetch

help:
	@$(PYTHON) scripts/setembrobr_v7_qwen_logreg.py --help

setup:
	$(PYTHON) -m pip install --requirement requirements.lock.txt

lint:
	$(PYTHON) -m py_compile scripts/setembrobr_v7_qwen_logreg.py scripts/setembrobr_v7_ensemble.py \
		scripts/setembrobr_v7_label_comparison.py scripts/promote_reports.py \
		scripts/promote_ensemble_reports.py scripts/promote_label_comparison_reports.py \
		tests/test_setembrobr_v7_qwen_logreg.py tests/test_setembrobr_v7_ensemble_contract.py \
		tests/test_setembrobr_v7_label_comparison.py
	$(PYTHON) -m ruff check scripts/setembrobr_v7_qwen_logreg.py scripts/setembrobr_v7_ensemble.py \
		scripts/setembrobr_v7_label_comparison.py scripts/promote_reports.py \
		scripts/promote_ensemble_reports.py scripts/promote_label_comparison_reports.py \
		tests/test_setembrobr_v7_qwen_logreg.py tests/test_setembrobr_v7_ensemble_contract.py \
		tests/test_setembrobr_v7_label_comparison.py

test: lint
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

validate:
	$(PIPELINE) validate --source-pkl $(SOURCE_PKL) --archive-root $(ARCHIVE_ROOT)

seal: validate
	$(PIPELINE) seal --source-pkl $(SOURCE_PKL)

pool-embeddings: seal
	$(PIPELINE) pool-embeddings --archive-root $(ARCHIVE_ROOT) --temporary-dir $(TEMPORARY_DIR)

train-oof: pool-embeddings
	$(PIPELINE) train-oof

audit-oof: train-oof
	$(PIPELINE) audit-oof

lock-threshold: audit-oof
	$(PIPELINE) lock

fit-full: lock-threshold
	$(PIPELINE) fit-full

score-test: fit-full
	$(PIPELINE) score-test

audit-test: score-test
	$(PIPELINE) audit-test

evaluate-test: audit-test
	$(PIPELINE) evaluate-test

materialize-corpus: evaluate-test
	$(PIPELINE) materialize-corpus --source-pkl $(SOURCE_PKL)

audit-final: materialize-corpus
	$(PIPELINE) audit-final --source-pkl $(SOURCE_PKL)

run: audit-final

promote-reports: audit-final
	$(PYTHON) scripts/promote_reports.py --output-dir $(OUTPUT_DIR) --artifacts-dir artifacts

fedora-sync:
	ssh $(FEDORA_HOST) mkdir -p $(FEDORA_PROJECT_DIR)
	rsync -az --exclude '.work/' --exclude '.venv/' ./ $(FEDORA_HOST):$(FEDORA_PROJECT_DIR)/

fedora-setup: fedora-sync
	ssh $(FEDORA_HOST) 'python3 -m venv $(FEDORA_PROJECT_DIR)/.venv && $(FEDORA_PROJECT_DIR)/.venv/bin/python -m pip install --requirement $(FEDORA_PROJECT_DIR)/requirements.lock.txt'

fedora-run: fedora-setup
	ssh $(FEDORA_HOST) 'cd $(FEDORA_PROJECT_DIR) && .venv/bin/python scripts/setembrobr_v7_qwen_logreg.py --help >/dev/null && scripts/run_fedora_pipeline.sh $(FEDORA_SOURCE_PKL) $(FEDORA_ARCHIVE_ROOT) $(FEDORA_OUTPUT_DIR) $(FEDORA_TEMPORARY_DIR)'

fedora-start: fedora-setup
	ssh $(FEDORA_HOST) 'systemctl --user reset-failed $(FEDORA_UNIT).service >/dev/null 2>&1 || true; systemd-run --user --unit=$(FEDORA_UNIT) --property=Restart=no $(FEDORA_PROJECT_DIR)/scripts/run_fedora_pipeline.sh $(FEDORA_SOURCE_PKL) $(FEDORA_ARCHIVE_ROOT) $(FEDORA_OUTPUT_DIR) $(FEDORA_TEMPORARY_DIR)'

fedora-status:
	ssh $(FEDORA_HOST) 'systemctl --user status $(FEDORA_UNIT).service --no-pager --lines=12 || true; test ! -f $(FEDORA_OUTPUT_DIR)/job/status.json || cat $(FEDORA_OUTPUT_DIR)/job/status.json'

fedora-fetch:
	mkdir -p $(OUTPUT_DIR)
	rsync -az --prune-empty-dirs \
		--include '/corpus/' --include '/corpus/*.pkl' \
		--include '/reports/' --include '/reports/*.json' \
		--include '/manifests/' --include '/manifests/*.json' \
		--include '/ensemble/' --include '/ensemble/*.json' \
		--exclude '*' \
		$(FEDORA_HOST):$(FEDORA_OUTPUT_DIR)/ $(OUTPUT_DIR)/
	$(PYTHON) scripts/promote_reports.py --output-dir $(OUTPUT_DIR) --artifacts-dir artifacts

ensemble-prepare-train:
	$(ENSEMBLE_PIPELINE) prepare-train --archive-root $(ARCHIVE_ROOT) --temporary-dir $(TEMPORARY_DIR)

ensemble-train-tabular-oof:
	$(ENSEMBLE_PIPELINE) train-tabular-oof

ensemble-train-sequence-oof:
	$(ENSEMBLE_PIPELINE) train-sequence-oof

ensemble-train-stack-oof:
	$(ENSEMBLE_PIPELINE) train-stack-oof

ensemble-audit-oof:
	$(ENSEMBLE_PIPELINE) audit-oof

ensemble-lock:
	$(ENSEMBLE_PIPELINE) lock

ensemble-prepare-test:
	$(ENSEMBLE_PIPELINE) prepare-test --archive-root $(ARCHIVE_ROOT) --temporary-dir $(TEMPORARY_DIR)

ensemble-score-test:
	$(ENSEMBLE_PIPELINE) score-test

ensemble-audit-test:
	$(ENSEMBLE_PIPELINE) audit-test

ensemble-evaluate:
	$(ENSEMBLE_PIPELINE) evaluate

ensemble-audit-feature-support:
	$(ENSEMBLE_PIPELINE) audit-feature-support

ensemble-run-oof:
	$(ENSEMBLE_PIPELINE) run-oof --archive-root $(ARCHIVE_ROOT) --temporary-dir $(TEMPORARY_DIR)

ensemble-run-test:
	$(ENSEMBLE_PIPELINE) run-test --archive-root $(ARCHIVE_ROOT) --temporary-dir $(TEMPORARY_DIR)

fedora-ensemble-sync:
	ssh $(FEDORA_ENSEMBLE_SSH_OPTS) $(FEDORA_ENSEMBLE_HOST) 'mkdir -p $(FEDORA_PROJECT_DIR)/shared'
	rsync -az --exclude '.work/' --exclude '.venv/' -e "ssh $(FEDORA_ENSEMBLE_SSH_OPTS)" ./ $(FEDORA_ENSEMBLE_HOST):$(FEDORA_PROJECT_DIR)/
	rsync -az -e "ssh $(FEDORA_ENSEMBLE_SSH_OPTS)" $(FEATURE_HELPER) $(FEDORA_ENSEMBLE_HOST):$(FEDORA_PROJECT_DIR)/shared/raw_ternary_prepare_setembrobr.py

fedora-ensemble-setup: fedora-ensemble-sync
	ssh $(FEDORA_ENSEMBLE_SSH_OPTS) $(FEDORA_ENSEMBLE_HOST) '$(FEDORA_ENSEMBLE_PYTHON) -m pip install --requirement $(FEDORA_PROJECT_DIR)/ensemble-requirements.lock.txt'

fedora-ensemble-start: fedora-ensemble-setup
	ssh $(FEDORA_ENSEMBLE_SSH_OPTS) $(FEDORA_ENSEMBLE_HOST) 'systemctl --user reset-failed $(FEDORA_ENSEMBLE_UNIT).service >/dev/null 2>&1 || true; systemd-run --user --unit=$(FEDORA_ENSEMBLE_UNIT) --property=Restart=no $(FEDORA_PROJECT_DIR)/scripts/run_fedora_ensemble.sh $(FEDORA_ENSEMBLE_PYTHON) $(FEDORA_ARCHIVE_ROOT) $(FEDORA_OUTPUT_DIR) $(FEDORA_ENSEMBLE_OUTPUT_DIR) $(FEDORA_ENSEMBLE_TEMPORARY_DIR)'

fedora-ensemble-status:
	ssh $(FEDORA_ENSEMBLE_SSH_OPTS) $(FEDORA_ENSEMBLE_HOST) 'systemctl --user status $(FEDORA_ENSEMBLE_UNIT).service --no-pager --lines=18 || true; test ! -f $(FEDORA_ENSEMBLE_OUTPUT_DIR)/job/status.json || cat $(FEDORA_ENSEMBLE_OUTPUT_DIR)/job/status.json'

fedora-ensemble-fetch:
	mkdir -p $(ENSEMBLE_OUTPUT_DIR)/reports $(ENSEMBLE_OUTPUT_DIR)/ensemble
	rsync -az -e "ssh $(FEDORA_ENSEMBLE_SSH_OPTS)" $(FEDORA_ENSEMBLE_HOST):$(FEDORA_ENSEMBLE_OUTPUT_DIR)/reports/ $(ENSEMBLE_OUTPUT_DIR)/reports/
	rsync -az -e "ssh $(FEDORA_ENSEMBLE_SSH_OPTS)" $(FEDORA_ENSEMBLE_HOST):$(FEDORA_ENSEMBLE_OUTPUT_DIR)/ensemble/ $(ENSEMBLE_OUTPUT_DIR)/ensemble/
	$(PYTHON) scripts/promote_ensemble_reports.py --output-dir $(ENSEMBLE_OUTPUT_DIR) --artifacts-dir artifacts/ensemble

label-comparison-prepare:
	$(LABEL_COMPARISON) prepare --source-pkl $(SOURCE_PKL) --specialist-labels $(SPECIALIST_LABELS)

label-comparison-train-oof: label-comparison-prepare
	$(LABEL_COMPARISON) train-oof

label-comparison-audit-oof: label-comparison-train-oof
	$(LABEL_COMPARISON) audit-oof

label-comparison-lock: label-comparison-audit-oof
	$(LABEL_COMPARISON) lock

label-comparison-fit-full: label-comparison-lock
	$(LABEL_COMPARISON) fit-full

label-comparison-score-test: label-comparison-fit-full
	$(LABEL_COMPARISON) score-test

label-comparison-audit-test: label-comparison-score-test
	$(LABEL_COMPARISON) audit-test

label-comparison-evaluate: label-comparison-audit-test
	$(LABEL_COMPARISON) evaluate

label-comparison-audit-final: label-comparison-evaluate
	$(LABEL_COMPARISON) audit-final --source-pkl $(SOURCE_PKL) --specialist-labels $(SPECIALIST_LABELS)

label-comparison-run: label-comparison-audit-final

promote-label-comparison-reports: label-comparison-audit-final
	$(PYTHON) scripts/promote_label_comparison_reports.py \
		--output-dir $(LABEL_COMPARISON_OUTPUT_DIR) --artifacts-dir artifacts/label-comparison

fedora-label-comparison-run: fedora-sync
	ssh $(FEDORA_HOST) 'test -f $(FEDORA_SPECIALIST_LABELS)'
	ssh $(FEDORA_HOST) 'cd $(FEDORA_PROJECT_DIR) && scripts/run_fedora_label_comparison.sh \
		$(FEDORA_PROJECT_DIR)/.venv/bin/python $(FEDORA_SOURCE_PKL) $(FEDORA_SPECIALIST_LABELS) \
		$(FEDORA_OUTPUT_DIR) $(FEDORA_LABEL_COMPARISON_OUTPUT_DIR)'

fedora-label-comparison-fetch:
	mkdir -p $(LABEL_COMPARISON_OUTPUT_DIR)/reports $(LABEL_COMPARISON_OUTPUT_DIR)/locks
	rsync -az $(FEDORA_HOST):$(FEDORA_LABEL_COMPARISON_OUTPUT_DIR)/reports/ $(LABEL_COMPARISON_OUTPUT_DIR)/reports/
	rsync -az $(FEDORA_HOST):$(FEDORA_LABEL_COMPARISON_OUTPUT_DIR)/locks/ $(LABEL_COMPARISON_OUTPUT_DIR)/locks/
	$(PYTHON) scripts/promote_label_comparison_reports.py \
		--output-dir $(LABEL_COMPARISON_OUTPUT_DIR) --artifacts-dir artifacts/label-comparison
