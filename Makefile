CONFIG ?= configs/setembrobr.seed42.strict-blind.json
TERNARY_CONFIG ?= ternary-classification/configs/setembrobr.seed42.ternary-strict-blind.json
FEDORA_HOST ?= fedora
FEDORA_RUN_DIR ?= ~/codex-runs/depression-prediction-setembrobr
FEDORA_TERNARY_RUN_DIR ?= ~/codex-runs/depression-prediction-setembrobr-ternary

.PHONY: lint typecheck test db-check-setembrobr manifest-setembrobr export-sequences-setembrobr train-tabular-oof-setembrobr train-seq-oof-setembrobr train-candidate-tabular-oof-setembrobr train-candidate-seq-oof-setembrobr fedora-candidate-seq-oof-setembrobr train-candidate-oof-setembrobr audit-oof-setembrobr select-ensemble-setembrobr evaluate-test-setembrobr reproduce-setembrobr ternary-markers-setembrobr ternary-manifest-setembrobr ternary-train-tabular-oof-setembrobr ternary-train-seq-oof-setembrobr fedora-ternary-train-seq-oof-setembrobr ternary-audit-oof-setembrobr ternary-select-ensemble-setembrobr ternary-robustness-setembrobr ternary-nested-oof-selection-setembrobr ternary-evaluate-test-setembrobr reproduce-ternary-setembrobr reproduce-ternary-setembrobr-gpu

lint:
	bun run lint

typecheck:
	bun run typecheck

test:
	bun run test

db-check-setembrobr:
	bun run db-check-setembrobr

manifest-setembrobr:
	bun run manifest-setembrobr

export-sequences-setembrobr:
	python3 scripts/export_sequences_setembrobr.py --config $(CONFIG)

train-tabular-oof-setembrobr:
	python3 scripts/train_tabular_oof_setembrobr.py --config $(CONFIG)

train-seq-oof-setembrobr:
	python3 scripts/train_seq_oof_setembrobr.py --config $(CONFIG)

train-candidate-tabular-oof-setembrobr:
	python3 scripts/train_candidate_tabular_oof_setembrobr.py --config $(CONFIG)

train-candidate-seq-oof-setembrobr:
	python3 scripts/train_candidate_seq_oof_setembrobr.py --config $(CONFIG)

fedora-candidate-seq-oof-setembrobr:
	ssh $(FEDORA_HOST) 'nvidia-smi'
	ssh $(FEDORA_HOST) "bash -lc 'mkdir -p $(FEDORA_RUN_DIR)'"
	rsync -azR requirements.txt configs scripts outputs/setembrobr/seed42_strict_blind/manifest outputs/setembrobr/seed42_strict_blind/sequences $(FEDORA_HOST):$(FEDORA_RUN_DIR)/
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RUN_DIR) && python3 -m venv .venv && . .venv/bin/activate && python -m pip install -U pip && python -m pip install \"numpy>=1.26\" && python -m pip install torch --index-url https://download.pytorch.org/whl/cu126'"
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RUN_DIR) && . .venv/bin/activate && python scripts/train_candidate_seq_oof_setembrobr.py --config $(CONFIG)'"
	rsync -az $(FEDORA_HOST):$(FEDORA_RUN_DIR)/outputs/setembrobr/seed42_strict_blind/scores/ outputs/setembrobr/seed42_strict_blind/scores/
	rsync -az $(FEDORA_HOST):$(FEDORA_RUN_DIR)/outputs/setembrobr/seed42_strict_blind/model-manifests/ outputs/setembrobr/seed42_strict_blind/model-manifests/

train-candidate-oof-setembrobr: train-candidate-tabular-oof-setembrobr fedora-candidate-seq-oof-setembrobr

audit-oof-setembrobr:
	bun run audit-oof-setembrobr

select-ensemble-setembrobr:
	bun run select-ensemble-setembrobr

evaluate-test-setembrobr:
	bun run evaluate-test-setembrobr

reproduce-setembrobr: lint typecheck test db-check-setembrobr manifest-setembrobr export-sequences-setembrobr train-tabular-oof-setembrobr train-seq-oof-setembrobr audit-oof-setembrobr select-ensemble-setembrobr evaluate-test-setembrobr

ternary-markers-setembrobr: manifest-setembrobr
	bun run ternary-markers-setembrobr

ternary-manifest-setembrobr: ternary-markers-setembrobr
	bun run ternary-manifest-setembrobr

ternary-train-tabular-oof-setembrobr: ternary-manifest-setembrobr
	python3 scripts/ternary_train_tabular_oof_setembrobr.py --config $(TERNARY_CONFIG)

ternary-train-seq-oof-setembrobr: ternary-manifest-setembrobr export-sequences-setembrobr
	python3 scripts/ternary_train_seq_oof_setembrobr.py --config $(TERNARY_CONFIG)

fedora-ternary-train-seq-oof-setembrobr: ternary-manifest-setembrobr export-sequences-setembrobr
	ssh $(FEDORA_HOST) 'nvidia-smi'
	ssh $(FEDORA_HOST) "bash -lc 'mkdir -p $(FEDORA_TERNARY_RUN_DIR)'"
	rsync -azR requirements.txt scripts/ternary_train_seq_oof_setembrobr.py scripts/write_gpu_run_manifest.py $(TERNARY_CONFIG) outputs/setembrobr/seed42_strict_blind/sequences/top128 outputs/setembrobr/seed42_ternary_strict_blind/manifest outputs/setembrobr/seed42_ternary_strict_blind/evidence-markers outputs/setembrobr/seed42_ternary_strict_blind/label-policies $(FEDORA_HOST):$(FEDORA_TERNARY_RUN_DIR)/
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_TERNARY_RUN_DIR) && python3 -m venv .venv && . .venv/bin/activate && python -m pip install -U pip && python -m pip install \"numpy>=1.26\" && python -m pip install torch --index-url https://download.pytorch.org/whl/cu126'"
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_TERNARY_RUN_DIR) && . .venv/bin/activate && python scripts/ternary_train_seq_oof_setembrobr.py --config $(TERNARY_CONFIG)'"
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_TERNARY_RUN_DIR) && . .venv/bin/activate && python scripts/write_gpu_run_manifest.py --config $(TERNARY_CONFIG) --host-label $(FEDORA_HOST)'"
	mkdir -p outputs/setembrobr/seed42_ternary_strict_blind/scores outputs/setembrobr/seed42_ternary_strict_blind/model-manifests outputs/setembrobr/seed42_ternary_strict_blind/gpu-runs
	rsync -az $(FEDORA_HOST):$(FEDORA_TERNARY_RUN_DIR)/outputs/setembrobr/seed42_ternary_strict_blind/scores/ outputs/setembrobr/seed42_ternary_strict_blind/scores/
	rsync -az $(FEDORA_HOST):$(FEDORA_TERNARY_RUN_DIR)/outputs/setembrobr/seed42_ternary_strict_blind/model-manifests/ outputs/setembrobr/seed42_ternary_strict_blind/model-manifests/
	rsync -az $(FEDORA_HOST):$(FEDORA_TERNARY_RUN_DIR)/outputs/setembrobr/seed42_ternary_strict_blind/gpu-runs/ outputs/setembrobr/seed42_ternary_strict_blind/gpu-runs/

ternary-audit-oof-setembrobr:
	bun run ternary-audit-oof-setembrobr

ternary-select-ensemble-setembrobr:
	bun run ternary-select-ensemble-setembrobr

ternary-robustness-setembrobr:
	bun run ternary-robustness-setembrobr

ternary-nested-oof-selection-setembrobr:
	bun run ternary-nested-oof-selection-setembrobr

ternary-evaluate-test-setembrobr:
	bun run ternary-evaluate-test-setembrobr

reproduce-ternary-setembrobr: lint typecheck test db-check-setembrobr ternary-manifest-setembrobr ternary-train-tabular-oof-setembrobr ternary-train-seq-oof-setembrobr ternary-audit-oof-setembrobr ternary-select-ensemble-setembrobr ternary-robustness-setembrobr ternary-nested-oof-selection-setembrobr ternary-evaluate-test-setembrobr

reproduce-ternary-setembrobr-gpu: lint typecheck test db-check-setembrobr ternary-manifest-setembrobr ternary-train-tabular-oof-setembrobr fedora-ternary-train-seq-oof-setembrobr ternary-audit-oof-setembrobr ternary-select-ensemble-setembrobr ternary-robustness-setembrobr ternary-nested-oof-selection-setembrobr ternary-evaluate-test-setembrobr
