CONFIG ?= configs/setembrobr.seed42.strict-blind.json
PYTHON ?= python3
TERNARY_CONFIG ?= ternary-classification/configs/setembrobr.seed42.ternary-strict-blind.json
RAW_EMBED_CONFIG ?= configs/setembrobr.seed42.raw-qwen3-embeddings.json
RAW_BINARY_CONFIG ?= configs/setembrobr.seed42.raw-qwen3-binary.json
RAW_BINARY_OUTPUT_DIR ?= outputs/setembrobr/seed42_raw_qwen3_binary
RAW_BINARY_RELEVANCE_CONFIG ?= configs/setembrobr.seed42.relevance-features-qwen3-binary.json
RAW_BINARY_RELEVANCE_OUTPUT_DIR ?= outputs/setembrobr/seed42_relevance_features_qwen3_binary
RAW_BINARY_TEMPORAL_CONFIG ?= configs/setembrobr.seed42.temporal-relevance-qwen3-binary.json
RAW_BINARY_TEMPORAL_OUTPUT_DIR ?= outputs/setembrobr/seed42_temporal_relevance_qwen3_binary
RAW_BINARY_PREP_STAMP ?= $(RAW_BINARY_OUTPUT_DIR)/reports/raw_binary_prepare_manifest.json
RAW_BINARY_TABULAR_STAMP ?= $(RAW_BINARY_OUTPUT_DIR)/reports/raw_binary_tabular_oof_manifest.json
RAW_BINARY_SEQ_STAMP ?= $(RAW_BINARY_OUTPUT_DIR)/reports/raw_binary_seq_oof_manifest.json
RAW_BINARY_STACK_STAMP ?= $(RAW_BINARY_OUTPUT_DIR)/reports/raw_binary_stack_oof_manifest.json
RAW_TERNARY_CONFIG ?= configs/setembrobr.seed42.raw-qwen3-ternary-diagnosed-only.json
RAW_TERNARY_SYMMETRIC_CONFIG ?= configs/setembrobr.seed42.raw-qwen3-ternary-symmetric.json
RAW_TERNARY_LEGACY_GROUP ?= legacy_cnn_logreg_mlp
RAW_TERNARY_LEGACY_LOCK ?= legacy-cnn-logreg-mlp-lock
RAW_TERNARY_LEGACY_TABULAR_MODELS ?= ternary_legacy_logreg_combined_s42 ternary_legacy_focal_combined_g1 ternary_legacy_mlp_combined_h128_a01_s42
RAW_TERNARY_LEGACY_SEQUENCE_MODELS ?= ternary_legacy_seq_cnn_top128_s13 ternary_legacy_seq_cnn_top128_s42
FEDORA_HOST ?= fedora
FEDORA_RUN_DIR ?= ~/codex-runs/depression-prediction-setembrobr
FEDORA_TERNARY_RUN_DIR ?= ~/codex-runs/depression-prediction-setembrobr-ternary
FEDORA_RAW_EMBED_RUN_DIR ?= ~/codex-runs/depression-prediction-setembrobr-raw-embeddings
FEDORA_RAW_EXPERIMENT_RUN_DIR ?= ~/codex-runs/depression-prediction-setembrobr-raw-experiments
FEDORA_RAW_BINARY_RUN_DIR ?= ~/codex-runs/depression-prediction-setembrobr-raw-binary
FEDORA_RAW_BINARY_PYTHON ?= /home/aluisioamorim/codex-runs/depression-prediction-setembrobr-raw-embeddings/.venv/bin/python

.PHONY: lint typecheck test docs-lock-results-summary db-check-setembrobr manifest-setembrobr export-sequences-setembrobr train-tabular-oof-setembrobr train-seq-oof-setembrobr train-candidate-tabular-oof-setembrobr train-candidate-seq-oof-setembrobr fedora-candidate-seq-oof-setembrobr train-candidate-oof-setembrobr audit-oof-setembrobr select-ensemble-setembrobr evaluate-test-setembrobr reproduce-setembrobr raw-validate-setembrobr fedora-raw-sync-setembrobr fedora-raw-setup-setembrobr fedora-raw-validate-setembrobr fedora-raw-embed-smoke-setembrobr fedora-raw-embed-setembrobr raw-binary-prepare-setembrobr raw-binary-train-tabular-oof-setembrobr raw-binary-train-seq-oof-setembrobr raw-binary-train-stack-oof-setembrobr raw-binary-audit-oof-setembrobr raw-binary-select-ensemble-setembrobr raw-binary-evaluate-test-setembrobr raw-binary-llm-oof-setembrobr raw-binary-llm-test-setembrobr raw-binary-evaluate-llm-setembrobr raw-binary-reproduce-setembrobr raw-binary-reproduce-llm-setembrobr raw-binary-relevance-oof-setembrobr raw-binary-temporal-relevance-oof-setembrobr fedora-raw-binary-sync-setembrobr fedora-raw-binary-reproduce-setembrobr fedora-raw-binary-relevance-oof-setembrobr fedora-raw-binary-temporal-relevance-oof-setembrobr fedora-raw-binary-llm-recompute-setembrobr raw-ternary-prepare-setembrobr raw-ternary-manifest-setembrobr raw-ternary-train-tabular-oof-setembrobr raw-ternary-train-seq-oof-setembrobr raw-ternary-train-stack-oof-setembrobr raw-ternary-train-legacy-tabular-oof-setembrobr raw-ternary-train-legacy-seq-oof-setembrobr raw-ternary-select-legacy-architecture-setembrobr raw-ternary-evaluate-legacy-architecture-setembrobr raw-ternary-legacy-architecture-setembrobr raw-ternary-audit-oof-setembrobr raw-ternary-select-ensemble-setembrobr raw-ternary-robustness-setembrobr raw-ternary-nested-oof-selection-setembrobr raw-ternary-oof-diagnostics-setembrobr raw-ternary-model-policy-leaderboard-setembrobr raw-ternary-family-ablation-setembrobr raw-ternary-evaluate-test-setembrobr raw-ternary-llm-oof-symmetric-setembrobr raw-ternary-llm-test-symmetric-setembrobr raw-ternary-evaluate-llm-symmetric-setembrobr raw-ternary-reproduce-setembrobr raw-ternary-reproduce-diagnosed-setembrobr raw-ternary-reproduce-symmetric-setembrobr fedora-raw-experiments-sync-setembrobr fedora-raw-experiments-setup-setembrobr fedora-raw-experiments-reproduce-setembrobr fedora-raw-legacy-architecture-recompute-setembrobr fedora-raw-symmetric-llm-recompute-setembrobr ternary-markers-setembrobr ternary-manifest-setembrobr ternary-train-tabular-oof-setembrobr ternary-train-seq-oof-setembrobr ternary-train-stack-oof-setembrobr fedora-ternary-train-seq-oof-setembrobr ternary-audit-oof-setembrobr ternary-select-ensemble-setembrobr ternary-robustness-setembrobr ternary-nested-oof-selection-setembrobr ternary-oof-diagnostics-setembrobr ternary-model-policy-leaderboard-setembrobr ternary-family-ablation-setembrobr ternary-evaluate-test-setembrobr reproduce-ternary-setembrobr reproduce-ternary-setembrobr-gpu

lint:
	bun run lint

typecheck:
	bun run typecheck

test:
	bun run test

docs-lock-results-summary:
	bun run docs-lock-results-summary

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

raw-validate-setembrobr:
	python3 scripts/raw_qwen3_embeddings_setembrobr.py --config $(RAW_EMBED_CONFIG) --mode validate

fedora-raw-sync-setembrobr:
	ssh $(FEDORA_HOST) "bash -lc 'mkdir -p $(FEDORA_RAW_EMBED_RUN_DIR)/repo $(FEDORA_RAW_EMBED_RUN_DIR)/data/depression_tweets $(FEDORA_RAW_EMBED_RUN_DIR)/data/relevance_score $(FEDORA_RAW_EMBED_RUN_DIR)/models/huggingface $(FEDORA_RAW_EMBED_RUN_DIR)/artifacts/manifests $(FEDORA_RAW_EMBED_RUN_DIR)/artifacts/shards $(FEDORA_RAW_EMBED_RUN_DIR)/artifacts/tweet_embeddings $(FEDORA_RAW_EMBED_RUN_DIR)/artifacts/pooled $(FEDORA_RAW_EMBED_RUN_DIR)/artifacts/reports $(FEDORA_RAW_EMBED_RUN_DIR)/logs'"
	rsync -azR requirements.txt requirements-raw-embeddings.txt package.json bun.lock tsconfig.json Makefile configs scripts src tests $(FEDORA_HOST):$(FEDORA_RAW_EMBED_RUN_DIR)/repo/
	cd dataset && rsync -azR depression_tweets/ $(FEDORA_HOST):$(FEDORA_RAW_EMBED_RUN_DIR)/data/
	cd dataset && rsync -azR relevance_score/ $(FEDORA_HOST):$(FEDORA_RAW_EMBED_RUN_DIR)/data/

fedora-raw-setup-setembrobr:
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EMBED_RUN_DIR) && python3 -m venv .venv && . .venv/bin/activate && python -m pip install -U pip uv && UV_HTTP_TIMEOUT=300 uv pip install \"torch>=2.2\" --index-url https://download.pytorch.org/whl/cu126 && UV_HTTP_TIMEOUT=300 uv pip install -r repo/requirements-raw-embeddings.txt'"

fedora-raw-validate-setembrobr: fedora-raw-sync-setembrobr fedora-raw-setup-setembrobr
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EMBED_RUN_DIR)/repo && . ../.venv/bin/activate && HF_HOME=../models/huggingface TRANSFORMERS_CACHE=../models/huggingface PYTHONUNBUFFERED=1 python scripts/raw_qwen3_embeddings_setembrobr.py --config $(RAW_EMBED_CONFIG) --mode validate --dataset-dir ../data/depression_tweets --relevance-dir ../data/relevance_score --output-dir ../artifacts'"

fedora-raw-embed-smoke-setembrobr: fedora-raw-sync-setembrobr fedora-raw-setup-setembrobr
	ssh $(FEDORA_HOST) 'nvidia-smi'
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EMBED_RUN_DIR)/repo && . ../.venv/bin/activate && HF_HOME=../models/huggingface TRANSFORMERS_CACHE=../models/huggingface PYTHONUNBUFFERED=1 python scripts/raw_qwen3_embeddings_setembrobr.py --config $(RAW_EMBED_CONFIG) --mode embed --dataset-dir ../data/depression_tweets --relevance-dir ../data/relevance_score --output-dir ../artifacts/smoke --smoke-users 2 --batch-size 2 --shard-users 2 --device cuda --force'"

fedora-raw-embed-setembrobr: fedora-raw-sync-setembrobr fedora-raw-setup-setembrobr
	ssh $(FEDORA_HOST) 'nvidia-smi'
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EMBED_RUN_DIR)/repo && . ../.venv/bin/activate && HF_HOME=../models/huggingface TRANSFORMERS_CACHE=../models/huggingface PYTHONUNBUFFERED=1 python scripts/raw_qwen3_embeddings_setembrobr.py --config $(RAW_EMBED_CONFIG) --mode embed --dataset-dir ../data/depression_tweets --relevance-dir ../data/relevance_score --output-dir ../artifacts --device cuda'"

raw-binary-prepare-setembrobr: $(RAW_BINARY_PREP_STAMP)

$(RAW_BINARY_PREP_STAMP): $(RAW_BINARY_CONFIG) scripts/raw_binary_prepare_setembrobr.py scripts/raw_ternary_prepare_setembrobr.py
	$(PYTHON) scripts/raw_binary_prepare_setembrobr.py --config $(RAW_BINARY_CONFIG) --mode all

raw-binary-train-tabular-oof-setembrobr: $(RAW_BINARY_TABULAR_STAMP)

$(RAW_BINARY_TABULAR_STAMP): $(RAW_BINARY_PREP_STAMP) scripts/binary_train_tabular_oof_setembrobr.py
	$(PYTHON) scripts/binary_train_tabular_oof_setembrobr.py --config $(RAW_BINARY_CONFIG)
	mkdir -p $(dir $@)
	touch $@

raw-binary-train-seq-oof-setembrobr: $(RAW_BINARY_SEQ_STAMP)

$(RAW_BINARY_SEQ_STAMP): $(RAW_BINARY_PREP_STAMP) scripts/binary_train_seq_oof_setembrobr.py
	$(PYTHON) scripts/binary_train_seq_oof_setembrobr.py --config $(RAW_BINARY_CONFIG)
	mkdir -p $(dir $@)
	touch $@

raw-binary-train-stack-oof-setembrobr: $(RAW_BINARY_STACK_STAMP)

$(RAW_BINARY_STACK_STAMP): $(RAW_BINARY_TABULAR_STAMP) scripts/binary_stack_oof_setembrobr.py
	$(PYTHON) scripts/binary_stack_oof_setembrobr.py --config $(RAW_BINARY_CONFIG)
	mkdir -p $(dir $@)
	touch $@

raw-binary-audit-oof-setembrobr:
	CONFIG=$(RAW_BINARY_CONFIG) bun run raw-binary-audit-setembrobr

raw-binary-select-ensemble-setembrobr:
	CONFIG=$(RAW_BINARY_CONFIG) bun run select-ensemble-setembrobr

raw-binary-evaluate-test-setembrobr:
	CONFIG=$(RAW_BINARY_CONFIG) bun run evaluate-test-setembrobr

raw-binary-llm-oof-setembrobr:
	CONFIG=$(RAW_BINARY_CONFIG) PYTHON=$(PYTHON) bun run scripts/binary-llm-disambiguator-setembrobr.ts --mode oof

raw-binary-llm-test-setembrobr:
	CONFIG=$(RAW_BINARY_CONFIG) PYTHON=$(PYTHON) bun run scripts/binary-llm-disambiguator-setembrobr.ts --mode test

raw-binary-evaluate-llm-setembrobr:
	CONFIG=$(RAW_BINARY_CONFIG) bun run scripts/binary-evaluate-llm-disambiguated-setembrobr.ts

raw-binary-reproduce-setembrobr: lint typecheck test raw-binary-prepare-setembrobr raw-binary-train-tabular-oof-setembrobr raw-binary-train-seq-oof-setembrobr raw-binary-train-stack-oof-setembrobr raw-binary-audit-oof-setembrobr raw-binary-select-ensemble-setembrobr raw-binary-evaluate-test-setembrobr

raw-binary-reproduce-llm-setembrobr: raw-binary-reproduce-setembrobr raw-binary-llm-oof-setembrobr raw-binary-llm-test-setembrobr raw-binary-evaluate-llm-setembrobr

raw-binary-relevance-oof-setembrobr:
	$(MAKE) RAW_BINARY_CONFIG=$(RAW_BINARY_RELEVANCE_CONFIG) RAW_BINARY_OUTPUT_DIR=$(RAW_BINARY_RELEVANCE_OUTPUT_DIR) lint typecheck test raw-binary-prepare-setembrobr raw-binary-train-tabular-oof-setembrobr raw-binary-train-seq-oof-setembrobr raw-binary-train-stack-oof-setembrobr raw-binary-audit-oof-setembrobr raw-binary-select-ensemble-setembrobr
	@jq -r '"relevance_features_oof_macro_f1=\(.oofMetrics.macroF1)"' $(RAW_BINARY_RELEVANCE_OUTPUT_DIR)/ensemble/ensemble-lock.json

raw-binary-temporal-relevance-oof-setembrobr:
	$(MAKE) RAW_BINARY_CONFIG=$(RAW_BINARY_TEMPORAL_CONFIG) RAW_BINARY_OUTPUT_DIR=$(RAW_BINARY_TEMPORAL_OUTPUT_DIR) lint typecheck test raw-binary-prepare-setembrobr raw-binary-train-tabular-oof-setembrobr raw-binary-train-seq-oof-setembrobr raw-binary-train-stack-oof-setembrobr raw-binary-audit-oof-setembrobr raw-binary-select-ensemble-setembrobr
	$(PYTHON) scripts/raw_binary_temporal_oof_diagnostics.py --config $(RAW_BINARY_TEMPORAL_CONFIG)
	@jq -r '"temporal_relevance_oof_macro_f1=\(.oofMetrics.macroF1)"' $(RAW_BINARY_TEMPORAL_OUTPUT_DIR)/ensemble/ensemble-lock.json

fedora-raw-binary-sync-setembrobr:
	ssh $(FEDORA_HOST) "bash -lc 'mkdir -p $(FEDORA_RAW_BINARY_RUN_DIR)/repo'"
	rsync -azR AGENTS.md requirements-raw-embeddings.txt package.json bun.lock tsconfig.json Makefile configs scripts src tests ternary-classification $(FEDORA_HOST):$(FEDORA_RAW_BINARY_RUN_DIR)/repo/
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_BINARY_RUN_DIR)/repo && export PATH=\$$HOME/.bun/bin:\$$PATH && bun install --frozen-lockfile'"

fedora-raw-binary-reproduce-setembrobr: fedora-raw-binary-sync-setembrobr
	ssh $(FEDORA_HOST) 'nvidia-smi'
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_BINARY_RUN_DIR)/repo && export PATH=\$$HOME/.bun/bin:\$$PATH && make PYTHON=$(FEDORA_RAW_BINARY_PYTHON) raw-binary-reproduce-setembrobr'"
	mkdir -p $(RAW_BINARY_OUTPUT_DIR)
	rsync -az --exclude 'features/' --exclude 'sequences/' --exclude 'llm-disambiguator/cache/' --exclude 'llm-disambiguator/*timeline_packs*.jsonl' $(FEDORA_HOST):$(FEDORA_RAW_BINARY_RUN_DIR)/repo/$(RAW_BINARY_OUTPUT_DIR)/ $(RAW_BINARY_OUTPUT_DIR)/

fedora-raw-binary-relevance-oof-setembrobr: fedora-raw-binary-sync-setembrobr
	ssh $(FEDORA_HOST) 'nvidia-smi'
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_BINARY_RUN_DIR)/repo && export PATH=\$$HOME/.bun/bin:\$$PATH && make PYTHON=$(FEDORA_RAW_BINARY_PYTHON) raw-binary-relevance-oof-setembrobr'"
	mkdir -p $(RAW_BINARY_RELEVANCE_OUTPUT_DIR)
	rsync -az --exclude 'features/' --exclude 'sequences/' --exclude 'llm-disambiguator/cache/' --exclude 'llm-disambiguator/*timeline_packs*.jsonl' $(FEDORA_HOST):$(FEDORA_RAW_BINARY_RUN_DIR)/repo/$(RAW_BINARY_RELEVANCE_OUTPUT_DIR)/ $(RAW_BINARY_RELEVANCE_OUTPUT_DIR)/
	@jq -r '"relevance_features_oof_macro_f1=\(.oofMetrics.macroF1)"' $(RAW_BINARY_RELEVANCE_OUTPUT_DIR)/ensemble/ensemble-lock.json

fedora-raw-binary-temporal-relevance-oof-setembrobr: fedora-raw-binary-sync-setembrobr
	ssh $(FEDORA_HOST) 'nvidia-smi'
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_BINARY_RUN_DIR)/repo && export PATH=\$$HOME/.bun/bin:\$$PATH && make PYTHON=$(FEDORA_RAW_BINARY_PYTHON) raw-binary-temporal-relevance-oof-setembrobr'"
	mkdir -p $(RAW_BINARY_TEMPORAL_OUTPUT_DIR)
	rsync -az --exclude 'features/' --exclude 'sequences/' --exclude 'llm-disambiguator/cache/' --exclude 'llm-disambiguator/*timeline_packs*.jsonl' $(FEDORA_HOST):$(FEDORA_RAW_BINARY_RUN_DIR)/repo/$(RAW_BINARY_TEMPORAL_OUTPUT_DIR)/ $(RAW_BINARY_TEMPORAL_OUTPUT_DIR)/
	@jq -r '"temporal_relevance_oof_macro_f1=\(.oofMetrics.macroF1)"' $(RAW_BINARY_TEMPORAL_OUTPUT_DIR)/ensemble/ensemble-lock.json

fedora-raw-binary-llm-recompute-setembrobr: fedora-raw-binary-sync-setembrobr
	@if [ ! -f .env ]; then echo "Missing .env with ANTHROPIC_API_KEY"; exit 1; fi
	rsync -az .env $(FEDORA_HOST):$(FEDORA_RAW_BINARY_RUN_DIR)/repo/.env
	ssh $(FEDORA_HOST) 'nvidia-smi'
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_BINARY_RUN_DIR)/repo && export PATH=\$$HOME/.bun/bin:\$$PATH && make PYTHON=$(FEDORA_RAW_BINARY_PYTHON) raw-binary-reproduce-setembrobr raw-binary-llm-test-setembrobr raw-binary-evaluate-llm-setembrobr'"
	mkdir -p $(RAW_BINARY_OUTPUT_DIR)
	rsync -az --exclude 'features/' --exclude 'sequences/' --exclude 'llm-disambiguator/cache/' --exclude 'llm-disambiguator/*timeline_packs*.jsonl' $(FEDORA_HOST):$(FEDORA_RAW_BINARY_RUN_DIR)/repo/$(RAW_BINARY_OUTPUT_DIR)/ $(RAW_BINARY_OUTPUT_DIR)/

raw-ternary-prepare-setembrobr:
	python3 scripts/raw_ternary_prepare_setembrobr.py --config $(RAW_TERNARY_CONFIG) --mode all

raw-ternary-manifest-setembrobr: raw-ternary-prepare-setembrobr
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) bun run ternary-manifest-setembrobr

raw-ternary-train-tabular-oof-setembrobr: raw-ternary-manifest-setembrobr
	python3 scripts/ternary_train_tabular_oof_setembrobr.py --config $(RAW_TERNARY_CONFIG)

raw-ternary-train-seq-oof-setembrobr: raw-ternary-manifest-setembrobr
	python3 scripts/ternary_train_seq_oof_setembrobr.py --config $(RAW_TERNARY_CONFIG)

raw-ternary-train-stack-oof-setembrobr: raw-ternary-train-tabular-oof-setembrobr
	python3 scripts/ternary_stack_oof_setembrobr.py --config $(RAW_TERNARY_CONFIG)

raw-ternary-train-legacy-tabular-oof-setembrobr: raw-ternary-manifest-setembrobr
	python3 scripts/ternary_train_tabular_oof_setembrobr.py --config $(RAW_TERNARY_CONFIG) --only $(RAW_TERNARY_LEGACY_TABULAR_MODELS)

raw-ternary-train-legacy-seq-oof-setembrobr: raw-ternary-manifest-setembrobr
	python3 scripts/ternary_train_seq_oof_setembrobr.py --config $(RAW_TERNARY_CONFIG) --only $(RAW_TERNARY_LEGACY_SEQUENCE_MODELS)

raw-ternary-audit-oof-setembrobr:
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) bun run ternary-audit-oof-setembrobr
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) bun run raw-ternary-audit-setembrobr

raw-ternary-select-legacy-architecture-setembrobr:
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) TERNARY_SELECTION_GROUP_ID=$(RAW_TERNARY_LEGACY_GROUP) TERNARY_LOCK_BASENAME=$(RAW_TERNARY_LEGACY_LOCK) bun run ternary-select-ensemble-setembrobr

raw-ternary-evaluate-legacy-architecture-setembrobr:
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) TERNARY_LOCK_BASENAME=$(RAW_TERNARY_LEGACY_LOCK) bun run ternary-evaluate-test-setembrobr

raw-ternary-legacy-architecture-setembrobr: raw-ternary-train-legacy-tabular-oof-setembrobr raw-ternary-train-legacy-seq-oof-setembrobr raw-ternary-audit-oof-setembrobr raw-ternary-select-legacy-architecture-setembrobr raw-ternary-evaluate-legacy-architecture-setembrobr

raw-ternary-select-ensemble-setembrobr:
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) bun run ternary-select-ensemble-setembrobr

raw-ternary-robustness-setembrobr:
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) bun run ternary-robustness-setembrobr

raw-ternary-nested-oof-selection-setembrobr:
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) bun run ternary-nested-oof-selection-setembrobr

raw-ternary-oof-diagnostics-setembrobr:
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) bun run ternary-oof-diagnostics-setembrobr

raw-ternary-model-policy-leaderboard-setembrobr:
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) bun run ternary-model-policy-leaderboard-setembrobr

raw-ternary-family-ablation-setembrobr:
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) bun run ternary-family-ablation-setembrobr

raw-ternary-evaluate-test-setembrobr:
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) bun run ternary-evaluate-test-setembrobr

raw-ternary-llm-oof-symmetric-setembrobr: RAW_TERNARY_CONFIG=$(RAW_TERNARY_SYMMETRIC_CONFIG)
raw-ternary-llm-oof-symmetric-setembrobr:
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) PYTHON=$(PYTHON) bun run scripts/ternary-llm-disambiguator-setembrobr.ts --mode oof

raw-ternary-llm-test-symmetric-setembrobr: RAW_TERNARY_CONFIG=$(RAW_TERNARY_SYMMETRIC_CONFIG)
raw-ternary-llm-test-symmetric-setembrobr:
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) PYTHON=$(PYTHON) bun run scripts/ternary-llm-disambiguator-setembrobr.ts --mode test

raw-ternary-evaluate-llm-symmetric-setembrobr: RAW_TERNARY_CONFIG=$(RAW_TERNARY_SYMMETRIC_CONFIG)
raw-ternary-evaluate-llm-symmetric-setembrobr:
	TERNARY_CONFIG=$(RAW_TERNARY_CONFIG) bun run scripts/ternary-evaluate-llm-disambiguated-setembrobr.ts

raw-ternary-reproduce-setembrobr: lint typecheck test raw-ternary-prepare-setembrobr raw-ternary-manifest-setembrobr raw-ternary-train-tabular-oof-setembrobr raw-ternary-train-seq-oof-setembrobr raw-ternary-train-stack-oof-setembrobr raw-ternary-audit-oof-setembrobr raw-ternary-select-ensemble-setembrobr raw-ternary-robustness-setembrobr raw-ternary-nested-oof-selection-setembrobr raw-ternary-oof-diagnostics-setembrobr raw-ternary-model-policy-leaderboard-setembrobr raw-ternary-family-ablation-setembrobr raw-ternary-evaluate-test-setembrobr

raw-ternary-reproduce-diagnosed-setembrobr: RAW_TERNARY_CONFIG=configs/setembrobr.seed42.raw-qwen3-ternary-diagnosed-only.json
raw-ternary-reproduce-diagnosed-setembrobr: raw-ternary-reproduce-setembrobr

raw-ternary-reproduce-symmetric-setembrobr: RAW_TERNARY_CONFIG=$(RAW_TERNARY_SYMMETRIC_CONFIG)
raw-ternary-reproduce-symmetric-setembrobr: raw-ternary-reproduce-setembrobr

fedora-raw-experiments-sync-setembrobr:
	ssh $(FEDORA_HOST) "bash -lc 'mkdir -p $(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo $(FEDORA_RAW_EXPERIMENT_RUN_DIR)/logs'"
	rsync -azR AGENTS.md requirements-raw-embeddings.txt package.json bun.lock tsconfig.json Makefile configs scripts src tests ternary-classification $(FEDORA_HOST):$(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo/

fedora-raw-experiments-setup-setembrobr:
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EXPERIMENT_RUN_DIR) && python3 -m venv .venv && . .venv/bin/activate && python -m pip install -U pip uv && UV_HTTP_TIMEOUT=300 uv pip install \"torch>=2.2\" --index-url https://download.pytorch.org/whl/cu126 && UV_HTTP_TIMEOUT=300 uv pip install -r repo/requirements-raw-embeddings.txt'"
	ssh $(FEDORA_HOST) "bash -lc 'command -v bun >/dev/null || curl -fsSL https://bun.sh/install | bash'"
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo && export PATH=\$$HOME/.bun/bin:\$$PATH && bun install'"

fedora-raw-experiments-reproduce-setembrobr: fedora-raw-experiments-sync-setembrobr fedora-raw-experiments-setup-setembrobr
	ssh $(FEDORA_HOST) 'nvidia-smi'
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo && . ../.venv/bin/activate && export PATH=\$$HOME/.bun/bin:\$$PATH && make raw-ternary-reproduce-diagnosed-setembrobr'"
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo && . ../.venv/bin/activate && export PATH=\$$HOME/.bun/bin:\$$PATH && make raw-ternary-reproduce-symmetric-setembrobr'"
	mkdir -p outputs/setembrobr/seed42_raw_qwen3_ternary_diagnosed_only outputs/setembrobr/seed42_raw_qwen3_ternary_symmetric
	rsync -az $(FEDORA_HOST):$(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo/outputs/setembrobr/seed42_raw_qwen3_ternary_diagnosed_only/ outputs/setembrobr/seed42_raw_qwen3_ternary_diagnosed_only/
	rsync -az $(FEDORA_HOST):$(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo/outputs/setembrobr/seed42_raw_qwen3_ternary_symmetric/ outputs/setembrobr/seed42_raw_qwen3_ternary_symmetric/

fedora-raw-legacy-architecture-recompute-setembrobr: fedora-raw-experiments-sync-setembrobr fedora-raw-experiments-setup-setembrobr
	ssh $(FEDORA_HOST) 'nvidia-smi'
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo && . ../.venv/bin/activate && export PATH=\$$HOME/.bun/bin:\$$PATH && make RAW_TERNARY_CONFIG=configs/setembrobr.seed42.raw-qwen3-ternary-diagnosed-only.json raw-ternary-legacy-architecture-setembrobr'"
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo && . ../.venv/bin/activate && export PATH=\$$HOME/.bun/bin:\$$PATH && make RAW_TERNARY_CONFIG=configs/setembrobr.seed42.raw-qwen3-ternary-symmetric.json raw-ternary-legacy-architecture-setembrobr'"
	mkdir -p outputs/setembrobr/seed42_raw_qwen3_ternary_diagnosed_only outputs/setembrobr/seed42_raw_qwen3_ternary_symmetric
	rsync -az $(FEDORA_HOST):$(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo/outputs/setembrobr/seed42_raw_qwen3_ternary_diagnosed_only/ outputs/setembrobr/seed42_raw_qwen3_ternary_diagnosed_only/
	rsync -az $(FEDORA_HOST):$(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo/outputs/setembrobr/seed42_raw_qwen3_ternary_symmetric/ outputs/setembrobr/seed42_raw_qwen3_ternary_symmetric/

fedora-raw-symmetric-llm-recompute-setembrobr: fedora-raw-experiments-sync-setembrobr fedora-raw-experiments-setup-setembrobr
	@if [ ! -f .env ]; then echo "Missing .env with ANTHROPIC_API_KEY"; exit 1; fi
	rsync -az .env $(FEDORA_HOST):$(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo/.env
	ssh $(FEDORA_HOST) 'nvidia-smi'
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo && . ../.venv/bin/activate && export PATH=\$$HOME/.bun/bin:\$$PATH && make raw-ternary-reproduce-symmetric-setembrobr'"
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo && . ../.venv/bin/activate && export PATH=\$$HOME/.bun/bin:\$$PATH && make raw-ternary-llm-oof-symmetric-setembrobr'"
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo && . ../.venv/bin/activate && export PATH=\$$HOME/.bun/bin:\$$PATH && make raw-ternary-llm-test-symmetric-setembrobr'"
	ssh $(FEDORA_HOST) "bash -lc 'cd $(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo && . ../.venv/bin/activate && export PATH=\$$HOME/.bun/bin:\$$PATH && make raw-ternary-evaluate-llm-symmetric-setembrobr'"
	mkdir -p outputs/setembrobr/seed42_raw_qwen3_ternary_symmetric
	rsync -az $(FEDORA_HOST):$(FEDORA_RAW_EXPERIMENT_RUN_DIR)/repo/outputs/setembrobr/seed42_raw_qwen3_ternary_symmetric/ outputs/setembrobr/seed42_raw_qwen3_ternary_symmetric/

ternary-markers-setembrobr: manifest-setembrobr
	bun run ternary-markers-setembrobr

ternary-manifest-setembrobr: ternary-markers-setembrobr
	bun run ternary-manifest-setembrobr

ternary-train-tabular-oof-setembrobr: ternary-manifest-setembrobr
	python3 scripts/ternary_train_tabular_oof_setembrobr.py --config $(TERNARY_CONFIG)

ternary-train-seq-oof-setembrobr: ternary-manifest-setembrobr export-sequences-setembrobr
	python3 scripts/ternary_train_seq_oof_setembrobr.py --config $(TERNARY_CONFIG)

ternary-train-stack-oof-setembrobr: ternary-train-tabular-oof-setembrobr
	python3 scripts/ternary_stack_oof_setembrobr.py --config $(TERNARY_CONFIG)

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

ternary-oof-diagnostics-setembrobr:
	bun run ternary-oof-diagnostics-setembrobr

ternary-model-policy-leaderboard-setembrobr:
	bun run ternary-model-policy-leaderboard-setembrobr

ternary-family-ablation-setembrobr:
	bun run ternary-family-ablation-setembrobr

ternary-evaluate-test-setembrobr:
	bun run ternary-evaluate-test-setembrobr

reproduce-ternary-setembrobr: lint typecheck test db-check-setembrobr ternary-manifest-setembrobr ternary-train-tabular-oof-setembrobr ternary-train-seq-oof-setembrobr ternary-train-stack-oof-setembrobr ternary-audit-oof-setembrobr ternary-select-ensemble-setembrobr ternary-robustness-setembrobr ternary-nested-oof-selection-setembrobr ternary-oof-diagnostics-setembrobr ternary-model-policy-leaderboard-setembrobr ternary-family-ablation-setembrobr ternary-evaluate-test-setembrobr

reproduce-ternary-setembrobr-gpu: lint typecheck test db-check-setembrobr ternary-manifest-setembrobr ternary-train-tabular-oof-setembrobr fedora-ternary-train-seq-oof-setembrobr ternary-train-stack-oof-setembrobr ternary-audit-oof-setembrobr ternary-select-ensemble-setembrobr ternary-robustness-setembrobr ternary-nested-oof-selection-setembrobr ternary-oof-diagnostics-setembrobr ternary-model-policy-leaderboard-setembrobr ternary-family-ablation-setembrobr ternary-evaluate-test-setembrobr
