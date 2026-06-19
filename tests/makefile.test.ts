import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";

describe("Makefile Fedora targets", () => {
  test("defines the strict-blind ternary GPU sequence path", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const target = extractTarget(makefile, "fedora-ternary-train-seq-oof-setembrobr");

    expect(makefile).toContain("FEDORA_TERNARY_RUN_DIR ?= ~/codex-runs/depression-prediction-setembrobr-ternary");
    expect(target).toContain("ssh $(FEDORA_HOST) 'nvidia-smi'");
    expect(target).not.toContain("127.0.0.1");
    expect(target).toContain("rsync -azR requirements.txt scripts/ternary_train_seq_oof_setembrobr.py scripts/write_gpu_run_manifest.py $(TERNARY_CONFIG)");
    expect(target).toContain("outputs/setembrobr/seed42_strict_blind/sequences/top128");
    expect(target).not.toContain("outputs/setembrobr/seed42_strict_blind/sequences ");
    expect(target).toContain("outputs/setembrobr/seed42_ternary_strict_blind/manifest");
    expect(target).toContain("outputs/setembrobr/seed42_ternary_strict_blind/evidence-markers");
    expect(target).toContain("outputs/setembrobr/seed42_ternary_strict_blind/label-policies");
    expect(target).toContain("python scripts/ternary_train_seq_oof_setembrobr.py --config $(TERNARY_CONFIG)");
    expect(target).toContain("python scripts/write_gpu_run_manifest.py --config $(TERNARY_CONFIG) --host-label $(FEDORA_HOST)");
    expect(target).toContain("outputs/setembrobr/seed42_ternary_strict_blind/scores/");
    expect(target).toContain("outputs/setembrobr/seed42_ternary_strict_blind/model-manifests/");
    expect(target).toContain("outputs/setembrobr/seed42_ternary_strict_blind/gpu-runs/");
  });

  test("defines Fedora raw Qwen3 embedding workspace targets", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const syncTarget = extractTarget(makefile, "fedora-raw-sync-setembrobr");
    const setupTarget = extractTarget(makefile, "fedora-raw-setup-setembrobr");
    const validateTarget = extractTarget(makefile, "fedora-raw-validate-setembrobr");
    const smokeTarget = extractTarget(makefile, "fedora-raw-embed-smoke-setembrobr");
    const fullTarget = extractTarget(makefile, "fedora-raw-embed-setembrobr");

    expect(makefile).toContain("RAW_EMBED_CONFIG ?= configs/setembrobr.seed42.raw-qwen3-embeddings.json");
    expect(makefile).toContain(
      "FEDORA_RAW_EMBED_RUN_DIR ?= ~/codex-runs/depression-prediction-setembrobr-raw-embeddings",
    );
    expect(syncTarget).toContain("$(FEDORA_RAW_EMBED_RUN_DIR)/repo");
    expect(syncTarget).toContain("$(FEDORA_RAW_EMBED_RUN_DIR)/data/depression_tweets");
    expect(syncTarget).toContain("$(FEDORA_RAW_EMBED_RUN_DIR)/data/relevance_score");
    expect(syncTarget).toContain("$(FEDORA_RAW_EMBED_RUN_DIR)/artifacts/tweet_embeddings");
    expect(syncTarget).toContain(
      "rsync -azR requirements.txt requirements-raw-embeddings.txt package.json bun.lock tsconfig.json Makefile configs scripts src tests",
    );
    expect(syncTarget).toContain("cd dataset && rsync -azR depression_tweets/");
    expect(syncTarget).toContain("cd dataset && rsync -azR relevance_score/");
    expect(setupTarget).toContain('UV_HTTP_TIMEOUT=300 uv pip install \\"torch>=2.2\\" --index-url https://download.pytorch.org/whl/cu126');
    expect(setupTarget).toContain("UV_HTTP_TIMEOUT=300 uv pip install -r repo/requirements-raw-embeddings.txt");
    expect(validateTarget).toContain("--dataset-dir ../data/depression_tweets");
    expect(validateTarget).toContain("--relevance-dir ../data/relevance_score");
    expect(validateTarget).toContain("--output-dir ../artifacts");
    expect(smokeTarget).toContain("ssh $(FEDORA_HOST) 'nvidia-smi'");
    expect(smokeTarget).toContain("--smoke-users 2");
    expect(smokeTarget).toContain("--device cuda");
    expect(fullTarget).toContain("--output-dir ../artifacts --device cuda");
    expect(syncTarget + smokeTarget + fullTarget).not.toContain("top128");
  });

  test("runs local selection and evaluation after Fedora sequence training", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const target = extractTarget(makefile, "reproduce-ternary-setembrobr-gpu");

    expect(target).toContain("lint");
    expect(target).toContain("typecheck");
    expect(target).toContain("test");
    expect(target).toContain("db-check-setembrobr");
    expect(target).toContain("ternary-train-tabular-oof-setembrobr");
    expect(target).toContain("fedora-ternary-train-seq-oof-setembrobr");
    expect(target).toContain("ternary-train-stack-oof-setembrobr");
    expect(target).toContain("ternary-audit-oof-setembrobr");
    expect(target).toContain("ternary-select-ensemble-setembrobr");
    expect(target).toContain("ternary-robustness-setembrobr");
    expect(target).toContain("ternary-nested-oof-selection-setembrobr");
    expect(target).toContain("ternary-oof-diagnostics-setembrobr");
    expect(target).toContain("ternary-model-policy-leaderboard-setembrobr");
    expect(target).toContain("ternary-family-ablation-setembrobr");
    expect(target).toContain("ternary-evaluate-test-setembrobr");
    expect(target.indexOf("ternary-train-stack-oof-setembrobr")).toBeLessThan(target.indexOf("ternary-audit-oof-setembrobr"));
    expect(target.indexOf("ternary-robustness-setembrobr")).toBeLessThan(target.indexOf("ternary-evaluate-test-setembrobr"));
    expect(target.indexOf("ternary-nested-oof-selection-setembrobr")).toBeLessThan(target.indexOf("ternary-evaluate-test-setembrobr"));
    expect(target.indexOf("ternary-oof-diagnostics-setembrobr")).toBeLessThan(target.indexOf("ternary-evaluate-test-setembrobr"));
    expect(target.indexOf("ternary-model-policy-leaderboard-setembrobr")).toBeLessThan(target.indexOf("ternary-evaluate-test-setembrobr"));
    expect(target.indexOf("ternary-family-ablation-setembrobr")).toBeLessThan(target.indexOf("ternary-evaluate-test-setembrobr"));
  });

  test("defines raw strict-blind Qwen3 recompute targets without db-check", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const reproduceTarget = extractTarget(makefile, "raw-ternary-reproduce-setembrobr");
    const auditTarget = extractTarget(makefile, "raw-ternary-audit-oof-setembrobr");
    const setupTarget = extractTarget(makefile, "fedora-raw-experiments-setup-setembrobr");
    const fedoraTarget = extractTarget(makefile, "fedora-raw-experiments-reproduce-setembrobr");

    expect(makefile).toContain("RAW_TERNARY_CONFIG ?= configs/setembrobr.seed42.raw-qwen3-ternary-diagnosed-only.json");
    expect(makefile).toContain("RAW_TERNARY_SYMMETRIC_CONFIG ?= configs/setembrobr.seed42.raw-qwen3-ternary-symmetric.json");
    expect(makefile).toContain(
      "FEDORA_RAW_EXPERIMENT_RUN_DIR ?= ~/codex-runs/depression-prediction-setembrobr-raw-experiments",
    );
    expect(extractTarget(makefile, "fedora-raw-experiments-sync-setembrobr")).toContain(
      "AGENTS.md requirements-raw-embeddings.txt package.json bun.lock tsconfig.json Makefile configs scripts src tests ternary-classification",
    );
    expect(reproduceTarget).toContain("lint");
    expect(reproduceTarget).toContain("typecheck");
    expect(reproduceTarget).toContain("test");
    expect(reproduceTarget).not.toContain("db-check-setembrobr");
    expect(reproduceTarget).toContain("raw-ternary-prepare-setembrobr");
    expect(reproduceTarget).toContain("raw-ternary-train-tabular-oof-setembrobr");
    expect(reproduceTarget).toContain("raw-ternary-train-seq-oof-setembrobr");
    expect(reproduceTarget).toContain("raw-ternary-train-stack-oof-setembrobr");
    expect(reproduceTarget).toContain("raw-ternary-audit-oof-setembrobr");
    expect(reproduceTarget.indexOf("raw-ternary-audit-oof-setembrobr")).toBeLessThan(
      reproduceTarget.indexOf("raw-ternary-select-ensemble-setembrobr"),
    );
    expect(reproduceTarget.indexOf("raw-ternary-select-ensemble-setembrobr")).toBeLessThan(
      reproduceTarget.indexOf("raw-ternary-evaluate-test-setembrobr"),
    );
    expect(auditTarget).toContain("bun run ternary-audit-oof-setembrobr");
    expect(auditTarget).toContain("bun run raw-ternary-audit-setembrobr");
    expect(setupTarget).toContain("bun install");
    expect(fedoraTarget).toContain("make raw-ternary-reproduce-diagnosed-setembrobr");
    expect(fedoraTarget).toContain("make raw-ternary-reproduce-symmetric-setembrobr");
    expect(fedoraTarget).toContain("nvidia-smi");
  });

  test("defines raw binary classifier and LLM disambiguator targets", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const reproduceTarget = extractTarget(makefile, "raw-binary-reproduce-setembrobr");
    const llmTarget = extractTarget(makefile, "raw-binary-reproduce-llm-setembrobr");

    expect(makefile).toContain("RAW_BINARY_CONFIG ?= configs/setembrobr.seed42.raw-qwen3-binary.json");
    expect(makefile).toContain("RAW_BINARY_PREP_STAMP ?= $(RAW_BINARY_OUTPUT_DIR)/reports/raw_binary_prepare_manifest.json");
    expect(makefile).toContain("RAW_BINARY_TABULAR_STAMP ?= $(RAW_BINARY_OUTPUT_DIR)/reports/raw_binary_tabular_oof_manifest.json");
    expect(makefile).toContain("RAW_BINARY_SEQ_STAMP ?= $(RAW_BINARY_OUTPUT_DIR)/reports/raw_binary_seq_oof_manifest.json");
    expect(makefile).toContain("RAW_BINARY_STACK_STAMP ?= $(RAW_BINARY_OUTPUT_DIR)/reports/raw_binary_stack_oof_manifest.json");
    expect(extractTarget(makefile, "raw-binary-prepare-setembrobr")).toContain("$(RAW_BINARY_PREP_STAMP)");
    expect(makefile).toContain("$(PYTHON) scripts/raw_binary_prepare_setembrobr.py --config $(RAW_BINARY_CONFIG) --mode all");
    expect(extractTarget(makefile, "raw-binary-train-tabular-oof-setembrobr")).toContain("$(RAW_BINARY_TABULAR_STAMP)");
    expect(makefile).toContain("$(PYTHON) scripts/binary_train_tabular_oof_setembrobr.py --config $(RAW_BINARY_CONFIG)");
    expect(extractTarget(makefile, "raw-binary-train-seq-oof-setembrobr")).toContain("$(RAW_BINARY_SEQ_STAMP)");
    expect(makefile).toContain("$(PYTHON) scripts/binary_train_seq_oof_setembrobr.py --config $(RAW_BINARY_CONFIG)");
    expect(extractTarget(makefile, "raw-binary-train-stack-oof-setembrobr")).toContain("$(RAW_BINARY_STACK_STAMP)");
    expect(makefile).toContain("$(PYTHON) scripts/binary_stack_oof_setembrobr.py --config $(RAW_BINARY_CONFIG)");
    expect(extractTarget(makefile, "raw-binary-audit-oof-setembrobr")).toContain("CONFIG=$(RAW_BINARY_CONFIG) bun run raw-binary-audit-setembrobr");
    expect(extractTarget(makefile, "raw-binary-select-ensemble-setembrobr")).toContain("CONFIG=$(RAW_BINARY_CONFIG) bun run select-ensemble-setembrobr");
    expect(extractTarget(makefile, "raw-binary-evaluate-test-setembrobr")).toContain("CONFIG=$(RAW_BINARY_CONFIG) bun run evaluate-test-setembrobr");
    expect(extractTarget(makefile, "raw-binary-llm-oof-setembrobr")).toContain("scripts/binary-llm-disambiguator-setembrobr.ts --mode oof");
    expect(extractTarget(makefile, "raw-binary-llm-test-setembrobr")).toContain("scripts/binary-llm-disambiguator-setembrobr.ts --mode test");
    expect(extractTarget(makefile, "raw-binary-evaluate-llm-setembrobr")).toContain("scripts/binary-evaluate-llm-disambiguated-setembrobr.ts");
    expect(reproduceTarget).toContain("lint");
    expect(reproduceTarget).toContain("typecheck");
    expect(reproduceTarget).toContain("test");
    expect(reproduceTarget).not.toContain("db-check-setembrobr");
    expect(reproduceTarget.indexOf("raw-binary-train-stack-oof-setembrobr")).toBeLessThan(
      reproduceTarget.indexOf("raw-binary-audit-oof-setembrobr"),
    );
    expect(reproduceTarget.indexOf("raw-binary-audit-oof-setembrobr")).toBeLessThan(
      reproduceTarget.indexOf("raw-binary-select-ensemble-setembrobr"),
    );
    expect(reproduceTarget.indexOf("raw-binary-select-ensemble-setembrobr")).toBeLessThan(
      reproduceTarget.indexOf("raw-binary-evaluate-test-setembrobr"),
    );
    expect(llmTarget.indexOf("raw-binary-reproduce-setembrobr")).toBeLessThan(llmTarget.indexOf("raw-binary-llm-oof-setembrobr"));
    expect(llmTarget.indexOf("raw-binary-llm-oof-setembrobr")).toBeLessThan(llmTarget.indexOf("raw-binary-llm-test-setembrobr"));
    expect(llmTarget.indexOf("raw-binary-llm-test-setembrobr")).toBeLessThan(llmTarget.indexOf("raw-binary-evaluate-llm-setembrobr"));
  });

  test("defines raw binary relevance-channel OOF-only targets", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const localTarget = extractTarget(makefile, "raw-binary-relevance-oof-setembrobr");
    const fedoraTarget = extractTarget(makefile, "fedora-raw-binary-relevance-oof-setembrobr");

    expect(makefile).toContain("RAW_BINARY_RELEVANCE_CONFIG ?= configs/setembrobr.seed42.relevance-features-qwen3-binary.json");
    expect(makefile).toContain("RAW_BINARY_RELEVANCE_OUTPUT_DIR ?= outputs/setembrobr/seed42_relevance_features_qwen3_binary");
    expect(localTarget).toContain("lint");
    expect(localTarget).toContain("typecheck");
    expect(localTarget).toContain("test");
    expect(localTarget).toContain("raw-binary-prepare-setembrobr");
    expect(localTarget).toContain("raw-binary-train-tabular-oof-setembrobr");
    expect(localTarget).toContain("raw-binary-train-seq-oof-setembrobr");
    expect(localTarget).toContain("raw-binary-train-stack-oof-setembrobr");
    expect(localTarget).toContain("raw-binary-audit-oof-setembrobr");
    expect(localTarget).toContain("raw-binary-select-ensemble-setembrobr");
    expect(localTarget).not.toContain("raw-binary-evaluate-test-setembrobr");
    expect(localTarget).not.toContain("raw-binary-llm");
    expect(fedoraTarget).toContain("raw-binary-relevance-oof-setembrobr");
    expect(fedoraTarget).toContain("$(RAW_BINARY_RELEVANCE_OUTPUT_DIR)");
    expect(fedoraTarget).toContain("--exclude 'features/'");
    expect(fedoraTarget).toContain("--exclude 'sequences/'");
    expect(fedoraTarget).not.toContain("raw-binary-evaluate-test-setembrobr");
    expect(fedoraTarget).not.toContain("raw-binary-llm");
  });

  test("defines raw binary temporal relevance OOF-only targets", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const localTarget = extractTarget(makefile, "raw-binary-temporal-relevance-oof-setembrobr");
    const fedoraTarget = extractTarget(makefile, "fedora-raw-binary-temporal-relevance-oof-setembrobr");

    expect(makefile).toContain("RAW_BINARY_TEMPORAL_CONFIG ?= configs/setembrobr.seed42.temporal-relevance-qwen3-binary.json");
    expect(makefile).toContain("RAW_BINARY_TEMPORAL_OUTPUT_DIR ?= outputs/setembrobr/seed42_temporal_relevance_qwen3_binary");
    expect(localTarget).toContain("lint");
    expect(localTarget).toContain("typecheck");
    expect(localTarget).toContain("test");
    expect(localTarget).toContain("raw-binary-prepare-setembrobr");
    expect(localTarget).toContain("raw-binary-train-tabular-oof-setembrobr");
    expect(localTarget).toContain("raw-binary-train-seq-oof-setembrobr");
    expect(localTarget).toContain("raw-binary-train-stack-oof-setembrobr");
    expect(localTarget).toContain("raw-binary-audit-oof-setembrobr");
    expect(localTarget).toContain("raw-binary-select-ensemble-setembrobr");
    expect(localTarget).toContain("raw_binary_temporal_oof_diagnostics.py");
    expect(localTarget.indexOf("raw-binary-select-ensemble-setembrobr")).toBeLessThan(
      localTarget.indexOf("raw_binary_temporal_oof_diagnostics.py"),
    );
    expect(localTarget).not.toContain("raw-binary-evaluate-test-setembrobr");
    expect(localTarget).not.toContain("raw-binary-llm");
    expect(fedoraTarget).toContain("raw-binary-temporal-relevance-oof-setembrobr");
    expect(fedoraTarget).toContain("$(RAW_BINARY_TEMPORAL_OUTPUT_DIR)");
    expect(fedoraTarget).toContain("--exclude 'features/'");
    expect(fedoraTarget).toContain("--exclude 'sequences/'");
    expect(fedoraTarget).not.toContain("raw-binary-evaluate-test-setembrobr");
    expect(fedoraTarget).not.toContain("raw-binary-llm");
  });

  test("defines Fedora raw binary recompute targets using the Qwen artifact venv", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const syncTarget = extractTarget(makefile, "fedora-raw-binary-sync-setembrobr");
    const reproduceTarget = extractTarget(makefile, "fedora-raw-binary-reproduce-setembrobr");
    const llmTarget = extractTarget(makefile, "fedora-raw-binary-llm-recompute-setembrobr");

    expect(makefile).toContain("FEDORA_RAW_BINARY_RUN_DIR ?= ~/codex-runs/depression-prediction-setembrobr-raw-binary");
    expect(makefile).toContain(
      "FEDORA_RAW_BINARY_PYTHON ?= /home/aluisioamorim/codex-runs/depression-prediction-setembrobr-raw-embeddings/.venv/bin/python",
    );
    expect(syncTarget).toContain("AGENTS.md requirements-raw-embeddings.txt package.json bun.lock tsconfig.json Makefile configs scripts src tests ternary-classification");
    expect(syncTarget).toContain("bun install --frozen-lockfile");
    expect(reproduceTarget).toContain("make PYTHON=$(FEDORA_RAW_BINARY_PYTHON) raw-binary-reproduce-setembrobr");
    expect(reproduceTarget).toContain("--exclude 'features/'");
    expect(reproduceTarget).toContain("--exclude 'sequences/'");
    expect(llmTarget).toContain("rsync -az .env");
    expect(llmTarget).toContain("raw-binary-llm-test-setembrobr");
    expect(llmTarget).toContain("raw-binary-evaluate-llm-setembrobr");
    expect(llmTarget).toContain("--exclude 'llm-disambiguator/cache/'");
  });

  test("defines a Fedora raw legacy architecture recompute target", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const target = extractTarget(makefile, "fedora-raw-legacy-architecture-recompute-setembrobr");

    expect(makefile).toContain("RAW_TERNARY_LEGACY_GROUP ?= legacy_cnn_logreg_mlp");
    expect(makefile).toContain("RAW_TERNARY_LEGACY_LOCK ?= legacy-cnn-logreg-mlp-lock");
    expect(extractTarget(makefile, "raw-ternary-train-legacy-tabular-oof-setembrobr")).toContain("--only $(RAW_TERNARY_LEGACY_TABULAR_MODELS)");
    expect(extractTarget(makefile, "raw-ternary-train-legacy-seq-oof-setembrobr")).toContain("--only $(RAW_TERNARY_LEGACY_SEQUENCE_MODELS)");
    expect(extractTarget(makefile, "raw-ternary-select-legacy-architecture-setembrobr")).toContain(
      "TERNARY_SELECTION_GROUP_ID=$(RAW_TERNARY_LEGACY_GROUP)",
    );
    expect(target).toContain("raw-ternary-legacy-architecture-setembrobr");
    expect(target).toContain("configs/setembrobr.seed42.raw-qwen3-ternary-diagnosed-only.json");
    expect(target).toContain("configs/setembrobr.seed42.raw-qwen3-ternary-symmetric.json");
    expect(target).toContain("nvidia-smi");
  });

  test("defines a Fedora raw symmetric LLM disambiguator recompute target", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const target = extractTarget(makefile, "fedora-raw-symmetric-llm-recompute-setembrobr");

    expect(target).toContain("rsync -az .env");
    expect(target).toContain("make raw-ternary-reproduce-symmetric-setembrobr");
    expect(target).toContain("make raw-ternary-llm-oof-symmetric-setembrobr");
    expect(target).toContain("make raw-ternary-llm-test-symmetric-setembrobr");
    expect(target).toContain("make raw-ternary-evaluate-llm-symmetric-setembrobr");
    expect(target.indexOf("raw-ternary-reproduce-symmetric-setembrobr")).toBeLessThan(
      target.indexOf("raw-ternary-llm-oof-symmetric-setembrobr"),
    );
    expect(target.indexOf("raw-ternary-llm-oof-symmetric-setembrobr")).toBeLessThan(
      target.indexOf("raw-ternary-llm-test-symmetric-setembrobr"),
    );
    expect(target.indexOf("raw-ternary-llm-test-symmetric-setembrobr")).toBeLessThan(
      target.indexOf("raw-ternary-evaluate-llm-symmetric-setembrobr"),
    );
  });

  test("defines train-only ternary robustness before final test evaluation", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const target = extractTarget(makefile, "ternary-robustness-setembrobr");

    expect(target).toContain("bun run ternary-robustness-setembrobr");
    expect(target).not.toContain("ternary-evaluate-test-setembrobr");
    expect(target).not.toContain("test_score");
  });

  test("defines train-only nested OOF split selection before final test evaluation", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const target = extractTarget(makefile, "ternary-nested-oof-selection-setembrobr");

    expect(target).toContain("bun run ternary-nested-oof-selection-setembrobr");
    expect(target).not.toContain("ternary-evaluate-test-setembrobr");
    expect(target).not.toContain("test_score");
  });

  test("defines train-only OOF diagnostics before final test evaluation", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const target = extractTarget(makefile, "ternary-oof-diagnostics-setembrobr");

    expect(target).toContain("bun run ternary-oof-diagnostics-setembrobr");
    expect(target).not.toContain("ternary-evaluate-test-setembrobr");
    expect(target).not.toContain("test_score");
  });

  test("defines train-only model policy leaderboard before final test evaluation", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const target = extractTarget(makefile, "ternary-model-policy-leaderboard-setembrobr");

    expect(target).toContain("bun run ternary-model-policy-leaderboard-setembrobr");
    expect(target).not.toContain("ternary-evaluate-test-setembrobr");
    expect(target).not.toContain("test_score");
  });

  test("defines train-only family ablation before final test evaluation", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const target = extractTarget(makefile, "ternary-family-ablation-setembrobr");

    expect(target).toContain("bun run ternary-family-ablation-setembrobr");
    expect(target).not.toContain("ternary-evaluate-test-setembrobr");
    expect(target).not.toContain("test_score");
  });

  test("defines strict-blind ternary stacking before audit and selection", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const stackTarget = extractTarget(makefile, "ternary-train-stack-oof-setembrobr");
    const reproduceTarget = extractTarget(makefile, "reproduce-ternary-setembrobr");

    expect(stackTarget).toContain("python3 scripts/ternary_stack_oof_setembrobr.py --config $(TERNARY_CONFIG)");
    expect(stackTarget).not.toContain("ternary-evaluate-test-setembrobr");
    expect(stackTarget).not.toContain("test_labels");
    expect(reproduceTarget).toContain("ternary-train-stack-oof-setembrobr");
    expect(reproduceTarget.indexOf("ternary-train-stack-oof-setembrobr")).toBeLessThan(
      reproduceTarget.indexOf("ternary-audit-oof-setembrobr"),
    );
    expect(reproduceTarget.indexOf("ternary-train-stack-oof-setembrobr")).toBeLessThan(
      reproduceTarget.indexOf("ternary-select-ensemble-setembrobr"),
    );
  });
});

function extractTarget(makefile: string, targetName: string): string {
  const pattern = new RegExp(`^${escapeRegExp(targetName)}:[\\s\\S]*?(?=\\n[A-Za-z0-9_.-]+:|\\n$)`, "mu");
  const match = makefile.match(pattern);
  if (!match) throw new Error(`Missing Makefile target: ${targetName}`);
  return match[0]!;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}
