import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { loadTernaryConfig } from "../src/ternary-config.ts";

describe("raw Qwen3 embedding configuration", () => {
  test("uses full tweet-level artifacts without a top-N policy", async () => {
    const config = JSON.parse(await readFile("configs/setembrobr.seed42.raw-qwen3-embeddings.json", "utf8"));

    expect(config.datasetDir).toBe("dataset/depression_tweets");
    expect(config.relevanceDir).toBe("dataset/relevance_score");
    expect(config.outputDir).toBe("outputs/setembrobr/seed42_raw_qwen3_embeddings");
    expect(config.embeddingModelId).toBe("Qwen/Qwen3-Embedding-4B");
    expect(config.embeddingDimension).toBe(2560);
    expect(config.embeddingStorageDtype).toBe("float16");
    expect(config.seed).toBe(42);
    expect(config).not.toHaveProperty("topN");
  });

  test("script writes the required tweet-level Parquet schema", async () => {
    const script = await readFile("scripts/raw_qwen3_embeddings_setembrobr.py", "utf8");

    expect(script).toContain('"tweet_embeddings/train"');
    expect(script).toContain('"tweet_embeddings/test"');
    expect(script).toContain('("user_id", pa.string())');
    expect(script).toContain('("tweet_index", pa.int32())');
    expect(script).toContain('("tweet_text", pa.string())');
    expect(script).toContain('("gpt5_relevance", pa.int16())');
    expect(script).toContain('("embedding", pa.list_(self.value_type, list_size=embedding_dimension))');
    expect(script).toContain('"columns": ["user_id", "tweet_index", "tweet_text", "gpt5_relevance", "embedding"]');
    expect(script).not.toContain("sequence_embeddings");
    expect(script).not.toContain("top_items");
  });

  test("defines a raw anxiety lane without synthetic relevance pools", async () => {
    const config = JSON.parse(
      await readFile("configs/setembrobr.seed42.raw-qwen3-anxiety-embeddings.json", "utf8"),
    );
    const script = await readFile("scripts/raw_qwen3_embeddings_anxiety_setembrobr.py", "utf8");

    expect(config.datasetId).toBe("setembrobr-anxiety");
    expect(config.expectedUsers).toEqual({ train: 14200, test: 3552 });
    expect(config.expectedTweets).toEqual({ train: 21769232, test: 5638808 });
    expect(config.tweetDelimiter).toBe(" # ");
    expect(config.includeRelevancePools).toBe(false);
    expect(config.redactTestLabels).toBe(true);
    expect(config.minFreeGiB).toBe(20);
    expect(script).toContain('relevance_raw=[None] * tweet_count');
    expect(script).toContain('"label": "" if redacted else record.label_code');
  });
});

describe("raw Qwen3 ternary experiment configuration", () => {
  test("defines separate diagnosed-only and symmetric strict-blind lanes", async () => {
    const diagnosed = await loadTernaryConfig("configs/setembrobr.seed42.raw-qwen3-ternary-diagnosed-only.json");
    const symmetric = await loadTernaryConfig("configs/setembrobr.seed42.raw-qwen3-ternary-symmetric.json");

    expect(diagnosed.featureSource).toBe("raw_artifacts");
    expect(diagnosed.rawArtifactsDir).toBe(
      "/home/aluisioamorim/codex-runs/depression-prediction-setembrobr-raw-embeddings/artifacts",
    );
    expect(diagnosed.outputDir).toBe("outputs/setembrobr/seed42_raw_qwen3_ternary_diagnosed_only");
    expect(symmetric.outputDir).toBe("outputs/setembrobr/seed42_raw_qwen3_ternary_symmetric");
    expect(diagnosed.database.embeddingDimension).toBe(2560);
    expect(diagnosed.candidateModels.tabular.every((candidate) => !candidate.featureBlocks.includes("v1"))).toBe(true);
    expect(diagnosed.candidateModels.tabular.every((candidate) => !candidate.featureBlocks.includes("v2"))).toBe(true);
    expect(diagnosed.candidateModels.tabular.every((candidate) => !candidate.featureBlocks.includes("rel5_combined"))).toBe(true);
    expect(symmetric.candidateModels.sequence.length).toBe(diagnosed.candidateModels.sequence.length);
    expect(diagnosed.labelPolicies.every((policy) => policy.appliesTo === "diagnosed_only")).toBe(true);
    expect(symmetric.labelPolicies.every((policy) => policy.appliesTo === "both_classes")).toBe(true);
    expect(symmetric.labelPolicies.find((policy) => policy.policyId === "sym_evidence_q20")?.quantileScope).toBe(
      "per_binary_label",
    );
    expect(symmetric.llmDisambiguator?.enabled).toBe(true);
    expect(symmetric.llmDisambiguator?.requestedModel).toBe("claude-sonnet-4-6[1M]");
    expect(symmetric.llmDisambiguator?.apiModel).toBe("claude-sonnet-4-6");
    expect(diagnosed.llmDisambiguator).toBeUndefined();
  });

  test("pre-registers the raw legacy CNN plus LogReg plus MLP architecture", async () => {
    const diagnosed = await loadTernaryConfig("configs/setembrobr.seed42.raw-qwen3-ternary-diagnosed-only.json");
    const legacyGroup = diagnosed.ensemble.selectionGroups?.find((group) => group.groupId === "legacy_cnn_logreg_mlp");

    expect(legacyGroup?.modelIds).toEqual([
      "ternary_legacy_seq_cnn_top128_s13",
      "ternary_legacy_seq_cnn_top128_s42",
      "ternary_legacy_focal_combined_g1",
      "ternary_legacy_logreg_combined_s42",
      "ternary_legacy_mlp_combined_h128_a01_s42",
    ]);
    expect(diagnosed.ensemble.exhaustiveModelLimit).toBeGreaterThanOrEqual(5);
    for (const modelId of legacyGroup?.modelIds ?? []) expect(diagnosed.models).toContain(modelId);

    const legacySequence = diagnosed.candidateModels.sequence.filter((candidate) => candidate.modelId.startsWith("ternary_legacy_seq_cnn"));
    expect(legacySequence).toHaveLength(2);
    expect(legacySequence.every((candidate) => candidate.family === "cnn")).toBe(true);
    expect(legacySequence.every((candidate) => candidate.topN === 128)).toBe(true);
    expect(legacySequence.every((candidate) => candidate.useRelevanceChannel === false)).toBe(true);
  });

  test("raw strict-blind scripts keep sealed labels out of training and selection", async () => {
    const prepScript = await readFile("scripts/raw_ternary_prepare_setembrobr.py", "utf8");
    const auditScript = await readFile("scripts/raw-ternary-audit-setembrobr.ts", "utf8");
    const evaluator = await readFile("scripts/ternary-evaluate-test-setembrobr.ts", "utf8");

    expect(prepScript).toContain("sealed_test_labels_seed");
    expect(prepScript).toContain("redacted_test_row_hash");
    expect(prepScript).toContain('"true_labels": np.asarray([row["label_code"] if split == "train" else -1');
    expect(auditScript).toContain("raw-test-label-redaction");
    expect(auditScript).toContain("raw-static-sealed-label-reference");
    expect(auditScript).toContain("auditLlmTestDecisionSchema");
    expect(evaluator).toContain("readRawSealedTestLabels");
    expect(evaluator).toContain("TERNARY_LOCK_BASENAME");
  });
});
