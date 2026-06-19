import { describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { loadConfig } from "../src/config.ts";
import {
  readRawBinaryAuditManifest,
  readRawBinarySealedTestLabels,
  readRawBinaryTrainManifest,
  rawBinaryStrictBlindManifestHash,
} from "../src/raw-binary.ts";
import { sha256File } from "../src/hash.ts";
import type { ProjectConfig } from "../src/types.ts";

describe("raw binary config", () => {
  test("registers binary ports of the winning ternary families", async () => {
    const config = await loadConfig("configs/setembrobr.seed42.raw-qwen3-binary.json");
    expect(config.featureSource).toBe("raw_artifacts");
    expect(config.outputDir).toBe("outputs/setembrobr/seed42_raw_qwen3_binary");
    expect(config.llmDisambiguator?.requestedModel).toBe("claude-sonnet-4-6[1M]");

    const tabular = config.candidateModels?.tabular ?? [];
    const sequence = config.candidateModels?.sequence ?? [];
    const stacking = config.candidateModels?.stacking ?? [];
    expect(tabular.map((model) => model.modelId)).toContain("binary_hgb_expanded_pca_s42");
    expect(tabular.map((model) => model.modelId)).toContain("binary_hier_logreg_gate");
    expect(tabular.map((model) => model.modelId)).toContain("binary_xgb_embedding_rich_s7");
    expect(new Set(sequence.map((model) => model.modelId))).toEqual(
      new Set([
        "binary_legacy_seq_cnn_top128_s13",
        "binary_legacy_seq_cnn_top128_s42",
        "binary_seq_cnn_top128_s42",
        "binary_seq_cnn_wide_top128_s13",
        "binary_seq_bilstm_top128_s13",
        "binary_seq_transformer_top128_s42",
      ]),
    );
    expect(stacking.map((model) => model.modelId)).toEqual([
      "binary_stack_logreg_xgb_tabular",
      "binary_stack_logreg_boosted_core",
      "binary_stack_logreg_xgb_variants",
    ]);
    for (const modelId of [...tabular, ...sequence, ...stacking].map((model) => model.modelId)) {
      expect(config.models).toContain(modelId);
    }
  });

  test("registers the relevance-channel OOF-only raw binary experiment", async () => {
    const baseline = await loadConfig("configs/setembrobr.seed42.raw-qwen3-binary.json");
    const relevance = await loadConfig("configs/setembrobr.seed42.relevance-features-qwen3-binary.json");

    expect(relevance.outputDir).toBe("outputs/setembrobr/seed42_relevance_features_qwen3_binary");
    expect(relevance.sealedTestLabelsPath).toBe(
      "outputs/setembrobr/seed42_relevance_features_qwen3_binary/manifest/sealed_test_labels_seed42.csv",
    );
    expect(relevance.candidateModels?.tabular).toEqual(baseline.candidateModels?.tabular);
    expect(relevance.candidateModels?.stacking).toEqual(baseline.candidateModels?.stacking);
    expect(relevance.models).toEqual(baseline.models);
    expect(relevance.candidateModels?.sequence.every((model) => model.useRelevanceChannel === true)).toBe(true);

    const baselineSequence = baseline.candidateModels?.sequence ?? [];
    expect(baselineSequence.filter((model) => model.modelId.startsWith("binary_legacy_seq_cnn")).map((model) => model.useRelevanceChannel)).toEqual([
      false,
      false,
    ]);
  });

  test("registers the temporal relevance OOF-only raw binary experiment", async () => {
    const baseline = await loadConfig("configs/setembrobr.seed42.raw-qwen3-binary.json");
    const relevance = await loadConfig("configs/setembrobr.seed42.relevance-features-qwen3-binary.json");
    const temporalText = await readFile("configs/setembrobr.seed42.temporal-relevance-qwen3-binary.json", "utf8");
    const temporal = await loadConfig("configs/setembrobr.seed42.temporal-relevance-qwen3-binary.json");

    expect(JSON.parse(temporalText)).not.toHaveProperty("extends");
    expect(temporal.outputDir).toBe("outputs/setembrobr/seed42_temporal_relevance_qwen3_binary");
    expect(temporal.sealedTestLabelsPath).toBe(
      "outputs/setembrobr/seed42_temporal_relevance_qwen3_binary/manifest/sealed_test_labels_seed42.csv",
    );
    expect(temporal.sequenceExport?.order).toBe("recent_chronological");
    expect(temporal.candidateModels?.sequence.every((model) => model.useRelevanceChannel === true)).toBe(true);
    expect(temporal.candidateModels?.tabular.every((model) => model.featureBlocks.includes("temporal_markers"))).toBe(true);
    expect(temporal.candidateModels?.tabular.map((model) => ({ ...model, featureBlocks: model.featureBlocks.filter((block) => block !== "temporal_markers") }))).toEqual(
      relevance.candidateModels?.tabular,
    );
    expect(temporal.candidateModels?.stacking).toEqual(relevance.candidateModels?.stacking);
    expect(temporal.models).toEqual(relevance.models);

    expect(baseline.sequenceExport).toBeUndefined();
    expect(relevance.sequenceExport).toBeUndefined();
    expect(baseline.candidateModels?.tabular.some((model) => model.featureBlocks.includes("temporal_markers"))).toBe(false);
    expect(relevance.candidateModels?.tabular.some((model) => model.featureBlocks.includes("temporal_markers"))).toBe(false);
  });

  test("raw prepare records temporal feature and sequence export metadata", async () => {
    const prepareScript = await readFile("scripts/raw_ternary_prepare_setembrobr.py", "utf8");

    expect(prepareScript).toContain('"temporalColumns": TEMPORAL_COLUMNS');
    expect(prepareScript).toContain('"temporal_markers": np.asarray([aggs[uid].temporal_markers()');
    expect(prepareScript).toContain('"sequenceOrder": sequence_order');
    expect(prepareScript).toContain('"recent_chronological"');
    expect(prepareScript).toContain('"tweet_index_desc_take_topN"');
  });

  test("reads raw binary manifests with sealed test labels isolated", async () => {
    const dir = await mkdtemp(join(tmpdir(), "raw-binary-"));
    const manifestDir = join(dir, "manifest");
    await mkdir(manifestDir, { recursive: true });
    const strictPath = join(manifestDir, "strict_blind_split_manifest_seed42.csv");
    await writeFile(
      strictPath,
      [
        "dataset,split,user_id,label,fold,row_hash",
        "setembrobr,train,D1,diagnosed,1,h1",
        "setembrobr,test,T1,-1,,redacted",
        "",
      ].join("\n"),
    );
    await writeFile(
      join(manifestDir, "train_binary_manifest_seed42.csv"),
      "dataset,split,label,user_id,row_hash,fold\nsetembrobr,train,diagnosed,D1,h1,1\n",
    );
    await writeFile(
      join(manifestDir, "test_inference_manifest_seed42.csv"),
      "dataset,split,user_id,label,fold,row_hash\nsetembrobr,test,T1,-1,,redacted\n",
    );
    await writeFile(join(manifestDir, "sealed_test_labels_seed42.csv"), "user_id,binary_label,label_code,row_hash\nT1,control,0,h2\n");

    const config: ProjectConfig = {
      dataset: "setembrobr" as const,
      seed: 42,
      foldCount: 5,
      outputDir: dir,
      featureSource: "raw_artifacts" as const,
      database: {
        tables: {
          trainUserEmb: "unused",
          testUserEmb: "unused",
          trainUserEmbRel3: "unused",
          testUserEmbRel3: "unused",
          trainSubFeatures: "unused",
          testSubFeatures: "unused",
          trainEmbeddings: "unused",
          testEmbeddings: "unused",
        },
        embeddingDimension: 0,
      },
      models: [],
      ensemble: { weightStep: 0.05, primaryMetric: "macroF1" as const, tieBreakers: ["diagnosedF1" as const] },
    };
    expect(await readRawBinaryTrainManifest(config)).toEqual([
      { dataset: "setembrobr", split: "train", label: "diagnosed", userId: "D1", rowHash: "h1", fold: 1 },
    ]);
    expect(await readRawBinarySealedTestLabels(config)).toEqual([{ userId: "T1", label: "control" }]);
    expect((await readRawBinaryAuditManifest(config)).map((row) => row.userId)).toEqual(["D1", "T1"]);
    expect(await rawBinaryStrictBlindManifestHash(config)).toBe(await sha256File(strictPath));
  });

  test("binary stacker source rejects unaligned or labeled test score inputs", async () => {
    const script = await Bun.file("scripts/binary_stack_oof_setembrobr.py").text();
    expect(script).toContain("train OOF user set mismatch");
    expect(script).toContain("label-free score user set mismatch");
    expect(script).toContain("label-free score contains forbidden columns");
    expect(script).toContain("usesTestScoresForTraining");
  });
});
