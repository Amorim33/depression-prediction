import { describe, expect, test } from "bun:test";
import { loadConfig } from "../src/config.ts";
import { loadTernaryConfig } from "../src/ternary-config.ts";

const config = await loadConfig();
const ternaryConfig = await loadTernaryConfig();
const tabular = config.candidateModels?.tabular ?? [];
const sequence = config.candidateModels?.sequence ?? [];
const candidateIds = [...tabular.map((model) => model.modelId), ...sequence.map((model) => model.modelId)];

describe("candidate registry", () => {
  test("pre-registers unique candidate model IDs in the model list", () => {
    expect(candidateIds.length).toBeGreaterThan(0);
    expect(new Set(candidateIds).size).toBe(candidateIds.length);
    expect(new Set(config.models).size).toBe(config.models.length);
    for (const modelId of candidateIds) {
      expect(config.models).toContain(modelId);
    }
  });

  test("covers the requested MLP diversity grid", () => {
    const mlp = tabular.filter((model) => model.family === "mlp");
    expect(new Set(mlp.map((model) => model.hiddenSize))).toEqual(new Set([64, 128, 192, 256]));
    expect(new Set(mlp.map((model) => model.alpha))).toEqual(new Set([0.001, 0.005, 0.01]));
    expect(new Set(mlp.map((model) => model.seed))).toEqual(new Set([7, 13, 42]));
    expect(mlp).toHaveLength(4 * 3 * 3);
  });

  test("covers focal and sequence candidate families", () => {
    const focalGammas = new Set(tabular.filter((model) => model.family === "focal_logreg").map((model) => model.gamma));
    expect(focalGammas).toEqual(new Set([0.5, 1, 2, 3]));
    expect(new Set(sequence.map((model) => model.family))).toEqual(new Set(["cnn", "cnn_wide", "bilstm", "tiny_transformer"]));
    expect(new Set(sequence.map((model) => model.topN))).toEqual(new Set([64, 128, 256]));
  });

  test("configures optional DB feature tables as train/test pairs", () => {
    const tables = config.database.tables;
    for (const [trainKey, testKey] of [
      ["trainUserEmbRel6", "testUserEmbRel6"],
      ["trainUserEmbRel7", "testUserEmbRel7"],
      ["trainV2SubFeatures", "testV2SubFeatures"],
      ["trainRel5CombinedFeatures", "testRel5CombinedFeatures"],
    ] as const) {
      expect(Boolean(tables[trainKey])).toBe(true);
      expect(Boolean(tables[testKey])).toBe(true);
    }
  });

  test("pre-registers ternary model selection groups", () => {
    const groups = ternaryConfig.ensemble.selectionGroups ?? [];
    const ternaryModelIds = new Set(ternaryConfig.models);
    expect(groups.map((group) => group.groupId)).toContain("all_models");
    expect(groups.map((group) => group.groupId)).toContain("tabular_all");
    expect(groups.map((group) => group.groupId)).toContain("sequence_all");
    expect(new Set(groups.map((group) => group.groupId)).size).toBe(groups.length);
    for (const group of groups) {
      expect(group.modelIds.length).toBeGreaterThan(0);
      for (const modelId of group.modelIds) expect(ternaryModelIds.has(modelId)).toBe(true);
    }
  });

  test("pre-registers bounded ternary local refinement groups", () => {
    const refinement = ternaryConfig.ensemble.localRefinement;
    expect(refinement?.policyIds).toEqual(["diag_evidence_q20"]);
    expect(refinement?.groupIds).toEqual(["tabular_core_xgb_s42_shallow", "tabular_core_xgb_s42_rich"]);
    expect(refinement?.weightStep).toBe(0.01);
    expect(refinement?.radius).toBe(0.03);
    expect(refinement?.maxModels).toBe(6);
    const groups = new Set((ternaryConfig.ensemble.selectionGroups ?? []).map((group) => group.groupId));
    for (const groupId of refinement?.groupIds ?? []) expect(groups.has(groupId)).toBe(true);
  });

  test("pre-registers ternary XGBoost tabular candidates", () => {
    const xgb = ternaryConfig.candidateModels.tabular.filter((model) => model.family === "xgboost");
    const tabularGroup = ternaryConfig.ensemble.selectionGroups?.find((group) => group.groupId === "tabular_all");
    const boostingGroup = ternaryConfig.ensemble.selectionGroups?.find((group) => group.groupId === "tabular_boosting");
    expect(new Set(xgb.map((model) => model.modelId))).toEqual(
      new Set([
        "ternary_xgb_tabular_markers_s42",
        "ternary_xgb_expanded_pca_s13",
        "ternary_xgb_expanded_pca_s42",
        "ternary_xgb_embedding_rich_s7",
        "ternary_xgb_shallow_pca_s99",
      ]),
    );
    for (const model of xgb) {
      expect(ternaryConfig.models).toContain(model.modelId);
      expect(tabularGroup?.modelIds).toContain(model.modelId);
      expect(boostingGroup?.modelIds).toContain(model.modelId);
      expect(model.nEstimators).toBeGreaterThan(0);
      expect(model.learningRate).toBeGreaterThan(0);
    }
  });

  test("pre-registers ternary histogram gradient boosting candidates", () => {
    const hgb = ternaryConfig.candidateModels.tabular.filter((model) => model.family === "hist_gradient_boosting");
    const tabularGroup = ternaryConfig.ensemble.selectionGroups?.find((group) => group.groupId === "tabular_all");
    const boostingGroup = ternaryConfig.ensemble.selectionGroups?.find((group) => group.groupId === "tabular_boosting");
    expect(new Set(hgb.map((model) => model.modelId))).toEqual(
      new Set(["ternary_hgb_expanded_pca_s42", "ternary_hgb_markers_s13"]),
    );
    for (const model of hgb) {
      expect(ternaryConfig.models).toContain(model.modelId);
      expect(tabularGroup?.modelIds).toContain(model.modelId);
      expect(boostingGroup?.modelIds).toContain(model.modelId);
      expect(model.maxIter).toBeGreaterThan(0);
      expect(model.maxLeafNodes).toBeGreaterThan(0);
      expect(model.learningRate).toBeGreaterThan(0);
    }
  });

  test("pre-registers ternary stacking candidates", () => {
    const stacking = ternaryConfig.candidateModels.stacking ?? [];
    const allModelsGroup = ternaryConfig.ensemble.selectionGroups?.find((group) => group.groupId === "all_models");
    const stackingGroup = ternaryConfig.ensemble.selectionGroups?.find((group) => group.groupId === "stacking_only");
    expect(stacking.map((model) => model.modelId)).toEqual([
      "ternary_stack_logreg_xgb_tabular",
      "ternary_stack_logreg_boosted_core",
      "ternary_stack_logreg_xgb_variants",
    ]);
    const baseModelsById = new Map(stacking.map((model) => [model.modelId, model.baseModelIds]));
    expect(baseModelsById.get("ternary_stack_logreg_xgb_tabular")).toEqual([
      "ternary_hier_logreg_gate",
      "ternary_mlp_h128_s42",
      "ternary_xgb_expanded_pca_s13",
      "ternary_xgb_tabular_markers_s42",
    ]);
    expect(baseModelsById.get("ternary_stack_logreg_boosted_core")).toEqual([
      "ternary_hier_logreg_gate",
      "ternary_mlp_h128_s42",
      "ternary_xgb_expanded_pca_s13",
      "ternary_xgb_shallow_pca_s99",
      "ternary_xgb_tabular_markers_s42",
    ]);
    expect(baseModelsById.get("ternary_stack_logreg_xgb_variants")).toEqual([
      "ternary_xgb_embedding_rich_s7",
      "ternary_xgb_expanded_pca_s13",
      "ternary_xgb_expanded_pca_s42",
      "ternary_xgb_shallow_pca_s99",
    ]);
    for (const model of stacking) {
      expect(model.family).toBe("stacking_logreg");
      expect(ternaryConfig.models).toContain(model.modelId);
      expect(allModelsGroup?.modelIds).toContain(model.modelId);
      expect(stackingGroup?.modelIds).toContain(model.modelId);
    }
  });
});
