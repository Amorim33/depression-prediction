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
});
