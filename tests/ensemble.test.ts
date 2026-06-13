import { describe, expect, test } from "bun:test";
import { selectEnsemble, evaluateLockedEnsemble } from "../src/ensemble.ts";
import type { BinaryLabel, ScoreRow } from "../src/types.ts";

describe("ensemble selection", () => {
  test("selects and evaluates from aligned OOF/test scores", () => {
    const oof: ScoreRow[] = [
      { userId: "A", label: "diagnosed", fold: 1, score: 0.9, modelId: "m1" },
      { userId: "B", label: "control", fold: 2, score: 0.1, modelId: "m1" },
    ];
    const lock = selectEnsemble({
      seed: 42,
      manifestHash: "manifest",
      oofByModel: new Map([["m1", oof]]),
      sourceHashes: { m1: "hash" },
      weightStep: 0.5,
      command: "test",
    });
    const labels = new Map<string, BinaryLabel>([
      ["A", "diagnosed"],
      ["B", "control"],
    ]);
    const metrics = evaluateLockedEnsemble(lock, new Map([["m1", [{ userId: "A", score: 0.9, modelId: "m1" }, { userId: "B", score: 0.1, modelId: "m1" }]]]), labels);
    expect(metrics.tp).toBe(1);
    expect(metrics.tn).toBe(1);
  });

  test("uses greedy pruning when the model set is larger than the exhaustive limit", () => {
    const labels: Array<[string, BinaryLabel]> = [
      ["A", "diagnosed"],
      ["B", "control"],
      ["C", "diagnosed"],
      ["D", "control"],
    ];
    const scoreTemplates = [
      [0.95, 0.05, 0.85, 0.15],
      [0.9, 0.1, 0.3, 0.2],
      [0.7, 0.4, 0.6, 0.3],
      [0.6, 0.2, 0.5, 0.4],
      [0.55, 0.45, 0.52, 0.48],
      [0.45, 0.55, 0.48, 0.52],
      [0.4, 0.6, 0.35, 0.65],
      [0.3, 0.7, 0.25, 0.75],
    ];
    const oofByModel = new Map<string, ScoreRow[]>();
    for (let index = 0; index < scoreTemplates.length; index += 1) {
      const modelId = `m${index}`;
      oofByModel.set(
        modelId,
        labels.map(([userId, label], rowIndex) => ({
          userId,
          label,
          fold: rowIndex % 2,
          score: scoreTemplates[index]![rowIndex]!,
          modelId,
        })),
      );
    }
    const lock = selectEnsemble({
      seed: 42,
      manifestHash: "manifest",
      oofByModel,
      sourceHashes: Object.fromEntries([...oofByModel.keys()].map((modelId) => [modelId, "hash"])),
      weightStep: 0.5,
      exhaustiveModelLimit: 3,
      candidatePruneTo: 4,
      maxModels: 3,
      command: "test",
    });
    expect(lock.selectionStrategy?.startsWith("ranked-prefix-pruned")).toBe(true);
    expect(lock.modelIds.length).toBeLessThanOrEqual(3);
    for (const modelId of lock.modelIds) {
      expect(lock.weights[modelId]).toBeGreaterThan(0);
    }
  });
});
