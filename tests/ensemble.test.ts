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
});

