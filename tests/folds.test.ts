import { describe, expect, test } from "bun:test";
import { buildTrainValidationSplit, createStratifiedFolds } from "../src/folds.ts";
import type { BinaryLabel } from "../src/types.ts";

describe("strict-blind folds", () => {
  test("cover each train row exactly once as validation", () => {
    const labels: BinaryLabel[] = [
      ...Array.from({ length: 10 }, () => "diagnosed" as const),
      ...Array.from({ length: 30 }, () => "control" as const),
    ];
    const folds = createStratifiedFolds(labels, 5, 42);
    const seen = new Set<number>();
    for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
      const split = buildTrainValidationSplit(folds, foldIndex, labels.length);
      const trainSet = new Set(split.train);
      for (const index of split.validation) {
        expect(trainSet.has(index)).toBe(false);
        seen.add(index);
      }
    }
    expect(seen.size).toBe(labels.length);
  });

  test("is deterministic for the same seed", () => {
    const labels: BinaryLabel[] = Array.from({ length: 20 }, (_, index) => (index % 4 === 0 ? "diagnosed" : "control"));
    expect(createStratifiedFolds(labels, 5, 42)).toEqual(createStratifiedFolds(labels, 5, 42));
  });
});

