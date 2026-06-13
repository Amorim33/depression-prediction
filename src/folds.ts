import { createSeededRandom, deriveSeed, shuffleInPlace } from "./random.ts";

export function createStratifiedFolds(labels: readonly string[], foldCount: number, seed: number): number[][] {
  const byLabel = new Map<string, number[]>();
  labels.forEach((label, index) => {
    const bucket = byLabel.get(label) ?? [];
    bucket.push(index);
    byLabel.set(label, bucket);
  });

  const folds = Array.from({ length: foldCount }, () => [] as number[]);
  for (const [labelIndex, label] of [...byLabel.keys()].sort().entries()) {
    const indexes = byLabel.get(label)!;
    shuffleInPlace(indexes, createSeededRandom(deriveSeed(seed, 1 + labelIndex)));
    for (const [index, rowIndex] of indexes.entries()) folds[index % foldCount]!.push(rowIndex);
  }
  for (const [index, fold] of folds.entries()) {
    shuffleInPlace(fold, createSeededRandom(deriveSeed(seed, 10 + index)));
  }
  return folds;
}

export function buildTrainValidationSplit(folds: readonly number[][], validationFoldIndex: number, totalSize: number) {
  const validation = [...folds[validationFoldIndex]!].sort((left, right) => left - right);
  const validationSet = new Set(validation);
  const train: number[] = [];
  for (let index = 0; index < totalSize; index += 1) {
    if (!validationSet.has(index)) train.push(index);
  }
  return { train, validation };
}
