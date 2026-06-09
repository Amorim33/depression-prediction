import type { BinaryLabel } from "./types.ts";
import { createSeededRandom, deriveSeed, shuffleInPlace } from "./random.ts";

export function createStratifiedFolds(labels: readonly BinaryLabel[], foldCount: number, seed: number): number[][] {
  const diagnosed: number[] = [];
  const control: number[] = [];
  labels.forEach((label, index) => {
    if (label === "diagnosed") diagnosed.push(index);
    else control.push(index);
  });

  shuffleInPlace(diagnosed, createSeededRandom(deriveSeed(seed, 1)));
  shuffleInPlace(control, createSeededRandom(deriveSeed(seed, 2)));

  const folds = Array.from({ length: foldCount }, () => [] as number[]);
  for (const [index, rowIndex] of diagnosed.entries()) folds[index % foldCount]!.push(rowIndex);
  for (const [index, rowIndex] of control.entries()) folds[index % foldCount]!.push(rowIndex);
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

