import type { BinaryLabel, Metrics } from "./types.ts";

export function predictedLabel(score: number, threshold: number): BinaryLabel {
  return score > threshold ? "diagnosed" : "control";
}

export function computeMetrics(actual: readonly BinaryLabel[], predicted: readonly BinaryLabel[]): Metrics {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const a = actual[index]!;
    const p = predicted[index]!;
    if (a === "diagnosed" && p === "diagnosed") tp += 1;
    else if (a === "control" && p === "diagnosed") fp += 1;
    else if (a === "control" && p === "control") tn += 1;
    else fn += 1;
  }

  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const diagnosedF1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const controlPrecision = tn + fn > 0 ? tn / (tn + fn) : 0;
  const controlRecall = tn + fp > 0 ? tn / (tn + fp) : 0;
  const controlF1 =
    controlPrecision + controlRecall > 0 ? (2 * controlPrecision * controlRecall) / (controlPrecision + controlRecall) : 0;
  return {
    tp,
    fp,
    tn,
    fn,
    precision,
    recall,
    diagnosedF1,
    controlF1,
    macroF1: (diagnosedF1 + controlF1) / 2,
    accuracy: actual.length > 0 ? (tp + tn) / actual.length : 0,
  };
}

export function sweepThreshold(
  rows: readonly { actual: BinaryLabel; score: number }[],
  optimize: "macroF1" | "diagnosedF1" = "macroF1",
): { threshold: number; metrics: Metrics } {
  const sortedScores = [...new Set(rows.map((row) => row.score))].sort((left, right) => left - right);
  const candidates = new Set<number>([-Number.EPSILON, 1 + Number.EPSILON]);
  for (let index = 0; index < sortedScores.length; index += 1) {
    const score = sortedScores[index]!;
    candidates.add(score);
    if (index + 1 < sortedScores.length) candidates.add((score + sortedScores[index + 1]!) / 2);
  }

  let bestThreshold = 0.5;
  let bestMetrics: Metrics | null = null;
  for (const threshold of [...candidates].sort((left, right) => left - right)) {
    const predictions = rows.map((row) => predictedLabel(row.score, threshold));
    const metrics = computeMetrics(rows.map((row) => row.actual), predictions);
    if (!bestMetrics || compareMetrics(metrics, bestMetrics, optimize) > 0) {
      bestThreshold = threshold;
      bestMetrics = metrics;
    }
  }

  if (!bestMetrics) throw new Error("No threshold candidates available");
  return { threshold: bestThreshold, metrics: bestMetrics };
}

function compareMetrics(left: Metrics, right: Metrics, optimize: "macroF1" | "diagnosedF1"): number {
  const keys: Array<keyof Metrics> =
    optimize === "macroF1" ? ["macroF1", "diagnosedF1", "precision", "recall"] : ["diagnosedF1", "macroF1", "precision", "recall"];
  for (const key of keys) {
    const diff = Number(left[key]) - Number(right[key]);
    if (Math.abs(diff) > 1e-12) return diff;
  }
  return 0;
}

