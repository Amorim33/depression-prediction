import type { BinaryLabel, EnsembleLock, Metrics, ScoreRow } from "./types.ts";
import { computeMetrics, predictedLabel, sweepThreshold } from "./metrics.ts";

interface SelectionInput {
  seed: number;
  manifestHash: string;
  oofByModel: ReadonlyMap<string, readonly ScoreRow[]>;
  sourceHashes: Record<string, string>;
  weightStep: number;
  command: string;
}

export function selectEnsemble(input: SelectionInput): EnsembleLock {
  const modelIds = [...input.oofByModel.keys()].sort();
  if (modelIds.length === 0) throw new Error("No OOF models available for ensemble selection");
  const users = alignedUsers(input.oofByModel, modelIds);
  const labels = users.map((userId) => {
    const label = input.oofByModel.get(modelIds[0]!)!.find((row) => row.userId === userId)?.label;
    if (!label) throw new Error(`Missing label for ${userId}`);
    return label;
  });

  let best: { weights: number[]; threshold: number; metrics: Metrics } | null = null;
  for (const weights of enumerateWeights(modelIds.length, input.weightStep)) {
    const scores = users.map((userId) => weightedScore(input.oofByModel, modelIds, weights, userId));
    const { threshold, metrics } = sweepThreshold(
      scores.map((score, index) => ({ score, actual: labels[index]! })),
      "macroF1",
    );
    if (!best || compareForSelection(metrics, best.metrics) > 0) best = { weights, threshold, metrics };
  }
  if (!best) throw new Error("No ensemble candidate selected");

  return {
    dataset: "setembrobr",
    seed: input.seed,
    manifestHash: input.manifestHash,
    modelIds,
    weights: Object.fromEntries(modelIds.map((modelId, index) => [modelId, Number(best!.weights[index]!.toFixed(6))])),
    threshold: best.threshold,
    oofMetrics: best.metrics,
    sourceHashes: input.sourceHashes,
    createdAt: new Date(0).toISOString(),
    command: input.command,
  };
}

export function evaluateLockedEnsemble(
  lock: EnsembleLock,
  testScoresByModel: ReadonlyMap<string, readonly ScoreRow[]>,
  labelsByUser: ReadonlyMap<string, BinaryLabel>,
): Metrics {
  const users = alignedUsers(testScoresByModel, lock.modelIds);
  const actual = users.map((userId) => {
    const label = labelsByUser.get(userId);
    if (!label) throw new Error(`Missing test label for ${userId}`);
    return label;
  });
  const predicted = users.map((userId) => {
    const score = lock.modelIds.reduce((sum, modelId) => {
      const modelRows = testScoresByModel.get(modelId);
      if (!modelRows) throw new Error(`Missing test scores for ${modelId}`);
      const row = modelRows.find((entry) => entry.userId === userId);
      if (!row) throw new Error(`Missing ${modelId} score for ${userId}`);
      return sum + row.score * (lock.weights[modelId] ?? 0);
    }, 0);
    return predictedLabel(score, lock.threshold);
  });
  return computeMetrics(actual, predicted);
}

function alignedUsers(scoreMap: ReadonlyMap<string, readonly ScoreRow[]>, modelIds: readonly string[]): string[] {
  const first = scoreMap.get(modelIds[0]!);
  if (!first) throw new Error(`Missing scores for ${modelIds[0]}`);
  const users = first.map((row) => row.userId).sort();
  for (const modelId of modelIds) {
    const rows = scoreMap.get(modelId);
    if (!rows) throw new Error(`Missing scores for ${modelId}`);
    const modelUsers = rows.map((row) => row.userId).sort();
    if (modelUsers.join("\n") !== users.join("\n")) throw new Error(`User alignment mismatch for ${modelId}`);
  }
  return users;
}

function weightedScore(scoreMap: ReadonlyMap<string, readonly ScoreRow[]>, modelIds: readonly string[], weights: readonly number[], userId: string): number {
  let score = 0;
  for (let index = 0; index < modelIds.length; index += 1) {
    const modelId = modelIds[index]!;
    const row = scoreMap.get(modelId)!.find((entry) => entry.userId === userId);
    if (!row) throw new Error(`Missing ${modelId} score for ${userId}`);
    score += row.score * weights[index]!;
  }
  return score;
}

function enumerateWeights(count: number, step: number): number[][] {
  const precision = Math.round(1 / step);
  const out: number[][] = [];
  function recurse(prefix: number[], remaining: number): void {
    if (prefix.length === count - 1) {
      out.push([...prefix, remaining / precision]);
      return;
    }
    for (let units = 0; units <= remaining; units += 1) recurse([...prefix, units / precision], remaining - units);
  }
  recurse([], precision);
  return out;
}

function compareForSelection(left: Metrics, right: Metrics): number {
  for (const key of ["macroF1", "diagnosedF1", "precision"] as const) {
    const diff = left[key] - right[key];
    if (Math.abs(diff) > 1e-12) return diff;
  }
  return 0;
}

