import type { BinaryLabel, EnsembleLock, Metrics, ScoreRow } from "./types.ts";
import { computeMetrics, predictedLabel } from "./metrics.ts";

type ScoreArray = Float64Array<ArrayBufferLike>;

interface SelectionInput {
  seed: number;
  manifestHash: string;
  oofByModel: ReadonlyMap<string, readonly ScoreRow[]>;
  sourceHashes: Record<string, string>;
  weightStep: number;
  command: string;
  exhaustiveModelLimit?: number | undefined;
  candidatePruneTo?: number | undefined;
  maxModels?: number | undefined;
}

interface SelectionData {
  labelCodes: Uint8Array;
  scoresByModel: ScoreArray[];
}

interface SelectionResult {
  indexes: number[];
  weights: number[];
  threshold: number;
  metrics: Metrics;
  strategy: string;
}

export function selectEnsemble(input: SelectionInput): EnsembleLock {
  const modelIds = [...input.oofByModel.keys()].sort();
  if (modelIds.length === 0) throw new Error("No OOF models available for ensemble selection");
  const selectionData = buildSelectionData(input.oofByModel, modelIds);

  const exhaustiveLimit = input.exhaustiveModelLimit ?? 7;
  const result =
    modelIds.length <= exhaustiveLimit
      ? selectExhaustive(selectionData, modelIds.map((_modelId, index) => index), input.weightStep, "exhaustive")
      : selectGreedyPruned(selectionData, modelIds, {
          weightStep: input.weightStep,
          candidatePruneTo: input.candidatePruneTo ?? 16,
          maxModels: input.maxModels ?? 8,
        });
  const selected = compactSelection(result, modelIds);

  return {
    dataset: "setembrobr",
    seed: input.seed,
    manifestHash: input.manifestHash,
    modelIds: selected.modelIds,
    weights: Object.fromEntries(selected.modelIds.map((modelId) => [modelId, Number(selected.weightsByModel.get(modelId)!.toFixed(6))])),
    threshold: result.threshold,
    oofMetrics: result.metrics,
    sourceHashes: Object.fromEntries(selected.modelIds.map((modelId) => [modelId, input.sourceHashes[modelId] ?? "missing"])),
    createdAt: new Date(0).toISOString(),
    command: input.command,
    selectionStrategy: result.strategy,
  };
}

function buildSelectionData(
  scoreMap: ReadonlyMap<string, readonly ScoreRow[]>,
  modelIds: readonly string[],
): SelectionData {
  const users = alignedUsers(scoreMap, modelIds);
  const labelCodes = new Uint8Array(users.length);
  const scoresByModel = modelIds.map((modelId) => {
    const rows = scoreMap.get(modelId);
    if (!rows) throw new Error(`Missing scores for ${modelId}`);
    const byUser = new Map(rows.map((row) => [row.userId, row]));
    const scores = new Float64Array(users.length);
    for (let index = 0; index < users.length; index += 1) {
      const userId = users[index]!;
      const row = byUser.get(userId);
      if (!row) throw new Error(`Missing ${modelId} score for ${userId}`);
      if (!Number.isFinite(row.score)) throw new Error(`Non-finite ${modelId} score for ${userId}`);
      scores[index] = row.score;
      if (modelId === modelIds[0]) {
        if (!row.label) throw new Error(`Missing label for ${userId}`);
        labelCodes[index] = row.label === "diagnosed" ? 1 : 0;
      }
    }
    return scores;
  });
  return { labelCodes, scoresByModel };
}

function selectExhaustive(data: SelectionData, indexes: readonly number[], step: number, strategy: string): SelectionResult {
  let best: { weights: number[]; threshold: number; metrics: Metrics } | null = null;
  const scoresByModel = indexes.map((index) => data.scoresByModel[index]!);
  for (const weights of enumerateWeights(indexes.length, step)) {
    const scores = weightedScores(scoresByModel, weights);
    const { threshold, metrics } = sweepThresholdFromAligned(scores, data.labelCodes);
    if (!best || compareForSelection(metrics, best.metrics) > 0) best = { weights, threshold, metrics };
  }
  if (!best) throw new Error("No ensemble candidate selected");
  return { indexes: [...indexes], weights: best.weights, threshold: best.threshold, metrics: best.metrics, strategy };
}

function selectGreedyPruned(
  data: SelectionData,
  modelIds: readonly string[],
  options: { weightStep: number; candidatePruneTo: number; maxModels: number },
): SelectionResult {
  const ranked = rankSingleModels(data, modelIds).slice(0, options.candidatePruneTo);
  if (ranked.length === 0) throw new Error("No ranked model candidates available");

  let best: SelectionResult | null = null;
  const maxPrefix = Math.min(options.maxModels, ranked.length);
  for (let size = 1; size <= maxPrefix; size += 1) {
    const prefix = ranked.slice(0, size);
    for (const weights of [equalWeights(size), metricWeights(prefix)]) {
      const scores = weightedScores(
        prefix.map((candidate) => data.scoresByModel[candidate.index]!),
        weights,
      );
      const { threshold, metrics } = sweepThresholdFromAligned(scores, data.labelCodes);
      if (!best || compareForSelection(metrics, best.metrics) > 0) {
        best = {
          indexes: prefix.map((candidate) => candidate.index),
          weights,
          threshold,
          metrics,
          strategy: `ranked-prefix-pruned(top=${ranked.length},max=${options.maxModels},step=${options.weightStep})`,
        };
      }
    }
  }

  if (!best) throw new Error("No pruned ensemble candidate selected");
  return best;
}

function rankSingleModels(
  data: SelectionData,
  modelIds: readonly string[],
): Array<{ index: number; threshold: number; metrics: Metrics }> {
  return modelIds
    .map((_modelId, index) => {
      const { threshold, metrics } = sweepThresholdFromAligned(data.scoresByModel[index]!, data.labelCodes);
      return { index, threshold, metrics };
    })
    .sort((left, right) => {
      const metricDiff = compareForSelection(right.metrics, left.metrics);
      if (metricDiff !== 0) return metricDiff;
      return modelIds[left.index]!.localeCompare(modelIds[right.index]!);
    });
}

function compactSelection(result: SelectionResult, modelIds: readonly string[]): { modelIds: string[]; weightsByModel: Map<string, number> } {
  const compacted = compactIndexWeights(result.indexes, result.weights);
  const weightsByModel = new Map<string, number>();
  for (let index = 0; index < compacted.indexes.length; index += 1) {
    weightsByModel.set(modelIds[compacted.indexes[index]!]!, compacted.weights[index]!);
  }
  const selectedModelIds = [...weightsByModel.keys()].sort();
  return { modelIds: selectedModelIds, weightsByModel };
}

function equalWeights(size: number): number[] {
  return Array.from({ length: size }, () => 1 / size);
}

function metricWeights(prefix: readonly { metrics: Metrics }[]): number[] {
  const raw = prefix.map((candidate) => Math.max(candidate.metrics.macroF1, 1e-12) ** 4);
  const total = raw.reduce((sum, value) => sum + value, 0);
  return total > 0 ? raw.map((value) => value / total) : equalWeights(prefix.length);
}

function compactIndexWeights(indexes: readonly number[], weights: readonly number[]): { indexes: number[]; weights: number[] } {
  const byIndex = new Map<number, number>();
  for (let index = 0; index < indexes.length; index += 1) {
    const weight = weights[index] ?? 0;
    if (weight <= 1e-12) continue;
    byIndex.set(indexes[index]!, (byIndex.get(indexes[index]!) ?? 0) + weight);
  }
  const compactedIndexes = [...byIndex.keys()].sort((left, right) => left - right);
  const weightSum = compactedIndexes.reduce((sum, index) => sum + (byIndex.get(index) ?? 0), 0);
  return {
    indexes: compactedIndexes,
    weights: compactedIndexes.map((index) => (byIndex.get(index) ?? 0) / weightSum),
  };
}

function weightedScores(scoresByModel: readonly ScoreArray[], weights: readonly number[]): ScoreArray {
  const userCount = scoresByModel[0]?.length ?? 0;
  const scores = new Float64Array(userCount);
  for (let modelIndex = 0; modelIndex < scoresByModel.length; modelIndex += 1) {
    const weight = weights[modelIndex] ?? 0;
    if (weight === 0) continue;
    const modelScores = scoresByModel[modelIndex]!;
    for (let userIndex = 0; userIndex < userCount; userIndex += 1) {
      scores[userIndex] = scores[userIndex]! + modelScores[userIndex]! * weight;
    }
  }
  return scores;
}

function sweepThresholdFromAligned(scores: ScoreArray, labelCodes: Uint8Array): { threshold: number; metrics: Metrics } {
  const order = [...scores.keys()].sort((left, right) => scores[left]! - scores[right]!);
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (let index = 0; index < scores.length; index += 1) {
    const actualDiagnosed = labelCodes[index] === 1;
    const predictedDiagnosed = scores[index]! > -Number.EPSILON;
    if (actualDiagnosed && predictedDiagnosed) tp += 1;
    else if (!actualDiagnosed && predictedDiagnosed) fp += 1;
    else if (!actualDiagnosed) tn += 1;
    else fn += 1;
  }

  let bestThreshold = -Number.EPSILON;
  let bestMetrics = metricsFromCounts(tp, fp, tn, fn);

  for (let orderIndex = 0; orderIndex < order.length; ) {
    const threshold = scores[order[orderIndex]!]!;
    while (orderIndex < order.length && scores[order[orderIndex]!] === threshold) {
      const userIndex = order[orderIndex]!;
      if (scores[userIndex]! > -Number.EPSILON) {
        if (labelCodes[userIndex] === 1) {
          tp -= 1;
          fn += 1;
        } else {
          fp -= 1;
          tn += 1;
        }
      }
      orderIndex += 1;
    }
    const metrics = metricsFromCounts(tp, fp, tn, fn);
    if (compareForThreshold(metrics, bestMetrics) > 0) {
      bestThreshold = threshold;
      bestMetrics = metrics;
    }
  }

  return { threshold: bestThreshold, metrics: bestMetrics };
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

function compareForThreshold(left: Metrics, right: Metrics): number {
  for (const key of ["macroF1", "diagnosedF1", "precision", "recall"] as const) {
    const diff = left[key] - right[key];
    if (Math.abs(diff) > 1e-12) return diff;
  }
  return 0;
}

function metricsFromCounts(tp: number, fp: number, tn: number, fn: number): Metrics {
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const diagnosedF1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const controlPrecision = tn + fn > 0 ? tn / (tn + fn) : 0;
  const controlRecall = tn + fp > 0 ? tn / (tn + fp) : 0;
  const controlF1 =
    controlPrecision + controlRecall > 0 ? (2 * controlPrecision * controlRecall) / (controlPrecision + controlRecall) : 0;
  const total = tp + fp + tn + fn;
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
    accuracy: total > 0 ? (tp + tn) / total : 0,
  };
}
