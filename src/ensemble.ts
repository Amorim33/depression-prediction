import type { BinaryLabel, BinaryLockedPredictionRow, EnsembleLock, Metrics, PredictionTarget, ScoreRow } from "./types.ts";
import { computeMetrics, predictedLabel } from "./metrics.ts";

type ScoreArray = Float64Array<ArrayBufferLike>;

interface SelectionInput {
  seed: number;
  predictionTarget?: PredictionTarget;
  manifestHash: string;
  oofByModel: ReadonlyMap<string, readonly ScoreRow[]>;
  sourceHashes: Record<string, string>;
  weightStep: number;
  command: string;
  exhaustiveModelLimit?: number | undefined;
  candidatePruneTo?: number | undefined;
  maxModels?: number | undefined;
  selectionMode?: "free" | "fixed_model_set" | undefined;
  requiredModelIds?: readonly string[] | undefined;
  minimumWeight?: number | undefined;
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
  const result = input.selectionMode === "fixed_model_set"
    ? selectFixedModelSet(selectionData, modelIds, input.weightStep, input.minimumWeight ?? input.weightStep)
    : modelIds.length <= exhaustiveLimit
      ? selectExhaustive(selectionData, modelIds.map((_modelId, index) => index), input.weightStep, "exhaustive")
      : selectGreedyPruned(selectionData, modelIds, {
          weightStep: input.weightStep,
          candidatePruneTo: input.candidatePruneTo ?? 16,
          maxModels: input.maxModels ?? 8,
        });
  const selected = compactSelection(result, modelIds);

  return {
    dataset: "setembrobr",
    predictionTarget: input.predictionTarget ?? "depression",
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

function selectFixedModelSet(data: SelectionData, modelIds: readonly string[], step: number, minimumWeight: number): SelectionResult {
  let best: { weights: number[]; threshold: number; metrics: Metrics } | null = null;
  for (const weights of enumerateWeights(modelIds.length, step, minimumWeight)) {
    const scores = weightedScores(data.scoresByModel, weights);
    const { threshold, metrics } = sweepThresholdFromAligned(scores, data.labelCodes);
    if (!best || compareForSelection(metrics, best.metrics) > 0) best = { weights, threshold, metrics };
  }
  if (!best) throw new Error("No fixed-model-set ensemble candidate selected");
  return {
    indexes: modelIds.map((_modelId, index) => index),
    weights: best.weights,
    threshold: best.threshold,
    metrics: best.metrics,
    strategy: `fixed-model-set(step=${step},min=${minimumWeight})`,
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
  const rows = predictLockedEnsembleRows(lock, testScoresByModel);
  const actual = rows.map((row) => {
    const label = labelsByUser.get(row.userId);
    if (!label) throw new Error(`Missing test label for ${row.userId}`);
    return label;
  });
  const predicted = rows.map((row) => row.predicted);
  return computeMetrics(actual, predicted);
}

export function predictLockedEnsembleRows(
  lock: EnsembleLock,
  scoresByModel: ReadonlyMap<string, readonly ScoreRow[]>,
): BinaryLockedPredictionRow[] {
  const users = alignedUsers(scoresByModel, lock.modelIds);
  const firstRows = scoresByModel.get(lock.modelIds[0]!);
  if (!firstRows) throw new Error(`Missing scores for ${lock.modelIds[0]}`);
  const firstByUser = new Map(firstRows.map((row) => [row.userId, row]));
  return users.map((userId) => {
    const base = firstByUser.get(userId);
    if (!base) throw new Error(`Missing base score for ${userId}`);
    const score = lock.modelIds.reduce((sum, modelId) => {
      const modelRows = scoresByModel.get(modelId);
      if (!modelRows) throw new Error(`Missing scores for ${modelId}`);
      const row = modelRows.find((entry) => entry.userId === userId);
      if (!row) throw new Error(`Missing ${modelId} score for ${userId}`);
      return sum + row.score * (lock.weights[modelId] ?? 0);
    }, 0);
    const prediction: BinaryLockedPredictionRow = {
      userId,
      score,
      predicted: predictedLabel(score, lock.threshold),
    };
    if (base.label) prediction.label = base.label;
    if (base.fold !== undefined) prediction.fold = base.fold;
    return prediction;
  });
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

function enumerateWeights(count: number, step: number, minimumWeight = 0): number[][] {
  const precision = Math.round(1 / step);
  const minimumUnits = Math.round(minimumWeight * precision);
  if (Math.abs(precision * step - 1) > 1e-9) throw new Error(`weightStep must divide 1 exactly: ${step}`);
  if (minimumUnits * count > precision) throw new Error("minimum ensemble weight is infeasible");
  const out: number[][] = [];
  function recurse(prefix: number[], remaining: number): void {
    if (prefix.length === count - 1) {
      if (remaining >= minimumUnits) out.push([...prefix, remaining / precision]);
      return;
    }
    const remainingSlots = count - prefix.length - 1;
    for (let units = minimumUnits; units <= remaining - remainingSlots * minimumUnits; units += 1) {
      recurse([...prefix, units / precision], remaining - units);
    }
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
