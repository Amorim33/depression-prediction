import { readFile } from "node:fs/promises";
import { parseCsv, writeCsv } from "./csv.ts";
import { sha256Text, stableJson } from "./hash.ts";
import type { AuditReport } from "./audit.ts";
import type {
  BinaryLabel,
  EvidenceMarker,
  TernaryDecisionRule,
  TernaryEnsembleLock,
  TernaryLabel,
  TernaryLabelPolicyConfig,
  TernaryLabelPolicyLock,
  TernaryManifestRow,
  TernaryMetrics,
  TernaryProbabilityRow,
} from "./types.ts";

export const TERNARY_LABELS = ["diagnosed", "control", "no-evidence"] as const satisfies readonly TernaryLabel[];
export const EVIDENCE_FORMULA_VERSION = "v1";

type ProbTriple = [number, number, number];

interface TernarySelectionInput {
  seed: number;
  originalManifestHash: string;
  labelPolicyId: string;
  labelPolicyHash: string;
  oofByModel: ReadonlyMap<string, readonly TernaryProbabilityRow[]>;
  sourceHashes: Record<string, string>;
  weightStep: number;
  decisionRules: readonly TernaryDecisionRule[];
  command: string;
  exhaustiveModelLimit?: number | undefined;
  candidatePruneTo?: number | undefined;
  maxModels?: number | undefined;
  refineWeightStep?: number | undefined;
  refineWeightRadius?: number | undefined;
  refineModelLimit?: number | undefined;
}

interface SelectionData {
  users: string[];
  labels: TernaryLabel[];
  probsByModel: ProbTriple[][];
}

interface SelectionResult {
  indexes: number[];
  weights: number[];
  decisionRule: TernaryDecisionRule;
  metrics: TernaryMetrics;
  strategy: string;
}

export function normalizeTernaryLabel(value: unknown): TernaryLabel {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (normalized === "diagnosed") return "diagnosed";
  if (normalized === "control") return "control";
  if (normalized === "no-evidence" || normalized === "no_evidence" || normalized === "no evidence") return "no-evidence";
  throw new Error(`Unknown ternary label: ${String(value)}`);
}

export function computeEvidenceScore(marker: Omit<EvidenceMarker, "evidenceScore">): number {
  return clamp01(
    0.3 * clamp01(marker.rel7Ratio * 10) +
      0.25 * clamp01(marker.rel5Ratio * 6) +
      0.15 * clamp01(marker.rel3Ratio * 3) +
      0.15 * clamp01(marker.top10AvgRelevance / 7) +
      0.15 * clamp01(marker.maxRelevance / 7),
  );
}

export function parseEvidenceMarkerRecord(record: Record<string, string>): EvidenceMarker {
  const totalTweets = numberField(record, "total_tweets");
  const base = {
    userId: required(record, "user_id"),
    totalTweets,
    maxRelevance: numberField(record, "max_relevance"),
    rel3Count: numberField(record, "rel3_count"),
    rel5Count: numberField(record, "rel5_count"),
    rel6Count: numberField(record, "rel6_count"),
    rel7Count: numberField(record, "rel7_count"),
    rel3Ratio: numberField(record, "rel3_ratio"),
    rel5Ratio: numberField(record, "rel5_ratio"),
    rel6Ratio: numberField(record, "rel6_ratio"),
    rel7Ratio: numberField(record, "rel7_ratio"),
    top10AvgRelevance: numberField(record, "top10_avg_relevance"),
  };
  const evidenceScore =
    record.evidence_score === undefined || record.evidence_score === ""
      ? computeEvidenceScore(base)
      : numberField(record, "evidence_score");
  return { ...base, totalTweets, evidenceScore };
}

export function evidenceMarkerToCsvRecord(marker: EvidenceMarker): Record<string, string | number> {
  return {
    user_id: marker.userId,
    total_tweets: marker.totalTweets,
    max_relevance: fixed(marker.maxRelevance),
    rel3_count: marker.rel3Count,
    rel5_count: marker.rel5Count,
    rel6_count: marker.rel6Count,
    rel7_count: marker.rel7Count,
    rel3_ratio: fixed(marker.rel3Ratio),
    rel5_ratio: fixed(marker.rel5Ratio),
    rel6_ratio: fixed(marker.rel6Ratio),
    rel7_ratio: fixed(marker.rel7Ratio),
    top10_avg_relevance: fixed(marker.top10AvgRelevance),
    evidence_score: fixed(marker.evidenceScore),
  };
}

export async function readEvidenceMarkers(path: string): Promise<EvidenceMarker[]> {
  return parseCsv(await readFile(path, "utf8")).map(parseEvidenceMarkerRecord);
}

export function lockTernaryLabelPolicy(
  policy: TernaryLabelPolicyConfig,
  trainRows: readonly { binaryLabel: BinaryLabel; marker: EvidenceMarker }[],
  originalManifestHash: string,
  seed: number,
): TernaryLabelPolicyLock {
  const cutoff =
    policy.kind === "evidence_quantile"
      ? quantile(
          trainRows.filter((row) => row.binaryLabel === "diagnosed").map((row) => row.marker.evidenceScore),
          policy.quantile ?? 0,
        )
      : undefined;
  const base = {
    ...policy,
    dataset: "setembrobr" as const,
    seed,
    originalManifestHash,
    evidenceFormulaVersion: EVIDENCE_FORMULA_VERSION as "v1",
    ...(cutoff === undefined ? {} : { cutoff }),
    createdAt: new Date(0).toISOString(),
  };
  return { ...base, policyHash: sha256Text(stableJson(base)) };
}

export function deriveTernaryLabel(
  binaryLabel: BinaryLabel,
  marker: EvidenceMarker,
  policy: TernaryLabelPolicyLock,
): TernaryLabel {
  if (binaryLabel === "control") return "control";
  if (policy.kind === "rel_count_zero") {
    return relevanceCount(marker, policy.relevanceThreshold ?? 3) === 0 ? "no-evidence" : "diagnosed";
  }
  if (policy.kind === "low_density") {
    return marker.rel3Ratio <= (policy.densityThreshold ?? 0.01) ? "no-evidence" : "diagnosed";
  }
  if (policy.kind === "top10_avg_lt") {
    return marker.top10AvgRelevance < (policy.top10AvgThreshold ?? 3) ? "no-evidence" : "diagnosed";
  }
  return marker.evidenceScore <= (policy.cutoff ?? 0) ? "no-evidence" : "diagnosed";
}

export function readTernaryManifestText(text: string): TernaryManifestRow[] {
  return parseCsv(text).map((record) => ({
    dataset: "setembrobr",
    split: "train",
    label: normalizeTernaryLabel(required(record, "label")),
    binaryLabel: normalizeBinaryLabel(required(record, "binary_label")),
    userId: required(record, "user_id"),
    rowHash: required(record, "row_hash"),
    fold: Number(required(record, "fold")),
    labelPolicyId: required(record, "label_policy_id"),
  }));
}

export async function readTernaryManifest(path: string): Promise<TernaryManifestRow[]> {
  return readTernaryManifestText(await readFile(path, "utf8"));
}

export function writeTernaryManifestCsv(rows: readonly TernaryManifestRow[]): string {
  return writeCsv(
    ["dataset", "split", "label", "binary_label", "user_id", "row_hash", "fold", "label_policy_id"],
    rows.map((row) => ({
      dataset: row.dataset,
      split: row.split,
      label: row.label,
      binary_label: row.binaryLabel,
      user_id: row.userId,
      row_hash: row.rowHash,
      fold: row.fold,
      label_policy_id: row.labelPolicyId,
    })),
  );
}

export async function readTernaryOofScores(path: string): Promise<TernaryProbabilityRow[]> {
  return readTernaryOofScoreText(await readFile(path, "utf8"));
}

export async function readTernaryTestScores(path: string): Promise<TernaryProbabilityRow[]> {
  return readTernaryTestScoreText(await readFile(path, "utf8"));
}

export function readTernaryOofScoreText(text: string): TernaryProbabilityRow[] {
  return parseCsv(text).map((record) => ({
    userId: required(record, "user_id"),
    label: normalizeTernaryLabel(required(record, "label")),
    fold: Number(required(record, "fold")),
    probDiagnosed: numberField(record, "prob_diagnosed"),
    probControl: numberField(record, "prob_control"),
    probNoEvidence: numberField(record, "prob_no_evidence"),
    modelId: required(record, "model_id"),
    labelPolicyId: required(record, "label_policy_id"),
  }));
}

export function readTernaryTestScoreText(text: string): TernaryProbabilityRow[] {
  return parseCsv(text).map((record) => ({
    userId: required(record, "user_id"),
    probDiagnosed: numberField(record, "prob_diagnosed"),
    probControl: numberField(record, "prob_control"),
    probNoEvidence: numberField(record, "prob_no_evidence"),
    modelId: required(record, "model_id"),
    labelPolicyId: required(record, "label_policy_id"),
  }));
}

export function predictedTernaryLabel(probs: ProbTriple, rule: TernaryDecisionRule): TernaryLabel {
  if (rule.kind === "diagnosed_margin") {
    const margin = rule.diagnosedMargin ?? 0;
    if (probs[0] >= Math.max(probs[1], probs[2]) + margin) return "diagnosed";
    return probs[1] >= probs[2] ? "control" : "no-evidence";
  }
  if (rule.kind === "no_evidence_gate") {
    const min = rule.noEvidenceMin ?? 0.5;
    if (probs[2] >= min && probs[2] >= Math.max(probs[0], probs[1])) return "no-evidence";
  }
  return TERNARY_LABELS[argmaxIndex(probs)]!;
}

export function computeTernaryMetrics(actual: readonly TernaryLabel[], predicted: readonly TernaryLabel[]): TernaryMetrics {
  if (actual.length !== predicted.length) throw new Error("Actual and predicted ternary labels must have the same length");
  const confusion = emptyConfusion();
  for (let index = 0; index < actual.length; index += 1) {
    confusion[actual[index]!]![predicted[index]!] += 1;
  }

  const perClass = Object.fromEntries(
    TERNARY_LABELS.map((label) => {
      const tp = confusion[label][label];
      const fp = TERNARY_LABELS.filter((other) => other !== label).reduce((sum, other) => sum + confusion[other][label], 0);
      const fn = TERNARY_LABELS.filter((other) => other !== label).reduce((sum, other) => sum + confusion[label][other], 0);
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
      const support = TERNARY_LABELS.reduce((sum, predictedLabel) => sum + confusion[label][predictedLabel], 0);
      return [label, { precision, recall, f1, support }];
    }),
  ) as TernaryMetrics["perClass"];
  const correct = TERNARY_LABELS.reduce((sum, label) => sum + confusion[label][label], 0);
  return {
    macroF1: TERNARY_LABELS.reduce((sum, label) => sum + perClass[label].f1, 0) / TERNARY_LABELS.length,
    accuracy: actual.length > 0 ? correct / actual.length : 0,
    diagnosedF1: perClass.diagnosed.f1,
    diagnosedPrecision: perClass.diagnosed.precision,
    diagnosedRecall: perClass.diagnosed.recall,
    perClass,
    confusion,
  };
}

export function compareTernaryMetrics(left: TernaryMetrics, right: TernaryMetrics): number {
  for (const key of ["macroF1", "diagnosedF1", "diagnosedPrecision", "accuracy"] as const) {
    const diff = left[key] - right[key];
    if (Math.abs(diff) > 1e-12) return diff;
  }
  return 0;
}

export function selectTernaryEnsemble(input: TernarySelectionInput): TernaryEnsembleLock {
  const modelIds = [...input.oofByModel.keys()].sort();
  if (modelIds.length === 0) throw new Error("No ternary OOF models available for ensemble selection");
  if (input.decisionRules.length === 0) throw new Error("No ternary decision rules configured");
  const data = buildSelectionData(input.oofByModel, modelIds, input.labelPolicyId);
  const exhaustiveLimit = input.exhaustiveModelLimit ?? 6;
  const coarseResult =
    modelIds.length <= exhaustiveLimit
      ? selectTernaryExhaustive(data, modelIds.map((_modelId, index) => index), input.weightStep, input.decisionRules, "exhaustive")
      : selectTernaryGreedyPruned(data, modelIds, input.decisionRules, {
          weightStep: input.weightStep,
          candidatePruneTo: input.candidatePruneTo ?? 16,
          maxModels: input.maxModels ?? 8,
        });
  const result = refineTernaryWeights(data, coarseResult, input.decisionRules, {
    weightStep: input.refineWeightStep,
    radius: input.refineWeightRadius,
    maxModels: input.refineModelLimit,
  });
  const compacted = compactIndexWeights(result.indexes, result.weights);
  const selectedModelIds = compacted.indexes.map((index) => modelIds[index]!);
  return {
    dataset: "setembrobr",
    seed: input.seed,
    originalManifestHash: input.originalManifestHash,
    labelPolicyId: input.labelPolicyId,
    labelPolicyHash: input.labelPolicyHash,
    modelIds: selectedModelIds,
    weights: Object.fromEntries(selectedModelIds.map((modelId, index) => [modelId, Number(compacted.weights[index]!.toFixed(6))])),
    decisionRule: result.decisionRule,
    oofMetrics: result.metrics,
    sourceHashes: Object.fromEntries(selectedModelIds.map((modelId) => [modelId, input.sourceHashes[modelId] ?? "missing"])),
    createdAt: new Date(0).toISOString(),
    command: input.command,
    selectionStrategy: result.strategy,
  };
}

function refineTernaryWeights(
  data: SelectionData,
  result: SelectionResult,
  decisionRules: readonly TernaryDecisionRule[],
  options: { weightStep?: number | undefined; radius?: number | undefined; maxModels?: number | undefined },
): SelectionResult {
  if (options.weightStep === undefined || options.radius === undefined || options.maxModels === undefined) return result;
  if (options.weightStep <= 0 || options.radius <= 0) return result;
  const refineStep = options.weightStep;
  const radius = options.radius;
  const maxModels = options.maxModels;
  const compacted = compactIndexWeights(result.indexes, result.weights);
  if (compacted.indexes.length < 2 || compacted.indexes.length > maxModels) return result;

  const precision = Math.round(1 / refineStep);
  if (precision < 1 || Math.abs(1 / precision - refineStep) > 1e-12) {
    throw new Error(`Refinement weight step must divide 1 exactly: ${refineStep}`);
  }
  const ranges = compacted.weights.map((weight) => ({
    min: Math.max(0, Math.floor((weight - radius) * precision)),
    max: Math.min(precision, Math.ceil((weight + radius) * precision)),
  }));
  const probsByModel = compacted.indexes.map((index) => data.probsByModel[index]!);
  let best: SelectionResult = { ...result };
  forEachBoundedWeight(ranges, precision, (weights) => {
    const candidate = evaluateWeightedCandidate(data.labels, probsByModel, weights, decisionRules);
    if (compareTernaryMetrics(candidate.metrics, best.metrics) > 0) {
      best = {
        indexes: compacted.indexes,
        weights,
        ...candidate,
        strategy: `${result.strategy}+local-refine(step=${refineStep},radius=${radius})`,
      };
    }
  });
  return best;
}

export function evaluateTernaryLockedEnsemble(
  lock: TernaryEnsembleLock,
  testScoresByModel: ReadonlyMap<string, readonly TernaryProbabilityRow[]>,
  labelsByUser: ReadonlyMap<string, TernaryLabel>,
): TernaryMetrics {
  const users = alignedUsers(testScoresByModel, lock.modelIds);
  const predicted = users.map((userId) => {
    const probs = weightedUserProbability(userId, lock.modelIds, lock.weights, testScoresByModel);
    return predictedTernaryLabel(probs, lock.decisionRule);
  });
  const actual = users.map((userId) => {
    const label = labelsByUser.get(userId);
    if (!label) throw new Error(`Missing ternary test label for ${userId}`);
    return label;
  });
  return computeTernaryMetrics(actual, predicted);
}

export function auditTernaryOofScores(
  trainManifestRows: readonly TernaryManifestRow[],
  testUsers: ReadonlySet<string>,
  oofByModel: ReadonlyMap<string, readonly TernaryProbabilityRow[]>,
  expectedPolicyId: string,
): AuditReport {
  const findings = [];
  const trainByUser = new Map(trainManifestRows.map((row) => [row.userId, row]));
  for (const [modelKey, rows] of oofByModel) {
    const seen = new Set<string>();
    for (const row of rows) {
      if (testUsers.has(row.userId)) findings.push(fail("ternary-oof-test-user", `${modelKey}: OOF row contains test user ${row.userId}`));
      const manifest = trainByUser.get(row.userId);
      if (!manifest) findings.push(fail("ternary-oof-unknown-user", `${modelKey}: ${row.userId}`));
      else {
        if (row.label !== manifest.label) findings.push(fail("ternary-oof-label-mismatch", `${modelKey}: ${row.userId}`));
        if (row.fold !== manifest.fold) findings.push(fail("ternary-oof-fold-mismatch", `${modelKey}: ${row.userId}`));
      }
      if (row.labelPolicyId !== expectedPolicyId) {
        findings.push(fail("ternary-oof-policy-mismatch", `${modelKey}: ${row.userId}`));
      }
      if (!probabilitiesValid(row)) findings.push(fail("ternary-oof-invalid-probs", `${modelKey}: ${row.userId}`));
      if (seen.has(row.userId)) findings.push(fail("ternary-oof-duplicate-user", `${modelKey}: ${row.userId}`));
      seen.add(row.userId);
    }
    for (const manifest of trainManifestRows) {
      if (!seen.has(manifest.userId)) findings.push(fail("ternary-oof-missing-user", `${modelKey}: ${manifest.userId}`));
    }
  }
  if (findings.length === 0) findings.push(pass("ternary-oof-integrity", `${expectedPolicyId}: OOF rows match train manifest exactly`));
  return { ok: findings.every((finding) => finding.ok), findings };
}

export function auditTernaryTestScoreSchema(fileName: string, csvText: string): AuditReport {
  const [headerLine = ""] = csvText.split(/\r?\n/u);
  const headers = headerLine.split(",");
  const findings = [];
  for (const forbidden of ["label", "actual", "predicted", "fold", "threshold", "macro_f1", "macroF1"]) {
    if (headers.includes(forbidden)) findings.push(fail("ternary-test-forbidden-column", `${fileName}: forbidden column ${forbidden}`));
  }
  for (const requiredHeader of ["user_id", "prob_diagnosed", "prob_control", "prob_no_evidence", "model_id", "label_policy_id"]) {
    if (!headers.includes(requiredHeader)) findings.push(fail("ternary-test-missing-column", `${fileName}: missing ${requiredHeader}`));
  }
  for (const [index, row] of readTernaryTestScoreText(csvText).entries()) {
    if (!probabilitiesValid(row)) findings.push(fail("ternary-test-invalid-probs", `${fileName}: row ${index + 2}`));
  }
  if (findings.length === 0) findings.push(pass("ternary-test-score-schema", `${fileName}: label-free probability schema`));
  return { ok: findings.every((finding) => finding.ok), findings };
}

function buildSelectionData(
  scoreMap: ReadonlyMap<string, readonly TernaryProbabilityRow[]>,
  modelIds: readonly string[],
  expectedPolicyId: string,
): SelectionData {
  const users = alignedUsers(scoreMap, modelIds);
  const labels: TernaryLabel[] = [];
  const probsByModel = modelIds.map((modelId) => {
    const rows = scoreMap.get(modelId);
    if (!rows) throw new Error(`Missing ternary scores for ${modelId}`);
    const byUser = new Map(rows.map((row) => [row.userId, row]));
    return users.map((userId, index) => {
      const row = byUser.get(userId);
      if (!row) throw new Error(`Missing ${modelId} ternary score for ${userId}`);
      if (row.labelPolicyId !== expectedPolicyId) throw new Error(`${modelId}: unexpected label policy ${row.labelPolicyId}`);
      if (!probabilitiesValid(row)) throw new Error(`${modelId}: invalid probabilities for ${userId}`);
      if (!row.label) throw new Error(`${modelId}: missing ternary label for ${userId}`);
      if (modelId === modelIds[0]) labels[index] = row.label;
      else if (labels[index] !== row.label) throw new Error(`${modelId}: label mismatch for ${userId}`);
      return rowToProbTriple(row);
    });
  });
  return { users, labels, probsByModel };
}

function selectTernaryExhaustive(
  data: SelectionData,
  indexes: readonly number[],
  step: number,
  decisionRules: readonly TernaryDecisionRule[],
  strategy: string,
): SelectionResult {
  let best: SelectionResult | null = null;
  const probsByModel = indexes.map((index) => data.probsByModel[index]!);
  for (const weights of enumerateWeights(indexes.length, step)) {
    const candidate = evaluateWeightedCandidate(data.labels, probsByModel, weights, decisionRules);
    if (!best || compareTernaryMetrics(candidate.metrics, best.metrics) > 0) {
      best = { indexes: [...indexes], weights, ...candidate, strategy };
    }
  }
  if (!best) throw new Error("No ternary ensemble candidate selected");
  return best;
}

function selectTernaryGreedyPruned(
  data: SelectionData,
  modelIds: readonly string[],
  decisionRules: readonly TernaryDecisionRule[],
  options: { weightStep: number; candidatePruneTo: number; maxModels: number },
): SelectionResult {
  const ranked = rankSingleTernaryModels(data, modelIds, decisionRules).slice(0, options.candidatePruneTo);
  let best: SelectionResult | null = null;
  for (let size = 1; size <= Math.min(options.maxModels, ranked.length); size += 1) {
    const prefix = ranked.slice(0, size);
    for (const weights of [equalWeights(size), metricWeights(prefix)]) {
      const candidate = evaluateWeightedCandidate(
        data.labels,
        prefix.map((item) => data.probsByModel[item.index]!),
        weights,
        decisionRules,
      );
      if (!best || compareTernaryMetrics(candidate.metrics, best.metrics) > 0) {
        best = {
          indexes: prefix.map((item) => item.index),
          weights,
          ...candidate,
          strategy: `ranked-prefix-pruned(top=${ranked.length},max=${options.maxModels},step=${options.weightStep})`,
        };
      }
    }
  }
  if (!best) throw new Error("No ternary pruned ensemble candidate selected");
  return best;
}

function rankSingleTernaryModels(
  data: SelectionData,
  modelIds: readonly string[],
  decisionRules: readonly TernaryDecisionRule[],
): Array<{ index: number; metrics: TernaryMetrics }> {
  return modelIds
    .map((_modelId, index) => ({
      index,
      metrics: evaluateWeightedCandidate(data.labels, [data.probsByModel[index]!], [1], decisionRules).metrics,
    }))
    .sort((left, right) => {
      const metricDiff = compareTernaryMetrics(right.metrics, left.metrics);
      if (metricDiff !== 0) return metricDiff;
      return modelIds[left.index]!.localeCompare(modelIds[right.index]!);
    });
}

function evaluateWeightedCandidate(
  actual: readonly TernaryLabel[],
  probsByModel: readonly ProbTriple[][],
  weights: readonly number[],
  decisionRules: readonly TernaryDecisionRule[],
): { decisionRule: TernaryDecisionRule; metrics: TernaryMetrics } {
  const probs = weightedProbabilities(probsByModel, weights);
  let best: { decisionRule: TernaryDecisionRule; metrics: TernaryMetrics } | null = null;
  for (const decisionRule of decisionRules) {
    const predicted = probs.map((prob) => predictedTernaryLabel(prob, decisionRule));
    const metrics = computeTernaryMetrics(actual, predicted);
    if (!best || compareTernaryMetrics(metrics, best.metrics) > 0) best = { decisionRule, metrics };
  }
  if (!best) throw new Error("No ternary decision rule selected");
  return best;
}

function weightedProbabilities(probsByModel: readonly ProbTriple[][], weights: readonly number[]): ProbTriple[] {
  const userCount = probsByModel[0]?.length ?? 0;
  const out = Array.from({ length: userCount }, () => [0, 0, 0] as ProbTriple);
  for (let modelIndex = 0; modelIndex < probsByModel.length; modelIndex += 1) {
    const weight = weights[modelIndex] ?? 0;
    if (weight === 0) continue;
    const modelProbs = probsByModel[modelIndex]!;
    for (let userIndex = 0; userIndex < userCount; userIndex += 1) {
      out[userIndex]![0] += modelProbs[userIndex]![0] * weight;
      out[userIndex]![1] += modelProbs[userIndex]![1] * weight;
      out[userIndex]![2] += modelProbs[userIndex]![2] * weight;
    }
  }
  return out;
}

function weightedUserProbability(
  userId: string,
  modelIds: readonly string[],
  weights: Readonly<Record<string, number>>,
  rowsByModel: ReadonlyMap<string, readonly TernaryProbabilityRow[]>,
): ProbTriple {
  const out: ProbTriple = [0, 0, 0];
  for (const modelId of modelIds) {
    const row = rowsByModel.get(modelId)?.find((entry) => entry.userId === userId);
    if (!row) throw new Error(`Missing ${modelId} ternary score for ${userId}`);
    const weight = weights[modelId] ?? 0;
    out[0] += row.probDiagnosed * weight;
    out[1] += row.probControl * weight;
    out[2] += row.probNoEvidence * weight;
  }
  return out;
}

function alignedUsers(scoreMap: ReadonlyMap<string, readonly TernaryProbabilityRow[]>, modelIds: readonly string[]): string[] {
  const first = scoreMap.get(modelIds[0]!);
  if (!first) throw new Error(`Missing ternary scores for ${modelIds[0]}`);
  const users = first.map((row) => row.userId).sort();
  for (const modelId of modelIds) {
    const rows = scoreMap.get(modelId);
    if (!rows) throw new Error(`Missing ternary scores for ${modelId}`);
    const modelUsers = rows.map((row) => row.userId).sort();
    if (modelUsers.join("\n") !== users.join("\n")) throw new Error(`Ternary user alignment mismatch for ${modelId}`);
  }
  return users;
}

function probabilitiesValid(row: TernaryProbabilityRow): boolean {
  const probs = [row.probDiagnosed, row.probControl, row.probNoEvidence];
  return probs.every((prob) => Number.isFinite(prob) && prob >= -1e-8 && prob <= 1 + 1e-8) && Math.abs(probs.reduce((sum, prob) => sum + prob, 0) - 1) <= 1e-4;
}

function rowToProbTriple(row: TernaryProbabilityRow): ProbTriple {
  return [row.probDiagnosed, row.probControl, row.probNoEvidence];
}

function emptyConfusion(): TernaryMetrics["confusion"] {
  return Object.fromEntries(
    TERNARY_LABELS.map((actual) => [actual, Object.fromEntries(TERNARY_LABELS.map((predicted) => [predicted, 0]))]),
  ) as TernaryMetrics["confusion"];
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

function forEachBoundedWeight(ranges: readonly { min: number; max: number }[], precision: number, callback: (weights: number[]) => void): void {
  function recurse(prefix: number[], remaining: number): void {
    const index = prefix.length;
    if (index === ranges.length - 1) {
      const range = ranges[index]!;
      if (remaining >= range.min && remaining <= range.max) callback([...prefix, remaining / precision]);
      return;
    }
    const range = ranges[index]!;
    for (let units = range.min; units <= Math.min(range.max, remaining); units += 1) {
      recurse([...prefix, units / precision], remaining - units);
    }
  }
  recurse([], precision);
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

function equalWeights(size: number): number[] {
  return Array.from({ length: size }, () => 1 / size);
}

function metricWeights(prefix: readonly { metrics: TernaryMetrics }[]): number[] {
  const raw = prefix.map((candidate) => Math.max(candidate.metrics.macroF1, 1e-12) ** 4);
  const total = raw.reduce((sum, value) => sum + value, 0);
  return total > 0 ? raw.map((value) => value / total) : equalWeights(prefix.length);
}

function relevanceCount(marker: EvidenceMarker, threshold: 3 | 5 | 6 | 7): number {
  if (threshold === 3) return marker.rel3Count;
  if (threshold === 5) return marker.rel5Count;
  if (threshold === 6) return marker.rel6Count;
  return marker.rel7Count;
}

function quantile(values: readonly number[], q: number): number {
  if (values.length === 0) throw new Error("Cannot derive evidence quantile policy without diagnosed train users");
  if (q < 0 || q > 1) throw new Error(`Quantile must be between 0 and 1, got ${q}`);
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * q;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower]!;
  return sorted[lower]! + (sorted[upper]! - sorted[lower]!) * (position - lower);
}

function argmaxIndex(values: readonly number[]): number {
  let bestIndex = 0;
  let bestValue = values[0] ?? -Infinity;
  for (let index = 1; index < values.length; index += 1) {
    const value = values[index] ?? -Infinity;
    if (value > bestValue + 1e-12) {
      bestIndex = index;
      bestValue = value;
    }
  }
  return bestIndex;
}

function normalizeBinaryLabel(value: unknown): BinaryLabel {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (normalized === "diagnosed" || normalized === "yes") return "diagnosed";
  if (normalized === "control" || normalized === "no") return "control";
  throw new Error(`Unknown binary label: ${String(value)}`);
}

function numberField(record: Record<string, string>, key: string): number {
  const value = Number(required(record, key));
  if (!Number.isFinite(value)) throw new Error(`Invalid numeric field ${key}: ${record[key]}`);
  return value;
}

function required(record: Record<string, string>, key: string): string {
  const value = record[key];
  if (value === undefined || value === "") throw new Error(`Missing CSV column/value: ${key}`);
  return value;
}

function fixed(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(8);
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function pass(code: string, detail: string) {
  return { ok: true, code, detail };
}

function fail(code: string, detail: string) {
  return { ok: false, code, detail };
}
