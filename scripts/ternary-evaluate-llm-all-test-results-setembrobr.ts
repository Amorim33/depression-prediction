import { readdir, readFile, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { listCsvFiles, writeJson } from "../src/artifacts.ts";
import { writeCsv } from "../src/csv.ts";
import { finalPredictionsWithLlm, readLlmTestDecisions } from "../src/llm-disambiguator.ts";
import { isRawTernaryConfig, readRawSealedTestLabels } from "../src/raw-ternary.ts";
import { loadTernaryConfig, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import {
  computeTernaryMetrics,
  deriveTernaryLabel,
  predictTernaryLockedEnsembleRows,
  predictedTernaryLabel,
  readEvidenceMarkers,
  readTernaryTestScores,
} from "../src/ternary.ts";
import type {
  TernaryDecisionRule,
  TernaryEnsembleLock,
  TernaryLabel,
  TernaryLabelPolicyLock,
  TernaryLockedPredictionRow,
  TernaryMetrics,
} from "../src/types.ts";

const config = await loadTernaryConfig();
if (!isRawTernaryConfig(config)) throw new Error("LLM all-results evaluation requires raw ternary config");
if (!config.llmDisambiguator?.enabled) throw new Error("llmDisambiguator.enabled must be true for this config");
if (!config.outputDir.includes("symmetric")) throw new Error("LLM all-results comparison is scoped to the symmetric raw lane");

const decisionRule: TernaryDecisionRule = { ruleId: "diagnosed_margin_010", kind: "diagnosed_margin", diagnosedMargin: 0.1 };
const decisionPath = resolveTernaryOutputPath(config, "llm-disambiguator", "test_all_score_decisions.csv");
const decisionsByUser = readLlmTestDecisions(await readFile(decisionPath, "utf8"));
const policyLocks = await readPolicyLocks();
const labelsByPolicy = await readLabelsByPolicy(policyLocks);

const individualResults: IndividualResultRow[] = [];
for (const path of await listCsvFiles(resolveTernaryOutputPath(config, "scores"), "test_score_")) {
  const rows = await readTernaryTestScores(path);
  if (rows.length === 0) continue;
  const modelId = rows[0]!.modelId;
  const labelPolicyId = rows[0]!.labelPolicyId;
  const labelsByUser = labelsByPolicy.get(labelPolicyId);
  if (!labelsByUser) throw new Error(`Missing policy labels for ${labelPolicyId}`);
  const actual: TernaryLabel[] = [];
  const basePredicted: TernaryLabel[] = [];
  const baseRows: TernaryLockedPredictionRow[] = [];
  for (const row of rows) {
    const label = labelsByUser.get(row.userId);
    if (!label) throw new Error(`Missing test label for ${row.userId} under ${labelPolicyId}`);
    const predicted = predictedTernaryLabel([row.probDiagnosed, row.probControl, row.probNoEvidence], decisionRule);
    actual.push(label);
    basePredicted.push(predicted);
    baseRows.push({
      userId: row.userId,
      predicted,
      probDiagnosed: row.probDiagnosed,
      probControl: row.probControl,
      probNoEvidence: row.probNoEvidence,
    });
  }
  const baseMetrics = computeTernaryMetrics(actual, basePredicted);
  const llmMetrics = computeTernaryMetrics(actual, finalPredictionsWithLlm(baseRows, decisionsByUser));
  individualResults.push({
    type: "individual_model",
    labelPolicyId,
    modelId,
    scoreFile: basename(path),
    baseDiagnosedPredictions: basePredicted.filter((label) => label === "diagnosed").length,
    switchedToControl: baseRows.filter((row) => row.predicted === "diagnosed" && decisionsByUser.get(row.userId)?.trueDepression === false).length,
    baseMetrics,
    llmMetrics,
  });
}

const lockResults: LockResultRow[] = [];
for (const lockPath of await listJsonFiles(resolveTernaryOutputPath(config, "ensemble"))) {
  const lock = JSON.parse(await readFile(lockPath, "utf8")) as TernaryEnsembleLock;
  const labelsByUser = labelsByPolicy.get(lock.labelPolicyId);
  if (!labelsByUser) throw new Error(`Missing policy labels for ${lock.labelPolicyId}`);
  const scoresByModel = new Map();
  for (const modelId of lock.modelIds) {
    const path = resolveTernaryOutputPath(config, "scores", `test_score_${lock.labelPolicyId}_${modelId}.csv`);
    scoresByModel.set(modelId, await readTernaryTestScores(path));
  }
  const baseRows = predictTernaryLockedEnsembleRows(lock, scoresByModel);
  const actual = baseRows.map((row) => {
    const label = labelsByUser.get(row.userId);
    if (!label) throw new Error(`Missing test label for ${row.userId} under ${lock.labelPolicyId}`);
    return label;
  });
  const basePredicted = baseRows.map((row) => row.predicted);
  const llmPredicted = finalPredictionsWithLlm(baseRows, decisionsByUser);
  lockResults.push({
    type: "locked_ensemble",
    lockBasename: basename(lockPath, ".json"),
    labelPolicyId: lock.labelPolicyId,
    modelIds: lock.modelIds,
    weights: lock.weights,
    decisionRule: lock.decisionRule,
    baseDiagnosedPredictions: basePredicted.filter((label) => label === "diagnosed").length,
    switchedToControl: baseRows.filter((row) => row.predicted === "diagnosed" && decisionsByUser.get(row.userId)?.trueDepression === false).length,
    baseMetrics: computeTernaryMetrics(actual, basePredicted),
    llmMetrics: computeTernaryMetrics(actual, llmPredicted),
  });
}

const bestByPolicy = [...groupBy(individualResults, (row) => row.labelPolicyId).entries()]
  .map(([labelPolicyId, rows]) => {
    const best = [...rows].sort(compareResultRows)[0]!;
    return {
      labelPolicyId,
      modelCount: rows.length,
      bestModelId: best.modelId,
      baseMacroF1: best.baseMetrics.macroF1,
      llmMacroF1: best.llmMetrics.macroF1,
      baseDiagnosedF1: best.baseMetrics.diagnosedF1,
      llmDiagnosedF1: best.llmMetrics.diagnosedF1,
      llmDiagnosedPrecision: best.llmMetrics.diagnosedPrecision,
      llmDiagnosedRecall: best.llmMetrics.diagnosedRecall,
      switchedToControl: best.switchedToControl,
    };
  })
  .sort((left, right) => metricSort(right.llmMacroF1, left.llmMacroF1) || metricSort(right.llmDiagnosedF1, left.llmDiagnosedF1));

const topIndividual = [...individualResults].sort(compareResultRows).slice(0, 30);
const report = {
  dataset: "setembrobr",
  seed: config.seed,
  lane: "raw_qwen3_symmetric",
  purpose: "post_hoc_test_comparison_not_model_selection",
  testLabelsSource: "sealed-final-eval-labels",
  decisionPath,
  decisionCount: decisionsByUser.size,
  resultCount: individualResults.length,
  topIndividual,
  bestByPolicy,
  lockResults: lockResults.sort(compareResultRows),
};

await writeJson(resolveTernaryOutputPath(config, "reports", "llm-disambiguator-all-test-results.json"), report);
await writeFile(resolveTernaryOutputPath(config, "reports", "llm-disambiguator-all-test-results.csv"), renderCsv([...topIndividual, ...lockResults]));
await writeFile(resolveTernaryOutputPath(config, "reports", "llm-disambiguator-all-test-results.md"), renderMarkdown(report));
console.log(JSON.stringify({ resultCount: individualResults.length, decisionCount: decisionsByUser.size, bestByPolicy, lockResults: report.lockResults }, null, 2));

async function readPolicyLocks(): Promise<Map<string, TernaryLabelPolicyLock>> {
  const out = new Map<string, TernaryLabelPolicyLock>();
  for (const path of await listJsonFiles(resolveTernaryOutputPath(config, "label-policies"))) {
    const lock = JSON.parse(await readFile(path, "utf8")) as TernaryLabelPolicyLock;
    out.set(lock.policyId, lock);
  }
  return out;
}

async function readLabelsByPolicy(policyLocks: ReadonlyMap<string, TernaryLabelPolicyLock>): Promise<Map<string, Map<string, TernaryLabel>>> {
  const testMarkers = await readEvidenceMarkers(resolveTernaryOutputPath(config, "evidence-markers", "test_markers.csv"));
  const markersByUser = new Map(testMarkers.map((marker) => [marker.userId, marker]));
  const sealedRows = await readRawSealedTestLabels(config);
  const out = new Map<string, Map<string, TernaryLabel>>();
  for (const [policyId, policyLock] of policyLocks) {
    const labelsByUser = new Map<string, TernaryLabel>();
    for (const row of sealedRows) {
      const marker = markersByUser.get(row.userId);
      if (!marker) throw new Error(`Missing test evidence marker for ${row.userId}`);
      labelsByUser.set(row.userId, deriveTernaryLabel(row.label, marker, policyLock));
    }
    out.set(policyId, labelsByUser);
  }
  return out;
}

async function listJsonFiles(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true }).catch(() => []);
  return entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
    .map((entry) => resolve(dir, entry.name))
    .sort();
}

function compareResultRows(left: ResultRow, right: ResultRow): number {
  return (
    metricSort(right.llmMetrics.macroF1, left.llmMetrics.macroF1) ||
    metricSort(right.llmMetrics.diagnosedF1, left.llmMetrics.diagnosedF1) ||
    metricSort(right.llmMetrics.diagnosedPrecision, left.llmMetrics.diagnosedPrecision) ||
    metricSort(right.llmMetrics.accuracy, left.llmMetrics.accuracy)
  );
}

function metricSort(left: number, right: number): number {
  const diff = left - right;
  return Math.abs(diff) > 1e-12 ? diff : 0;
}

function groupBy<T, K>(rows: readonly T[], keyFn: (row: T) => K): Map<K, T[]> {
  const out = new Map<K, T[]>();
  for (const row of rows) out.set(keyFn(row), [...(out.get(keyFn(row)) ?? []), row]);
  return out;
}

function renderCsv(rows: readonly ResultRow[]): string {
  return writeCsv(
    [
      "type",
      "label_policy_id",
      "model_or_lock",
      "base_macro_f1",
      "llm_macro_f1",
      "base_diagnosed_f1",
      "llm_diagnosed_f1",
      "llm_diagnosed_precision",
      "llm_diagnosed_recall",
      "base_accuracy",
      "llm_accuracy",
      "base_diagnosed_predictions",
      "switched_to_control",
    ],
    rows.map((row) => ({
      type: row.type,
      label_policy_id: row.labelPolicyId,
      model_or_lock: "modelId" in row ? row.modelId : row.lockBasename,
      base_macro_f1: row.baseMetrics.macroF1.toFixed(6),
      llm_macro_f1: row.llmMetrics.macroF1.toFixed(6),
      base_diagnosed_f1: row.baseMetrics.diagnosedF1.toFixed(6),
      llm_diagnosed_f1: row.llmMetrics.diagnosedF1.toFixed(6),
      llm_diagnosed_precision: row.llmMetrics.diagnosedPrecision.toFixed(6),
      llm_diagnosed_recall: row.llmMetrics.diagnosedRecall.toFixed(6),
      base_accuracy: row.baseMetrics.accuracy.toFixed(6),
      llm_accuracy: row.llmMetrics.accuracy.toFixed(6),
      base_diagnosed_predictions: row.baseDiagnosedPredictions,
      switched_to_control: row.switchedToControl,
    })),
  );
}

function renderMarkdown(reportValue: typeof report): string {
  return `# Symmetric LLM All Test Results

Post-hoc diagnostic comparison. These test-label metrics are not used for model selection.

## Locked Ensembles

| Lock | Policy | Base Macro F1 | LLM Macro F1 | Base Diagnosed F1 | LLM Diagnosed F1 | LLM P | LLM R | Switched |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
${reportValue.lockResults
  .map(
    (row) =>
      `| ${row.lockBasename} | ${row.labelPolicyId} | ${f(row.baseMetrics.macroF1)} | ${f(row.llmMetrics.macroF1)} | ${f(row.baseMetrics.diagnosedF1)} | ${f(row.llmMetrics.diagnosedF1)} | ${f(row.llmMetrics.diagnosedPrecision)} | ${f(row.llmMetrics.diagnosedRecall)} | ${row.switchedToControl} |`,
  )
  .join("\n")}

## Best Individual Model By Policy

| Policy | Model | Base Macro F1 | LLM Macro F1 | Base Diagnosed F1 | LLM Diagnosed F1 | LLM P | LLM R | Switched |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
${reportValue.bestByPolicy
  .map(
    (row) =>
      `| ${row.labelPolicyId} | ${row.bestModelId} | ${f(row.baseMacroF1)} | ${f(row.llmMacroF1)} | ${f(row.baseDiagnosedF1)} | ${f(row.llmDiagnosedF1)} | ${f(row.llmDiagnosedPrecision)} | ${f(row.llmDiagnosedRecall)} | ${row.switchedToControl} |`,
  )
  .join("\n")}

## Top Individual Models After LLM

| Policy | Model | Base Macro F1 | LLM Macro F1 | Base Diagnosed F1 | LLM Diagnosed F1 | LLM P | LLM R | Switched |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
${reportValue.topIndividual
  .slice(0, 20)
  .map(
    (row) =>
      `| ${row.labelPolicyId} | ${row.modelId} | ${f(row.baseMetrics.macroF1)} | ${f(row.llmMetrics.macroF1)} | ${f(row.baseMetrics.diagnosedF1)} | ${f(row.llmMetrics.diagnosedF1)} | ${f(row.llmMetrics.diagnosedPrecision)} | ${f(row.llmMetrics.diagnosedRecall)} | ${row.switchedToControl} |`,
  )
  .join("\n")}
`;
}

function f(value: number): string {
  return value.toFixed(4);
}

type IndividualResultRow = {
  type: "individual_model";
  labelPolicyId: string;
  modelId: string;
  scoreFile: string;
  baseDiagnosedPredictions: number;
  switchedToControl: number;
  baseMetrics: TernaryMetrics;
  llmMetrics: TernaryMetrics;
};

type LockResultRow = {
  type: "locked_ensemble";
  lockBasename: string;
  labelPolicyId: string;
  modelIds: string[];
  weights: Record<string, number>;
  decisionRule: TernaryDecisionRule;
  baseDiagnosedPredictions: number;
  switchedToControl: number;
  baseMetrics: TernaryMetrics;
  llmMetrics: TernaryMetrics;
};

type ResultRow = IndividualResultRow | LockResultRow;
