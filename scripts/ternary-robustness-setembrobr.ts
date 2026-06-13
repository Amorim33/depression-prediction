import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { listCsvFiles, writeJson } from "../src/artifacts.ts";
import { loadTernaryConfig, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import {
  compareTernaryMetrics,
  computeTernaryMetrics,
  predictedTernaryLabel,
  readTernaryOofScores,
} from "../src/ternary.ts";
import type { TernaryDecisionRule, TernaryEnsembleLock, TernaryLabel, TernaryMetrics, TernaryProbabilityRow } from "../src/types.ts";

type SourceKind = "single-model" | "locked-ensemble";

interface ScalarStats {
  mean: number;
  min: number;
  max: number;
  std: number;
}

interface RobustnessStats {
  foldCount: number;
  combinationCount: number;
  macroF1: ScalarStats;
  diagnosedF1: ScalarStats;
  diagnosedPrecision: ScalarStats;
  accuracy: ScalarStats;
}

interface RobustnessEntry {
  source: SourceKind;
  labelPolicyId: string;
  modelId: string;
  ruleId: string;
  overallMetrics: TernaryMetrics;
  foldCombinationStats: RobustnessStats;
}

const config = await loadTernaryConfig();
const scoresDir = resolveTernaryOutputPath(config, "scores");
const entries: RobustnessEntry[] = [];

for (const policy of config.labelPolicies) {
  const oofFiles = await listCsvFiles(scoresDir, `train_oof_${policy.policyId}_`);
  for (const path of oofFiles) {
    const rows = await readTernaryOofScores(path);
    const modelId = rows[0]?.modelId;
    if (!modelId) throw new Error(`Missing model_id in ${path}`);
    for (const rule of config.ensemble.decisionRules) {
      entries.push({
        source: "single-model",
        labelPolicyId: policy.policyId,
        modelId,
        ruleId: rule.ruleId,
        overallMetrics: evaluateRows(rows, rule),
        foldCombinationStats: summarizeFoldCombinations(rows, rule),
      });
    }
  }
}

const lockedEntry = await lockedEnsembleEntry().catch((error: unknown) => {
  console.warn(`skipping locked ensemble robustness: ${error instanceof Error ? error.message : String(error)}`);
  return null;
});
if (lockedEntry) entries.push(lockedEntry);

if (entries.length === 0) throw new Error("No ternary train OOF rows found. Run ternary training first.");

entries.sort(compareRobustnessEntries);

const outJson = resolveTernaryOutputPath(config, "reports", "ternary-robustness.json");
const outMd = resolveTernaryOutputPath(config, "reports", "ternary-robustness.md");
const report = {
  dataset: config.dataset,
  seed: config.seed,
  usesTestLabels: false,
  usesTestScores: false,
  sourceArtifacts: ["train_oof_*.csv", "ensemble-lock.json"],
  rankedBy: ["mean fold-combination Macro F1", "minimum fold-combination Macro F1", "overall OOF Macro F1"],
  candidateCount: entries.length,
  generatedAt: new Date(0).toISOString(),
  topCandidates: entries.slice(0, 40),
  lockedEnsemble: lockedEntry,
};
await writeJson(outJson, report);
await mkdir(dirname(outMd), { recursive: true });
await writeFile(outMd, renderMarkdown(entries, lockedEntry));
console.log(`wrote ${outJson}`);
console.log(`wrote ${outMd}`);
console.log(JSON.stringify(entries[0], null, 2));

async function lockedEnsembleEntry(): Promise<RobustnessEntry | null> {
  const lockPath = resolveTernaryOutputPath(config, "ensemble", "ensemble-lock.json");
  const lock = (await Bun.file(lockPath).json()) as TernaryEnsembleLock;
  const rowsByModel = new Map<string, TernaryProbabilityRow[]>();
  for (const modelId of lock.modelIds) {
    const path = resolveTernaryOutputPath(config, "scores", `train_oof_${lock.labelPolicyId}_${modelId}.csv`);
    rowsByModel.set(modelId, await readTernaryOofScores(path));
  }
  const rows = lockedRows(lock, rowsByModel);
  return {
    source: "locked-ensemble",
    labelPolicyId: lock.labelPolicyId,
    modelId: "locked_ensemble",
    ruleId: lock.decisionRule.ruleId,
    overallMetrics: evaluateRows(rows, lock.decisionRule),
    foldCombinationStats: summarizeFoldCombinations(rows, lock.decisionRule),
  };
}

function lockedRows(
  lock: TernaryEnsembleLock,
  rowsByModel: ReadonlyMap<string, readonly TernaryProbabilityRow[]>,
): TernaryProbabilityRow[] {
  const firstRows = rowsByModel.get(lock.modelIds[0]!);
  if (!firstRows) throw new Error("Locked ensemble has no first model rows");
  const indexed = new Map(
    lock.modelIds.map((modelId) => {
      const rows = rowsByModel.get(modelId);
      if (!rows) throw new Error(`Missing locked OOF rows for ${modelId}`);
      return [modelId, new Map(rows.map((row) => [row.userId, row]))] as const;
    }),
  );

  return firstRows.map((base) => {
    if (!base.label || base.fold === undefined) throw new Error(`Locked base row is missing OOF label/fold for ${base.userId}`);
    const out: TernaryProbabilityRow = {
      userId: base.userId,
      label: base.label,
      fold: base.fold,
      probDiagnosed: 0,
      probControl: 0,
      probNoEvidence: 0,
      modelId: "locked_ensemble",
      labelPolicyId: lock.labelPolicyId,
    };
    for (const modelId of lock.modelIds) {
      const row = indexed.get(modelId)?.get(base.userId);
      if (!row) throw new Error(`Missing ${modelId} locked row for ${base.userId}`);
      if (row.label !== base.label || row.fold !== base.fold) throw new Error(`Locked row alignment mismatch for ${base.userId}`);
      const weight = lock.weights[modelId] ?? 0;
      out.probDiagnosed += row.probDiagnosed * weight;
      out.probControl += row.probControl * weight;
      out.probNoEvidence += row.probNoEvidence * weight;
    }
    return out;
  });
}

function evaluateRows(rows: readonly TernaryProbabilityRow[], rule: TernaryDecisionRule): TernaryMetrics {
  const actual: TernaryLabel[] = [];
  const predicted: TernaryLabel[] = [];
  for (const row of rows) {
    if (!row.label) throw new Error(`Missing OOF label for ${row.userId}`);
    actual.push(row.label);
    predicted.push(predictedTernaryLabel([row.probDiagnosed, row.probControl, row.probNoEvidence], rule));
  }
  return computeTernaryMetrics(actual, predicted);
}

function summarizeFoldCombinations(rows: readonly TernaryProbabilityRow[], rule: TernaryDecisionRule): RobustnessStats {
  const folds = [...new Set(rows.map((row) => row.fold))].filter((fold): fold is number => fold !== undefined).sort((a, b) => a - b);
  const metrics: TernaryMetrics[] = [];
  for (const subset of nonEmptySubsets(folds)) {
    const selected = new Set(subset);
    metrics.push(evaluateRows(rows.filter((row) => selected.has(row.fold ?? -1)), rule));
  }
  return {
    foldCount: folds.length,
    combinationCount: metrics.length,
    macroF1: stats(metrics.map((metric) => metric.macroF1)),
    diagnosedF1: stats(metrics.map((metric) => metric.diagnosedF1)),
    diagnosedPrecision: stats(metrics.map((metric) => metric.diagnosedPrecision)),
    accuracy: stats(metrics.map((metric) => metric.accuracy)),
  };
}

function nonEmptySubsets(values: readonly number[]): number[][] {
  const subsets: number[][] = [];
  for (let mask = 1; mask < 2 ** values.length; mask += 1) {
    const subset: number[] = [];
    for (let index = 0; index < values.length; index += 1) {
      if ((mask & (1 << index)) !== 0) subset.push(values[index]!);
    }
    subsets.push(subset);
  }
  return subsets;
}

function stats(values: readonly number[]): ScalarStats {
  if (values.length === 0) throw new Error("Cannot summarize empty metric list");
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    mean,
    min: Math.min(...values),
    max: Math.max(...values),
    std: Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length),
  };
}

function compareRobustnessEntries(left: RobustnessEntry, right: RobustnessEntry): number {
  const robustDiff =
    right.foldCombinationStats.macroF1.mean - left.foldCombinationStats.macroF1.mean ||
    right.foldCombinationStats.macroF1.min - left.foldCombinationStats.macroF1.min ||
    compareTernaryMetrics(right.overallMetrics, left.overallMetrics);
  if (Math.abs(robustDiff) > 1e-12) return robustDiff;
  return `${left.labelPolicyId}:${left.modelId}:${left.ruleId}`.localeCompare(`${right.labelPolicyId}:${right.modelId}:${right.ruleId}`);
}

function renderMarkdown(allEntries: readonly RobustnessEntry[], locked: RobustnessEntry | null): string {
  const lines = [
    "# SetembroBR Ternary Train-Only Robustness",
    "",
    "This report uses train OOF rows only. It does not read test score files, test labels, final test reports, or test prevalence.",
    "",
    "## Top Train-Only Candidates",
    "",
    "| Rank | Source | Label policy | Model | Rule | Mean Macro F1 | Min Macro F1 | Overall Macro F1 | Mean diagnosed F1 | Mean diagnosed precision |",
    "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ...allEntries.slice(0, 20).map(
      (entry, index) =>
        `| ${index + 1} | ${entry.source} | \`${entry.labelPolicyId}\` | \`${entry.modelId}\` | \`${entry.ruleId}\` | ${fmt(entry.foldCombinationStats.macroF1.mean)} | ${fmt(entry.foldCombinationStats.macroF1.min)} | ${fmt(entry.overallMetrics.macroF1)} | ${fmt(entry.foldCombinationStats.diagnosedF1.mean)} | ${fmt(entry.foldCombinationStats.diagnosedPrecision.mean)} |`,
    ),
    "",
  ];
  if (locked) {
    lines.push(
      "## Locked Ensemble Train-Only Stability",
      "",
      `- Policy: \`${locked.labelPolicyId}\``,
      `- Rule: \`${locked.ruleId}\``,
      `- Overall OOF Macro F1: \`${fmt(locked.overallMetrics.macroF1)}\``,
      `- Mean fold-combination Macro F1: \`${fmt(locked.foldCombinationStats.macroF1.mean)}\``,
      `- Minimum fold-combination Macro F1: \`${fmt(locked.foldCombinationStats.macroF1.min)}\``,
      "",
    );
  }
  return `${lines.join("\n")}\n`;
}

function fmt(value: number): string {
  return value.toFixed(6);
}
