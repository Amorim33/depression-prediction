import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { listCsvFiles, writeJson } from "../src/artifacts.ts";
import { loadTernaryConfig, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import { compareTernaryMetrics, computeTernaryMetrics, predictedTernaryLabel, readTernaryOofScores } from "../src/ternary.ts";
import type { TernaryDecisionRule, TernaryLabel, TernaryMetrics, TernaryProbabilityRow } from "../src/types.ts";

interface ModelMetadata {
  modelId: string;
  family: string;
  source: "tabular" | "sequence" | "unknown";
}

interface LeaderboardEntry {
  labelPolicyId: string;
  modelId: string;
  family: string;
  source: string;
  ruleId: string;
  metrics: TernaryMetrics;
}

interface GroupSummary {
  groupId: string;
  candidateCount: number;
  best: LeaderboardEntry;
  meanMacroF1: number;
  meanDiagnosedF1: number;
  meanDiagnosedPrecision: number;
}

const config = await loadTernaryConfig();
const metadataByModel = modelMetadata();
const entries: LeaderboardEntry[] = [];

for (const policy of config.labelPolicies) {
  const files = await listCsvFiles(resolveTernaryOutputPath(config, "scores"), `train_oof_${policy.policyId}_`);
  for (const path of files) {
    const rows = await readTernaryOofScores(path);
    const modelId = rows[0]?.modelId;
    if (!modelId) throw new Error(`Missing model_id in ${path}`);
    const metadata = metadataByModel.get(modelId) ?? { modelId, family: "unknown", source: "unknown" as const };
    for (const rule of config.ensemble.decisionRules) {
      entries.push({
        labelPolicyId: policy.policyId,
        modelId,
        family: metadata.family,
        source: metadata.source,
        ruleId: rule.ruleId,
        metrics: evaluateRows(rows, rule),
      });
    }
  }
}

if (entries.length === 0) throw new Error("No ternary OOF rows found. Run ternary training first.");
entries.sort(compareEntries);

const bestPerModelPolicy = bestBy(entries, (entry) => `${entry.labelPolicyId}:${entry.modelId}`);
const report = {
  dataset: config.dataset,
  seed: config.seed,
  usesTestLabels: false,
  usesTestScores: false,
  sourceArtifacts: ["train_oof_*.csv", "ternary config", "model-manifests/*.json"],
  rankedBy: ["OOF Macro F1", "OOF diagnosed F1", "OOF diagnosed precision", "OOF accuracy"],
  evaluatedCombinationCount: entries.length,
  modelPolicyCount: bestPerModelPolicy.length,
  topModelPolicyRules: bestPerModelPolicy.slice(0, 50),
  policySummaries: summarize(bestPerModelPolicy, (entry) => entry.labelPolicyId),
  familySummaries: summarize(bestPerModelPolicy, (entry) => `${entry.source}:${entry.family}`),
  generatedAt: new Date(0).toISOString(),
};

const outJson = resolveTernaryOutputPath(config, "reports", "ternary-model-policy-leaderboard.json");
const outMd = resolveTernaryOutputPath(config, "reports", "ternary-model-policy-leaderboard.md");
await writeJson(outJson, report);
await mkdir(dirname(outMd), { recursive: true });
await writeFile(outMd, renderMarkdown(report));
console.log(`wrote ${outJson}`);
console.log(`wrote ${outMd}`);
console.log(JSON.stringify(report.topModelPolicyRules[0], null, 2));

function modelMetadata(): Map<string, ModelMetadata> {
  const out = new Map<string, ModelMetadata>();
  for (const candidate of config.candidateModels.tabular) {
    out.set(candidate.modelId, { modelId: candidate.modelId, family: candidate.family, source: "tabular" });
  }
  for (const candidate of config.candidateModels.sequence) {
    out.set(candidate.modelId, { modelId: candidate.modelId, family: candidate.family, source: "sequence" });
  }
  return out;
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

function bestBy(entries: readonly LeaderboardEntry[], keyFn: (entry: LeaderboardEntry) => string): LeaderboardEntry[] {
  const best = new Map<string, LeaderboardEntry>();
  for (const entry of entries) {
    const key = keyFn(entry);
    const current = best.get(key);
    if (!current || compareEntries(entry, current) < 0) best.set(key, entry);
  }
  return [...best.values()].sort(compareEntries);
}

function summarize(entries: readonly LeaderboardEntry[], keyFn: (entry: LeaderboardEntry) => string): GroupSummary[] {
  const groups = new Map<string, LeaderboardEntry[]>();
  for (const entry of entries) {
    const key = keyFn(entry);
    groups.set(key, [...(groups.get(key) ?? []), entry]);
  }
  return [...groups.entries()]
    .map(([groupId, groupEntries]) => ({
      groupId,
      candidateCount: groupEntries.length,
      best: [...groupEntries].sort(compareEntries)[0]!,
      meanMacroF1: average(groupEntries.map((entry) => entry.metrics.macroF1)),
      meanDiagnosedF1: average(groupEntries.map((entry) => entry.metrics.diagnosedF1)),
      meanDiagnosedPrecision: average(groupEntries.map((entry) => entry.metrics.diagnosedPrecision)),
    }))
    .sort((left, right) => compareEntries(left.best, right.best));
}

function compareEntries(left: LeaderboardEntry, right: LeaderboardEntry): number {
  const metricDiff = compareTernaryMetrics(right.metrics, left.metrics);
  if (metricDiff !== 0) return metricDiff;
  return `${left.labelPolicyId}:${left.modelId}:${left.ruleId}`.localeCompare(`${right.labelPolicyId}:${right.modelId}:${right.ruleId}`);
}

function average(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function renderMarkdown(report: {
  topModelPolicyRules: LeaderboardEntry[];
  policySummaries: GroupSummary[];
  familySummaries: GroupSummary[];
}): string {
  const lines = [
    "# SetembroBR Ternary Model And Policy Leaderboard",
    "",
    "This report uses train OOF probabilities only. It does not read test score files, test labels, final test reports, or test prevalence.",
    "",
    "## Top Model/Policy/Rule Candidates",
    "",
    "| Rank | Policy | Model | Family | Rule | Macro F1 | Diagnosed F1 | Diagnosed precision | Accuracy |",
    "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ...report.topModelPolicyRules.slice(0, 20).map(
      (entry, index) =>
        `| ${index + 1} | \`${entry.labelPolicyId}\` | \`${entry.modelId}\` | \`${entry.source}:${entry.family}\` | \`${entry.ruleId}\` | ${fmt(entry.metrics.macroF1)} | ${fmt(entry.metrics.diagnosedF1)} | ${fmt(entry.metrics.diagnosedPrecision)} | ${fmt(entry.metrics.accuracy)} |`,
    ),
    "",
    "## Policy Summary",
    "",
    "| Policy | Candidates | Best model | Best Macro F1 | Mean Macro F1 |",
    "| --- | ---: | --- | ---: | ---: |",
    ...report.policySummaries.map(
      (summary) =>
        `| \`${summary.groupId}\` | ${summary.candidateCount} | \`${summary.best.modelId}\` | ${fmt(summary.best.metrics.macroF1)} | ${fmt(summary.meanMacroF1)} |`,
    ),
    "",
    "## Family Summary",
    "",
    "| Family | Candidates | Best policy/model | Best Macro F1 | Mean Macro F1 |",
    "| --- | ---: | --- | ---: | ---: |",
    ...report.familySummaries.map(
      (summary) =>
        `| \`${summary.groupId}\` | ${summary.candidateCount} | \`${summary.best.labelPolicyId}/${summary.best.modelId}\` | ${fmt(summary.best.metrics.macroF1)} | ${fmt(summary.meanMacroF1)} |`,
    ),
    "",
  ];
  return `${lines.join("\n")}\n`;
}

function fmt(value: number): string {
  return value.toFixed(6);
}
