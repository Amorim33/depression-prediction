import { readFile, writeFile } from "node:fs/promises";
import { listCsvFiles, writeJson } from "../src/artifacts.ts";
import {
  LLM_PROMPT_METADATA,
  finalPredictionsWithLlm,
  readLlmTestDecisions,
} from "../src/llm-disambiguator.ts";
import { isRawTernaryConfig, readRawSealedTestLabels } from "../src/raw-ternary.ts";
import { loadTernaryConfig, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import {
  computeTernaryMetrics,
  deriveTernaryLabel,
  predictTernaryLockedEnsembleRows,
  readEvidenceMarkers,
  readTernaryTestScores,
} from "../src/ternary.ts";
import type { EvidenceMarker, TernaryEnsembleLock, TernaryLabel, TernaryLabelPolicyLock, TernaryMetrics } from "../src/types.ts";

const config = await loadTernaryConfig();
if (!isRawTernaryConfig(config)) throw new Error("LLM-disambiguated evaluation requires raw ternary config");
if (!config.llmDisambiguator?.enabled) throw new Error("llmDisambiguator.enabled must be true for this config");
if (!config.outputDir.includes("symmetric")) throw new Error("LLM-disambiguated evaluation is scoped to the symmetric raw lane");

const lockBasename = process.env.TERNARY_LOCK_BASENAME?.trim() || "ensemble-lock";
const reportBasename = process.env.TERNARY_REPORT_BASENAME?.trim() || "final-test-report-llm-disambiguated";
const lock = JSON.parse(await readFile(resolveTernaryOutputPath(config, "ensemble", `${lockBasename}.json`), "utf8")) as TernaryEnsembleLock;
const policyLock = JSON.parse(
  await readFile(resolveTernaryOutputPath(config, "label-policies", `${lock.labelPolicyId}.json`), "utf8"),
) as TernaryLabelPolicyLock;
if (policyLock.policyHash !== lock.labelPolicyHash) throw new Error("Locked ensemble label policy hash does not match policy lock");

const testMarkers = await readEvidenceMarkers(resolveTernaryOutputPath(config, "evidence-markers", "test_markers.csv"));
const markersByUser = new Map(testMarkers.map((marker) => [marker.userId, marker]));
const labelsByUser = new Map<string, TernaryLabel>();
for (const row of await readRawSealedTestLabels(config)) {
  const marker = markersByUser.get(row.userId);
  if (!marker) throw new Error(`Missing test evidence marker for ${row.userId}`);
  labelsByUser.set(row.userId, deriveTernaryLabel(row.label, marker, policyLock));
}

const testScoresByModel = new Map();
for (const path of await listCsvFiles(resolveTernaryOutputPath(config, "scores"), `test_score_${lock.labelPolicyId}_`)) {
  const rows = await readTernaryTestScores(path);
  const modelId = rows[0]?.modelId;
  if (modelId && lock.modelIds.includes(modelId)) testScoresByModel.set(modelId, rows);
}
for (const modelId of lock.modelIds) {
  if (!testScoresByModel.has(modelId)) throw new Error(`Missing label-free ternary test scores for locked model ${modelId}`);
}

const baseRows = predictTernaryLockedEnsembleRows(lock, testScoresByModel);
const decisionPath = resolveTernaryOutputPath(config, "llm-disambiguator", `test_decisions_${lockBasename}.csv`);
const decisionsByUser = readLlmTestDecisions(await readFile(decisionPath, "utf8"));
const actual = baseRows.map((row) => {
  const label = labelsByUser.get(row.userId);
  if (!label) throw new Error(`Missing ternary test label for ${row.userId}`);
  return label;
});
const basePredicted = baseRows.map((row) => row.predicted);
const llmPredicted = finalPredictionsWithLlm(baseRows, decisionsByUser);
const baseMetrics = computeTernaryMetrics(actual, basePredicted);
const llmMetrics = computeTernaryMetrics(actual, llmPredicted);
const markerSummary = summarizeMarkersByLabel(testMarkers, labelsByUser);

const report = {
  dataset: "setembrobr",
  seed: config.seed,
  originalManifestHash: lock.originalManifestHash,
  labelPolicyId: lock.labelPolicyId,
  labelPolicyHash: lock.labelPolicyHash,
  testLabelsSource: "sealed-final-eval-labels",
  lock,
  llmDisambiguator: {
    promptId: LLM_PROMPT_METADATA.promptId,
    promptVersion: LLM_PROMPT_METADATA.promptVersion,
    promptHash: LLM_PROMPT_METADATA.promptHash,
    requestedModel: config.llmDisambiguator.requestedModel,
    apiModel: config.llmDisambiguator.apiModel,
    decisionPath,
    decisionCount: decisionsByUser.size,
    switchedToControl: [...decisionsByUser.values()].filter((decision) => !decision.trueDepression).length,
  },
  baseTestMetrics: baseMetrics,
  testMetrics: llmMetrics,
  evidenceMarkerSummary: markerSummary,
};

const reportPath = resolveTernaryOutputPath(config, "reports", `${reportBasename}.json`);
await writeJson(reportPath, report);
await writeFile(resolveTernaryOutputPath(config, "reports", `${reportBasename}.md`), renderMarkdownReport(report));
console.log(JSON.stringify(llmMetrics, null, 2));

function summarizeMarkersByLabel(markers: readonly EvidenceMarker[], labelsByUserMap: ReadonlyMap<string, TernaryLabel>) {
  const groups = new Map<TernaryLabel, EvidenceMarker[]>();
  for (const marker of markers) {
    const label = labelsByUserMap.get(marker.userId);
    if (!label) continue;
    groups.set(label, [...(groups.get(label) ?? []), marker]);
  }
  return Object.fromEntries(
    [...groups.entries()].map(([label, rows]) => [
      label,
      {
        count: rows.length,
        avgEvidenceScore: average(rows.map((row) => row.evidenceScore)),
        avgTop10Relevance: average(rows.map((row) => row.top10AvgRelevance)),
        avgTotalTweets: average(rows.map((row) => row.totalTweets)),
        avgRel3Ratio: average(rows.map((row) => row.rel3Ratio)),
      },
    ]),
  );
}

function average(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function renderMarkdownReport(reportValue: {
  labelPolicyId: string;
  labelPolicyHash: string;
  lock: TernaryEnsembleLock;
  llmDisambiguator: { promptVersion: string; promptHash: string; decisionCount: number; switchedToControl: number };
  baseTestMetrics: TernaryMetrics;
  testMetrics: TernaryMetrics;
}): string {
  const metrics = reportValue.testMetrics;
  const base = reportValue.baseTestMetrics;
  return `# SetembroBR Symmetric LLM-Disambiguated Final Test Report

## Lock

- Label policy: \`${reportValue.labelPolicyId}\`
- Label policy hash: \`${reportValue.labelPolicyHash}\`
- Selection strategy: \`${reportValue.lock.selectionStrategy ?? "unknown"}\`
- Decision rule: \`${reportValue.lock.decisionRule.ruleId}\`
- LLM prompt version: \`${reportValue.llmDisambiguator.promptVersion}\`
- LLM prompt hash: \`${reportValue.llmDisambiguator.promptHash}\`
- LLM decisions: \`${reportValue.llmDisambiguator.decisionCount}\`
- Switched diagnosed to control: \`${reportValue.llmDisambiguator.switchedToControl}\`

## Test Metrics

| Metric | Base | LLM-disambiguated |
| --- | ---: | ---: |
| Macro F1 | \`${base.macroF1.toFixed(6)}\` | \`${metrics.macroF1.toFixed(6)}\` |
| Diagnosed F1 | \`${base.diagnosedF1.toFixed(6)}\` | \`${metrics.diagnosedF1.toFixed(6)}\` |
| Diagnosed precision | \`${base.diagnosedPrecision.toFixed(6)}\` | \`${metrics.diagnosedPrecision.toFixed(6)}\` |
| Diagnosed recall | \`${base.diagnosedRecall.toFixed(6)}\` | \`${metrics.diagnosedRecall.toFixed(6)}\` |
| Accuracy | \`${base.accuracy.toFixed(6)}\` | \`${metrics.accuracy.toFixed(6)}\` |

## Confusion Matrix

| Actual \\ Predicted | diagnosed | control | no-evidence |
| --- | ---: | ---: | ---: |
${(["diagnosed", "control", "no-evidence"] as const)
  .map(
    (label) =>
      `| ${label} | ${metrics.confusion[label].diagnosed} | ${metrics.confusion[label].control} | ${metrics.confusion[label]["no-evidence"]} |`,
  )
  .join("\n")}
`;
}
