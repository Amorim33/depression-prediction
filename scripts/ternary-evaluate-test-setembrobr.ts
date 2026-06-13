import { readFile, writeFile } from "node:fs/promises";
import { listCsvFiles, writeJson } from "../src/artifacts.ts";
import { readManifest } from "../src/manifest.ts";
import { loadTernaryConfig, resolveSourceOutputPath, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import {
  deriveTernaryLabel,
  evaluateTernaryLockedEnsemble,
  readEvidenceMarkers,
  readTernaryTestScores,
} from "../src/ternary.ts";
import type { EvidenceMarker, TernaryEnsembleLock, TernaryLabel, TernaryLabelPolicyLock, TernaryMetrics } from "../src/types.ts";

const config = await loadTernaryConfig();
const lock = JSON.parse(await readFile(resolveTernaryOutputPath(config, "ensemble", "ensemble-lock.json"), "utf8")) as TernaryEnsembleLock;
const policyLock = JSON.parse(
  await readFile(resolveTernaryOutputPath(config, "label-policies", `${lock.labelPolicyId}.json`), "utf8"),
) as TernaryLabelPolicyLock;
if (policyLock.policyHash !== lock.labelPolicyHash) throw new Error("Locked ensemble label policy hash does not match policy lock");

const sourceRows = await readManifest(resolveSourceOutputPath(config, "manifest", `split_manifest_seed${config.seed}.csv`));
const testMarkers = await readEvidenceMarkers(resolveTernaryOutputPath(config, "evidence-markers", "test_markers.csv"));
const markersByUser = new Map(testMarkers.map((marker) => [marker.userId, marker]));
const labelsByUser = new Map<string, TernaryLabel>();
for (const row of sourceRows.filter((entry) => entry.split === "test")) {
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

const metrics = evaluateTernaryLockedEnsemble(lock, testScoresByModel, labelsByUser);
const markerSummary = summarizeMarkersByLabel(testMarkers, labelsByUser);
const report = {
  dataset: "setembrobr",
  seed: config.seed,
  originalManifestHash: lock.originalManifestHash,
  labelPolicyId: lock.labelPolicyId,
  labelPolicyHash: lock.labelPolicyHash,
  lock,
  testMetrics: metrics,
  evidenceMarkerSummary: markerSummary,
};
const reportPath = resolveTernaryOutputPath(config, "reports", "final-test-report.json");
await writeJson(reportPath, report);
await writeFile(resolveTernaryOutputPath(config, "reports", "final-test-report.md"), renderMarkdownReport(report));
console.log(JSON.stringify(metrics, null, 2));

function summarizeMarkersByLabel(markers: readonly EvidenceMarker[], labelsByUser: ReadonlyMap<string, TernaryLabel>) {
  const groups = new Map<TernaryLabel, EvidenceMarker[]>();
  for (const marker of markers) {
    const label = labelsByUser.get(marker.userId);
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

function renderMarkdownReport(report: {
  labelPolicyId: string;
  labelPolicyHash: string;
  lock: TernaryEnsembleLock;
  testMetrics: TernaryMetrics;
}): string {
  const metrics = report.testMetrics;
  return `# SetembroBR Ternary Final Test Report

## Lock

- Label policy: \`${report.labelPolicyId}\`
- Label policy hash: \`${report.labelPolicyHash}\`
- Selection strategy: \`${report.lock.selectionStrategy ?? "unknown"}\`
- Decision rule: \`${report.lock.decisionRule.ruleId}\`

## Test Metrics

| Metric | Value |
| --- | ---: |
| Macro F1 | \`${metrics.macroF1.toFixed(6)}\` |
| Diagnosed F1 | \`${metrics.diagnosedF1.toFixed(6)}\` |
| Diagnosed precision | \`${metrics.diagnosedPrecision.toFixed(6)}\` |
| Diagnosed recall | \`${metrics.diagnosedRecall.toFixed(6)}\` |
| Accuracy | \`${metrics.accuracy.toFixed(6)}\` |

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
