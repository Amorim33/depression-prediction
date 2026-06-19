import { readFile, writeFile } from "node:fs/promises";
import { listCsvFiles, readTestScores, writeJson } from "../src/artifacts.ts";
import {
  LLM_PROMPT_METADATA,
  finalBinaryPredictionsWithLlm,
  readLlmTestDecisions,
} from "../src/llm-disambiguator.ts";
import { loadConfig, resolveOutputPath } from "../src/config.ts";
import { predictLockedEnsembleRows } from "../src/ensemble.ts";
import { computeMetrics } from "../src/metrics.ts";
import { isRawBinaryConfig, readRawBinarySealedTestLabels } from "../src/raw-binary.ts";
import type { EnsembleLock, Metrics } from "../src/types.ts";

const config = await loadConfig();
if (!isRawBinaryConfig(config)) throw new Error("LLM-disambiguated binary evaluation requires raw binary config");
if (!config.llmDisambiguator?.enabled) throw new Error("llmDisambiguator.enabled must be true for this config");

const lockBasename = process.env.BINARY_LOCK_BASENAME?.trim() || "ensemble-lock";
const reportBasename = process.env.BINARY_REPORT_BASENAME?.trim() || "final-test-report-llm-disambiguated";
const lock = JSON.parse(await readFile(resolveOutputPath(config, "ensemble", `${lockBasename}.json`), "utf8")) as EnsembleLock;
const labelsByUser = new Map((await readRawBinarySealedTestLabels(config)).map((row) => [row.userId, row.label]));

const testScoresByModel = new Map();
for (const path of await listCsvFiles(resolveOutputPath(config, "scores"), "test_score_")) {
  const rows = await readTestScores(path);
  const modelId = rows[0]?.modelId;
  if (modelId && lock.modelIds.includes(modelId)) testScoresByModel.set(modelId, rows);
}
for (const modelId of lock.modelIds) {
  if (!testScoresByModel.has(modelId)) throw new Error(`Missing label-free test scores for locked model ${modelId}`);
}

const baseRows = predictLockedEnsembleRows(lock, testScoresByModel);
const decisionPath = resolveOutputPath(config, "llm-disambiguator", `test_decisions_${lockBasename}.csv`);
const decisionsByUser = readLlmTestDecisions(await readFile(decisionPath, "utf8"));
const actual = baseRows.map((row) => {
  const label = labelsByUser.get(row.userId);
  if (!label) throw new Error(`Missing binary test label for ${row.userId}`);
  return label;
});
const basePredicted = baseRows.map((row) => row.predicted);
const llmPredicted = finalBinaryPredictionsWithLlm(baseRows, decisionsByUser);
const baseMetrics = computeMetrics(actual, basePredicted);
const llmMetrics = computeMetrics(actual, llmPredicted);

const report = {
  dataset: "setembrobr",
  seed: config.seed,
  manifestHash: lock.manifestHash,
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
};

const reportPath = resolveOutputPath(config, "reports", `${reportBasename}.json`);
await writeJson(reportPath, report);
await writeFile(resolveOutputPath(config, "reports", `${reportBasename}.md`), renderMarkdownReport(report));
console.log(JSON.stringify(llmMetrics, null, 2));

function renderMarkdownReport(reportValue: {
  lock: EnsembleLock;
  llmDisambiguator: { promptVersion: string; promptHash: string; decisionCount: number; switchedToControl: number };
  baseTestMetrics: Metrics;
  testMetrics: Metrics;
}): string {
  const metrics = reportValue.testMetrics;
  const base = reportValue.baseTestMetrics;
  return `# SetembroBR Binary LLM-Disambiguated Final Test Report

## Lock

- Selection strategy: \`${reportValue.lock.selectionStrategy ?? "unknown"}\`
- Threshold: \`${reportValue.lock.threshold.toFixed(8)}\`
- LLM prompt version: \`${reportValue.llmDisambiguator.promptVersion}\`
- LLM prompt hash: \`${reportValue.llmDisambiguator.promptHash}\`
- LLM decisions: \`${reportValue.llmDisambiguator.decisionCount}\`
- Switched diagnosed to control: \`${reportValue.llmDisambiguator.switchedToControl}\`

## Test Metrics

| Metric | Base | LLM-disambiguated |
| --- | ---: | ---: |
| Macro F1 | \`${base.macroF1.toFixed(6)}\` | \`${metrics.macroF1.toFixed(6)}\` |
| Diagnosed F1 | \`${base.diagnosedF1.toFixed(6)}\` | \`${metrics.diagnosedF1.toFixed(6)}\` |
| Diagnosed precision | \`${base.precision.toFixed(6)}\` | \`${metrics.precision.toFixed(6)}\` |
| Diagnosed recall | \`${base.recall.toFixed(6)}\` | \`${metrics.recall.toFixed(6)}\` |
| Accuracy | \`${base.accuracy.toFixed(6)}\` | \`${metrics.accuracy.toFixed(6)}\` |

## Confusion Matrix

|  | Pred diagnosed | Pred control |
| --- | ---: | ---: |
| Actual diagnosed | ${metrics.tp} | ${metrics.fn} |
| Actual control | ${metrics.fp} | ${metrics.tn} |
`;
}
