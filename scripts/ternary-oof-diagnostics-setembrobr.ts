import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { writeJson } from "../src/artifacts.ts";
import { sha256Text } from "../src/hash.ts";
import { loadTernaryConfig, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import { computeTernaryMetrics, predictedTernaryLabel, readTernaryOofScores, TERNARY_LABELS } from "../src/ternary.ts";
import type { TernaryDecisionRule, TernaryEnsembleLock, TernaryLabel, TernaryMetrics, TernaryProbabilityRow } from "../src/types.ts";

type ProbTriple = [number, number, number];

interface DiagnosticRow {
  userId: string;
  fold: number;
  actual: TernaryLabel;
  predicted: TernaryLabel;
  probs: ProbTriple;
  predictedProbability: number;
  maxProbability: number;
  actualProbability: number;
  correct: boolean;
}

interface ConfidenceBin {
  lower: number;
  upper: number;
  count: number;
  accuracy: number;
  averageConfidence: number;
  calibrationGap: number;
}

interface FoldSummary {
  fold: number;
  count: number;
  accuracy: number;
  brierScore: number;
  predictedCounts: Record<TernaryLabel, number>;
  actualCounts: Record<TernaryLabel, number>;
}

interface ClassSummary {
  actualCount: number;
  predictedCount: number;
  averageActualProbability: number;
  averagePredictedProbability: number;
}

interface OofDiagnosticsReport {
  dataset: "setembrobr";
  seed: number;
  usesTestLabels: boolean;
  usesTestScores: boolean;
  sourceArtifacts: string[];
  labelPolicyId: string;
  modelIds: string[];
  decisionRule: TernaryDecisionRule;
  oofMetrics: TernaryMetrics;
  probabilityDiagnostics: {
    brierScore: number;
    negativeLogLikelihood: number;
    expectedCalibrationError: number;
    confidenceBins: ConfidenceBin[];
    meanPredictedProbability: Record<TernaryLabel, number>;
  };
  foldSummaries: FoldSummary[];
  classSummaries: Record<TernaryLabel, ClassSummary>;
  highConfidenceErrors: Array<{
    userIdHash: string;
    fold: number;
    actual: TernaryLabel;
    predicted: TernaryLabel;
    predictedProbability: number;
    actualProbability: number;
  }>;
  generatedAt: string;
}

const config = await loadTernaryConfig();
const lock = (await Bun.file(resolveTernaryOutputPath(config, "ensemble", "ensemble-lock.json")).json()) as TernaryEnsembleLock;
const rows = await lockedRows(lock);
const metrics = evaluateRows(rows, lock.decisionRule);
const diagnostics = diagnosticRows(rows, lock.decisionRule);
const report: OofDiagnosticsReport = {
  dataset: config.dataset,
  seed: config.seed,
  usesTestLabels: false,
  usesTestScores: false,
  sourceArtifacts: ["train_oof_*.csv", "ensemble-lock.json"],
  labelPolicyId: lock.labelPolicyId,
  modelIds: lock.modelIds,
  decisionRule: lock.decisionRule,
  oofMetrics: metrics,
  probabilityDiagnostics: {
    brierScore: brierScore(diagnostics),
    negativeLogLikelihood: negativeLogLikelihood(diagnostics),
    expectedCalibrationError: expectedCalibrationError(confidenceBins(diagnostics)),
    confidenceBins: confidenceBins(diagnostics),
    meanPredictedProbability: meanPredictedProbability(diagnostics),
  },
  foldSummaries: foldSummaries(diagnostics),
  classSummaries: classSummaries(diagnostics),
  highConfidenceErrors: diagnostics
    .filter((row) => !row.correct)
    .sort((left, right) => right.predictedProbability - left.predictedProbability || left.userId.localeCompare(right.userId))
    .slice(0, 25)
    .map(({ userId, fold, actual, predicted, predictedProbability, actualProbability }) => ({
      userIdHash: sha256Text(userId),
      fold,
      actual,
      predicted,
      predictedProbability,
      actualProbability,
    })),
  generatedAt: new Date(0).toISOString(),
};

const outJson = resolveTernaryOutputPath(config, "reports", "ternary-oof-diagnostics.json");
const outMd = resolveTernaryOutputPath(config, "reports", "ternary-oof-diagnostics.md");
await writeJson(outJson, report);
await mkdir(dirname(outMd), { recursive: true });
await writeFile(outMd, renderMarkdown(report));
console.log(`wrote ${outJson}`);
console.log(`wrote ${outMd}`);
console.log(JSON.stringify(report.probabilityDiagnostics, null, 2));

async function lockedRows(lock: TernaryEnsembleLock): Promise<TernaryProbabilityRow[]> {
  const rowsByModel = new Map<string, TernaryProbabilityRow[]>();
  for (const modelId of lock.modelIds) {
    rowsByModel.set(
      modelId,
      await readTernaryOofScores(resolveTernaryOutputPath(config, "scores", `train_oof_${lock.labelPolicyId}_${modelId}.csv`)),
    );
  }
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
    predicted.push(predictedTernaryLabel(rowToProbs(row), rule));
  }
  return computeTernaryMetrics(actual, predicted);
}

function diagnosticRows(rows: readonly TernaryProbabilityRow[], rule: TernaryDecisionRule): DiagnosticRow[] {
  return rows.map((row) => {
    if (!row.label || row.fold === undefined) throw new Error(`Missing OOF label/fold for ${row.userId}`);
    const probs = rowToProbs(row);
    const predicted = predictedTernaryLabel(probs, rule);
    const predictedProbability = probabilityForLabel(probs, predicted);
    return {
      userId: row.userId,
      fold: row.fold,
      actual: row.label,
      predicted,
      probs,
      predictedProbability,
      maxProbability: Math.max(...probs),
      actualProbability: probabilityForLabel(probs, row.label),
      correct: predicted === row.label,
    };
  });
}

function confidenceBins(rows: readonly DiagnosticRow[]): ConfidenceBin[] {
  const edges = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001];
  return edges.slice(0, -1).map((lower, index) => {
    const upper = edges[index + 1]!;
    const selected = rows.filter((row) => row.predictedProbability >= lower && row.predictedProbability < upper);
    const accuracy = average(selected.map((row) => (row.correct ? 1 : 0)));
    const averageConfidence = average(selected.map((row) => row.predictedProbability));
    return {
      lower,
      upper: upper > 1 ? 1 : upper,
      count: selected.length,
      accuracy,
      averageConfidence,
      calibrationGap: averageConfidence - accuracy,
    };
  });
}

function expectedCalibrationError(bins: readonly ConfidenceBin[]): number {
  const total = bins.reduce((sum, bin) => sum + bin.count, 0);
  if (total === 0) return 0;
  return bins.reduce((sum, bin) => sum + (bin.count / total) * Math.abs(bin.calibrationGap), 0);
}

function brierScore(rows: readonly DiagnosticRow[]): number {
  return average(
    rows.map((row) =>
      TERNARY_LABELS.reduce((sum, label, index) => sum + (row.probs[index]! - (row.actual === label ? 1 : 0)) ** 2, 0),
    ),
  );
}

function negativeLogLikelihood(rows: readonly DiagnosticRow[]): number {
  return average(rows.map((row) => -Math.log(Math.max(row.actualProbability, 1e-12))));
}

function meanPredictedProbability(rows: readonly DiagnosticRow[]) {
  return Object.fromEntries(
    TERNARY_LABELS.map((label, index) => [label, average(rows.map((row) => row.probs[index]!))]),
  ) as Record<TernaryLabel, number>;
}

function foldSummaries(rows: readonly DiagnosticRow[]): FoldSummary[] {
  const folds = [...new Set(rows.map((row) => row.fold))].sort((a, b) => a - b);
  return folds.map((fold) => {
    const selected = rows.filter((row) => row.fold === fold);
    return {
      fold,
      count: selected.length,
      accuracy: average(selected.map((row) => (row.correct ? 1 : 0))),
      brierScore: brierScore(selected),
      predictedCounts: labelCounts(selected.map((row) => row.predicted)),
      actualCounts: labelCounts(selected.map((row) => row.actual)),
    };
  });
}

function classSummaries(rows: readonly DiagnosticRow[]): Record<TernaryLabel, ClassSummary> {
  return Object.fromEntries(
    TERNARY_LABELS.map((label) => {
      const actualRows = rows.filter((row) => row.actual === label);
      const predictedRows = rows.filter((row) => row.predicted === label);
      return [
        label,
        {
          actualCount: actualRows.length,
          predictedCount: predictedRows.length,
          averageActualProbability: average(actualRows.map((row) => row.actualProbability)),
          averagePredictedProbability: average(predictedRows.map((row) => row.predictedProbability)),
        },
      ];
    }),
  ) as Record<TernaryLabel, ClassSummary>;
}

function rowToProbs(row: TernaryProbabilityRow): ProbTriple {
  return [row.probDiagnosed, row.probControl, row.probNoEvidence];
}

function probabilityForLabel(probs: ProbTriple, label: TernaryLabel): number {
  return probs[TERNARY_LABELS.indexOf(label)]!;
}

function labelCounts(labels: readonly TernaryLabel[]): Record<TernaryLabel, number> {
  return Object.fromEntries(TERNARY_LABELS.map((label) => [label, labels.filter((entry) => entry === label).length])) as Record<TernaryLabel, number>;
}

function average(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function renderMarkdown(report: OofDiagnosticsReport): string {
  const lines = [
    "# SetembroBR Ternary OOF Diagnostics",
    "",
    "This diagnostic uses train OOF probabilities only. It does not read test score files, test labels, final test reports, or test prevalence.",
    "",
    "## Probability Diagnostics",
    "",
    `- Brier score: \`${fmt(report.probabilityDiagnostics.brierScore)}\``,
    `- Negative log likelihood: \`${fmt(report.probabilityDiagnostics.negativeLogLikelihood)}\``,
    `- Expected calibration error: \`${fmt(report.probabilityDiagnostics.expectedCalibrationError)}\``,
    "",
    "## Confidence Bins",
    "",
    "| Probability bin | Count | Accuracy | Avg confidence | Gap |",
    "| --- | ---: | ---: | ---: | ---: |",
    ...report.probabilityDiagnostics.confidenceBins.map(
      (bin) => `| [${fmt(bin.lower)}, ${fmt(bin.upper)}) | ${bin.count} | ${fmt(bin.accuracy)} | ${fmt(bin.averageConfidence)} | ${fmt(bin.calibrationGap)} |`,
    ),
    "",
    "## Fold Summaries",
    "",
    "| Fold | Count | Accuracy | Brier score |",
    "| ---: | ---: | ---: | ---: |",
    ...report.foldSummaries.map((fold) => `| ${fold.fold} | ${fold.count} | ${fmt(fold.accuracy)} | ${fmt(fold.brierScore)} |`),
    "",
  ];
  return `${lines.join("\n")}\n`;
}

function fmt(value: number): string {
  return value.toFixed(6);
}
