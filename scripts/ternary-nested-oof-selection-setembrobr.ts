import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { listCsvFiles, writeJson } from "../src/artifacts.ts";
import { sha256File } from "../src/hash.ts";
import { loadTernaryConfig, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import {
  compareTernaryMetrics,
  computeTernaryMetrics,
  predictedTernaryLabel,
  readTernaryOofScores,
  selectTernaryEnsemble,
} from "../src/ternary.ts";
import type {
  TernaryEnsembleLock,
  TernaryLabel,
  TernaryLabelPolicyLock,
  TernaryMetrics,
  TernaryModelSelectionGroup,
  TernaryProbabilityRow,
} from "../src/types.ts";

interface OuterFoldResult {
  heldoutFold: number;
  selectedLabelPolicyId: string;
  selectedSelectionGroupId: string;
  selectedCandidateModelIds: string[];
  selectedModelIds: string[];
  selectedDecisionRuleId: string;
  innerSelectionMetrics: TernaryMetrics;
  heldoutMetrics: TernaryMetrics;
}

interface PolicyRows {
  lock: TernaryLabelPolicyLock;
  rowsByModel: Map<string, TernaryProbabilityRow[]>;
  sourceHashes: Record<string, string>;
  folds: number[];
}

const config = await loadTernaryConfig();
const nestedSelector = {
  exhaustiveModelLimit: Math.min(config.ensemble.exhaustiveModelLimit ?? 4, 4),
  candidatePruneTo: Math.min(config.ensemble.candidatePruneTo ?? 8, 8),
  maxModels: Math.min(config.ensemble.maxModels ?? 6, 6),
};
const policyRows = await loadPolicyRows();
const folds = [...new Set([...policyRows.values()].flatMap((policy) => policy.folds))].sort((a, b) => a - b);
const outerFoldResults: OuterFoldResult[] = [];

for (const heldoutFold of folds) {
  const locks: TernaryEnsembleLock[] = [];
  for (const policy of policyRows.values()) {
    const innerRows = filterRowsByFold(policy.rowsByModel, heldoutFold, false);
    if (innerRows.size === 0) continue;
    locks.push(...selectionGroups([...innerRows.keys()].sort()).map((group) => selectForGroup(policy, innerRows, group, heldoutFold)));
  }

  const selected = bestLock(locks);
  const selectedPolicy = policyRows.get(selected.labelPolicyId);
  if (!selectedPolicy) throw new Error(`Missing selected policy rows for ${selected.labelPolicyId}`);
  const heldoutRows = filterRowsByFold(selectedPolicy.rowsByModel, heldoutFold, true);
  outerFoldResults.push({
    heldoutFold,
    selectedLabelPolicyId: selected.labelPolicyId,
    selectedSelectionGroupId: selected.selectionGroupId ?? "all_models",
    selectedCandidateModelIds: selected.candidateModelIds ?? selected.modelIds,
    selectedModelIds: selected.modelIds,
    selectedDecisionRuleId: selected.decisionRule.ruleId,
    innerSelectionMetrics: selected.oofMetrics,
    heldoutMetrics: evaluateHeldout(selected, heldoutRows),
  });
}

if (outerFoldResults.length === 0) throw new Error("No outer-fold results were generated");

const aggregate = aggregateMetrics(outerFoldResults.map((result) => result.heldoutMetrics));
const selectionCounts = countSelections(outerFoldResults);
const report = {
  dataset: config.dataset,
  seed: config.seed,
  usesTestLabels: false,
  usesTestScores: false,
  sourceArtifacts: ["train_oof_*.csv", "label-policies/*.json", "ternary config selectionGroups"],
  method: "For each train fold, select policy/model-group/models/weights/rule on the other train OOF folds, then evaluate the locked inner choice on the held-out train fold OOF rows.",
  boundedSelector: {
    ...nestedSelector,
    reason: "Nested diagnostics evaluate every label policy and selection group, but force larger groups through the existing greedy-pruned selector to keep routine strict-blind reproduction bounded.",
  },
  outerFoldCount: outerFoldResults.length,
  selectionCounts,
  aggregateHeldoutMetrics: aggregate,
  outerFoldResults,
  generatedAt: new Date(0).toISOString(),
};

const outJson = resolveTernaryOutputPath(config, "reports", "ternary-nested-oof-selection.json");
const outMd = resolveTernaryOutputPath(config, "reports", "ternary-nested-oof-selection.md");
await writeJson(outJson, report);
await mkdir(dirname(outMd), { recursive: true });
await writeFile(outMd, renderMarkdown(report));
console.log(`wrote ${outJson}`);
console.log(`wrote ${outMd}`);
console.log(JSON.stringify(aggregate, null, 2));

async function loadPolicyRows(): Promise<Map<string, PolicyRows>> {
  const scoresDir = resolveTernaryOutputPath(config, "scores");
  const out = new Map<string, PolicyRows>();
  for (const policy of config.labelPolicies) {
    const lock = JSON.parse(
      await readFile(resolveTernaryOutputPath(config, "label-policies", `${policy.policyId}.json`), "utf8"),
    ) as TernaryLabelPolicyLock;
    const rowsByModel = new Map<string, TernaryProbabilityRow[]>();
    const sourceHashes: Record<string, string> = {};
    const seenFolds = new Set<number>();
    for (const path of await listCsvFiles(scoresDir, `train_oof_${policy.policyId}_`)) {
      const rows = await readTernaryOofScores(path);
      const modelId = rows[0]?.modelId;
      if (!modelId) throw new Error(`Missing model_id in ${path}`);
      for (const row of rows) {
        if (row.fold === undefined) throw new Error(`Missing fold in ${path} for ${row.userId}`);
        seenFolds.add(row.fold);
      }
      rowsByModel.set(modelId, rows);
      sourceHashes[modelId] = await sha256File(path);
    }
    if (rowsByModel.size > 0) out.set(policy.policyId, { lock, rowsByModel, sourceHashes, folds: [...seenFolds].sort((a, b) => a - b) });
  }
  return out;
}

function filterRowsByFold(
  rowsByModel: ReadonlyMap<string, readonly TernaryProbabilityRow[]>,
  fold: number,
  includeHeldout: boolean,
): Map<string, TernaryProbabilityRow[]> {
  const out = new Map<string, TernaryProbabilityRow[]>();
  for (const [modelId, rows] of rowsByModel) {
    out.set(
      modelId,
      rows.filter((row) => (includeHeldout ? row.fold === fold : row.fold !== fold)),
    );
  }
  return out;
}

function selectionGroups(availableModelIds: readonly string[]): TernaryModelSelectionGroup[] {
  const configured = config.ensemble.selectionGroups?.length
    ? config.ensemble.selectionGroups
    : [{ groupId: "all_models", description: "All available models.", modelIds: [...availableModelIds] }];
  return configured
    .map((group) => ({
      ...group,
      modelIds: group.modelIds.filter((modelId) => availableModelIds.includes(modelId)).sort(),
    }))
    .filter((group) => group.modelIds.length > 0);
}

function selectForGroup(
  policy: PolicyRows,
  innerRows: ReadonlyMap<string, readonly TernaryProbabilityRow[]>,
  group: TernaryModelSelectionGroup,
  heldoutFold: number,
): TernaryEnsembleLock {
  const groupedRows = new Map<string, readonly TernaryProbabilityRow[]>();
  const groupedHashes: Record<string, string> = {};
  for (const modelId of group.modelIds) {
    const rows = innerRows.get(modelId);
    if (!rows) throw new Error(`${policy.lock.policyId}/${group.groupId}: missing inner rows for ${modelId}`);
    groupedRows.set(modelId, rows);
    groupedHashes[modelId] = policy.sourceHashes[modelId] ?? "missing";
  }
  const lock = selectTernaryEnsemble({
    seed: config.seed,
    originalManifestHash: policy.lock.originalManifestHash,
    labelPolicyId: policy.lock.policyId,
    labelPolicyHash: policy.lock.policyHash,
    oofByModel: groupedRows,
    sourceHashes: groupedHashes,
    weightStep: config.ensemble.weightStep,
    decisionRules: config.ensemble.decisionRules,
    command: `make ternary-nested-oof-selection-setembrobr heldout_fold=${heldoutFold} group=${group.groupId}`,
    exhaustiveModelLimit: nestedSelector.exhaustiveModelLimit,
    candidatePruneTo: nestedSelector.candidatePruneTo,
    maxModels: nestedSelector.maxModels,
    ...localRefinementForGroup(policy.lock.policyId, group.groupId),
  });
  return {
    ...lock,
    selectionGroupId: group.groupId,
    selectionGroupDescription: group.description,
    candidateModelIds: group.modelIds,
  };
}

function localRefinementForGroup(policyId: string, groupId: string) {
  const refinement = config.ensemble.localRefinement;
  if (!refinement?.groupIds.includes(groupId)) return {};
  if (refinement.policyIds && !refinement.policyIds.includes(policyId)) return {};
  return {
    refineWeightStep: refinement.weightStep,
    refineWeightRadius: refinement.radius,
    refineModelLimit: Math.min(refinement.maxModels, nestedSelector.maxModels),
  };
}

function bestLock(locks: readonly TernaryEnsembleLock[]): TernaryEnsembleLock {
  if (locks.length === 0) throw new Error("No inner locks generated");
  return [...locks].sort((left, right) => {
    const metricDiff = compareTernaryMetrics(right.oofMetrics, left.oofMetrics);
    if (metricDiff !== 0) return metricDiff;
    return `${left.labelPolicyId}:${left.selectionGroupId ?? ""}`.localeCompare(`${right.labelPolicyId}:${right.selectionGroupId ?? ""}`);
  })[0]!;
}

function evaluateHeldout(
  lock: TernaryEnsembleLock,
  rowsByModel: ReadonlyMap<string, readonly TernaryProbabilityRow[]>,
): TernaryMetrics {
  const firstRows = rowsByModel.get(lock.modelIds[0]!);
  if (!firstRows) throw new Error(`Missing held-out rows for ${lock.modelIds[0]}`);
  const indexed = new Map(
    lock.modelIds.map((modelId) => {
      const rows = rowsByModel.get(modelId);
      if (!rows) throw new Error(`Missing held-out rows for ${modelId}`);
      return [modelId, new Map(rows.map((row) => [row.userId, row]))] as const;
    }),
  );
  const actual: TernaryLabel[] = [];
  const predicted: TernaryLabel[] = [];
  for (const base of firstRows) {
    if (!base.label) throw new Error(`Missing held-out label for ${base.userId}`);
    const probs: [number, number, number] = [0, 0, 0];
    for (const modelId of lock.modelIds) {
      const row = indexed.get(modelId)?.get(base.userId);
      if (!row) throw new Error(`Missing ${modelId} held-out row for ${base.userId}`);
      if (row.label !== base.label || row.fold !== base.fold) throw new Error(`Held-out row alignment mismatch for ${base.userId}`);
      const weight = lock.weights[modelId] ?? 0;
      probs[0] += row.probDiagnosed * weight;
      probs[1] += row.probControl * weight;
      probs[2] += row.probNoEvidence * weight;
    }
    actual.push(base.label);
    predicted.push(predictedTernaryLabel(probs, lock.decisionRule));
  }
  return computeTernaryMetrics(actual, predicted);
}

function aggregateMetrics(metrics: readonly TernaryMetrics[]) {
  return {
    macroF1: scalar(metrics.map((metric) => metric.macroF1)),
    diagnosedF1: scalar(metrics.map((metric) => metric.diagnosedF1)),
    diagnosedPrecision: scalar(metrics.map((metric) => metric.diagnosedPrecision)),
    diagnosedRecall: scalar(metrics.map((metric) => metric.diagnosedRecall)),
    accuracy: scalar(metrics.map((metric) => metric.accuracy)),
  };
}

function scalar(values: readonly number[]) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    mean,
    min: Math.min(...values),
    max: Math.max(...values),
    std: Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length),
  };
}

function countSelections(results: readonly OuterFoldResult[]) {
  const policies: Record<string, number> = {};
  const groups: Record<string, number> = {};
  const rules: Record<string, number> = {};
  for (const result of results) {
    policies[result.selectedLabelPolicyId] = (policies[result.selectedLabelPolicyId] ?? 0) + 1;
    groups[result.selectedSelectionGroupId] = (groups[result.selectedSelectionGroupId] ?? 0) + 1;
    rules[result.selectedDecisionRuleId] = (rules[result.selectedDecisionRuleId] ?? 0) + 1;
  }
  return { policies, groups, rules };
}

function renderMarkdown(report: {
  aggregateHeldoutMetrics: ReturnType<typeof aggregateMetrics>;
  outerFoldResults: OuterFoldResult[];
  selectionCounts: ReturnType<typeof countSelections>;
  boundedSelector: { exhaustiveModelLimit: number; candidatePruneTo: number; maxModels: number; reason: string };
}): string {
  const lines = [
    "# SetembroBR Ternary Nested OOF Selection",
    "",
    "This report is train-only. Each outer split selects policy, model group, weights, and rule on four train OOF folds, then evaluates on the remaining train OOF fold.",
    "It does not read test score files, test labels, final test reports, or test prevalence.",
    `Nested selector bounds: exhaustive groups up to ${report.boundedSelector.exhaustiveModelLimit} models; prune to ${report.boundedSelector.candidatePruneTo}; max selected models ${report.boundedSelector.maxModels}.`,
    "",
    "## Aggregate Held-Out Train-Fold Metrics",
    "",
    "| Metric | Mean | Min | Max | Std |",
    "| --- | ---: | ---: | ---: | ---: |",
    `| Macro F1 | ${fmt(report.aggregateHeldoutMetrics.macroF1.mean)} | ${fmt(report.aggregateHeldoutMetrics.macroF1.min)} | ${fmt(report.aggregateHeldoutMetrics.macroF1.max)} | ${fmt(report.aggregateHeldoutMetrics.macroF1.std)} |`,
    `| Diagnosed F1 | ${fmt(report.aggregateHeldoutMetrics.diagnosedF1.mean)} | ${fmt(report.aggregateHeldoutMetrics.diagnosedF1.min)} | ${fmt(report.aggregateHeldoutMetrics.diagnosedF1.max)} | ${fmt(report.aggregateHeldoutMetrics.diagnosedF1.std)} |`,
    `| Diagnosed precision | ${fmt(report.aggregateHeldoutMetrics.diagnosedPrecision.mean)} | ${fmt(report.aggregateHeldoutMetrics.diagnosedPrecision.min)} | ${fmt(report.aggregateHeldoutMetrics.diagnosedPrecision.max)} | ${fmt(report.aggregateHeldoutMetrics.diagnosedPrecision.std)} |`,
    `| Accuracy | ${fmt(report.aggregateHeldoutMetrics.accuracy.mean)} | ${fmt(report.aggregateHeldoutMetrics.accuracy.min)} | ${fmt(report.aggregateHeldoutMetrics.accuracy.max)} | ${fmt(report.aggregateHeldoutMetrics.accuracy.std)} |`,
    "",
    "## Outer Fold Results",
    "",
    "| Held-out fold | Selected policy | Group | Rule | Candidate models | Selected models | Inner Macro F1 | Held-out Macro F1 | Held-out diagnosed precision |",
    "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ...report.outerFoldResults.map(
      (result) =>
        `| ${result.heldoutFold} | \`${result.selectedLabelPolicyId}\` | \`${result.selectedSelectionGroupId}\` | \`${result.selectedDecisionRuleId}\` | ${result.selectedCandidateModelIds.length} | ${result.selectedModelIds.length} | ${fmt(result.innerSelectionMetrics.macroF1)} | ${fmt(result.heldoutMetrics.macroF1)} | ${fmt(result.heldoutMetrics.diagnosedPrecision)} |`,
    ),
    "",
    "## Selection Counts",
    "",
    `- Policies: ${JSON.stringify(report.selectionCounts.policies)}`,
    `- Groups: ${JSON.stringify(report.selectionCounts.groups)}`,
    `- Rules: ${JSON.stringify(report.selectionCounts.rules)}`,
    "",
  ];
  return `${lines.join("\n")}\n`;
}

function fmt(value: number): string {
  return value.toFixed(6);
}
