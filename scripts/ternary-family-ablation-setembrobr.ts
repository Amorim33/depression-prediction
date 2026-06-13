import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { listCsvFiles, writeJson } from "../src/artifacts.ts";
import { sha256File } from "../src/hash.ts";
import { loadTernaryConfig, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import { compareTernaryMetrics, readTernaryOofScores, selectTernaryEnsemble } from "../src/ternary.ts";
import type { TernaryEnsembleLock, TernaryLabelPolicyLock, TernaryProbabilityRow } from "../src/types.ts";

interface ModelMetadata {
  modelId: string;
  family: string;
  source: "tabular" | "sequence";
}

interface GroupSpec {
  groupId: string;
  description: string;
  modelIds: string[];
}

interface AblationEntry {
  labelPolicyId: string;
  groupId: string;
  description: string;
  inputModelIds: string[];
  selectedLock: TernaryEnsembleLock;
  deltaVsFullPolicyMacroF1: number;
}

const config = await loadTernaryConfig();
const metadataByModel = modelMetadata();
const groups = groupSpecs();
const entries: AblationEntry[] = [];

for (const policy of config.labelPolicies) {
  const policyLock = JSON.parse(
    await readFile(resolveTernaryOutputPath(config, "label-policies", `${policy.policyId}.json`), "utf8"),
  ) as TernaryLabelPolicyLock;
  const { rowsByModel, sourceHashes } = await loadPolicyRows(policy.policyId);
  const fullLock = selectForGroup(policyLock, rowsByModel, sourceHashes, "all_models", [...rowsByModel.keys()].sort());

  for (const group of groups) {
    const available = group.modelIds.filter((modelId) => rowsByModel.has(modelId)).sort();
    if (available.length === 0) continue;
    const selectedLock = group.groupId === "all_models" ? fullLock : selectForGroup(policyLock, rowsByModel, sourceHashes, group.groupId, available);
    entries.push({
      labelPolicyId: policy.policyId,
      groupId: group.groupId,
      description: group.description,
      inputModelIds: available,
      selectedLock,
      deltaVsFullPolicyMacroF1: selectedLock.oofMetrics.macroF1 - fullLock.oofMetrics.macroF1,
    });
  }
}

if (entries.length === 0) throw new Error("No ternary OOF rows found. Run ternary training first.");
entries.sort(compareEntries);

const report = {
  dataset: config.dataset,
  seed: config.seed,
  usesTestLabels: false,
  usesTestScores: false,
  sourceArtifacts: ["train_oof_*.csv", "label-policies/*.json", "ternary config"],
  rankedBy: ["OOF Macro F1", "OOF diagnosed F1", "OOF diagnosed precision", "OOF accuracy"],
  groupCount: groups.length,
  evaluatedGroupCount: entries.length,
  topAblations: entries,
  bestByGroup: bestBy(entries, (entry) => entry.groupId),
  bestByPolicy: bestBy(entries, (entry) => entry.labelPolicyId),
  generatedAt: new Date(0).toISOString(),
};

const outJson = resolveTernaryOutputPath(config, "reports", "ternary-family-ablation.json");
const outMd = resolveTernaryOutputPath(config, "reports", "ternary-family-ablation.md");
await writeJson(outJson, report);
await mkdir(dirname(outMd), { recursive: true });
await writeFile(outMd, renderMarkdown(report));
console.log(`wrote ${outJson}`);
console.log(`wrote ${outMd}`);
console.log(JSON.stringify(entries[0], null, 2));

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

function groupSpecs(): GroupSpec[] {
  if (config.ensemble.selectionGroups?.length) return config.ensemble.selectionGroups;
  const allModelIds = [...metadataByModel.keys()].sort();
  const tabular = modelsWhere((metadata) => metadata.source === "tabular");
  const tabularWithoutBaseline = modelsWhere((metadata) => metadata.source === "tabular" && metadata.family !== "relevance_baseline");
  const sequence = modelsWhere((metadata) => metadata.source === "sequence");
  const cnnSequence = modelsWhere((metadata) => metadata.source === "sequence" && (metadata.family === "cnn" || metadata.family === "cnn_wide"));
  const nonCnnSequence = modelsWhere((metadata) => metadata.source === "sequence" && metadata.family !== "cnn" && metadata.family !== "cnn_wide");
  const evidenceBaseline = modelsWhere((metadata) => metadata.family === "relevance_baseline");
  return [
    { groupId: "all_models", description: "All pre-registered ternary models.", modelIds: allModelIds },
    { groupId: "tabular_all", description: "All tabular ternary models.", modelIds: tabular },
    { groupId: "tabular_without_relevance_baseline", description: "Tabular models excluding the relevance-only baseline.", modelIds: tabularWithoutBaseline },
    { groupId: "sequence_all", description: "All Fedora-trained sequence models.", modelIds: sequence },
    { groupId: "sequence_cnn_family", description: "CNN and wide-CNN sequence models.", modelIds: cnnSequence },
    { groupId: "sequence_bilstm_transformer", description: "BiLSTM and tiny-transformer sequence models.", modelIds: nonCnnSequence },
    { groupId: "relevance_baseline_only", description: "Relevance-only no-evidence baseline.", modelIds: evidenceBaseline },
  ];
}

function modelsWhere(predicate: (metadata: ModelMetadata) => boolean): string[] {
  return [...metadataByModel.values()].filter(predicate).map((metadata) => metadata.modelId).sort();
}

async function loadPolicyRows(policyId: string): Promise<{
  rowsByModel: Map<string, TernaryProbabilityRow[]>;
  sourceHashes: Record<string, string>;
}> {
  const rowsByModel = new Map<string, TernaryProbabilityRow[]>();
  const sourceHashes: Record<string, string> = {};
  for (const path of await listCsvFiles(resolveTernaryOutputPath(config, "scores"), `train_oof_${policyId}_`)) {
    const rows = await readTernaryOofScores(path);
    const modelId = rows[0]?.modelId;
    if (!modelId) throw new Error(`Missing model_id in ${path}`);
    rowsByModel.set(modelId, rows);
    sourceHashes[modelId] = await sha256File(path);
  }
  return { rowsByModel, sourceHashes };
}

function selectForGroup(
  policyLock: TernaryLabelPolicyLock,
  rowsByModel: ReadonlyMap<string, readonly TernaryProbabilityRow[]>,
  sourceHashes: Record<string, string>,
  groupId: string,
  modelIds: readonly string[],
): TernaryEnsembleLock {
  const restrictedRows = new Map<string, readonly TernaryProbabilityRow[]>();
  const restrictedHashes: Record<string, string> = {};
  for (const modelId of modelIds) {
    const rows = rowsByModel.get(modelId);
    if (!rows) throw new Error(`${groupId}: missing rows for ${modelId}`);
    restrictedRows.set(modelId, rows);
    restrictedHashes[modelId] = sourceHashes[modelId] ?? "missing";
  }
  return selectTernaryEnsemble({
    seed: config.seed,
    originalManifestHash: policyLock.originalManifestHash,
    labelPolicyId: policyLock.policyId,
    labelPolicyHash: policyLock.policyHash,
    oofByModel: restrictedRows,
    sourceHashes: restrictedHashes,
    weightStep: config.ensemble.weightStep,
    decisionRules: config.ensemble.decisionRules,
    command: `make ternary-family-ablation-setembrobr group=${groupId}`,
    exhaustiveModelLimit: config.ensemble.exhaustiveModelLimit,
    candidatePruneTo: config.ensemble.candidatePruneTo,
    maxModels: config.ensemble.maxModels,
  });
}

function bestBy(entries: readonly AblationEntry[], keyFn: (entry: AblationEntry) => string): AblationEntry[] {
  const best = new Map<string, AblationEntry>();
  for (const entry of entries) {
    const key = keyFn(entry);
    const current = best.get(key);
    if (!current || compareEntries(entry, current) < 0) best.set(key, entry);
  }
  return [...best.values()].sort(compareEntries);
}

function compareEntries(left: AblationEntry, right: AblationEntry): number {
  const metricDiff = compareTernaryMetrics(right.selectedLock.oofMetrics, left.selectedLock.oofMetrics);
  if (metricDiff !== 0) return metricDiff;
  return `${left.labelPolicyId}:${left.groupId}`.localeCompare(`${right.labelPolicyId}:${right.groupId}`);
}

function renderMarkdown(report: {
  topAblations: AblationEntry[];
  bestByGroup: AblationEntry[];
  bestByPolicy: AblationEntry[];
}): string {
  const lines = [
    "# SetembroBR Ternary Family Ablation",
    "",
    "This report uses train OOF probabilities only. It does not read test score files, test labels, final test reports, or test prevalence.",
    "",
    "## Top Group Ensembles",
    "",
    "| Rank | Policy | Group | Models in group | Selected models | Macro F1 | Diagnosed F1 | Diagnosed precision | Delta vs full policy |",
    "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...report.topAblations.slice(0, 20).map(
      (entry, index) =>
        `| ${index + 1} | \`${entry.labelPolicyId}\` | \`${entry.groupId}\` | ${entry.inputModelIds.length} | ${entry.selectedLock.modelIds.length} | ${fmt(entry.selectedLock.oofMetrics.macroF1)} | ${fmt(entry.selectedLock.oofMetrics.diagnosedF1)} | ${fmt(entry.selectedLock.oofMetrics.diagnosedPrecision)} | ${fmt(entry.deltaVsFullPolicyMacroF1)} |`,
    ),
    "",
    "## Best By Group",
    "",
    "| Group | Best policy | Macro F1 | Selected models |",
    "| --- | --- | ---: | ---: |",
    ...report.bestByGroup.map(
      (entry) =>
        `| \`${entry.groupId}\` | \`${entry.labelPolicyId}\` | ${fmt(entry.selectedLock.oofMetrics.macroF1)} | ${entry.selectedLock.modelIds.length} |`,
    ),
    "",
    "## Best By Policy",
    "",
    "| Policy | Best group | Macro F1 | Selected models |",
    "| --- | --- | ---: | ---: |",
    ...report.bestByPolicy.map(
      (entry) =>
        `| \`${entry.labelPolicyId}\` | \`${entry.groupId}\` | ${fmt(entry.selectedLock.oofMetrics.macroF1)} | ${entry.selectedLock.modelIds.length} |`,
    ),
    "",
  ];
  return `${lines.join("\n")}\n`;
}

function fmt(value: number): string {
  return value.toFixed(6);
}
