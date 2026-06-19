import { mkdir, readFile } from "node:fs/promises";
import { dirname } from "node:path";
import { listCsvFiles, writeJson } from "../src/artifacts.ts";
import { sha256File } from "../src/hash.ts";
import { loadTernaryConfig, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import { compareTernaryMetrics, readTernaryOofScores, selectTernaryEnsemble } from "../src/ternary.ts";
import type { TernaryEnsembleLock, TernaryLabelPolicyLock, TernaryModelSelectionGroup, TernaryProbabilityRow } from "../src/types.ts";

const config = await loadTernaryConfig();
const scoresDir = resolveTernaryOutputPath(config, "scores");
const candidates: TernaryEnsembleLock[] = [];
const selectionGroupFilter = process.env.TERNARY_SELECTION_GROUP_ID?.trim();
const lockBasename = process.env.TERNARY_LOCK_BASENAME?.trim() || "ensemble-lock";

for (const policy of config.labelPolicies) {
  const policyLock = JSON.parse(
    await readFile(resolveTernaryOutputPath(config, "label-policies", `${policy.policyId}.json`), "utf8"),
  ) as TernaryLabelPolicyLock;
  const oofFiles = await listCsvFiles(scoresDir, `train_oof_${policy.policyId}_`);
  if (oofFiles.length === 0) continue;

  const oofByModel = new Map<string, TernaryProbabilityRow[]>();
  const sourceHashes: Record<string, string> = {};
  for (const path of oofFiles) {
    const rows = await readTernaryOofScores(path);
    const modelId = rows[0]?.modelId;
    if (!modelId) throw new Error(`Missing model_id in ${path}`);
    oofByModel.set(modelId, rows);
    sourceHashes[modelId] = await sha256File(path);
  }

  const groups = selectionGroups([...oofByModel.keys()].sort());
  const selectedGroups = selectionGroupFilter ? groups.filter((group) => group.groupId === selectionGroupFilter) : groups;
  if (selectionGroupFilter && selectedGroups.length === 0) {
    throw new Error(`No available ternary selection group matched ${selectionGroupFilter} for policy ${policy.policyId}`);
  }
  const groupLocks = selectedGroups.map((group) => selectForGroup(policyLock, oofByModel, sourceHashes, group));
  candidates.push(...groupLocks);
}

if (candidates.length === 0) throw new Error("No ternary OOF files found. Run ternary training and audit first.");

const best = candidates.sort((left, right) => {
  const metricDiff = compareTernaryMetrics(right.oofMetrics, left.oofMetrics);
  if (metricDiff !== 0) return metricDiff;
  return `${left.labelPolicyId}:${left.selectionGroupId ?? ""}`.localeCompare(`${right.labelPolicyId}:${right.selectionGroupId ?? ""}`);
})[0]!;

const outPath = resolveTernaryOutputPath(config, "ensemble", `${lockBasename}.json`);
await mkdir(dirname(outPath), { recursive: true });
await writeJson(outPath, best);
const candidatesReportName = lockBasename === "ensemble-lock" ? "ensemble-candidates.json" : `${lockBasename}-candidates.json`;
await writeJson(resolveTernaryOutputPath(config, "reports", candidatesReportName), {
  dataset: "setembrobr",
  seed: config.seed,
  selectedPolicyId: best.labelPolicyId,
  selectedSelectionGroupId: best.selectionGroupId,
  selectionGroupFilter: selectionGroupFilter ?? null,
  candidates,
});
console.log(`wrote ${outPath}`);
console.log(JSON.stringify(best.oofMetrics, null, 2));

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
  policyLock: TernaryLabelPolicyLock,
  oofByModel: ReadonlyMap<string, readonly TernaryProbabilityRow[]>,
  sourceHashes: Record<string, string>,
  group: TernaryModelSelectionGroup,
): TernaryEnsembleLock {
  const groupedOof = new Map<string, readonly TernaryProbabilityRow[]>();
  const groupedHashes: Record<string, string> = {};
  for (const modelId of group.modelIds) {
    const rows = oofByModel.get(modelId);
    if (!rows) throw new Error(`${group.groupId}: missing rows for ${modelId}`);
    groupedOof.set(modelId, rows);
    groupedHashes[modelId] = sourceHashes[modelId] ?? "missing";
  }
  const lock = selectTernaryEnsemble({
    seed: config.seed,
    originalManifestHash: policyLock.originalManifestHash,
    labelPolicyId: policyLock.policyId,
    labelPolicyHash: policyLock.policyHash,
    oofByModel: groupedOof,
    sourceHashes: groupedHashes,
    weightStep: config.ensemble.weightStep,
    decisionRules: config.ensemble.decisionRules,
    command: `make ternary-select-ensemble-setembrobr group=${group.groupId}`,
    exhaustiveModelLimit: config.ensemble.exhaustiveModelLimit,
    candidatePruneTo: config.ensemble.candidatePruneTo,
    maxModels: config.ensemble.maxModels,
    ...localRefinementForGroup(policyLock.policyId, group.groupId),
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
    refineModelLimit: refinement.maxModels,
  };
}
