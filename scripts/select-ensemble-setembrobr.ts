import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { sha256File } from "../src/hash.ts";
import { listCsvFiles, readOofScores, writeJson } from "../src/artifacts.ts";
import { loadConfig, resolveOutputPath } from "../src/config.ts";
import { readManifest, manifestHash } from "../src/manifest.ts";
import { selectEnsemble } from "../src/ensemble.ts";
import { isRawBinaryConfig, rawBinaryStrictBlindManifestHash } from "../src/raw-binary.ts";

const config = await loadConfig();
const lockBasename = process.env.BINARY_LOCK_BASENAME?.trim() || "ensemble-lock";
const selectionGroupId = process.env.BINARY_SELECTION_GROUP_ID?.trim();
const selectionGroup = selectionGroupId
  ? config.ensemble.selectionGroups?.find((group) => group.groupId === selectionGroupId)
  : undefined;
if (selectionGroupId && !selectionGroup) throw new Error(`Unknown binary selection group: ${selectionGroupId}`);
const allowedModelIds = selectionGroup ? new Set(selectionGroup.modelIds) : undefined;
const fixedModelIds = config.ensemble.selectionMode === "fixed_model_set" ? config.ensemble.requiredModelIds : undefined;
const requiredModelIds = fixedModelIds ?? selectionGroup?.modelIds;
const effectiveAllowedModelIds = requiredModelIds ? new Set(requiredModelIds) : allowedModelIds;
const manifestHashValue = isRawBinaryConfig(config)
  ? await rawBinaryStrictBlindManifestHash(config)
  : manifestHash(await readManifest(resolveOutputPath(config, "manifest", "split_manifest_seed42.csv")));
const scoresDir = resolveOutputPath(config, "scores");
const oofFiles = await listCsvFiles(scoresDir, "train_oof_");
if (oofFiles.length === 0) throw new Error("No train_oof_*.csv files found. Run training and audit first.");

const oofByModel = new Map();
const sourceHashes: Record<string, string> = {};
for (const path of oofFiles) {
  const rows = await readOofScores(path);
  const modelId = rows[0]?.modelId;
  if (!modelId) throw new Error(`Missing model_id in ${path}`);
  if (effectiveAllowedModelIds && !effectiveAllowedModelIds.has(modelId)) continue;
  oofByModel.set(modelId, rows);
  sourceHashes[modelId] = await sha256File(path);
}
if (oofByModel.size === 0) throw new Error(selectionGroupId ? `No OOF files found for group ${selectionGroupId}` : "No OOF files selected");
if (requiredModelIds) {
  for (const modelId of requiredModelIds) {
    if (!oofByModel.has(modelId)) throw new Error(`Missing required OOF model for fixed selection: ${modelId}`);
  }
}

const lock = selectEnsemble({
  seed: config.seed,
  predictionTarget: config.predictionTarget ?? "depression",
  manifestHash: manifestHashValue,
  oofByModel,
  sourceHashes,
  weightStep: config.ensemble.weightStep,
  command: selectionGroupId ? `make select-ensemble-setembrobr BINARY_SELECTION_GROUP_ID=${selectionGroupId}` : "make select-ensemble-setembrobr",
  exhaustiveModelLimit: config.ensemble.exhaustiveModelLimit,
  candidatePruneTo: config.ensemble.candidatePruneTo,
  maxModels: config.ensemble.maxModels,
  selectionMode: config.ensemble.selectionMode,
  requiredModelIds,
  minimumWeight: config.ensemble.minimumWeight,
});
const proxyDefinitionPath = config.relevanceProxy ? resolveOutputPath(config, "relevance-proxy", "proxy-definition.json") : undefined;
const enrichedLock = config.relevanceProxy && proxyDefinitionPath
  ? {
      ...lock,
      relevanceProxyProvenance: {
        kind: config.relevanceProxy.kind,
        definitionSha256: await sha256File(proxyDefinitionPath),
        usesLabels: false as const,
      },
      artifactHashes: {
        strictBlindManifestSha256: manifestHashValue,
        rawEmbeddingManifestSha256: config.rawEmbeddingManifestSha256 ?? "",
        rawSplitManifestSha256: config.rawSplitManifestSha256 ?? "",
        relevanceProxyDefinitionSha256: await sha256File(proxyDefinitionPath),
      },
    }
  : lock;
const outValue = selectionGroup
  ? { ...enrichedLock, selectionGroupId: selectionGroup.groupId, selectionGroupDescription: selectionGroup.description, candidateModelIds: selectionGroup.modelIds }
  : enrichedLock;
const outPath = resolveOutputPath(config, "ensemble", `${lockBasename}.json`);
await mkdir(dirname(outPath), { recursive: true });
await writeJson(outPath, outValue);
console.log(`wrote ${outPath}`);
console.log(JSON.stringify(lock.oofMetrics, null, 2));
