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
  if (allowedModelIds && !allowedModelIds.has(modelId)) continue;
  oofByModel.set(modelId, rows);
  sourceHashes[modelId] = await sha256File(path);
}
if (oofByModel.size === 0) throw new Error(selectionGroupId ? `No OOF files found for group ${selectionGroupId}` : "No OOF files selected");

const lock = selectEnsemble({
  seed: config.seed,
  manifestHash: manifestHashValue,
  oofByModel,
  sourceHashes,
  weightStep: config.ensemble.weightStep,
  command: selectionGroupId ? `make select-ensemble-setembrobr BINARY_SELECTION_GROUP_ID=${selectionGroupId}` : "make select-ensemble-setembrobr",
  exhaustiveModelLimit: config.ensemble.exhaustiveModelLimit,
  candidatePruneTo: config.ensemble.candidatePruneTo,
  maxModels: config.ensemble.maxModels,
});
const outValue = selectionGroup
  ? { ...lock, selectionGroupId: selectionGroup.groupId, selectionGroupDescription: selectionGroup.description, candidateModelIds: selectionGroup.modelIds }
  : lock;
const outPath = resolveOutputPath(config, "ensemble", `${lockBasename}.json`);
await mkdir(dirname(outPath), { recursive: true });
await writeJson(outPath, outValue);
console.log(`wrote ${outPath}`);
console.log(JSON.stringify(lock.oofMetrics, null, 2));
