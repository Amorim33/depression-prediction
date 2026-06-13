import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { sha256File } from "../src/hash.ts";
import { listCsvFiles, readOofScores, writeJson } from "../src/artifacts.ts";
import { loadConfig, resolveOutputPath } from "../src/config.ts";
import { readManifest, manifestHash } from "../src/manifest.ts";
import { selectEnsemble } from "../src/ensemble.ts";

const config = await loadConfig();
const manifestRows = await readManifest(resolveOutputPath(config, "manifest", "split_manifest_seed42.csv"));
const scoresDir = resolveOutputPath(config, "scores");
const oofFiles = await listCsvFiles(scoresDir, "train_oof_");
if (oofFiles.length === 0) throw new Error("No train_oof_*.csv files found. Run training and audit first.");

const oofByModel = new Map();
const sourceHashes: Record<string, string> = {};
for (const path of oofFiles) {
  const rows = await readOofScores(path);
  const modelId = rows[0]?.modelId;
  if (!modelId) throw new Error(`Missing model_id in ${path}`);
  oofByModel.set(modelId, rows);
  sourceHashes[modelId] = await sha256File(path);
}

const lock = selectEnsemble({
  seed: config.seed,
  manifestHash: manifestHash(manifestRows),
  oofByModel,
  sourceHashes,
  weightStep: config.ensemble.weightStep,
  command: "make select-ensemble-setembrobr",
  exhaustiveModelLimit: config.ensemble.exhaustiveModelLimit,
  candidatePruneTo: config.ensemble.candidatePruneTo,
  maxModels: config.ensemble.maxModels,
});
const outPath = resolveOutputPath(config, "ensemble", "ensemble-lock.json");
await mkdir(dirname(outPath), { recursive: true });
await writeJson(outPath, lock);
console.log(`wrote ${outPath}`);
console.log(JSON.stringify(lock.oofMetrics, null, 2));
