import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { auditModelManifest, auditOofScores, auditStaticGuards, auditTestScoreSchema, mergeReports } from "../src/audit.ts";
import { listCsvFiles, readOofScores } from "../src/artifacts.ts";
import { loadConfig, resolveOutputPath } from "../src/config.ts";
import { readManifest } from "../src/manifest.ts";

const config = await loadConfig();
const manifestPath = resolveOutputPath(config, "manifest", "split_manifest_seed42.csv");
const scoresDir = resolveOutputPath(config, "scores");
const modelManifestDir = resolveOutputPath(config, "model-manifests");
const manifestRows = await readManifest(manifestPath);

const oofFiles = await listCsvFiles(scoresDir, "train_oof_");
const oofByModel = new Map();
for (const path of oofFiles) {
  const rows = await readOofScores(path);
  const modelId = rows[0]?.modelId;
  if (!modelId) throw new Error(`OOF file has no rows/model_id: ${path}`);
  oofByModel.set(modelId, rows);
}

const testScoreReports = [];
for (const path of await listCsvFiles(scoresDir, "test_score_")) {
  testScoreReports.push(auditTestScoreSchema(path, await readFile(path, "utf8")));
}

const modelManifestReports = [];
const modelManifestEntries = await readdir(modelManifestDir, { withFileTypes: true }).catch(() => []);
for (const entry of modelManifestEntries) {
  if (entry.isFile() && entry.name.endsWith(".json")) {
    modelManifestReports.push(await auditModelManifest(join(modelManifestDir, entry.name), config, manifestRows));
  }
}

const staticReport = await auditStaticGuards([
  "scripts/train_tabular_oof_setembrobr.py",
  "scripts/train_seq_oof_setembrobr.py",
  "scripts/select-ensemble-setembrobr.ts",
]);

const report = mergeReports([
  auditOofScores(manifestRows, oofByModel),
  ...testScoreReports,
  ...modelManifestReports,
  staticReport,
]);

for (const finding of report.findings) {
  console.log(`${finding.ok ? "ok" : "FAIL"} ${finding.code} ${finding.detail}`);
}
if (!report.ok) process.exit(1);

