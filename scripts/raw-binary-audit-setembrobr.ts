import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { auditOofScores, auditStaticGuards, auditTestScoreSchema, mergeReports, type AuditReport } from "../src/audit.ts";
import { listCsvFiles, readOofScores } from "../src/artifacts.ts";
import { loadConfig, resolveOutputPath } from "../src/config.ts";
import { readRawBinaryAuditManifest, rawBinaryStrictBlindManifestHash } from "../src/raw-binary.ts";

const config = await loadConfig();
const scoresDir = resolveOutputPath(config, "scores");
const modelManifestDir = resolveOutputPath(config, "model-manifests");
const manifestRows = await readRawBinaryAuditManifest(config);
const strictBlindHash = await rawBinaryStrictBlindManifestHash(config);

const oofFiles = await listCsvFiles(scoresDir, "train_oof_");
if (oofFiles.length === 0) throw new Error("No raw binary train_oof_*.csv files found. Run training first.");
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
if (!modelManifestEntries.some((entry) => entry.isFile() && entry.name.endsWith(".json"))) {
  throw new Error("No raw binary model manifests found. Run training first.");
}
for (const entry of modelManifestEntries) {
  if (entry.isFile() && entry.name.endsWith(".json")) {
    modelManifestReports.push(await auditRawBinaryModelManifest(join(modelManifestDir, entry.name), strictBlindHash));
  }
}

const staticReport = await auditStaticGuards([
  "scripts/binary_train_tabular_oof_setembrobr.py",
  "scripts/binary_train_seq_oof_setembrobr.py",
  "scripts/binary_stack_oof_setembrobr.py",
  "scripts/select-ensemble-setembrobr.ts",
  "scripts/binary-llm-disambiguator-setembrobr.ts",
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

async function auditRawBinaryModelManifest(path: string, expectedManifestHash: string): Promise<AuditReport> {
  const manifest = JSON.parse(await readFile(path, "utf8")) as {
    manifestHash?: string;
    featureSource?: string;
    scoreSchema?: string;
    usesTestLabelsForTraining?: boolean;
    usesTestScoresForTraining?: boolean;
  };
  const findings = [];
  if (manifest.manifestHash !== expectedManifestHash) {
    findings.push({ ok: false, code: "raw-binary-model-manifest-hash", detail: `${path}: manifest hash mismatch` });
  }
  if (manifest.featureSource !== "raw_artifacts") {
    findings.push({ ok: false, code: "raw-binary-model-feature-source", detail: `${path}: expected raw_artifacts` });
  }
  if (manifest.scoreSchema !== "binary-score-v1") {
    findings.push({ ok: false, code: "raw-binary-model-score-schema", detail: `${path}: expected binary-score-v1` });
  }
  if (manifest.usesTestLabelsForTraining) {
    findings.push({ ok: false, code: "raw-binary-model-test-labels", detail: `${path}: declares test-label use` });
  }
  if (manifest.usesTestScoresForTraining) {
    findings.push({ ok: false, code: "raw-binary-model-test-scores", detail: `${path}: declares test-score use` });
  }
  if (findings.length === 0) findings.push({ ok: true, code: "raw-binary-model-manifest", detail: `${path}: manifest OK` });
  return { ok: findings.every((finding) => finding.ok), findings };
}
