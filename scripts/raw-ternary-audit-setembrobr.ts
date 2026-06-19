import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { mergeReports, type AuditReport } from "../src/audit.ts";
import { listCsvFiles } from "../src/artifacts.ts";
import { parseCsv } from "../src/csv.ts";
import { sha256Text } from "../src/hash.ts";
import { auditLlmTestDecisionSchema } from "../src/llm-disambiguator.ts";
import { isRawTernaryConfig, rawManifestPath } from "../src/raw-ternary.ts";
import { loadTernaryConfig, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import { auditTernaryTestScoreSchema } from "../src/ternary.ts";

const config = await loadTernaryConfig();
if (!isRawTernaryConfig(config)) throw new Error("raw-ternary-audit-setembrobr requires featureSource=raw_artifacts");

const reports: AuditReport[] = [];
reports.push(await auditRedactedTestManifest());
reports.push(await auditRawModelManifests());
reports.push(await auditNoSealedLabelsInPreLockScripts());

for (const path of await listCsvFiles(resolveTernaryOutputPath(config, "scores"), "test_score_")) {
  reports.push(auditTernaryTestScoreSchema(path, await readFile(path, "utf8")));
}
for (const path of await listCsvFiles(resolveTernaryOutputPath(config, "llm-disambiguator"), "test_decisions_")) {
  reports.push(auditLlmTestDecisionSchema(path, await readFile(path, "utf8")));
}

const report = mergeReports(reports);
for (const finding of report.findings) {
  console.log(`${finding.ok ? "ok" : "FAIL"} ${finding.code} ${finding.detail}`);
}
if (!report.ok) process.exit(1);

async function auditRedactedTestManifest(): Promise<AuditReport> {
  const findings = [];
  const path = rawManifestPath(config, `test_inference_manifest_seed${config.seed}.csv`);
  const rows = parseCsv(await readFile(path, "utf8"));
  if (rows.length !== 2696) findings.push(fail("raw-test-user-count", `${path}: expected 2696 test users, got ${rows.length}`));
  for (const [index, row] of rows.entries()) {
    if (row.label !== "-1") findings.push(fail("raw-test-label-redaction", `${path}: row ${index + 2} label is not -1`));
    if ((row.fold ?? "") !== "") findings.push(fail("raw-test-fold-redaction", `${path}: row ${index + 2} fold is not blank`));
    if (
      row.row_hash === sha256Text(`setembrobr|raw-qwen3|test|0|${row.user_id}`) ||
      row.row_hash === sha256Text(`setembrobr|raw-qwen3|test|1|${row.user_id}`)
    ) {
      findings.push(fail("raw-test-row-hash-label-leak", `${path}: row ${index + 2} row_hash is label-derived`));
    }
  }
  if (findings.length === 0) findings.push(pass("raw-test-manifest-redacted", "Raw test inference manifest is label/fold redacted"));
  return { ok: findings.every((finding) => finding.ok), findings };
}

async function auditRawModelManifests(): Promise<AuditReport> {
  const findings = [];
  const dir = resolveTernaryOutputPath(config, "model-manifests");
  const entries = await readdir(dir, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".json")) continue;
    const path = join(dir, entry.name);
    const manifest = JSON.parse(await readFile(path, "utf8")) as {
      featureSource?: string;
      dbTables?: Record<string, string>;
      usesTestLabelsForTraining?: boolean;
      usesTestScoresForTraining?: boolean;
    };
    if (manifest.featureSource !== "raw_artifacts") {
      findings.push(fail("raw-model-feature-source", `${path}: featureSource must be raw_artifacts`));
    }
    if (Object.keys(manifest.dbTables ?? {}).length > 0) {
      findings.push(fail("raw-model-db-tables", `${path}: raw model manifest must not declare dbTables`));
    }
    if (manifest.usesTestLabelsForTraining) {
      findings.push(fail("raw-model-test-labels", `${path}: declares test-label use`));
    }
    if (manifest.usesTestScoresForTraining) {
      findings.push(fail("raw-model-test-scores", `${path}: declares test-score training use`));
    }
  }
  if (findings.length === 0) findings.push(pass("raw-model-manifests", "Raw model manifests do not declare DB or test-label training"));
  return { ok: findings.every((finding) => finding.ok), findings };
}

async function auditNoSealedLabelsInPreLockScripts(): Promise<AuditReport> {
  const findings = [];
  const paths = [
    "scripts/ternary-manifest-setembrobr.ts",
    "scripts/ternary_train_tabular_oof_setembrobr.py",
    "scripts/ternary_train_seq_oof_setembrobr.py",
    "scripts/ternary_stack_oof_setembrobr.py",
    "scripts/ternary-select-ensemble-setembrobr.ts",
    "scripts/ternary-robustness-setembrobr.ts",
    "scripts/ternary-nested-oof-selection-setembrobr.ts",
    "scripts/ternary-oof-diagnostics-setembrobr.ts",
    "scripts/ternary-model-policy-leaderboard-setembrobr.ts",
    "scripts/ternary-family-ablation-setembrobr.ts",
    "scripts/ternary-llm-disambiguator-setembrobr.ts",
  ];
  for (const path of paths) {
    const text = await readFile(path, "utf8");
    if (text.includes("sealed_test_labels") || text.includes("readRawSealedTestLabels")) {
      findings.push(fail("raw-static-sealed-label-reference", `${path}: pre-lock script references sealed test labels`));
    }
  }
  if (findings.length === 0) findings.push(pass("raw-static-sealed-labels", "Pre-lock scripts do not reference sealed test labels"));
  return { ok: findings.every((finding) => finding.ok), findings };
}

function pass(code: string, detail: string) {
  return { ok: true, code, detail };
}

function fail(code: string, detail: string) {
  return { ok: false, code, detail };
}
