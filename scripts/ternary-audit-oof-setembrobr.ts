import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { auditStaticGuards, mergeReports, type AuditReport } from "../src/audit.ts";
import { listCsvFiles } from "../src/artifacts.ts";
import { manifestHash, readManifest } from "../src/manifest.ts";
import { loadTernaryConfig, resolveSourceOutputPath, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import {
  auditTernaryOofScores,
  auditTernaryTestScoreSchema,
  readTernaryManifest,
  readTernaryOofScores,
} from "../src/ternary.ts";
import type { TernaryLabelPolicyLock } from "../src/types.ts";

const config = await loadTernaryConfig();
const sourceRows = await readManifest(resolveSourceOutputPath(config, "manifest", `split_manifest_seed${config.seed}.csv`));
const sourceManifestHash = manifestHash(sourceRows);
const testUsers = new Set(sourceRows.filter((row) => row.split === "test").map((row) => row.userId));
const reports: AuditReport[] = [];
const policyLocks = new Map<string, TernaryLabelPolicyLock>();

for (const policy of config.labelPolicies) {
  const policyLockPath = resolveTernaryOutputPath(config, "label-policies", `${policy.policyId}.json`);
  const trainManifestPath = resolveTernaryOutputPath(config, "manifest", `train_manifest_${policy.policyId}_seed${config.seed}.csv`);
  const policyLockText = await readFile(policyLockPath, "utf8").catch(() => null);
  if (!policyLockText) {
    reports.push(singleFinding(false, "ternary-policy-lock-missing", `${policy.policyId}: missing ${policyLockPath}`));
    continue;
  }
  const policyLock = JSON.parse(policyLockText) as TernaryLabelPolicyLock;
  policyLocks.set(policy.policyId, policyLock);
  const trainManifest = await readTernaryManifest(trainManifestPath).catch(() => null);
  if (!trainManifest) {
    reports.push(singleFinding(false, "ternary-train-manifest-missing", `${policy.policyId}: missing ${trainManifestPath}`));
    continue;
  }
  const oofByModel = new Map();
  for (const path of await listCsvFiles(resolveTernaryOutputPath(config, "scores"), `train_oof_${policy.policyId}_`)) {
    const rows = await readTernaryOofScores(path);
    const modelId = rows[0]?.modelId;
    if (!modelId) throw new Error(`OOF file has no rows/model_id: ${path}`);
    oofByModel.set(modelId, rows);
  }
  if (oofByModel.size > 0) reports.push(auditTernaryOofScores(trainManifest, testUsers, oofByModel, policy.policyId));
}

for (const path of await listCsvFiles(resolveTernaryOutputPath(config, "scores"), "test_score_")) {
  reports.push(auditTernaryTestScoreSchema(path, await readFile(path, "utf8")));
}

reports.push(await auditTernaryModelManifests(sourceManifestHash, policyLocks));
reports.push(
  await auditStaticGuards([
    "scripts/ternary_train_tabular_oof_setembrobr.py",
    "scripts/ternary_train_seq_oof_setembrobr.py",
    "scripts/ternary-select-ensemble-setembrobr.ts",
    "scripts/ternary-robustness-setembrobr.ts",
    "scripts/ternary-nested-oof-selection-setembrobr.ts",
  ]),
);
reports.push(
  await auditExtraStaticGuards([
    "scripts/ternary_train_tabular_oof_setembrobr.py",
    "scripts/ternary_train_seq_oof_setembrobr.py",
    "scripts/ternary-select-ensemble-setembrobr.ts",
    "scripts/ternary-robustness-setembrobr.ts",
    "scripts/ternary-nested-oof-selection-setembrobr.ts",
  ]),
);

const report = mergeReports(reports);
for (const finding of report.findings) {
  console.log(`${finding.ok ? "ok" : "FAIL"} ${finding.code} ${finding.detail}`);
}
if (!report.ok) process.exit(1);

async function auditTernaryModelManifests(
  expectedOriginalManifestHash: string,
  locks: ReadonlyMap<string, TernaryLabelPolicyLock>,
): Promise<AuditReport> {
  const findings = [];
  const modelManifestDir = resolveTernaryOutputPath(config, "model-manifests");
  const entries = await readdir(modelManifestDir, { withFileTypes: true }).catch(() => []);
  const allowedTables = new Set(Object.values(config.database.tables));
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".json")) continue;
    const path = join(modelManifestDir, entry.name);
    const manifest = JSON.parse(await readFile(path, "utf8")) as {
      originalManifestHash?: string;
      labelPolicyId?: string;
      labelPolicyHash?: string;
      dbTables?: Record<string, string>;
      usesTestLabelsForTraining?: boolean;
    };
    if (manifest.originalManifestHash !== expectedOriginalManifestHash) {
      findings.push(fail("ternary-model-original-manifest-hash", `${path}: original manifest hash mismatch`));
    }
    const lock = manifest.labelPolicyId ? locks.get(manifest.labelPolicyId) : undefined;
    if (!lock) findings.push(fail("ternary-model-policy", `${path}: unknown label policy ${manifest.labelPolicyId ?? "missing"}`));
    else if (manifest.labelPolicyHash !== lock.policyHash) {
      findings.push(fail("ternary-model-policy-hash", `${path}: label policy hash mismatch`));
    }
    if (manifest.usesTestLabelsForTraining) {
      findings.push(fail("ternary-model-test-labels", `${path}: declares test-label use`));
    }
    for (const table of Object.values(manifest.dbTables ?? {})) {
      if (!allowedTables.has(table)) findings.push(fail("ternary-model-table", `${path}: non-config table ${table}`));
    }
  }
  if (findings.length === 0) findings.push(pass("ternary-model-manifest", "Ternary model manifests OK"));
  return { ok: findings.every((finding) => finding.ok), findings };
}

async function auditExtraStaticGuards(paths: readonly string[]): Promise<AuditReport> {
  const forbidden = [
    'split_test["labels"]',
    'test_split["labels"]',
    "testLabels",
    "test_labels",
    "evaluateTernaryLockedEnsemble(",
  ];
  const findings = [];
  for (const path of paths) {
    const text = await readFile(path, "utf8");
    for (const pattern of forbidden) {
      if (text.includes(pattern)) findings.push(fail("ternary-static-forbidden-pattern", `${path} contains ${pattern}`));
    }
  }
  if (findings.length === 0) findings.push(pass("ternary-static-guards", "No forbidden ternary strict-blind patterns"));
  return { ok: findings.every((finding) => finding.ok), findings };
}

function pass(code: string, detail: string) {
  return { ok: true, code, detail };
}

function fail(code: string, detail: string) {
  return { ok: false, code, detail };
}

function singleFinding(ok: boolean, code: string, detail: string): AuditReport {
  return { ok, findings: [{ ok, code, detail }] };
}
