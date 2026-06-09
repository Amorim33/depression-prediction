import { readFile } from "node:fs/promises";
import { basename } from "node:path";
import { parseCsv } from "./csv.ts";
import { manifestHash } from "./manifest.ts";
import type { ManifestRow, ProjectConfig, ScoreRow } from "./types.ts";

export interface AuditFinding {
  ok: boolean;
  code: string;
  detail: string;
}

export interface AuditReport {
  ok: boolean;
  findings: AuditFinding[];
}

const TEST_FORBIDDEN_COLUMNS = new Set(["label", "actual", "predicted", "threshold", "tp", "fp", "tn", "fn", "macro_f1", "macroF1"]);

export function auditOofScores(manifestRows: readonly ManifestRow[], oofByModel: ReadonlyMap<string, readonly ScoreRow[]>): AuditReport {
  const findings: AuditFinding[] = [];
  const train = manifestRows.filter((row) => row.split === "train");
  const testUsers = new Set(manifestRows.filter((row) => row.split === "test").map((row) => row.userId));
  const trainByUser = new Map(train.map((row) => [row.userId, row]));

  for (const [modelId, rows] of oofByModel) {
    const seen = new Set<string>();
    for (const row of rows) {
      if (testUsers.has(row.userId)) {
        findings.push(fail("oof-test-user", `${modelId}: OOF row contains test user ${row.userId}`));
      }
      const manifest = trainByUser.get(row.userId);
      if (!manifest) findings.push(fail("oof-unknown-user", `${modelId}: OOF row not in train manifest ${row.userId}`));
      else {
        if (row.label !== manifest.label) findings.push(fail("oof-label-mismatch", `${modelId}: ${row.userId}`));
        if (row.fold !== manifest.fold) findings.push(fail("oof-fold-mismatch", `${modelId}: ${row.userId}`));
      }
      if (seen.has(row.userId)) findings.push(fail("oof-duplicate-user", `${modelId}: duplicate ${row.userId}`));
      seen.add(row.userId);
    }
    for (const row of train) {
      if (!seen.has(row.userId)) findings.push(fail("oof-missing-user", `${modelId}: missing ${row.userId}`));
    }
  }

  if (findings.length === 0) findings.push(pass("oof-integrity", "OOF rows match train manifest exactly"));
  return { ok: findings.every((finding) => finding.ok), findings };
}

export function auditTestScoreSchema(fileName: string, csvText: string): AuditReport {
  const [headerLine = ""] = csvText.split(/\r?\n/u);
  const headers = parseCsv(`${headerLine}\n`).length === 0 ? headerLine.split(",") : headerLine.split(",");
  const findings: AuditFinding[] = [];
  for (const header of headers) {
    if (TEST_FORBIDDEN_COLUMNS.has(header)) {
      findings.push(fail("test-score-forbidden-column", `${fileName}: forbidden column ${header}`));
    }
  }
  for (const required of ["user_id", "score", "model_id"]) {
    if (!headers.includes(required)) findings.push(fail("test-score-missing-column", `${fileName}: missing ${required}`));
  }
  if (findings.length === 0) findings.push(pass("test-score-schema", `${fileName}: label-free schema`));
  return { ok: findings.every((finding) => finding.ok), findings };
}

export async function auditStaticGuards(paths: readonly string[]): Promise<AuditReport> {
  const forbidden = [
    "sweepThreshold(test",
    'test_data["labels"]',
    "train_one(model, device, train_data, test_data",
    "train_one(...test_data",
    "test_*_scores",
  ];
  const findings: AuditFinding[] = [];
  for (const path of paths) {
    const text = await readFile(path, "utf8");
    for (const pattern of forbidden) {
      if (text.includes(pattern)) findings.push(fail("static-forbidden-pattern", `${basename(path)} contains ${pattern}`));
    }
  }
  if (findings.length === 0) findings.push(pass("static-guards", "No forbidden patterns in strict-blind scripts"));
  return { ok: findings.every((finding) => finding.ok), findings };
}

export async function auditModelManifest(path: string, config: ProjectConfig, manifestRows: readonly ManifestRow[]): Promise<AuditReport> {
  const manifest = JSON.parse(await readFile(path, "utf8")) as {
    manifestHash?: string;
    dbTables?: Record<string, string>;
    usesTestLabelsForTraining?: boolean;
  };
  const findings: AuditFinding[] = [];
  const expectedManifestHash = manifestHash(manifestRows);
  if (manifest.manifestHash !== expectedManifestHash) {
    findings.push(fail("model-manifest-hash", `${path}: manifest hash mismatch`));
  }
  if (manifest.usesTestLabelsForTraining) {
    findings.push(fail("model-manifest-test-labels", `${path}: declares test-label use`));
  }
  const allowedTables = new Set(Object.values(config.database.tables));
  for (const table of Object.values(manifest.dbTables ?? {})) {
    if (!allowedTables.has(table)) findings.push(fail("model-manifest-table", `${path}: non-config table ${table}`));
  }
  if (findings.length === 0) findings.push(pass("model-manifest", `${path}: manifest OK`));
  return { ok: findings.every((finding) => finding.ok), findings };
}

export function mergeReports(reports: readonly AuditReport[]): AuditReport {
  const findings = reports.flatMap((report) => report.findings);
  return { ok: findings.every((finding) => finding.ok), findings };
}

function pass(code: string, detail: string): AuditFinding {
  return { ok: true, code, detail };
}

function fail(code: string, detail: string): AuditFinding {
  return { ok: false, code, detail };
}

