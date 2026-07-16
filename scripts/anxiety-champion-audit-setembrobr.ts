import { existsSync } from "node:fs";
import { readdir, readFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { auditOofScores, auditTestScoreSchema, mergeReports, type AuditFinding, type AuditReport } from "../src/audit.ts";
import { validateNestedCrossFitRecords, type NestedCrossFitRecord } from "../src/anxiety.ts";
import { listCsvFiles, readOofScores, writeJson } from "../src/artifacts.ts";
import { loadConfig, resolveOutputPath } from "../src/config.ts";
import { parseCsv } from "../src/csv.ts";
import { sha256File } from "../src/hash.ts";
import { readRawBinaryAuditManifest, readRawBinaryTestUsers, rawBinaryStrictBlindManifestHash } from "../src/raw-binary.ts";
import type { EnsembleLock } from "../src/types.ts";

const mode = process.argv.includes("--mode=test") ? "test" : "oof";
const config = await loadConfig();
if ((config.predictionTarget ?? "depression") !== "anxiety") throw new Error("anxiety audit requires predictionTarget=anxiety");
if (!config.relevanceProxy || !config.expectedUsers) throw new Error("anxiety audit config is incomplete");
const relevanceProxy = config.relevanceProxy;
const expectedUserCounts = config.expectedUsers;

if (mode === "oof") await auditOof();
else await auditTest();

async function auditOof(): Promise<void> {
  const scoresDir = resolveOutputPath(config, "scores");
  const expectedModels = new Set(config.models);
  const findings: AuditFinding[] = [];
  const manifestRows = await readRawBinaryAuditManifest(config);
  const strictBlindHash = await rawBinaryStrictBlindManifestHash(config);
  const oofFiles = await listCsvFiles(scoresDir, "train_oof_");
  const actualModels = new Set<string>();
  const oofByModel = new Map();
  for (const path of oofFiles) {
    const rows = await readOofScores(path);
    const modelId = rows[0]?.modelId;
    if (!modelId) throw new Error(`OOF file has no model_id: ${path}`);
    if (rows.some((row) => row.modelId !== modelId)) findings.push(fail("mixed-model-score-file", path));
    actualModels.add(modelId);
    oofByModel.set(modelId, rows);
  }
  compareSets(expectedModels, actualModels, "oof-model-set", findings);
  if (oofFiles.length !== expectedModels.size) findings.push(fail("oof-file-count", `expected ${expectedModels.size}, got ${oofFiles.length}`));
  const trainRows = manifestRows.filter((row) => row.split === "train");
  if (trainRows.length !== expectedUserCounts.train) {
    findings.push(fail("train-user-count", `expected ${expectedUserCounts.train}, got ${trainRows.length}`));
  }

  const forbidden = [
    ...(await listCsvFiles(scoresDir, "test_score_")),
    resolveOutputPath(config, "work", "features", "test_raw_features.npz"),
    resolveOutputPath(config, "work", "sequences", "top128", "test_seq.npz"),
    resolveOutputPath(config, "relevance-proxy", "sidecars", "test"),
    resolveOutputPath(config, "relevance-proxy", "test-proxy-manifest.json"),
    resolveOutputPath(config, "sealed"),
    resolveOutputPath(config, "reports", "final-test-report.json"),
    resolveOutputPath(config, "reports", "label-free-test-score-manifest.json"),
  ];
  for (const path of forbidden) {
    if (existsSync(path)) findings.push(fail("oof-created-test-artifact", path));
  }

  const proxyDefinitionPath = resolveOutputPath(config, "relevance-proxy", "proxy-definition.json");
  const proxyDefinition = JSON.parse(await readFile(proxyDefinitionPath, "utf8")) as Record<string, unknown>;
  if (proxyDefinition.kind !== relevanceProxy.kind || proxyDefinition.usesTrainLabels !== false || proxyDefinition.usesTestLabels !== false) {
    findings.push(fail("proxy-label-independence", proxyDefinitionPath));
  }
  if (proxyDefinition.rawEmbeddingManifestSha256 !== config.rawEmbeddingManifestSha256) {
    findings.push(fail("proxy-raw-manifest-hash", proxyDefinitionPath));
  }
  const trainProxyManifest = JSON.parse(
    await readFile(resolveOutputPath(config, "relevance-proxy", "train-proxy-manifest.json"), "utf8"),
  ) as {
    expectedUsers?: number;
    pooledHashes?: Record<string, string>;
    shards?: Array<{
      rowCount?: number;
      sourceSha256?: string;
      sidecar?: string;
      sidecarSha256?: string;
      alignmentSha256?: string;
    }>;
  };
  if (trainProxyManifest.expectedUsers !== expectedUserCounts.train || !trainProxyManifest.shards?.length) {
    findings.push(fail("proxy-train-manifest", "missing shards or wrong user count"));
  }
  for (const shard of trainProxyManifest.shards ?? []) {
    if (!shard.rowCount || !shard.sourceSha256 || !shard.sidecar || !shard.sidecarSha256 || !shard.alignmentSha256) {
      findings.push(fail("proxy-shard-provenance", JSON.stringify(shard)));
      continue;
    }
    const sidecarPath = resolveOutputPath(config, "relevance-proxy", shard.sidecar);
    if (!existsSync(sidecarPath) || await sha256File(sidecarPath) !== shard.sidecarSha256) {
      findings.push(fail("proxy-sidecar-hash", sidecarPath));
    }
  }
  for (const threshold of relevanceProxy.poolThresholds) {
    const path = resolveOutputPath(config, "relevance-proxy", `train-rel${threshold}-pool-manifest.json`);
    const pool = JSON.parse(await readFile(path, "utf8")) as { pooledSha256?: string; zeroVectorFallback?: boolean };
    if (pool.zeroVectorFallback !== true || pool.pooledSha256 !== trainProxyManifest.pooledHashes?.[`rel${threshold}`]) {
      findings.push(fail("proxy-pool-manifest", path));
    }
  }

  const manifestEntries = await readdir(resolveOutputPath(config, "model-manifests"), { withFileTypes: true }).catch(() => []);
  const manifestModels = new Set<string>();
  for (const entry of manifestEntries) {
    if (!entry.isFile() || !entry.name.endsWith(".json")) continue;
    const path = join(resolveOutputPath(config, "model-manifests"), entry.name);
    const model = JSON.parse(await readFile(path, "utf8")) as {
      modelId?: string;
      predictionTarget?: string;
      manifestHash?: string;
      relevanceProxyKind?: string;
      relevanceProxyDefinitionHash?: string;
      scoreSchema?: string;
      usesTestLabelsForTraining?: boolean;
      usesTestScoresForTraining?: boolean;
      testArtifactsReadDuringOof?: boolean;
      checkpointHashes?: Record<string, string>;
      artifactHashes?: Record<string, string>;
      nestedCrossFitting?: boolean;
      family?: string;
    };
    if (model.modelId) manifestModels.add(model.modelId);
    if (
      model.predictionTarget !== "anxiety" ||
      model.manifestHash !== strictBlindHash ||
      model.relevanceProxyKind !== relevanceProxy.kind ||
      model.relevanceProxyDefinitionHash !== (await sha256File(proxyDefinitionPath)) ||
      model.scoreSchema !== "binary-score-v1" ||
      model.usesTestLabelsForTraining !== false ||
      model.usesTestScoresForTraining !== false ||
      model.testArtifactsReadDuringOof !== false ||
      Object.keys(model.checkpointHashes ?? {}).length !== config.foldCount ||
      !model.artifactHashes ||
      Object.values(model.artifactHashes).some((hash) => !/^[a-f0-9]{64}$/u.test(hash))
    ) {
      findings.push(fail("anxiety-model-manifest", path));
    }
    for (const [fold, expectedHash] of Object.entries(model.checkpointHashes ?? {})) {
      const checkpointPath = model.modelId?.startsWith("binary_stack_")
        ? resolveOutputPath(config, "checkpoints", "stacking", model.modelId, `outer-${fold}.joblib`)
        : model.family === "cnn"
          ? resolveOutputPath(config, "checkpoints", "sequence", model.modelId ?? "", `fold-${fold}.pt`)
          : resolveOutputPath(config, "checkpoints", "tabular", model.modelId ?? "", `fold-${fold}.joblib`);
      if (!existsSync(checkpointPath) || await sha256File(checkpointPath) !== expectedHash) {
        findings.push(fail("checkpoint-hash", checkpointPath));
      }
    }
    if (model.modelId?.startsWith("binary_stack_") && model.nestedCrossFitting !== true) {
      findings.push(fail("nested-stacking-flag", path));
    }
  }
  compareSets(expectedModels, manifestModels, "model-manifest-set", findings);

  const nested = JSON.parse(
    await readFile(resolveOutputPath(config, "reports", "nested-stacking-provenance.json"), "utf8"),
  ) as {
    allOuterFoldsExcluded?: boolean;
    allInnerValidationFoldsExcluded?: boolean;
    records?: Array<NestedCrossFitRecord & {
      baseModelId: string;
      checkpointSha256: string;
    }>;
  };
  if (
    !nested.allOuterFoldsExcluded ||
    !nested.allInnerValidationFoldsExcluded ||
    !validateNestedCrossFitRecords(nested.records ?? [])
  ) {
    findings.push(fail("nested-stacking-exclusions", "outer/inner validation folds were not proven excluded"));
  }
  for (const record of nested.records ?? []) {
    const checkpointPath = resolveOutputPath(
      config,
      "checkpoints",
      "nested-stack-bases",
      record.baseModelId,
      `outer-${record.outerFold}`,
      `inner-${record.innerValidationFold}.joblib`,
    );
    if (!existsSync(checkpointPath) || await sha256File(checkpointPath) !== record.checkpointSha256) {
      findings.push(fail("nested-checkpoint-hash", checkpointPath));
    }
  }
  if (findings.length === 0) {
    findings.push({
      ok: true,
      code: "anxiety-strict-blind-provenance",
      detail: "model sets, test-artifact absence, proxy sidecars/pools, checkpoints, and nested exclusions passed",
    });
  }

  const report = mergeReports([auditOofScores(manifestRows, oofByModel), { ok: findings.every((finding) => finding.ok), findings }]);
  const payload = {
    ...report,
    predictionTarget: "anxiety",
    strictBlindManifestSha256: strictBlindHash,
    rawEmbeddingManifestSha256: config.rawEmbeddingManifestSha256,
    rawSplitManifestSha256: config.rawSplitManifestSha256,
    relevanceProxyDefinitionSha256: await sha256File(proxyDefinitionPath),
    trainUsers: trainRows.length,
    modelIds: [...actualModels].sort(),
  };
  await writeJson(resolveOutputPath(config, "reports", "oof-audit.json"), payload);
  printReport(report);
}

async function auditTest(): Promise<void> {
  const findings: AuditFinding[] = [];
  const lockPath = resolveOutputPath(config, "ensemble", "ensemble-lock.json");
  const auditPath = resolveOutputPath(config, "reports", "oof-audit.json");
  const lock = JSON.parse(await readFile(lockPath, "utf8")) as EnsembleLock & {
    relevanceProxyProvenance?: { kind?: string; definitionSha256?: string };
  };
  const oofAudit = JSON.parse(await readFile(auditPath, "utf8")) as { ok?: boolean; strictBlindManifestSha256?: string };
  if (!oofAudit.ok) findings.push(fail("oof-audit-required", auditPath));
  const required = config.ensemble.requiredModelIds ?? [];
  compareSets(new Set(required), new Set(lock.modelIds), "locked-model-set", findings);
  if (lock.predictionTarget !== "anxiety" || lock.manifestHash !== oofAudit.strictBlindManifestSha256) {
    findings.push(fail("lock-provenance", lockPath));
  }
  if (lock.relevanceProxyProvenance?.kind !== relevanceProxy.kind) findings.push(fail("lock-proxy", lockPath));
  const sum = Object.values(lock.weights).reduce((total, weight) => total + weight, 0);
  for (const modelId of required) {
    const weight = lock.weights[modelId] ?? 0;
    const units = weight / config.ensemble.weightStep;
    if (weight < (config.ensemble.minimumWeight ?? config.ensemble.weightStep) || Math.abs(units - Math.round(units)) > 1e-9) {
      findings.push(fail("lock-weight", `${modelId}: ${weight}`));
    }
  }
  if (Math.abs(sum - 1) > 1e-9) findings.push(fail("lock-weight-sum", String(sum)));

  const testUsers = await readRawBinaryTestUsers(config);
  const expectedUsers = new Set(testUsers);
  const sourceHashes: Record<string, string> = {};
  const scoreReports: AuditReport[] = [];
  const actualModels = new Set<string>();
  for (const path of await listCsvFiles(resolveOutputPath(config, "scores"), "test_score_")) {
    const text = await readFile(path, "utf8");
    scoreReports.push(auditTestScoreSchema(path, text));
    const [header = ""] = text.split(/\r?\n/u);
    if (header !== "user_id,score,model_id") findings.push(fail("test-score-exact-schema", `${basename(path)}: ${header}`));
    const rows = parseCsv(text);
    const modelId = rows[0]?.model_id;
    if (!modelId) throw new Error(`test score has no model_id: ${path}`);
    actualModels.add(modelId);
    const rowUsers = rows.map((row) => row.user_id ?? "");
    if (rows.length !== expectedUserCounts.test || new Set(rowUsers).size !== rows.length) {
      findings.push(fail("test-score-row-count", `${modelId}: ${rows.length}`));
    }
    if (rows.some((row) => row.model_id !== modelId) || !setsEqual(expectedUsers, new Set(rowUsers))) {
      findings.push(fail("test-score-user-set", modelId));
    }
    sourceHashes[modelId] = await sha256File(path);
  }
  compareSets(new Set(config.models), actualModels, "test-score-model-set", findings);
  const report = mergeReports([...scoreReports, { ok: findings.every((finding) => finding.ok), findings }]);
  if (report.ok) {
    await writeJson(resolveOutputPath(config, "reports", "label-free-test-score-manifest.json"), {
      ok: true,
      predictionTarget: "anxiety",
      lockSha256: await sha256File(lockPath),
      oofAuditSha256: await sha256File(auditPath),
      testInferenceManifestSha256: await sha256File(
        resolveOutputPath(config, "manifest", `test_inference_manifest_seed${config.seed}.csv`),
      ),
      testUsers: testUsers.length,
      modelIds: [...actualModels].sort(),
      sourceHashes,
      labelsPresent: false,
    });
  }
  printReport(report);
}

function compareSets(expected: Set<string>, actual: Set<string>, code: string, findings: AuditFinding[]): void {
  if (!setsEqual(expected, actual)) {
    findings.push(fail(code, `expected=${JSON.stringify([...expected].sort())} actual=${JSON.stringify([...actual].sort())}`));
  }
}

function setsEqual(left: Set<string>, right: Set<string>): boolean {
  return left.size === right.size && [...left].every((value) => right.has(value));
}

function fail(code: string, detail: string): AuditFinding {
  return { ok: false, code, detail };
}

function printReport(report: AuditReport): void {
  for (const finding of report.findings) console.log(`${finding.ok ? "ok" : "FAIL"} ${finding.code} ${finding.detail}`);
  if (!report.ok) process.exit(1);
}
