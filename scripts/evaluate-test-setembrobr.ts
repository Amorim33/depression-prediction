import { readFile } from "node:fs/promises";
import { listCsvFiles, readTestScores, writeJson } from "../src/artifacts.ts";
import { loadConfig, resolveOutputPath } from "../src/config.ts";
import { evaluateLockedEnsemble } from "../src/ensemble.ts";
import { readManifest } from "../src/manifest.ts";
import { isRawBinaryConfig, readRawBinarySealedTestLabels } from "../src/raw-binary.ts";
import type { EnsembleLock } from "../src/types.ts";

const config = await loadConfig();
const lockBasename = process.env.BINARY_LOCK_BASENAME?.trim() || "ensemble-lock";
const reportBasename = process.env.BINARY_REPORT_BASENAME?.trim() || (lockBasename === "ensemble-lock" ? "final-test-report" : `${lockBasename}-final-test-report`);
const lock = JSON.parse(await readFile(resolveOutputPath(config, "ensemble", `${lockBasename}.json`), "utf8")) as EnsembleLock;
const labelsByUser = isRawBinaryConfig(config)
  ? new Map((await readRawBinarySealedTestLabels(config)).map((row) => [row.userId, row.label]))
  : new Map((await readManifest(resolveOutputPath(config, "manifest", "split_manifest_seed42.csv"))).filter((row) => row.split === "test").map((row) => [row.userId, row.label]));

const testScoresByModel = new Map();
for (const path of await listCsvFiles(resolveOutputPath(config, "scores"), "test_score_")) {
  const rows = await readTestScores(path);
  const modelId = rows[0]?.modelId;
  if (modelId && lock.modelIds.includes(modelId)) testScoresByModel.set(modelId, rows);
}
for (const modelId of lock.modelIds) {
  if (!testScoresByModel.has(modelId)) throw new Error(`Missing label-free test scores for locked model ${modelId}`);
}

const metrics = evaluateLockedEnsemble(lock, testScoresByModel, labelsByUser);
const report = {
  dataset: "setembrobr",
  predictionTarget: config.predictionTarget ?? "depression",
  seed: config.seed,
  manifestHash: lock.manifestHash,
  lock,
  testMetrics: metrics,
};
await writeJson(resolveOutputPath(config, "reports", `${reportBasename}.json`), report);
console.log(JSON.stringify(metrics, null, 2));
