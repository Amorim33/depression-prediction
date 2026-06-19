import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { writeJson } from "../src/artifacts.ts";
import { createStratifiedFolds } from "../src/folds.ts";
import { sha256Text } from "../src/hash.ts";
import { manifestHash, readManifest } from "../src/manifest.ts";
import { isRawTernaryConfig, rawStrictBlindManifestHash, readRawTrainManifest } from "../src/raw-ternary.ts";
import { loadTernaryConfig, resolveSourceOutputPath, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import {
  deriveTernaryLabel,
  lockTernaryLabelPolicy,
  readEvidenceMarkers,
  writeTernaryManifestCsv,
} from "../src/ternary.ts";
import type { TernaryManifestRow } from "../src/types.ts";

const config = await loadTernaryConfig();
const sourceRows = isRawTernaryConfig(config)
  ? await readRawTrainManifest(config)
  : await readManifest(resolveSourceOutputPath(config, "manifest", `split_manifest_seed${config.seed}.csv`));
const sourceManifestHash = isRawTernaryConfig(config) ? await rawStrictBlindManifestHash(config) : manifestHash(sourceRows);
const trainRows = sourceRows.filter((row) => row.split === "train");
const trainMarkers = await readEvidenceMarkers(resolveTernaryOutputPath(config, "evidence-markers", "train_markers.csv"));
const markersByUser = new Map(trainMarkers.map((marker) => [marker.userId, marker]));
const summary = [];

for (const policy of config.labelPolicies) {
  const policyInputRows = trainRows.map((row) => {
    const marker = markersByUser.get(row.userId);
    if (!marker) throw new Error(`${policy.policyId}: missing train evidence marker for ${row.userId}`);
    return { binaryLabel: row.label, marker };
  });
  const policyLock = lockTernaryLabelPolicy(policy, policyInputRows, sourceManifestHash, config.seed);
  const labels = trainRows.map((row) => {
    const marker = markersByUser.get(row.userId);
    if (!marker) throw new Error(`${policy.policyId}: missing train evidence marker for ${row.userId}`);
    return deriveTernaryLabel(row.label, marker, policyLock);
  });
  const folds = createStratifiedFolds(labels, config.foldCount, config.seed);
  const foldByIndex = new Map<number, number>();
  for (const [foldIndex, fold] of folds.entries()) {
    for (const rowIndex of fold) foldByIndex.set(rowIndex, foldIndex + 1);
  }
  const manifestRows: TernaryManifestRow[] = trainRows.map((row, index) => {
    const label = labels[index]!;
    return {
      dataset: "setembrobr",
      split: "train",
      label,
      binaryLabel: row.label,
      userId: row.userId,
      rowHash: sha256Text(`setembrobr|ternary|${policy.policyId}|train|${label}|${row.userId}|${row.rowHash}`),
      fold: foldByIndex.get(index) ?? 1,
      labelPolicyId: policy.policyId,
    };
  });
  const manifestPath = resolveTernaryOutputPath(config, "manifest", `train_manifest_${policy.policyId}_seed${config.seed}.csv`);
  await mkdir(dirname(manifestPath), { recursive: true });
  await writeFile(manifestPath, writeTernaryManifestCsv(manifestRows));
  const lockPath = resolveTernaryOutputPath(config, "label-policies", `${policy.policyId}.json`);
  await writeJson(lockPath, policyLock);
  const counts = countLabels(manifestRows);
  summary.push({
    policyId: policy.policyId,
    policyHash: policyLock.policyHash,
    cutoff: policyLock.cutoff ?? null,
    counts,
    manifestPath,
    lockPath,
  });
  console.log(`wrote ${manifestPath}`);
  console.log(`wrote ${lockPath}`);
}

await writeJson(resolveTernaryOutputPath(config, "reports", "label-policy-summary.json"), {
  dataset: "setembrobr",
  seed: config.seed,
  sourceManifestHash,
  policies: summary,
  createdAt: new Date(0).toISOString(),
});

function countLabels(rows: readonly TernaryManifestRow[]) {
  return rows.reduce(
    (counts, row) => {
      counts[row.label] += 1;
      return counts;
    },
    { diagnosed: 0, control: 0, "no-evidence": 0 },
  );
}
