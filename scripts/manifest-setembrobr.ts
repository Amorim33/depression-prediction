import { loadConfig, loadDatabaseUrl, resolveOutputPath } from "../src/config.ts";
import { createDatabaseClient } from "../src/db.ts";
import { buildManifest, manifestHash, writeManifest } from "../src/manifest.ts";
import { writeJson } from "../src/artifacts.ts";

const config = await loadConfig();
const databaseUrl = await loadDatabaseUrl();
const client = createDatabaseClient(databaseUrl);

try {
  const rows = await buildManifest(client, config);
  const csvPath = resolveOutputPath(config, "manifest", "split_manifest_seed42.csv");
  await writeManifest(csvPath, rows);
  await writeJson(resolveOutputPath(config, "manifest", "split_manifest_seed42.meta.json"), {
    dataset: config.dataset,
    seed: config.seed,
    foldCount: config.foldCount,
    manifestHash: manifestHash(rows),
    rows: rows.length,
    trainRows: rows.filter((row) => row.split === "train").length,
    testRows: rows.filter((row) => row.split === "test").length,
  });
  console.log(`wrote ${csvPath}`);
} finally {
  await client.end();
}

