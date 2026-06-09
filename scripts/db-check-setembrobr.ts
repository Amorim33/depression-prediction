import { writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { loadConfig, loadDatabaseUrl, resolveOutputPath } from "../src/config.ts";
import { createDatabaseClient } from "../src/db.ts";
import { validateDbContract } from "../src/db-contract.ts";

const config = await loadConfig();
const databaseUrl = await loadDatabaseUrl();
const client = createDatabaseClient(databaseUrl);

try {
  const report = await validateDbContract(client, config).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      checks: [{ name: "database-connection", ok: false, detail: message || "Unable to connect to DATABASE_URL" }],
    };
  });
  const outPath = resolveOutputPath(config, "db-contract-report.json");
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, `${JSON.stringify(report, null, 2)}\n`);
  for (const check of report.checks) {
    console.log(`${check.ok ? "ok" : "FAIL"} ${check.name} ${check.detail}`);
  }
  if (!report.ok) process.exit(1);
} finally {
  await client.end();
}
