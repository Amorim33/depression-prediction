import type { DbTables, ProjectConfig } from "./types.ts";
import { assertSafeIdentifier, type DatabaseClient } from "./db.ts";

const REQUIRED_COLUMNS: Record<keyof DbTables, string[]> = {
  trainUserEmb: ["user_id", "label", "embedding"],
  testUserEmb: ["user_id", "label", "embedding"],
  trainUserEmbRel3: ["user_id", "label", "embedding"],
  testUserEmbRel3: ["user_id", "label", "embedding"],
  trainSubFeatures: [
    "user_id",
    "actual",
    "max_therapy",
    "max_medication",
    "max_selfharm",
    "max_suicidal",
    "max_emptiness",
    "max_depr_self",
    "max_insomnia",
    "max_crying",
    "total_tweets",
  ],
  testSubFeatures: [
    "user_id",
    "actual",
    "max_therapy",
    "max_medication",
    "max_selfharm",
    "max_suicidal",
    "max_emptiness",
    "max_depr_self",
    "max_insomnia",
    "max_crying",
    "total_tweets",
  ],
  trainEmbeddings: ["user_id", "tweet_index", "tweet_text", "embedding", "gpt_3_5_relevance"],
  testEmbeddings: ["user_id", "tweet_index", "tweet_text", "embedding", "gpt_3_5_relevance"],
};

export interface DbContractReport {
  ok: boolean;
  checks: Array<{ name: string; ok: boolean; detail: string }>;
}

export async function validateDbContract(client: DatabaseClient, config: ProjectConfig): Promise<DbContractReport> {
  const checks: DbContractReport["checks"] = [];

  for (const [logicalName, tableName] of Object.entries(config.database.tables) as Array<[keyof DbTables, string]>) {
    assertSafeIdentifier(tableName);
    const tableExists = await existsTable(client, tableName);
    checks.push({ name: `table:${logicalName}`, ok: tableExists, detail: tableName });
    if (!tableExists) continue;

    const columns = await tableColumns(client, tableName);
    for (const column of REQUIRED_COLUMNS[logicalName]) {
      checks.push({
        name: `column:${logicalName}.${column}`,
        ok: columns.has(column),
        detail: `${tableName}.${column}`,
      });
    }
  }

  checks.push(await embeddingDimCheck(client, config.database.tables.trainUserEmb, config.database.embeddingDimension));
  checks.push(await embeddingDimCheck(client, config.database.tables.testUserEmb, config.database.embeddingDimension));
  checks.push(await disjointUsersCheck(client, config.database.tables.trainUserEmb, config.database.tables.testUserEmb));
  checks.push(await labelAgreementCheck(client, config.database.tables.trainUserEmb, config.database.tables.trainSubFeatures));
  checks.push(await labelAgreementCheck(client, config.database.tables.testUserEmb, config.database.tables.testSubFeatures));

  return { ok: checks.every((check) => check.ok), checks };
}

async function existsTable(client: DatabaseClient, tableName: string): Promise<boolean> {
  const rows = await client.sql`
    select count(*)::int as count
    from information_schema.tables
    where table_schema = 'public' and table_name = ${tableName}
  `;
  return Number(rows[0]?.count ?? 0) === 1;
}

async function tableColumns(client: DatabaseClient, tableName: string): Promise<Set<string>> {
  const rows = await client.sql`
    select column_name
    from information_schema.columns
    where table_schema = 'public' and table_name = ${tableName}
  `;
  return new Set(rows.map((row) => String(row.column_name)));
}

async function embeddingDimCheck(client: DatabaseClient, tableName: string, expected: number) {
  assertSafeIdentifier(tableName);
  const rows = await client.sql.unsafe(`select embedding::text as embedding from ${tableName} where embedding is not null limit 1`);
  const dim = countVectorDimensions(String(rows[0]?.embedding ?? ""));
  return {
    name: `embedding-dim:${tableName}`,
    ok: dim === expected,
    detail: `expected=${expected} actual=${dim}`,
  };
}

async function disjointUsersCheck(client: DatabaseClient, trainTable: string, testTable: string) {
  assertSafeIdentifier(trainTable);
  assertSafeIdentifier(testTable);
  const rows = await client.sql.unsafe(`
    select count(*)::int as overlap
    from ${trainTable} train
    inner join ${testTable} test using (user_id)
  `);
  const overlap = Number(rows[0]?.overlap ?? 0);
  return { name: "train-test-user-disjoint", ok: overlap === 0, detail: `overlap=${overlap}` };
}

async function labelAgreementCheck(client: DatabaseClient, userEmbTable: string, featureTable: string) {
  assertSafeIdentifier(userEmbTable);
  assertSafeIdentifier(featureTable);
  const rows = await client.sql.unsafe(`
    select count(*)::int as mismatches
    from ${userEmbTable} emb
    inner join ${featureTable} feat using (user_id)
    where lower(emb.label::text) <> lower(feat.actual::text)
  `);
  const mismatches = Number(rows[0]?.mismatches ?? 0);
  return {
    name: `label-agreement:${userEmbTable}:${featureTable}`,
    ok: mismatches === 0,
    detail: `mismatches=${mismatches}`,
  };
}

export function countVectorDimensions(raw: string): number {
  const trimmed = raw.trim().replace(/^\[/u, "").replace(/\]$/u, "");
  if (!trimmed) return 0;
  return trimmed.split(",").length;
}
