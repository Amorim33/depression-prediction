import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { parseCsv, writeCsv } from "./csv.ts";
import { createStratifiedFolds } from "./folds.ts";
import { sha256Text } from "./hash.ts";
import type { BinaryLabel, ManifestRow, ProjectConfig } from "./types.ts";
import { assertSafeIdentifier, type DatabaseClient } from "./db.ts";

export async function buildManifest(client: DatabaseClient, config: ProjectConfig): Promise<ManifestRow[]> {
  const train = await loadManifestSplit(client, config.database.tables.trainUserEmb, config.database.tables.trainSubFeatures, "train");
  const test = await loadManifestSplit(client, config.database.tables.testUserEmb, config.database.tables.testSubFeatures, "test");
  const folds = createStratifiedFolds(train.map((row) => row.label), config.foldCount, config.seed);
  const foldByIndex = new Map<number, number>();
  for (let foldIndex = 0; foldIndex < folds.length; foldIndex += 1) {
    for (const rowIndex of folds[foldIndex]!) foldByIndex.set(rowIndex, foldIndex + 1);
  }
  return [
    ...train.map((row, index) => ({ ...row, fold: foldByIndex.get(index) ?? null })),
    ...test,
  ];
}

export async function writeManifest(path: string, rows: readonly ManifestRow[]): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(
    path,
    writeCsv(
      ["dataset", "split", "label", "user_id", "row_hash", "fold"],
      rows.map((row) => ({
        dataset: row.dataset,
        split: row.split,
        label: row.label,
        user_id: row.userId,
        row_hash: row.rowHash,
        fold: row.fold ?? "",
      })),
    ),
  );
}

export async function readManifest(path: string): Promise<ManifestRow[]> {
  const rows = parseCsv(await readFile(path, "utf8"));
  return rows.map((row) => ({
    dataset: "setembrobr",
    split: row.split === "test" ? "test" : "train",
    label: normalizeLabel(row.label),
    userId: row.user_id ?? "",
    rowHash: row.row_hash ?? "",
    fold: row.fold ? Number(row.fold) : null,
  }));
}

export function manifestHash(rows: readonly ManifestRow[]): string {
  const text = writeCsv(
    ["dataset", "split", "label", "user_id", "row_hash", "fold"],
    rows.map((row) => ({
      dataset: row.dataset,
      split: row.split,
      label: row.label,
      user_id: row.userId,
      row_hash: row.rowHash,
      fold: row.fold ?? "",
    })),
  );
  return sha256Text(text);
}

async function loadManifestSplit(
  client: DatabaseClient,
  userEmbTable: string,
  featureTable: string,
  split: "train" | "test",
): Promise<ManifestRow[]> {
  assertSafeIdentifier(userEmbTable);
  assertSafeIdentifier(featureTable);
  const rows = await client.sql.unsafe(`
    select emb.user_id, emb.label
    from ${userEmbTable} emb
    inner join ${featureTable} feat using (user_id)
    where lower(emb.label::text) = lower(feat.actual::text)
    order by emb.user_id
  `);
  return rows.map((row) => {
    const userId = String(row.user_id);
    const label = normalizeLabel(row.label);
    return {
      dataset: "setembrobr",
      split,
      label,
      userId,
      rowHash: sha256Text(`setembrobr|${split}|${label}|${userId}`),
      fold: null,
    };
  });
}

function normalizeLabel(value: unknown): BinaryLabel {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (normalized === "diagnosed" || normalized === "yes") return "diagnosed";
  if (normalized === "control" || normalized === "no") return "control";
  throw new Error(`Unknown label: ${String(value)}`);
}

