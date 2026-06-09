import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { parseCsv, writeCsv } from "./csv.ts";
import type { BinaryLabel, ScoreRow } from "./types.ts";

export async function listCsvFiles(dir: string, prefix: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true }).catch(() => []);
  return entries
    .filter((entry) => entry.isFile() && entry.name.startsWith(prefix) && entry.name.endsWith(".csv"))
    .map((entry) => join(dir, entry.name))
    .sort();
}

export async function readOofScores(path: string): Promise<ScoreRow[]> {
  const records = parseCsv(await readFile(path, "utf8"));
  return records.map((record) => ({
    userId: required(record, "user_id"),
    label: normalizeLabel(required(record, "label")),
    fold: Number(required(record, "fold")),
    score: Number(required(record, "score")),
    modelId: required(record, "model_id"),
  }));
}

export async function readTestScores(path: string): Promise<ScoreRow[]> {
  const records = parseCsv(await readFile(path, "utf8"));
  return records.map((record) => ({
    userId: required(record, "user_id"),
    score: Number(required(record, "score")),
    modelId: required(record, "model_id"),
  }));
}

export async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}

export async function writeScoreCsv(path: string, rows: readonly ScoreRow[], includeLabels: boolean): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const headers = includeLabels ? ["user_id", "label", "fold", "score", "model_id"] : ["user_id", "score", "model_id"];
  await writeFile(
    path,
    writeCsv(
      headers,
      rows.map((row) =>
        includeLabels
          ? {
              user_id: row.userId,
              label: row.label ?? "",
              fold: row.fold ?? "",
              score: row.score.toFixed(8),
              model_id: row.modelId,
            }
          : {
              user_id: row.userId,
              score: row.score.toFixed(8),
              model_id: row.modelId,
            },
      ),
    ),
  );
}

export function normalizeLabel(value: unknown): BinaryLabel {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (normalized === "diagnosed" || normalized === "yes") return "diagnosed";
  if (normalized === "control" || normalized === "no") return "control";
  throw new Error(`Unknown label: ${String(value)}`);
}

function required(record: Record<string, string>, key: string): string {
  const value = record[key];
  if (value === undefined || value === "") throw new Error(`Missing CSV column/value: ${key}`);
  return value;
}

