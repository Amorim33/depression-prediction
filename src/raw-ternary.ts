import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseCsv } from "./csv.ts";
import { sha256File } from "./hash.ts";
import type { BinaryLabel, ManifestRow, TernaryProjectConfig } from "./types.ts";

export function isRawTernaryConfig(config: TernaryProjectConfig): boolean {
  return config.featureSource === "raw_artifacts";
}

export function rawManifestPath(config: TernaryProjectConfig, fileName: string): string {
  return resolve(process.cwd(), config.outputDir, "manifest", fileName);
}

export async function readRawTrainManifest(config: TernaryProjectConfig): Promise<ManifestRow[]> {
  const text = await readFile(rawManifestPath(config, `train_binary_manifest_seed${config.seed}.csv`), "utf8");
  return parseCsv(text).map((row) => ({
    dataset: "setembrobr",
    split: "train",
    label: normalizeBinaryLabel(row.label),
    userId: required(row, "user_id"),
    rowHash: required(row, "row_hash"),
    fold: Number(required(row, "fold")),
  }));
}

export async function readRawTestUsers(config: TernaryProjectConfig): Promise<string[]> {
  const text = await readFile(rawManifestPath(config, `test_inference_manifest_seed${config.seed}.csv`), "utf8");
  return parseCsv(text).map((row) => required(row, "user_id"));
}

export async function readRawSealedTestLabels(config: TernaryProjectConfig): Promise<Array<{ userId: string; label: BinaryLabel }>> {
  const configured = config.sealedTestLabelsPath;
  const path = configured ? resolve(process.cwd(), configured) : rawManifestPath(config, `sealed_test_labels_seed${config.seed}.csv`);
  const text = await readFile(path, "utf8");
  return parseCsv(text).map((row) => ({
    userId: required(row, "user_id"),
    label: normalizeBinaryLabel(row.binary_label ?? row.label),
  }));
}

export async function rawStrictBlindManifestHash(config: TernaryProjectConfig): Promise<string> {
  return sha256File(rawManifestPath(config, `strict_blind_split_manifest_seed${config.seed}.csv`));
}

function normalizeBinaryLabel(value: unknown): BinaryLabel {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (normalized === "1" || normalized === "diagnosed") return "diagnosed";
  if (normalized === "0" || normalized === "control") return "control";
  throw new Error(`Unknown binary label: ${String(value)}`);
}

function required(record: Record<string, string>, key: string): string {
  const value = record[key];
  if (value === undefined || value === "") throw new Error(`Missing required raw manifest field: ${key}`);
  return value;
}
