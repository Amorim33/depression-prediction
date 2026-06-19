import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseCsv } from "./csv.ts";
import { sha256File } from "./hash.ts";
import type { BinaryLabel, ManifestRow, ProjectConfig } from "./types.ts";

export function isRawBinaryConfig(config: ProjectConfig): boolean {
  return config.featureSource === "raw_artifacts";
}

export function rawBinaryManifestPath(config: ProjectConfig, fileName: string): string {
  return resolve(process.cwd(), config.outputDir, "manifest", fileName);
}

export async function readRawBinaryTrainManifest(config: ProjectConfig): Promise<ManifestRow[]> {
  const text = await readFile(rawBinaryManifestPath(config, `train_binary_manifest_seed${config.seed}.csv`), "utf8");
  return parseCsv(text).map((row) => ({
    dataset: "setembrobr",
    split: "train",
    label: normalizeBinaryLabel(required(row, "label")),
    userId: required(row, "user_id"),
    rowHash: required(row, "row_hash"),
    fold: Number(required(row, "fold")),
  }));
}

export async function readRawBinaryTestUsers(config: ProjectConfig): Promise<string[]> {
  const text = await readFile(rawBinaryManifestPath(config, `test_inference_manifest_seed${config.seed}.csv`), "utf8");
  return parseCsv(text).map((row) => required(row, "user_id"));
}

export async function readRawBinarySealedTestLabels(config: ProjectConfig): Promise<Array<{ userId: string; label: BinaryLabel }>> {
  const configured = config.sealedTestLabelsPath;
  const path = configured ? resolve(process.cwd(), configured) : rawBinaryManifestPath(config, `sealed_test_labels_seed${config.seed}.csv`);
  const text = await readFile(path, "utf8");
  return parseCsv(text).map((row) => ({
    userId: required(row, "user_id"),
    label: normalizeBinaryLabel(row.binary_label ?? row.label),
  }));
}

export async function rawBinaryStrictBlindManifestHash(config: ProjectConfig): Promise<string> {
  return sha256File(rawBinaryManifestPath(config, `strict_blind_split_manifest_seed${config.seed}.csv`));
}

export async function readRawBinaryAuditManifest(config: ProjectConfig): Promise<ManifestRow[]> {
  const trainRows = await readRawBinaryTrainManifest(config);
  const testRows = (await readRawBinaryTestUsers(config)).map((userId): ManifestRow => ({
    dataset: "setembrobr",
    split: "test",
    label: "control",
    userId,
    rowHash: "",
    fold: null,
  }));
  return [...trainRows, ...testRows];
}

function normalizeBinaryLabel(value: unknown): BinaryLabel {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (normalized === "1" || normalized === "diagnosed") return "diagnosed";
  if (normalized === "0" || normalized === "control") return "control";
  throw new Error(`Unknown binary label: ${String(value)}`);
}

function required(record: Record<string, string>, key: string): string {
  const value = record[key];
  if (value === undefined || value === "") throw new Error(`Missing required raw binary manifest field: ${key}`);
  return value;
}
