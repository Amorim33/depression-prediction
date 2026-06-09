import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

export function sha256Text(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

export async function sha256File(path: string): Promise<string> {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

export function stableJson(value: unknown): string {
  return JSON.stringify(sortForJson(value), null, 2);
}

function sortForJson(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortForJson);
  if (!value || typeof value !== "object") return value;

  const input = value as Record<string, unknown>;
  const output: Record<string, unknown> = {};
  for (const key of Object.keys(input).sort()) {
    output[key] = sortForJson(input[key]);
  }
  return output;
}

