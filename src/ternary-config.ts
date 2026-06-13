import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { TernaryProjectConfig } from "./types.ts";

export const DEFAULT_TERNARY_CONFIG_PATH = "ternary-classification/configs/setembrobr.seed42.ternary-strict-blind.json";

export async function loadTernaryConfig(path = DEFAULT_TERNARY_CONFIG_PATH, cwd = process.cwd()): Promise<TernaryProjectConfig> {
  return JSON.parse(await readFile(resolve(cwd, path), "utf8")) as TernaryProjectConfig;
}

export function resolveTernaryOutputPath(config: TernaryProjectConfig, ...parts: string[]): string {
  return resolve(process.cwd(), config.outputDir, ...parts);
}

export function resolveSourceOutputPath(config: TernaryProjectConfig, ...parts: string[]): string {
  return resolve(process.cwd(), config.sourceOutputDir, ...parts);
}
