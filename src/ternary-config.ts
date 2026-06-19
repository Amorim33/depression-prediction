import { readFile } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";
import type { TernaryProjectConfig } from "./types.ts";

export const DEFAULT_TERNARY_CONFIG_PATH = "ternary-classification/configs/setembrobr.seed42.ternary-strict-blind.json";

export async function loadTernaryConfig(path = configuredPath(), cwd = process.cwd()): Promise<TernaryProjectConfig> {
  return (await loadConfigFile(resolve(cwd, path))) as TernaryProjectConfig;
}

export function resolveTernaryOutputPath(config: TernaryProjectConfig, ...parts: string[]): string {
  return resolve(process.cwd(), config.outputDir, ...parts);
}

export function resolveSourceOutputPath(config: TernaryProjectConfig, ...parts: string[]): string {
  return resolve(process.cwd(), config.sourceOutputDir, ...parts);
}

function configuredPath(): string {
  const index = process.argv.indexOf("--config");
  const cliPath = index >= 0 ? process.argv[index + 1] : undefined;
  if (cliPath) return cliPath;
  return process.env.TERNARY_CONFIG || DEFAULT_TERNARY_CONFIG_PATH;
}

async function loadConfigFile(path: string): Promise<unknown> {
  const parsed = JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
  const parentPath = typeof parsed.extends === "string" ? parsed.extends : undefined;
  if (!parentPath) return parsed;
  const parent = (await loadConfigFile(isAbsolute(parentPath) ? parentPath : resolve(dirname(path), parentPath))) as Record<string, unknown>;
  const { extends: _extends, ...child } = parsed;
  return deepMerge(parent, child);
}

function deepMerge(parent: Record<string, unknown>, child: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...parent };
  for (const [key, value] of Object.entries(child)) {
    const parentValue = out[key];
    out[key] =
      isPlainObject(parentValue) && isPlainObject(value)
        ? deepMerge(parentValue as Record<string, unknown>, value as Record<string, unknown>)
        : value;
  }
  return out;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
