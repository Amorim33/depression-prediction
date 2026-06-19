import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { ProjectConfig } from "./types.ts";

export const DEFAULT_CONFIG_PATH = "configs/setembrobr.seed42.strict-blind.json";

export async function loadConfig(path = process.env.CONFIG?.trim() || DEFAULT_CONFIG_PATH, cwd = process.cwd()): Promise<ProjectConfig> {
  const configPath = resolve(cwd, path);
  return JSON.parse(await readFile(configPath, "utf8")) as ProjectConfig;
}

export async function loadEnv(cwd = process.cwd()): Promise<Record<string, string>> {
  const envPath = resolve(cwd, ".env");
  const fileEnv = existsSync(envPath) ? parseEnv(await readFile(envPath, "utf8")) : {};
  return {
    ...fileEnv,
    ...Object.fromEntries(Object.entries(process.env).filter((entry): entry is [string, string] => typeof entry[1] === "string")),
  };
}

export async function loadDatabaseUrl(cwd = process.cwd()): Promise<string> {
  const env = await loadEnv(cwd);
  const url = env.DATABASE_URL?.trim();
  if (!url) throw new Error("DATABASE_URL is required. Copy .env.example to .env or set the variable.");
  return url;
}

export function resolveOutputPath(config: ProjectConfig, ...parts: string[]): string {
  return resolve(process.cwd(), config.outputDir, ...parts);
}

function parseEnv(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split(/\r?\n/u)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq < 1) continue;
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    out[key] = value;
  }
  return out;
}
