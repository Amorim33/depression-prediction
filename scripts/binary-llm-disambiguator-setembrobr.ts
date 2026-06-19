import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { writeCsv } from "../src/csv.ts";
import { sha256File } from "../src/hash.ts";
import {
  LLM_DECISION_TOOL,
  LLM_DECISION_TOOL_NAME,
  LLM_PROMPT_METADATA,
  apiModelFromRequested,
  applyBinaryLlmDisambiguation,
  buildLlmUserMessage,
  llmRequestHash,
  parseAnthropicDecisionResponse,
  writeLlmDecisionCsv,
  type LlmDecision,
  type LlmTimelinePack,
} from "../src/llm-disambiguator.ts";
import { listCsvFiles, readOofScores, readTestScores } from "../src/artifacts.ts";
import { loadConfig, resolveOutputPath } from "../src/config.ts";
import { predictLockedEnsembleRows } from "../src/ensemble.ts";
import { computeMetrics } from "../src/metrics.ts";
import type { BinaryLockedPredictionRow, EnsembleLock, TernaryLabel } from "../src/types.ts";

type Mode = "oof" | "test";

const mode = arg("--mode") as Mode | undefined;
if (mode !== "oof" && mode !== "test") throw new Error("Usage: bun run scripts/binary-llm-disambiguator-setembrobr.ts --mode oof|test");

await loadDotEnv();

const config = await loadConfig();
if (!config.llmDisambiguator?.enabled) throw new Error("llmDisambiguator.enabled must be true for this config");
if (config.featureSource !== "raw_artifacts") throw new Error("Binary LLM disambiguator requires raw_artifacts config");
const llmConfig = config.llmDisambiguator;
const requestedModel = llmConfig.requestedModel;
const apiModel = llmConfig.apiModel || apiModelFromRequested(requestedModel);
if (apiModel !== apiModelFromRequested(requestedModel)) throw new Error("llmDisambiguator.apiModel must match requestedModel without [1M]");

const lockBasename = process.env.BINARY_LOCK_BASENAME?.trim() || "ensemble-lock";
const configPath = process.env.CONFIG?.trim() || arg("--config") || "configs/setembrobr.seed42.raw-qwen3-binary.json";
const limit = Number(arg("--limit") ?? process.env.LLM_DISAMBIGUATOR_LIMIT ?? "0");
const mock = process.env.LLM_DISAMBIGUATOR_MOCK === "true";
const pythonCommand = process.env.PYTHON?.trim() || "python3";
const laneId = "raw_qwen3_binary";

const lock = JSON.parse(await readFile(resolveOutputPath(config, "ensemble", `${lockBasename}.json`), "utf8")) as EnsembleLock;
const scoresByModel = await readLockedScores(mode, lock);
const baseRows = predictLockedEnsembleRows(lock, scoresByModel);
const candidates = baseRows.filter((row) => row.predicted === "diagnosed").slice(0, limit > 0 ? limit : undefined);

const llmDir = resolveOutputPath(config, "llm-disambiguator");
const usersPath = resolve(llmDir, `${mode}_candidate_users_${lockBasename}.csv`);
const packsPath = resolve(llmDir, `${mode}_timeline_packs_${lockBasename}.jsonl`);
await mkdir(llmDir, { recursive: true });
await writeFile(usersPath, writeCsv(["user_id"], candidates.map((row) => ({ user_id: row.userId }))));
await exportTimelinePacks(mode === "oof" ? "train" : "test", usersPath, packsPath);
const packs = await readPacks(packsPath);

const decisions = await mapWithConcurrency(candidates, llmConfig.concurrency, async (row) => {
  const pack = packs.get(row.userId);
  if (!pack) throw new Error(`Missing LLM timeline pack for ${row.userId}`);
  return decide(row, pack);
});

const decisionPath = resolve(llmDir, `${mode === "oof" ? "train_oof" : "test"}_decisions_${lockBasename}.csv`);
const trainLabels =
  mode === "oof"
    ? new Map<string, { label: TernaryLabel; fold: number }>(
        baseRows
          .filter((row) => row.label && row.fold !== undefined)
          .map((row) => [row.userId, { label: row.label!, fold: row.fold! }]),
      )
    : undefined;
await writeFile(decisionPath, writeLlmDecisionCsv(decisions, mode === "oof", trainLabels));

const promptLock = {
  dataset: "setembrobr",
  seed: config.seed,
  lane: laneId,
  lockBasename,
  promptId: LLM_PROMPT_METADATA.promptId,
  promptVersion: LLM_PROMPT_METADATA.promptVersion,
  promptHash: LLM_PROMPT_METADATA.promptHash,
  requestedModel,
  apiModel,
  maxEvidenceTweets: llmConfig.maxEvidenceTweets,
  createdAt: new Date(0).toISOString(),
};
await writeJson(resolve(llmDir, "prompt-lock.json"), promptLock);

if (mode === "oof") {
  const decisionsByUser = new Map(decisions.map((decision) => [decision.userId, decision]));
  const actual = baseRows.map((row) => {
    if (!row.label) throw new Error(`Missing OOF label for ${row.userId}`);
    return row.label;
  });
  const basePredicted = baseRows.map((row) => row.predicted);
  const llmPredicted = baseRows.map((row) => {
    const decision = decisionsByUser.get(row.userId);
    return applyBinaryLlmDisambiguation(row.predicted, decision?.trueDepression);
  });
  await writeJson(resolveOutputPath(config, "reports", "binary-llm-disambiguator-oof-report.json"), {
    dataset: "setembrobr",
    seed: config.seed,
    lane: laneId,
    lockBasename,
    promptLock,
    sourceScores: await sourceHashes(mode, lock),
    candidateUsers: candidates.length,
    llmTrueCount: decisions.filter((decision) => decision.trueDepression).length,
    switchedToControl: decisions.filter((decision) => !decision.trueDepression).length,
    baseMetrics: computeMetrics(actual, basePredicted),
    llmMetrics: computeMetrics(actual, llmPredicted),
    decisionPath,
  });
} else {
  await writeJson(resolveOutputPath(config, "reports", "binary-llm-disambiguator-test-manifest.json"), {
    dataset: "setembrobr",
    seed: config.seed,
    lane: laneId,
    lockBasename,
    promptLock,
    candidateUsers: candidates.length,
    llmTrueCount: decisions.filter((decision) => decision.trueDepression).length,
    switchedToControl: decisions.filter((decision) => !decision.trueDepression).length,
    testLabelsRead: false,
    decisionPath,
  });
}

console.log(JSON.stringify({ mode, candidates: candidates.length, decisions: decisions.length, decisionPath }, null, 2));

async function readLockedScores(modeValue: Mode, currentLock: EnsembleLock) {
  const out = new Map();
  for (const modelId of currentLock.modelIds) {
    const path = resolveOutputPath(config, "scores", `${modeValue === "oof" ? "train_oof" : "test_score"}_${modelId}.csv`);
    const rows = modeValue === "oof" ? await readOofScores(path) : await readTestScores(path);
    out.set(modelId, rows);
  }
  return out;
}

async function exportTimelinePacks(split: "train" | "test", usersFile: string, outputJsonl: string): Promise<void> {
  const proc = Bun.spawn({
    cmd: [
      pythonCommand,
      "scripts/export_llm_timeline_packs_setembrobr.py",
      "--config",
      configPath,
      "--split",
      split,
      "--users-file",
      usersFile,
      "--output-jsonl",
      outputJsonl,
      "--max-evidence-tweets",
      String(llmConfig.maxEvidenceTweets),
    ],
    stdout: "pipe",
    stderr: "pipe",
  });
  const [exitCode, stdout, stderr] = await Promise.all([
    proc.exited,
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  if (exitCode !== 0) {
    throw new Error(`timeline pack export failed (${exitCode}): ${stderr || stdout}`);
  }
}

async function decide(row: BinaryLockedPredictionRow, pack: LlmTimelinePack): Promise<LlmDecision> {
  const requestHash = llmRequestHash(apiModel, pack);
  if (mock) return decisionFromPayload(row, requestHash, mockPayload(pack), false);

  const cachePath = resolve(llmDir, "cache", `${requestHash}.json`);
  const cached = await readJsonIfExists(cachePath);
  if (cached) return decisionFromPayload(row, requestHash, cached, true);

  const payload = await callAnthropic(pack);
  await writeJson(cachePath, payload);
  return decisionFromPayload(row, requestHash, payload, false);
}

function decisionFromPayload(
  row: BinaryLockedPredictionRow,
  requestHash: string,
  payload: { trueDepression: boolean; confidence: number; reason: string },
  cacheHit: boolean,
): LlmDecision {
  return {
    userId: row.userId,
    baseDecision: row.predicted,
    trueDepression: payload.trueDepression,
    confidence: payload.confidence,
    reason: payload.reason,
    promptHash: LLM_PROMPT_METADATA.promptHash,
    requestHash,
    cacheHit,
    requestedModel,
    apiModel,
    labelPolicyId: "binary_raw",
    lockBasename,
  };
}

async function callAnthropic(pack: LlmTimelinePack): Promise<{ trueDepression: boolean; confidence: number; reason: string }> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY is missing; add it to .env or environment");
  const body = {
    model: apiModel,
    max_tokens: llmConfig.maxTokens,
    system: LLM_PROMPT_METADATA.promptText,
    tools: [LLM_DECISION_TOOL],
    tool_choice: { type: "tool", name: LLM_DECISION_TOOL_NAME },
    messages: [{ role: "user", content: buildLlmUserMessage(pack) }],
  };
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(body),
    });
    const text = await response.text();
    if (response.ok) {
      try {
        return parseAnthropicDecisionResponse(text);
      } catch (error) {
        if (attempt === 4) throw error;
        await sleep(1000 * 2 ** attempt);
        continue;
      }
    }
    if (![408, 409, 429, 500, 502, 503, 504].includes(response.status) || attempt === 4) {
      throw new Error(`Anthropic API failed ${response.status}: ${text.slice(0, 500)}`);
    }
    await sleep(1000 * 2 ** attempt);
  }
  throw new Error("Anthropic retry loop exhausted");
}

function mockPayload(pack: LlmTimelinePack): { trueDepression: boolean; confidence: number; reason: string } {
  const joined = pack.selected_tweets.map((tweet) => tweet.tweet_text.toLowerCase()).join("\n");
  const trueDepression = /(me matar|suic[ií]d|quero morrer|vontade de morrer|não quero mais viver|nao quero mais viver)/iu.test(joined);
  return {
    trueDepression,
    confidence: trueDepression ? 0.8 : 0.7,
    reason: trueDepression ? "Mock matched first-person ideation phrase." : "Mock did not match first-person ideation phrase.",
  };
}

async function readPacks(path: string): Promise<Map<string, LlmTimelinePack>> {
  const text = await readFile(path, "utf8");
  const out = new Map<string, LlmTimelinePack>();
  for (const line of text.split(/\r?\n/u)) {
    if (!line.trim()) continue;
    const pack = JSON.parse(line) as LlmTimelinePack;
    out.set(pack.user_id, pack);
  }
  return out;
}

async function sourceHashes(modeValue: Mode, currentLock: EnsembleLock): Promise<Record<string, string>> {
  const out: Record<string, string> = {};
  for (const modelId of currentLock.modelIds) {
    const path = resolveOutputPath(config, "scores", `${modeValue === "oof" ? "train_oof" : "test_score"}_${modelId}.csv`);
    out[modelId] = await sha256File(path);
  }
  return out;
}

async function mapWithConcurrency<T, U>(items: readonly T[], concurrency: number, fn: (item: T) => Promise<U>): Promise<U[]> {
  const out: U[] = [];
  let index = 0;
  async function worker(): Promise<void> {
    while (index < items.length) {
      const current = index;
      index += 1;
      out[current] = await fn(items[current]!);
    }
  }
  const workers = Array.from({ length: Math.max(1, concurrency) }, () => worker());
  await Promise.all(workers);
  return out;
}

async function loadDotEnv(): Promise<void> {
  const text = await readFile(".env", "utf8").catch(() => "");
  for (const line of text.split(/\r?\n/u)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq <= 0) continue;
    const key = trimmed.slice(0, eq).trim();
    const value = trimmed.slice(eq + 1).trim().replace(/^["']|["']$/gu, "");
    if (!process.env[key]) process.env[key] = value;
  }
}

async function readJsonIfExists(path: string): Promise<{ trueDepression: boolean; confidence: number; reason: string } | null> {
  const text = await readFile(path, "utf8").catch(() => null);
  if (!text) return null;
  return JSON.parse(text) as { trueDepression: boolean; confidence: number; reason: string };
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}

function arg(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}
