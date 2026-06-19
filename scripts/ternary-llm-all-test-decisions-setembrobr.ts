import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { listCsvFiles, writeJson } from "../src/artifacts.ts";
import { writeCsv } from "../src/csv.ts";
import {
  LLM_DECISION_TOOL,
  LLM_DECISION_TOOL_NAME,
  LLM_PROMPT_METADATA,
  apiModelFromRequested,
  buildLlmUserMessage,
  llmRequestHash,
  parseAnthropicDecisionResponse,
  writeLlmDecisionCsv,
  type LlmDecision,
  type LlmTimelinePack,
} from "../src/llm-disambiguator.ts";
import { loadTernaryConfig, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import { predictedTernaryLabel, readTernaryTestScores } from "../src/ternary.ts";
import type { TernaryDecisionRule } from "../src/types.ts";

await loadDotEnv();

const config = await loadTernaryConfig();
if (!config.llmDisambiguator?.enabled) throw new Error("llmDisambiguator.enabled must be true for this config");
if (!config.outputDir.includes("symmetric")) throw new Error("LLM all-results comparison is scoped to the symmetric raw lane");

const llmConfig = config.llmDisambiguator;
const requestedModel = llmConfig.requestedModel;
const apiModel = llmConfig.apiModel || apiModelFromRequested(requestedModel);
if (apiModel !== apiModelFromRequested(requestedModel)) throw new Error("llmDisambiguator.apiModel must match requestedModel without [1M]");

const configPath = process.env.TERNARY_CONFIG?.trim() || arg("--config") || "configs/setembrobr.seed42.raw-qwen3-ternary-symmetric.json";
const mock = process.env.LLM_DISAMBIGUATOR_MOCK === "true";
const limit = Number(arg("--limit") ?? process.env.LLM_DISAMBIGUATOR_LIMIT ?? "0");
const decisionRule: TernaryDecisionRule = { ruleId: "diagnosed_margin_010", kind: "diagnosed_margin", diagnosedMargin: 0.1 };

const scoreFiles = await listCsvFiles(resolveTernaryOutputPath(config, "scores"), "test_score_");
const candidateUsers = new Set<string>();
const fileSummaries = [];
for (const path of scoreFiles) {
  const rows = await readTernaryTestScores(path);
  const modelId = rows[0]?.modelId ?? "unknown";
  const labelPolicyId = rows[0]?.labelPolicyId ?? "unknown";
  let diagnosedPredictions = 0;
  for (const row of rows) {
    const predicted = predictedTernaryLabel([row.probDiagnosed, row.probControl, row.probNoEvidence], decisionRule);
    if (predicted !== "diagnosed") continue;
    diagnosedPredictions += 1;
    candidateUsers.add(row.userId);
  }
  fileSummaries.push({ labelPolicyId, modelId, diagnosedPredictions });
}

const candidateList = [...candidateUsers].sort().slice(0, limit > 0 ? limit : undefined);
const llmDir = resolveTernaryOutputPath(config, "llm-disambiguator");
const usersPath = resolve(llmDir, "test_all_score_candidate_users.csv");
const packsPath = resolve(llmDir, "test_all_score_timeline_packs.jsonl");
await mkdir(llmDir, { recursive: true });
await writeFile(usersPath, writeCsv(["user_id"], candidateList.map((userId) => ({ user_id: userId }))));
await exportTimelinePacks(usersPath, packsPath);
const packs = await readPacks(packsPath);

let completed = 0;
const decisions = await mapWithConcurrency(candidateList, llmConfig.concurrency, async (userId) => {
  const pack = packs.get(userId);
  if (!pack) throw new Error(`Missing LLM timeline pack for ${userId}`);
  const decision = await decide(userId, pack);
  completed += 1;
  if (completed % 25 === 0 || completed === candidateList.length) {
    console.error(`llm all-score decisions: ${completed}/${candidateList.length}`);
  }
  return decision;
});

const decisionPath = resolve(llmDir, "test_all_score_decisions.csv");
await writeFile(decisionPath, writeLlmDecisionCsv(decisions, false));
await writeJson(resolveTernaryOutputPath(config, "reports", "llm-disambiguator-all-test-manifest.json"), {
  dataset: "setembrobr",
  seed: config.seed,
  lane: "raw_qwen3_symmetric",
  purpose: "post_hoc_test_comparison_not_model_selection",
  testLabelsRead: false,
  scoreFileCount: scoreFiles.length,
  totalDiagnosedPredictionsAcrossScoreFiles: fileSummaries.reduce((sum, row) => sum + row.diagnosedPredictions, 0),
  candidateUsers: candidateUsers.size,
  generatedUsers: candidateList.length,
  promptId: LLM_PROMPT_METADATA.promptId,
  promptVersion: LLM_PROMPT_METADATA.promptVersion,
  promptHash: LLM_PROMPT_METADATA.promptHash,
  requestedModel,
  apiModel,
  maxEvidenceTweets: llmConfig.maxEvidenceTweets,
  llmTrueCount: decisions.filter((decision) => decision.trueDepression).length,
  switchedToControl: decisions.filter((decision) => !decision.trueDepression).length,
  cacheHitCount: decisions.filter((decision) => decision.cacheHit).length,
  decisionPath,
});

console.log(
  JSON.stringify(
    {
      scoreFiles: scoreFiles.length,
      candidateUsers: candidateUsers.size,
      decisions: decisions.length,
      cacheHitCount: decisions.filter((decision) => decision.cacheHit).length,
      decisionPath,
    },
    null,
    2,
  ),
);

async function exportTimelinePacks(usersFile: string, outputJsonl: string): Promise<void> {
  const proc = Bun.spawn({
    cmd: [
      "python3",
      "scripts/export_llm_timeline_packs_setembrobr.py",
      "--config",
      configPath,
      "--split",
      "test",
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

async function decide(userId: string, pack: LlmTimelinePack): Promise<LlmDecision> {
  const requestHash = llmRequestHash(apiModel, pack);
  const cachePath = resolve(llmDir, "cache", `${requestHash}.json`);
  const cached = await readJsonIfExists(cachePath);
  if (cached) return decisionFromPayload(userId, requestHash, cached, true);

  const payload = mock ? mockPayload(pack) : await callAnthropic(pack);
  await writeJson(cachePath, payload);
  return decisionFromPayload(userId, requestHash, payload, false);
}

function decisionFromPayload(
  userId: string,
  requestHash: string,
  payload: { trueDepression: boolean; confidence: number; reason: string },
  cacheHit: boolean,
): LlmDecision {
  return {
    userId,
    baseDecision: "diagnosed",
    trueDepression: payload.trueDepression,
    confidence: payload.confidence,
    reason: payload.reason,
    promptHash: LLM_PROMPT_METADATA.promptHash,
    requestHash,
    cacheHit,
    requestedModel,
    apiModel,
    labelPolicyId: "all_test_score_union",
    lockBasename: "all-test-score-results",
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
    if (response.ok) return parseAnthropicDecisionResponse(text);
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

async function readJsonIfExists(path: string): Promise<{ trueDepression: boolean; confidence: number; reason: string } | null> {
  const text = await readFile(path, "utf8").catch(() => null);
  if (!text) return null;
  return JSON.parse(text) as { trueDepression: boolean; confidence: number; reason: string };
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
  await Promise.all(Array.from({ length: Math.max(1, concurrency) }, () => worker()));
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

function arg(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

