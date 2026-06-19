import { parseCsv, writeCsv } from "./csv.ts";
import { sha256Text, stableJson } from "./hash.ts";
import { PROMPT_HASH, PROMPT_ID, PROMPT_TEXT, PROMPT_VERSION } from "./llm-disambiguator-prompt.ts";
import type { AuditReport } from "./audit.ts";
import type { BinaryLabel, BinaryLockedPredictionRow, TernaryLabel, TernaryLockedPredictionRow } from "./types.ts";

export interface LlmTweetEvidence {
  tweet_index: number;
  tweet_text: string;
  gpt5_relevance: number;
}

export interface LlmTimelinePack {
  user_id: string;
  split: "train" | "test";
  selected_tweets: LlmTweetEvidence[];
  total_selected: number;
}

export interface LlmDecision {
  userId: string;
  baseDecision: TernaryLabel;
  trueDepression: boolean;
  confidence: number;
  reason: string;
  promptHash: string;
  requestHash: string;
  cacheHit: boolean;
  requestedModel: string;
  apiModel: string;
  labelPolicyId: string;
  lockBasename: string;
}

export interface LlmPromptMetadata {
  promptId: string;
  promptVersion: string;
  promptHash: string;
  promptText: string;
}

export const LLM_DECISION_TOOL_NAME = "classify_true_depression";

export const LLM_DECISION_TOOL = {
  name: LLM_DECISION_TOOL_NAME,
  description: "Return the strict-blind depression disambiguation decision for one user's selected tweet timeline.",
  input_schema: {
    type: "object",
    additionalProperties: false,
    properties: {
      true_depression: {
        type: "boolean",
        description: "True only when the timeline shows genuine first-person depressive evidence under the system rubric.",
      },
      confidence: {
        type: "number",
        minimum: 0,
        maximum: 1,
        description: "Confidence in the decision from 0 to 1.",
      },
      reason: {
        type: "string",
        description: "Short reason grounded in the provided tweets; do not include labels or metrics.",
      },
    },
    required: ["true_depression", "confidence", "reason"],
  },
} as const;

export const LLM_PROMPT_METADATA: LlmPromptMetadata = {
  promptId: PROMPT_ID,
  promptVersion: PROMPT_VERSION,
  promptHash: PROMPT_HASH,
  promptText: PROMPT_TEXT,
};

export function apiModelFromRequested(requestedModel: string): string {
  return requestedModel.replace(/\[1m\]$/iu, "");
}

export function applyLlmDisambiguation(baseDecision: TernaryLabel, trueDepression: boolean | undefined): TernaryLabel {
  if (baseDecision === "diagnosed" && trueDepression === false) return "control";
  return baseDecision;
}

export function applyBinaryLlmDisambiguation(baseDecision: BinaryLabel, trueDepression: boolean | undefined): BinaryLabel {
  if (baseDecision === "diagnosed" && trueDepression === false) return "control";
  return baseDecision;
}

export function buildLlmUserMessage(pack: LlmTimelinePack): string {
  const tweets = pack.selected_tweets
    .map((tweet) => {
      const text = tweet.tweet_text.replace(/\s+/gu, " ").trim();
      return `- idx=${tweet.tweet_index} rel=${tweet.gpt5_relevance}: ${text}`;
    })
    .join("\n");
  return `Classify this user's timeline.\n\nuser_id: ${pack.user_id}\nselected_tweets: ${pack.total_selected}\n\n${tweets}`;
}

export function llmRequestHash(apiModel: string, pack: LlmTimelinePack): string {
  return sha256Text(
    stableJson({
      apiModel,
      promptHash: PROMPT_HASH,
      userId: pack.user_id,
      tweets: pack.selected_tweets,
    }),
  );
}

export function parseLlmJsonResponse(text: string): { trueDepression: boolean; confidence: number; reason: string } {
  const trimmed = text.trim();
  const jsonText = trimmed.startsWith("{") ? trimmed : trimmed.slice(trimmed.indexOf("{"), trimmed.lastIndexOf("}") + 1);
  if (!jsonText.startsWith("{") || !jsonText.endsWith("}")) {
    throw new Error(`LLM response did not contain a JSON object: ${text.slice(0, 120)}`);
  }
  const parsed = JSON.parse(jsonText) as { true_depression?: unknown; confidence?: unknown; reason?: unknown };
  if (typeof parsed.true_depression !== "boolean") throw new Error("LLM response missing boolean true_depression");
  const confidence = Number(parsed.confidence ?? 0);
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) throw new Error("LLM response confidence must be between 0 and 1");
  const reason = String(parsed.reason ?? "").replace(/\s+/gu, " ").trim();
  if (!reason) throw new Error("LLM response missing reason");
  return { trueDepression: parsed.true_depression, confidence, reason };
}

export function parseLlmDecisionObject(value: unknown): { trueDepression: boolean; confidence: number; reason: string } {
  const parsed = value as { true_depression?: unknown; confidence?: unknown; reason?: unknown } | null;
  if (typeof parsed?.true_depression !== "boolean") throw new Error("LLM response missing boolean true_depression");
  const confidence = Number(parsed.confidence ?? 0);
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) throw new Error("LLM response confidence must be between 0 and 1");
  const reason = String(parsed.reason ?? "").replace(/\s+/gu, " ").trim();
  if (!reason) throw new Error("LLM response missing reason");
  return { trueDepression: parsed.true_depression, confidence, reason };
}

export function parseAnthropicDecisionResponse(text: string): { trueDepression: boolean; confidence: number; reason: string } {
  const parsed = JSON.parse(text) as { content?: Array<{ type?: string; name?: string; text?: string; input?: unknown }> };
  const toolUse = parsed.content?.find((item) => item.type === "tool_use" && item.name === LLM_DECISION_TOOL_NAME);
  if (toolUse) return parseLlmDecisionObject(toolUse.input);
  const answer = parsed.content?.find((item) => item.type === "text" && item.text)?.text;
  if (!answer) throw new Error(`Anthropic response missing tool_use or text content: ${text.slice(0, 200)}`);
  return parseLlmJsonResponse(answer);
}

export function writeLlmDecisionCsv(rows: readonly LlmDecision[], includeTrainLabels: boolean, labelsByUser?: ReadonlyMap<string, { label: TernaryLabel; fold: number }>): string {
  const trainHeaders = [
    "user_id",
    "actual_label",
    "fold",
    "base_decision",
    "final_decision",
    "llm_true_depression",
    "llm_confidence",
    "llm_reason",
    "prompt_hash",
    "request_hash",
    "cache_hit",
    "requested_model",
    "api_model",
    "label_policy_id",
    "lock_basename",
  ];
  const testHeaders = [
    "user_id",
    "base_decision",
    "llm_true_depression",
    "llm_confidence",
    "llm_reason",
    "prompt_hash",
    "request_hash",
    "cache_hit",
    "requested_model",
    "api_model",
    "label_policy_id",
    "lock_basename",
  ];
  const headers = includeTrainLabels ? trainHeaders : testHeaders;
  return writeCsv(
    headers,
    rows.map((row) => {
      const trainLabel = labelsByUser?.get(row.userId);
      const base = {
        user_id: row.userId,
        base_decision: row.baseDecision,
        llm_true_depression: String(row.trueDepression),
        llm_confidence: row.confidence.toFixed(4),
        llm_reason: row.reason,
        prompt_hash: row.promptHash,
        request_hash: row.requestHash,
        cache_hit: String(row.cacheHit),
        requested_model: row.requestedModel,
        api_model: row.apiModel,
        label_policy_id: row.labelPolicyId,
        lock_basename: row.lockBasename,
      };
      if (!includeTrainLabels) return base;
      if (!trainLabel) throw new Error(`Missing train label for LLM decision ${row.userId}`);
      return {
        user_id: row.userId,
        actual_label: trainLabel.label,
        fold: trainLabel.fold,
        base_decision: row.baseDecision,
        final_decision: applyLlmDisambiguation(row.baseDecision, row.trueDepression),
        llm_true_depression: String(row.trueDepression),
        llm_confidence: row.confidence.toFixed(4),
        llm_reason: row.reason,
        prompt_hash: row.promptHash,
        request_hash: row.requestHash,
        cache_hit: String(row.cacheHit),
        requested_model: row.requestedModel,
        api_model: row.apiModel,
        label_policy_id: row.labelPolicyId,
        lock_basename: row.lockBasename,
      };
    }),
  );
}

export function readLlmTestDecisions(text: string): Map<string, LlmDecision> {
  const rows = parseCsv(text);
  const out = new Map<string, LlmDecision>();
  for (const row of rows) {
    const userId = required(row, "user_id");
    if (out.has(userId)) throw new Error(`Duplicate LLM decision for ${userId}`);
    out.set(userId, {
      userId,
      baseDecision: normalizeTernaryDecision(required(row, "base_decision")),
      trueDepression: parseBoolean(required(row, "llm_true_depression")),
      confidence: Number(required(row, "llm_confidence")),
      reason: required(row, "llm_reason"),
      promptHash: required(row, "prompt_hash"),
      requestHash: required(row, "request_hash"),
      cacheHit: parseBoolean(required(row, "cache_hit")),
      requestedModel: required(row, "requested_model"),
      apiModel: required(row, "api_model"),
      labelPolicyId: required(row, "label_policy_id"),
      lockBasename: required(row, "lock_basename"),
    });
  }
  return out;
}

export function finalPredictionsWithLlm(
  baseRows: readonly TernaryLockedPredictionRow[],
  decisionsByUser: ReadonlyMap<string, LlmDecision>,
): TernaryLabel[] {
  return baseRows.map((row) => {
    if (row.predicted !== "diagnosed") return row.predicted;
    const decision = decisionsByUser.get(row.userId);
    if (!decision) throw new Error(`Missing LLM decision for diagnosed test user ${row.userId}`);
    if (decision.baseDecision !== "diagnosed") throw new Error(`LLM decision for ${row.userId} was not based on diagnosed prediction`);
    return applyLlmDisambiguation(row.predicted, decision.trueDepression);
  });
}

export function finalBinaryPredictionsWithLlm(
  baseRows: readonly BinaryLockedPredictionRow[],
  decisionsByUser: ReadonlyMap<string, LlmDecision>,
): BinaryLabel[] {
  return baseRows.map((row) => {
    if (row.predicted !== "diagnosed") return row.predicted;
    const decision = decisionsByUser.get(row.userId);
    if (!decision) throw new Error(`Missing LLM decision for diagnosed test user ${row.userId}`);
    if (decision.baseDecision !== "diagnosed") throw new Error(`LLM decision for ${row.userId} was not based on diagnosed prediction`);
    return applyBinaryLlmDisambiguation(row.predicted, decision.trueDepression);
  });
}

export function auditLlmTestDecisionSchema(fileName: string, csvText: string): AuditReport {
  const [headerLine = ""] = csvText.split(/\r?\n/u);
  const headers = headerLine.split(",");
  const findings = [];
  for (const forbidden of ["label", "actual", "actual_label", "fold", "metric", "macro_f1", "macroF1", "diagnosedF1"]) {
    if (headers.includes(forbidden)) findings.push(fail("llm-test-forbidden-column", `${fileName}: forbidden column ${forbidden}`));
  }
  for (const requiredHeader of [
    "user_id",
    "base_decision",
    "llm_true_depression",
    "llm_confidence",
    "llm_reason",
    "prompt_hash",
    "request_hash",
    "cache_hit",
    "requested_model",
    "api_model",
    "label_policy_id",
    "lock_basename",
  ]) {
    if (!headers.includes(requiredHeader)) findings.push(fail("llm-test-missing-column", `${fileName}: missing ${requiredHeader}`));
  }
  for (const [index, row] of parseCsv(csvText).entries()) {
    try {
      normalizeTernaryDecision(required(row, "base_decision"));
      parseBoolean(required(row, "llm_true_depression"));
      const confidence = Number(required(row, "llm_confidence"));
      if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
        findings.push(fail("llm-test-invalid-confidence", `${fileName}: row ${index + 2}`));
      }
    } catch (error) {
      findings.push(fail("llm-test-invalid-row", `${fileName}: row ${index + 2}: ${error instanceof Error ? error.message : String(error)}`));
    }
  }
  if (findings.length === 0) findings.push(pass("llm-test-decision-schema", `${fileName}: label-free LLM decision schema`));
  return { ok: findings.every((finding) => finding.ok), findings };
}

function normalizeTernaryDecision(value: string): TernaryLabel {
  if (value === "diagnosed" || value === "control" || value === "no-evidence") return value;
  throw new Error(`Unknown ternary decision: ${value}`);
}

function parseBoolean(value: string): boolean {
  if (value === "true") return true;
  if (value === "false") return false;
  throw new Error(`Expected boolean string, got ${value}`);
}

function required(record: Record<string, string>, key: string): string {
  const value = record[key];
  if (value === undefined || value === "") throw new Error(`Missing CSV column/value: ${key}`);
  return value;
}

function pass(code: string, detail: string) {
  return { ok: true, code, detail };
}

function fail(code: string, detail: string) {
  return { ok: false, code, detail };
}
