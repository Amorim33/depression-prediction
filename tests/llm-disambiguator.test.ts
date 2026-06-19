import { describe, expect, test } from "bun:test";
import { sha256Text } from "../src/hash.ts";
import {
  LLM_PROMPT_METADATA,
  applyBinaryLlmDisambiguation,
  apiModelFromRequested,
  applyLlmDisambiguation,
  auditLlmTestDecisionSchema,
  buildLlmUserMessage,
  finalBinaryPredictionsWithLlm,
  finalPredictionsWithLlm,
  parseAnthropicDecisionResponse,
  parseLlmJsonResponse,
  readLlmTestDecisions,
} from "../src/llm-disambiguator.ts";
import { PROMPT_HASH, PROMPT_ID, PROMPT_TEXT, PROMPT_VERSION } from "../src/llm-disambiguator-prompt.ts";
import type { BinaryLockedPredictionRow, TernaryLockedPredictionRow } from "../src/types.ts";

describe("LLM disambiguator", () => {
  test("normalizes the requested 1M model suffix for API calls", () => {
    expect(apiModelFromRequested("claude-sonnet-4-6[1M]")).toBe("claude-sonnet-4-6");
    expect(apiModelFromRequested("claude-sonnet-4-6")).toBe("claude-sonnet-4-6");
  });

  test("only changes diagnosed predictions when the LLM rejects true depression", () => {
    expect(applyLlmDisambiguation("diagnosed", false)).toBe("control");
    expect(applyLlmDisambiguation("diagnosed", true)).toBe("diagnosed");
    expect(applyLlmDisambiguation("control", false)).toBe("control");
    expect(applyLlmDisambiguation("no-evidence", false)).toBe("no-evidence");
    expect(applyBinaryLlmDisambiguation("diagnosed", false)).toBe("control");
    expect(applyBinaryLlmDisambiguation("diagnosed", true)).toBe("diagnosed");
    expect(applyBinaryLlmDisambiguation("control", false)).toBe("control");
  });

  test("requires LLM decisions for every base diagnosed prediction", () => {
    const rows: TernaryLockedPredictionRow[] = [
      { userId: "A", predicted: "diagnosed", probDiagnosed: 0.8, probControl: 0.2, probNoEvidence: 0 },
      { userId: "B", predicted: "control", probDiagnosed: 0.2, probControl: 0.8, probNoEvidence: 0 },
    ];
    const decisions = readLlmTestDecisions(
      [
        "user_id,base_decision,llm_true_depression,llm_confidence,llm_reason,prompt_hash,request_hash,cache_hit,requested_model,api_model,label_policy_id,lock_basename",
        "A,diagnosed,false,0.9000,clear false positive,h,r,false,claude-sonnet-4-6[1M],claude-sonnet-4-6,p,ensemble-lock",
        "",
      ].join("\n"),
    );
    expect(finalPredictionsWithLlm(rows, decisions)).toEqual(["control", "control"]);
    expect(() => finalPredictionsWithLlm(rows, new Map())).toThrow("Missing LLM decision");
  });

  test("finalizes binary LLM decisions without introducing no-evidence", () => {
    const rows: BinaryLockedPredictionRow[] = [
      { userId: "A", predicted: "diagnosed", score: 0.8 },
      { userId: "B", predicted: "control", score: 0.2 },
    ];
    const decisions = readLlmTestDecisions(
      [
        "user_id,base_decision,llm_true_depression,llm_confidence,llm_reason,prompt_hash,request_hash,cache_hit,requested_model,api_model,label_policy_id,lock_basename",
        "A,diagnosed,false,0.9000,clear false positive,h,r,false,claude-sonnet-4-6[1M],claude-sonnet-4-6,binary_raw,ensemble-lock",
        "",
      ].join("\n"),
    );
    expect(finalBinaryPredictionsWithLlm(rows, decisions)).toEqual(["control", "control"]);
    expect(() => finalBinaryPredictionsWithLlm(rows, new Map())).toThrow("Missing LLM decision");
  });

  test("audits test decisions as label-free and fold-free", () => {
    const ok = auditLlmTestDecisionSchema(
      "test_decisions.csv",
      [
        "user_id,base_decision,llm_true_depression,llm_confidence,llm_reason,prompt_hash,request_hash,cache_hit,requested_model,api_model,label_policy_id,lock_basename",
        "A,diagnosed,true,0.9000,reason,h,r,false,claude-sonnet-4-6[1M],claude-sonnet-4-6,p,ensemble-lock",
        "",
      ].join("\n"),
    );
    expect(ok.ok).toBe(true);

    const bad = auditLlmTestDecisionSchema(
      "test_decisions.csv",
      "user_id,label,fold,base_decision,llm_true_depression,llm_confidence,llm_reason,prompt_hash,request_hash,cache_hit,requested_model,api_model,label_policy_id,lock_basename\nA,control,1,diagnosed,true,0.9,reason,h,r,false,claude-sonnet-4-6[1M],claude-sonnet-4-6,p,ensemble-lock\n",
    );
    expect(bad.ok).toBe(false);
    expect(bad.findings.some((finding) => finding.code === "llm-test-forbidden-column")).toBe(true);
  });

  test("keeps a deterministic hardcoded prompt hash", () => {
    expect(PROMPT_HASH).toBe(sha256Text(`${PROMPT_ID}\n${PROMPT_VERSION}\n${PROMPT_TEXT}`));
    expect(LLM_PROMPT_METADATA.promptHash).toBe(PROMPT_HASH);
    expect(PROMPT_TEXT).toContain("Few-shot examples from train OOF error analysis");
  });

  test("builds a Portuguese tweet evidence prompt", () => {
    const message = buildLlmUserMessage({
      user_id: "U",
      split: "test",
      total_selected: 1,
      selected_tweets: [{ tweet_index: 7, tweet_text: "eu quero morrer", gpt5_relevance: 8 }],
    });
    expect(message).toContain("idx=7 rel=8");
    expect(message).toContain("eu quero morrer");
  });

  test("parses strict JSON responses", () => {
    expect(parseLlmJsonResponse('{"true_depression":false,"confidence":0.7,"reason":"joke"}')).toEqual({
      trueDepression: false,
      confidence: 0.7,
      reason: "joke",
    });
  });

  test("parses Anthropic tool-use decisions", () => {
    const response = JSON.stringify({
      content: [
        {
          type: "tool_use",
          name: "classify_true_depression",
          input: { true_depression: true, confidence: 0.91, reason: "first-person persistent ideation" },
        },
      ],
    });
    expect(parseAnthropicDecisionResponse(response)).toEqual({
      trueDepression: true,
      confidence: 0.91,
      reason: "first-person persistent ideation",
    });
  });
});
