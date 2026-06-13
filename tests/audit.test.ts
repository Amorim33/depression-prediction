import { describe, expect, test } from "bun:test";
import { auditOofScores, auditTestScoreSchema } from "../src/audit.ts";
import type { ManifestRow, ScoreRow } from "../src/types.ts";

const manifest: ManifestRow[] = [
  { dataset: "setembrobr", split: "train", label: "diagnosed", userId: "D1", rowHash: "h1", fold: 1 },
  { dataset: "setembrobr", split: "train", label: "control", userId: "C1", rowHash: "h2", fold: 2 },
  { dataset: "setembrobr", split: "test", label: "control", userId: "TC1", rowHash: "h3", fold: null },
];

describe("OOF audit", () => {
  test("passes valid OOF rows", () => {
    const rows: ScoreRow[] = [
      { userId: "D1", label: "diagnosed", fold: 1, score: 0.9, modelId: "m" },
      { userId: "C1", label: "control", fold: 2, score: 0.1, modelId: "m" },
    ];
    expect(auditOofScores(manifest, new Map([["m", rows]])).ok).toBe(true);
  });

  test("fails when OOF contains test user", () => {
    const rows: ScoreRow[] = [
      { userId: "D1", label: "diagnosed", fold: 1, score: 0.9, modelId: "m" },
      { userId: "TC1", label: "control", fold: 2, score: 0.1, modelId: "m" },
    ];
    const report = auditOofScores(manifest, new Map([["m", rows]]));
    expect(report.ok).toBe(false);
    expect(report.findings.some((finding) => finding.code === "oof-test-user")).toBe(true);
  });

  test("rejects labels in test score files", () => {
    const report = auditTestScoreSchema("test_score_m.csv", "user_id,label,score,model_id\nU,control,0.1,m\n");
    expect(report.ok).toBe(false);
  });

  test("rejects non-finite scores", () => {
    const oofReport = auditOofScores(manifest, new Map([["m", [{ userId: "D1", label: "diagnosed", fold: 1, score: NaN, modelId: "m" }, { userId: "C1", label: "control", fold: 2, score: 0.1, modelId: "m" }]]]));
    expect(oofReport.ok).toBe(false);
    expect(oofReport.findings.some((finding) => finding.code === "oof-nonfinite-score")).toBe(true);

    const testReport = auditTestScoreSchema("test_score_m.csv", "user_id,score,model_id\nU,nan,m\n");
    expect(testReport.ok).toBe(false);
    expect(testReport.findings.some((finding) => finding.code === "test-score-nonfinite-score")).toBe(true);
  });
});
