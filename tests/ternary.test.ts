import { describe, expect, test } from "bun:test";
import {
  auditTernaryOofScores,
  auditTernaryTestScoreSchema,
  computeTernaryMetrics,
  deriveTernaryLabel,
  evaluateTernaryLockedEnsemble,
  lockTernaryLabelPolicy,
  selectTernaryEnsemble,
} from "../src/ternary.ts";
import type {
  EvidenceMarker,
  TernaryLabel,
  TernaryLabelPolicyConfig,
  TernaryManifestRow,
  TernaryProbabilityRow,
} from "../src/types.ts";

describe("ternary label policies", () => {
  test("never converts controls to no-evidence and converts diagnosed low evidence users", () => {
    const policy = lockTernaryLabelPolicy(
      {
        policyId: "diag_rel3_zero",
        kind: "rel_count_zero",
        relevanceThreshold: 3,
        description: "test",
      },
      [
        { binaryLabel: "diagnosed", marker: marker({ userId: "D", rel3Count: 0 }) },
        { binaryLabel: "control", marker: marker({ userId: "C", rel3Count: 0 }) },
      ],
      "manifest",
      42,
    );

    expect(deriveTernaryLabel("diagnosed", marker({ rel3Count: 0 }), policy)).toBe("no-evidence");
    expect(deriveTernaryLabel("diagnosed", marker({ rel3Count: 1 }), policy)).toBe("diagnosed");
    expect(deriveTernaryLabel("control", marker({ rel3Count: 0 }), policy)).toBe("control");
  });

  test("derives quantile cutoffs from train diagnosed users only", () => {
    const config = {
      policyId: "diag_evidence_q50",
      kind: "evidence_quantile",
      quantile: 0.5,
      description: "test",
    } satisfies TernaryLabelPolicyConfig;
    const policy = lockTernaryLabelPolicy(
      config,
      [
        { binaryLabel: "diagnosed", marker: marker({ userId: "D1", evidenceScore: 0.1 }) },
        { binaryLabel: "diagnosed", marker: marker({ userId: "D2", evidenceScore: 0.3 }) },
        { binaryLabel: "diagnosed", marker: marker({ userId: "D3", evidenceScore: 0.9 }) },
        { binaryLabel: "control", marker: marker({ userId: "C1", evidenceScore: 0.0 }) },
      ],
      "manifest",
      42,
    );

    expect(policy.cutoff).toBe(0.3);
    expect(deriveTernaryLabel("diagnosed", marker({ evidenceScore: 0.2 }), policy)).toBe("no-evidence");
    expect(deriveTernaryLabel("diagnosed", marker({ evidenceScore: 0.5 }), policy)).toBe("diagnosed");
  });
});

describe("ternary metrics and artifacts", () => {
  test("computes macro and diagnosed metrics", () => {
    const actual: TernaryLabel[] = ["diagnosed", "control", "no-evidence"];
    const predicted: TernaryLabel[] = ["diagnosed", "no-evidence", "no-evidence"];
    const metrics = computeTernaryMetrics(actual, predicted);
    expect(metrics.diagnosedF1).toBe(1);
    expect(metrics.perClass.control.f1).toBe(0);
    expect(metrics.perClass["no-evidence"].f1).toBeCloseTo(2 / 3);
    expect(metrics.macroF1).toBeCloseTo(5 / 9);
  });

  test("rejects labels and invalid probabilities in ternary test scores", () => {
    expect(
      auditTernaryTestScoreSchema(
        "test_score_m.csv",
        "user_id,label,prob_diagnosed,prob_control,prob_no_evidence,model_id,label_policy_id\nU,control,0.2,0.3,0.5,m,p\n",
      ).ok,
    ).toBe(false);
    expect(
      auditTernaryTestScoreSchema(
        "test_score_m.csv",
        "user_id,prob_diagnosed,prob_control,prob_no_evidence,model_id,label_policy_id\nU,0.2,0.3,0.2,m,p\n",
      ).ok,
    ).toBe(false);
  });

  test("audits OOF rows against a ternary train manifest", () => {
    const manifestRows = ternaryManifest();
    const rows = ternaryRows("m1");
    const report = auditTernaryOofScores(manifestRows, new Set(["T"]), new Map([["m1", rows]]), "p");
    expect(report.ok).toBe(true);

    const badReport = auditTernaryOofScores(
      manifestRows,
      new Set(["T"]),
      new Map([["m1", [{ ...rows[0]!, userId: "T" }, ...rows.slice(1)]]]),
      "p",
    );
    expect(badReport.ok).toBe(false);
    expect(badReport.findings.some((finding) => finding.code === "ternary-oof-test-user")).toBe(true);
  });
});

describe("ternary ensemble selection", () => {
  test("selects and evaluates a locked ternary ensemble", () => {
    const oofRows = ternaryRows("m1");
    const lock = selectTernaryEnsemble({
      seed: 42,
      originalManifestHash: "manifest",
      labelPolicyId: "p",
      labelPolicyHash: "policy",
      oofByModel: new Map([["m1", oofRows]]),
      sourceHashes: { m1: "hash" },
      weightStep: 0.5,
      decisionRules: [{ ruleId: "argmax", kind: "argmax" }],
      command: "test",
    });
    const metrics = evaluateTernaryLockedEnsemble(
      lock,
      new Map([["m1", oofRows.map(({ userId, probDiagnosed, probControl, probNoEvidence, modelId, labelPolicyId }) => ({ userId, probDiagnosed, probControl, probNoEvidence, modelId, labelPolicyId }))]]),
      new Map([
        ["A", "diagnosed"],
        ["B", "control"],
        ["C", "no-evidence"],
      ]),
    );
    expect(lock.modelIds).toEqual(["m1"]);
    expect(metrics.macroF1).toBe(1);
  });

  test("locally refines ternary ensemble weights from train OOF only", () => {
    const labels: TernaryLabel[] = ["diagnosed", "control", "no-evidence", "diagnosed", "control", "no-evidence"];
    const m1 = ternaryRowsFromProbs("m1", labels, [
      [0.29970438728381416, 0.31041908736529367, 0.38987652535089234],
      [0.01119046301401629, 0.35448640458375646, 0.6343231324022272],
      [0.6794043259574124, 0.15850712114884224, 0.1620885528937454],
      [0.89445485545787, 0.0600389282089413, 0.04550621633318868],
      [0.7806317072012139, 0.08037286017607773, 0.13899543262270841],
      [0.2796146364241091, 0.19067766842510728, 0.5297076951507835],
    ]);
    const m2 = ternaryRowsFromProbs("m2", labels, [
      [0.4154874090003096, 0.26138437916822543, 0.323128211831465],
      [0.6788747376240226, 0.28682780475234443, 0.03429745762363305],
      [0.14732529578889636, 0.23177586339779552, 0.6208988408133082],
      [0.3306863689342943, 0.10747429370312139, 0.5618393373625844],
      [0.6056446494966663, 0.11871788459571377, 0.2756374659076199],
      [0.37756236476433863, 0.22997778254535808, 0.39245985269030326],
    ]);

    const coarse = selectTernaryEnsemble({
      seed: 42,
      originalManifestHash: "manifest",
      labelPolicyId: "p",
      labelPolicyHash: "policy",
      oofByModel: new Map([
        ["m1", m1],
        ["m2", m2],
      ]),
      sourceHashes: { m1: "hash1", m2: "hash2" },
      weightStep: 0.5,
      decisionRules: [{ ruleId: "argmax", kind: "argmax" }],
      command: "test",
    });
    const refined = selectTernaryEnsemble({
      seed: 42,
      originalManifestHash: "manifest",
      labelPolicyId: "p",
      labelPolicyHash: "policy",
      oofByModel: new Map([
        ["m1", m1],
        ["m2", m2],
      ]),
      sourceHashes: { m1: "hash1", m2: "hash2" },
      weightStep: 0.5,
      refineWeightStep: 0.25,
      refineWeightRadius: 0.5,
      refineModelLimit: 2,
      decisionRules: [{ ruleId: "argmax", kind: "argmax" }],
      command: "test",
    });

    expect(coarse.weights).toEqual({ m1: 0.5, m2: 0.5 });
    expect(refined.selectionStrategy).toContain("local-refine(step=0.25,radius=0.5)");
    expect(refined.weights).toEqual({ m1: 0.25, m2: 0.75 });
    expect(refined.oofMetrics.macroF1).toBeGreaterThan(coarse.oofMetrics.macroF1);
  });
});

function ternaryManifest(): TernaryManifestRow[] {
  return [
    { dataset: "setembrobr", split: "train", label: "diagnosed", binaryLabel: "diagnosed", userId: "A", rowHash: "a", fold: 1, labelPolicyId: "p" },
    { dataset: "setembrobr", split: "train", label: "control", binaryLabel: "control", userId: "B", rowHash: "b", fold: 2, labelPolicyId: "p" },
    { dataset: "setembrobr", split: "train", label: "no-evidence", binaryLabel: "diagnosed", userId: "C", rowHash: "c", fold: 3, labelPolicyId: "p" },
  ];
}

function ternaryRows(modelId: string): TernaryProbabilityRow[] {
  return [
    {
      userId: "A",
      label: "diagnosed",
      fold: 1,
      probDiagnosed: 0.8,
      probControl: 0.1,
      probNoEvidence: 0.1,
      modelId,
      labelPolicyId: "p",
    },
    {
      userId: "B",
      label: "control",
      fold: 2,
      probDiagnosed: 0.1,
      probControl: 0.8,
      probNoEvidence: 0.1,
      modelId,
      labelPolicyId: "p",
    },
    {
      userId: "C",
      label: "no-evidence",
      fold: 3,
      probDiagnosed: 0.1,
      probControl: 0.1,
      probNoEvidence: 0.8,
      modelId,
      labelPolicyId: "p",
    },
  ];
}

function ternaryRowsFromProbs(modelId: string, labels: readonly TernaryLabel[], probs: readonly (readonly [number, number, number])[]): TernaryProbabilityRow[] {
  return probs.map((prob, index) => ({
    userId: `U${index}`,
    label: labels[index]!,
    fold: index % 3,
    probDiagnosed: prob[0],
    probControl: prob[1],
    probNoEvidence: prob[2],
    modelId,
    labelPolicyId: "p",
  }));
}

function marker(overrides: Partial<EvidenceMarker> = {}): EvidenceMarker {
  return {
    userId: "U",
    totalTweets: 10,
    maxRelevance: 0,
    rel3Count: 0,
    rel5Count: 0,
    rel6Count: 0,
    rel7Count: 0,
    rel3Ratio: 0,
    rel5Ratio: 0,
    rel6Ratio: 0,
    rel7Ratio: 0,
    top10AvgRelevance: 0,
    evidenceScore: 0,
    ...overrides,
  };
}
