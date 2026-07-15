import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { validateNestedCrossFitRecords } from "../src/anxiety.ts";
import { loadConfig } from "../src/config.ts";
import { selectEnsemble } from "../src/ensemble.ts";
import type { ScoreRow } from "../src/types.ts";

const configPath = "configs/setembrobr.seed42.anxiety-temporal-champion-qwen3-binary.json";
const championIds = [
  "binary_legacy_focal_combined_g1",
  "binary_legacy_logreg_combined_s42",
  "binary_legacy_seq_cnn_top128_s13",
  "binary_stack_logreg_boosted_core",
  "binary_stack_logreg_xgb_tabular",
];

describe("strict-blind anxiety champion", () => {
  test("pins the target, raw provenance, proxy, model set, and chronological sequence", async () => {
    const config = await loadConfig(configPath);
    expect(config.predictionTarget).toBe("anxiety");
    expect(config.seed).toBe(42);
    expect(config.foldCount).toBe(5);
    expect(config.expectedUsers).toEqual({ train: 14200, test: 3552 });
    expect(config.rawEmbeddingManifestSha256).toBe("c8cd40827f91b298d61a3d2845bb4337c0ae3e8abbc9bf4fdfc552b1aad3c410");
    expect(config.rawSplitManifestSha256).toBe("9457cacad5c1d9e378bfc4aff8c3ac8acdcac5f2eb75ad0926780e7838f6fc61");
    expect(config.rawEmbeddingModelRevision).toBe("5cf2132abc99cad020ac570b19d031efec650f2b");
    expect(config.relevanceProxy?.kind).toBe("anxiety-lexical-v1");
    expect(config.relevanceProxy?.poolThresholds).toEqual([3, 6, 7]);
    expect(config.sequenceExport?.order).toBe("recent_chronological");
    expect(config.ensemble.selectionMode).toBe("fixed_model_set");
    expect(config.ensemble.weightStep).toBe(0.05);
    expect(config.ensemble.minimumWeight).toBe(0.05);
    expect(config.ensemble.requiredModelIds).toEqual(championIds);
    const supportIds = config.candidateModels?.stacking?.flatMap((model) => model.baseModelIds) ?? [];
    expect(new Set(supportIds)).toEqual(
      new Set([
        "binary_hier_logreg_gate",
        "binary_mlp_h128_s42",
        "binary_xgb_expanded_pca_s13",
        "binary_xgb_tabular_markers_s42",
        "binary_xgb_shallow_pca_s99",
      ]),
    );
  });

  test("implements deterministic accent-insensitive lexical boundary scores and source guards", async () => {
    const source = String.raw`
import json, tempfile
from pathlib import Path
import numpy as np
from anxiety_champion_relevance import anxiety_relevance_score, alignment_hash, sha256_file, verify_raw_artifacts

samples = {
  "zero": "que dia bonito",
  "three": "ansiedade",
  "six": "eu estou com ansiedade",
  "seven_plus": "Eu estou com PÂNICO e coração acelerado agora, preciso de ajuda",
  "accent_lower": "pânico e coração acelerado",
  "accent_upper": "PÂNICO E CORAÇÃO ACELERADO",
  "third_party": "minha mãe está com ansiedade e falta de ar agora, procure ajuda",
}
scores = {key: anxiety_relevance_score(value) for key, value in samples.items()}
users = ["a", "b"]
indexes = [3, 9]
alignment = alignment_hash(users, indexes)
with tempfile.TemporaryDirectory() as tmp:
  raw = Path(tmp)
  (raw / "reports").mkdir()
  (raw / "manifests").mkdir()
  (raw / "reports" / "embedding_generation_manifest.json").write_text(json.dumps({"embedding": {"modelRevision": "rev"}}))
  (raw / "manifests" / "raw_split_manifest_seed42.csv").write_text("x\\n")
  cfg = {
    "seed": 42,
    "rawEmbeddingManifestSha256": sha256_file(raw / "reports" / "embedding_generation_manifest.json"),
    "rawSplitManifestSha256": sha256_file(raw / "manifests" / "raw_split_manifest_seed42.csv"),
    "rawEmbeddingModelRevision": "rev",
  }
  verify_raw_artifacts(cfg, raw)
  (raw / "manifests" / "raw_split_manifest_seed42.csv").write_text("corrupt\\n")
  rejected = False
  try:
    verify_raw_artifacts(cfg, raw)
  except RuntimeError:
    rejected = True
print(json.dumps({"scores": scores, "alignment": alignment, "repeat": alignment_hash(users, indexes), "rejected": rejected}))
`;
    const child = Bun.spawn([Bun.env.PYTHON ?? "python3", "-c", source], {
      env: { ...Bun.env, PYTHONPATH: "scripts" },
      stdout: "pipe",
      stderr: "pipe",
    });
    const stdout = await new Response(child.stdout).text();
    const stderr = await new Response(child.stderr).text();
    expect(await child.exited, stderr).toBe(0);
    const result = JSON.parse(stdout) as {
      scores: Record<string, number>;
      alignment: string;
      repeat: string;
      rejected: boolean;
    };
    expect(result.scores.zero).toBe(0);
    expect(result.scores.three).toBe(3);
    expect(result.scores.six).toBe(6);
    expect(result.scores.seven_plus).toBeGreaterThanOrEqual(7);
    expect(result.scores.accent_lower).toBe(result.scores.accent_upper);
    expect(result.scores.third_party).toBeLessThan(7);
    expect(result.alignment).toBe(result.repeat);
    expect(result.alignment).toHaveLength(64);
    expect(result.rejected).toBe(true);
  });

  test("rejects nested base fits containing either validation fold", () => {
    expect(
      validateNestedCrossFitRecords([
        { outerFold: 1, innerValidationFold: 2, fitFolds: [3, 4, 5] },
        { outerFold: 1, innerValidationFold: 3, fitFolds: [2, 4, 5] },
      ]),
    ).toBe(true);
    expect(validateNestedCrossFitRecords([{ outerFold: 1, innerValidationFold: 2, fitFolds: [1, 3, 4] }])).toBe(false);
    expect(validateNestedCrossFitRecords([{ outerFold: 1, innerValidationFold: 2, fitFolds: [2, 3, 4] }])).toBe(false);
  });

  test("locks exactly five positive step-aligned weights from OOF", () => {
    const oofByModel = new Map<string, ScoreRow[]>();
    for (const [modelIndex, modelId] of championIds.entries()) {
      oofByModel.set(
        modelId,
        Array.from({ length: 20 }, (_, index) => ({
          userId: `u${index}`,
          label: index % 4 === 0 ? "diagnosed" as const : "control" as const,
          fold: (index % 5) + 1,
          score: Math.min(0.99, Math.max(0.01, (index % 4 === 0 ? 0.75 : 0.25) + modelIndex * 0.01)),
          modelId,
        })),
      );
    }
    const lock = selectEnsemble({
      seed: 42,
      predictionTarget: "anxiety",
      manifestHash: "strict",
      oofByModel,
      sourceHashes: Object.fromEntries(championIds.map((modelId) => [modelId, "hash"])),
      weightStep: 0.05,
      minimumWeight: 0.05,
      selectionMode: "fixed_model_set",
      requiredModelIds: championIds,
      command: "test",
    });
    expect(lock.predictionTarget).toBe("anxiety");
    expect(lock.modelIds).toEqual([...championIds].sort());
    expect(Object.values(lock.weights).reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 12);
    for (const weight of Object.values(lock.weights)) {
      expect(weight).toBeGreaterThanOrEqual(0.05);
      expect(weight / 0.05).toBeCloseTo(Math.round(weight / 0.05), 12);
    }
    expect(lock.threshold).toBeNumber();
    expect(lock.selectionStrategy).toContain("fixed-model-set");
  });

  test("keeps OOF, test scoring, and sealed evaluation in separate stages", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const oof = extractTarget(makefile, "anxiety-champion-oof-setembrobr");
    const score = extractTarget(makefile, "anxiety-champion-score-test-setembrobr");
    const evaluate = extractTarget(makefile, "anxiety-champion-evaluate-test-setembrobr");
    expect(oof).toContain("--split train");
    expect(oof).toContain("--stage oof");
    expect(oof).toContain("--mode=oof");
    expect(oof).not.toContain("--split test");
    expect(oof).not.toContain("--stage score-test");
    expect(oof).not.toContain("anxiety_champion_evaluate.py");
    expect(score).toContain("--split test");
    expect(score).toContain("--stage score-test");
    expect(score).toContain("--mode=test");
    expect(score).not.toContain("anxiety_champion_evaluate.py");
    expect(evaluate).toContain("anxiety_champion_evaluate.py");
    expect(evaluate).toContain("docs-lock-results-summary");
    expect(extractTarget(makefile, "fedora-anxiety-champion-oof-setembrobr")).toContain("FEDORA_HOST=fedora.local");
    expect(extractTarget(makefile, "fedora-anxiety-champion-score-test-setembrobr")).toContain("FEDORA_HOST=fedora.local");
    expect(extractTarget(makefile, "fedora-anxiety-champion-evaluate-test-setembrobr")).toContain("FEDORA_HOST=fedora.local");
    const fedoraOof = extractTarget(makefile, "_fedora-anxiety-champion-oof-setembrobr");
    expect(fedoraOof).toContain("anxiety-champion-oof-setembrobr");
    expect(fedoraOof).toContain("--exclude 'work/'");
    expect(fedoraOof).toContain("--exclude 'checkpoints/'");
  });
});

function extractTarget(makefile: string, targetName: string): string {
  const pattern = new RegExp(`^${targetName}:[\\s\\S]*?(?=\\n[A-Za-z0-9_.-]+:|\\n$)`, "mu");
  const match = makefile.match(pattern);
  if (!match) throw new Error(`missing Makefile target ${targetName}`);
  return match[0]!;
}
