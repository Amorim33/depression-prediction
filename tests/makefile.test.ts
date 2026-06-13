import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";

describe("Makefile Fedora targets", () => {
  test("defines the strict-blind ternary GPU sequence path", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const target = extractTarget(makefile, "fedora-ternary-train-seq-oof-setembrobr");

    expect(makefile).toContain("FEDORA_TERNARY_RUN_DIR ?= ~/codex-runs/depression-prediction-setembrobr-ternary");
    expect(target).toContain("ssh $(FEDORA_HOST) 'nvidia-smi'");
    expect(target).not.toContain("127.0.0.1");
    expect(target).toContain("rsync -azR requirements.txt scripts/ternary_train_seq_oof_setembrobr.py scripts/write_gpu_run_manifest.py $(TERNARY_CONFIG)");
    expect(target).toContain("outputs/setembrobr/seed42_strict_blind/sequences/top128");
    expect(target).not.toContain("outputs/setembrobr/seed42_strict_blind/sequences ");
    expect(target).toContain("outputs/setembrobr/seed42_ternary_strict_blind/manifest");
    expect(target).toContain("outputs/setembrobr/seed42_ternary_strict_blind/evidence-markers");
    expect(target).toContain("outputs/setembrobr/seed42_ternary_strict_blind/label-policies");
    expect(target).toContain("python scripts/ternary_train_seq_oof_setembrobr.py --config $(TERNARY_CONFIG)");
    expect(target).toContain("python scripts/write_gpu_run_manifest.py --config $(TERNARY_CONFIG) --host-label $(FEDORA_HOST)");
    expect(target).toContain("outputs/setembrobr/seed42_ternary_strict_blind/scores/");
    expect(target).toContain("outputs/setembrobr/seed42_ternary_strict_blind/model-manifests/");
    expect(target).toContain("outputs/setembrobr/seed42_ternary_strict_blind/gpu-runs/");
  });

  test("runs local selection and evaluation after Fedora sequence training", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const target = extractTarget(makefile, "reproduce-ternary-setembrobr-gpu");

    expect(target).toContain("lint");
    expect(target).toContain("typecheck");
    expect(target).toContain("test");
    expect(target).toContain("db-check-setembrobr");
    expect(target).toContain("ternary-train-tabular-oof-setembrobr");
    expect(target).toContain("fedora-ternary-train-seq-oof-setembrobr");
    expect(target).toContain("ternary-audit-oof-setembrobr");
    expect(target).toContain("ternary-select-ensemble-setembrobr");
    expect(target).toContain("ternary-robustness-setembrobr");
    expect(target).toContain("ternary-nested-oof-selection-setembrobr");
    expect(target).toContain("ternary-evaluate-test-setembrobr");
    expect(target.indexOf("ternary-robustness-setembrobr")).toBeLessThan(target.indexOf("ternary-evaluate-test-setembrobr"));
    expect(target.indexOf("ternary-nested-oof-selection-setembrobr")).toBeLessThan(target.indexOf("ternary-evaluate-test-setembrobr"));
  });

  test("defines train-only ternary robustness before final test evaluation", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const target = extractTarget(makefile, "ternary-robustness-setembrobr");

    expect(target).toContain("bun run ternary-robustness-setembrobr");
    expect(target).not.toContain("ternary-evaluate-test-setembrobr");
    expect(target).not.toContain("test_score");
  });

  test("defines train-only nested OOF split selection before final test evaluation", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const target = extractTarget(makefile, "ternary-nested-oof-selection-setembrobr");

    expect(target).toContain("bun run ternary-nested-oof-selection-setembrobr");
    expect(target).not.toContain("ternary-evaluate-test-setembrobr");
    expect(target).not.toContain("test_score");
  });
});

function extractTarget(makefile: string, targetName: string): string {
  const pattern = new RegExp(`^${escapeRegExp(targetName)}:[\\s\\S]*?(?=\\n[A-Za-z0-9_.-]+:|\\n$)`, "mu");
  const match = makefile.match(pattern);
  if (!match) throw new Error(`Missing Makefile target: ${targetName}`);
  return match[0]!;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}
