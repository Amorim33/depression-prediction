import { describe, expect, test } from "bun:test";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

describe("lock results summary docs", () => {
  test("defines a deterministic docs renderer target", async () => {
    const makefile = await readFile("Makefile", "utf8");
    const packageJson = JSON.parse(await readFile("package.json", "utf8")) as { scripts?: Record<string, string> };

    expect(makefile).toContain("docs-lock-results-summary:");
    expect(makefile).toContain("bun run docs-lock-results-summary");
    expect(packageJson.scripts?.["docs-lock-results-summary"]).toBe("bun run scripts/render-lock-results-summary.ts");
  });

  test("documents every current SetembroBR lock or final-report family", async () => {
    const html = await readFile("docs/lock-results-summary.html", "utf8");
    const families = await collectResultFamilies("outputs/setembrobr");

    expect(families.size).toBeGreaterThan(0);
    for (const family of [...families].sort()) {
      expect(html).toContain(family);
    }
  });

  test("documents temporal-relevance and full LLM metrics exactly", async () => {
    const html = await readFile("docs/lock-results-summary.html", "utf8");
    const lock = JSON.parse(
      await readFile("outputs/setembrobr/seed42_temporal_relevance_qwen3_binary/ensemble/ensemble-lock.json", "utf8"),
    ) as { oofMetrics: { macroF1: number } };
    const llmReport = JSON.parse(
      await readFile(
        "outputs/setembrobr/seed42_temporal_relevance_qwen3_binary/reports/final-test-report-llm-disambiguated.json",
        "utf8",
      ),
    ) as { testMetrics: { macroF1: number }; llmDisambiguator: { decisionCount: number; switchedToControl: number } };

    expect(html).toContain(lock.oofMetrics.macroF1.toFixed(6));
    expect(html).toContain(llmReport.testMetrics.macroF1.toFixed(6));
    expect(html).toContain(String(llmReport.llmDisambiguator.decisionCount));
    expect(html).toContain(String(llmReport.llmDisambiguator.switchedToControl));
  });

  test("includes the temporal FP/FN exploration taxonomy", async () => {
    const html = await readFile("docs/lock-results-summary.html", "utf8");

    expect(html).toContain("Exploração temporal de FP / FN");
    expect(html).toContain("Controles parecidos com verdadeiros positivos");
    expect(html).toContain("Sinal passado, fechamento recente tipo controle");
    expect(html).toContain("Falsos alarmes lexicais");
    expect(html).toContain("Contaminação por terceiros/conta de apoio");
    expect(html).toContain("Pouca evidência visível de depressão");
    expect(html).toContain("Tom depressivo indireto");
    expect(html).toContain("Evidência forte anterior, silêncio recente");
    expect(html).toContain("Alta relevância em contexto não clínico");
  });

  test("keeps docs copy and output mirror identical", async () => {
    const docs = await readFile("docs/lock-results-summary.html", "utf8");
    const mirror = await readFile("outputs/setembrobr/lock-results-summary.html", "utf8");
    expect(docs).toBe(mirror);
  });
});

async function collectResultFamilies(root: string): Promise<Set<string>> {
  const out = new Set<string>();
  for (const entry of await readdir(root, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const experimentDir = join(root, entry.name);
    const hasLock = await hasFileUnder(join(experimentDir, "ensemble"), (name) => name.endsWith(".json"));
    const hasFinalReport = await hasFileUnder(join(experimentDir, "reports"), (name) => name.includes("final-test-report") && name.endsWith(".json"));
    if (hasLock || hasFinalReport) out.add(entry.name);
  }
  return out;
}

async function hasFileUnder(dir: string, predicate: (name: string) => boolean): Promise<boolean> {
  const entries = await readdir(dir, { withFileTypes: true }).catch(() => []);
  return entries.some((entry) => entry.isFile() && predicate(entry.name));
}
