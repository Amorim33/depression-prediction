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

  test("documents anxiety OOF and sealed-test results without depression ranking", async () => {
    const html = await readFile("docs/lock-results-summary.html", "utf8");
    const binarySection = html.slice(html.indexOf('<section id="binary">'), html.indexOf('<section id="ternary">'));
    const anxietySection = html.slice(html.indexOf('<section id="anxiety">'), html.indexOf('<section id="llm">'));

    expect(anxietySection).toContain("Resultados de ansiedade");
    expect(anxietySection).toContain("0.679394");
    expect(anxietySection).toContain("0.662850");
    expect(anxietySection).toContain("0.414163");
    expect(anxietySection).toContain("0.395492");
    expect(anxietySection).toContain("0.434685");
    expect(anxietySection).toContain("VP 193");
    expect(anxietySection).toContain("FP 295");
    expect(anxietySection).toContain("VN 2813");
    expect(anxietySection).toContain("FN 251");
    expect(anxietySection).toContain('<span class="chip">Focal LogReg</span><span class="chip">LogReg</span><span class="chip">CNN</span><span class="chip">Stacking ×2</span>');
    expect(binarySection).not.toContain("— ansiedade");
    expect(html).toContain("Este resultado não participa dos rankings de depressão");
  });

  test("documents measured anxiety embedding time and Fedora hardware provenance", async () => {
    const html = await readFile("docs/lock-results-summary.html", "utf8");
    const provenance = JSON.parse(
      await readFile("docs/anxiety-embedding-generation-provenance.json", "utf8"),
    ) as {
      timing: { startedAt: string; completedAt: string; durationSeconds: number };
      hardware: { cpu: string; gpu: string; gpuMemoryMiB: number; memoryGiB: number; driverVersion: string; cudaVersion: string };
      workload: { totalTweets: number };
      sourceEvidence: { rawEmbeddingManifestSha256: string; jobExitCode: number };
    };
    const finalReport = JSON.parse(
      await readFile(
        "outputs/setembrobr/seed42_anxiety_temporal_champion_qwen3_binary/reports/final-test-report.json",
        "utf8",
      ),
    ) as { artifactHashes: { rawEmbeddingManifestSha256: string } };

    const elapsedSeconds = Math.round(
      (Date.parse(provenance.timing.completedAt) - Date.parse(provenance.timing.startedAt)) / 1000,
    );
    expect(provenance.timing.durationSeconds).toBe(elapsedSeconds);
    expect(provenance.sourceEvidence.rawEmbeddingManifestSha256).toBe(
      finalReport.artifactHashes.rawEmbeddingManifestSha256,
    );
    expect(provenance.sourceEvidence.jobExitCode).toBe(0);
    expect(provenance.workload.totalTweets).toBe(27_408_040);
    expect(html).toContain("Geração dos embeddings de ansiedade");
    expect(html).toContain("52h 13min 22s");
    expect(html).toContain("2026-07-12 11:39:03 BRT");
    expect(html).toContain("2026-07-14 15:52:25 BRT");
    expect(html).toContain(provenance.hardware.gpu);
    expect(html).toContain(`${provenance.hardware.gpuMemoryMiB.toLocaleString("pt-BR")} MiB`);
    expect(html).toContain(provenance.hardware.cpu);
    expect(html).toContain(`${provenance.hardware.memoryGiB} GiB`);
    expect(html).toContain(`Driver ${provenance.hardware.driverVersion} · CUDA ${provenance.hardware.cudaVersion}`);
    expect(html).toContain("docs/anxiety-embedding-generation-provenance.json");
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
