import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

const outputRoot = "outputs/setembrobr";
const docsPath = "docs/lock-results-summary.html";
const mirrorPath = "outputs/setembrobr/lock-results-summary.html";
const anxietyEmbeddingProvenancePath = "docs/anxiety-embedding-generation-provenance.json";
const anxietyThesisComparisonPath = "docs/anxiety-thesis-comparison.json";
const rawBinaryBaselineMacro = 0.6987181018176564;

type JsonRecord = Record<string, unknown>;

interface NormalizedMetrics {
  macroF1: number;
  diagnosedF1: number;
  controlF1: number | null;
  precision: number;
  recall: number;
  accuracy: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

type Track = "binary" | "ternary" | "anxiety";

interface LockEntry {
  experimentId: string;
  experimentTitle: string;
  lockBasename: string;
  lockPath: string;
  track: Track;
  modelCount: number;
  modelIds: string[];
  weights: Record<string, number>;
  policy: string;
  group: string;
  groupDescription: string;
  decision: string;
  selection: string;
  oof: NormalizedMetrics | null;
}

interface ResultEntry {
  experimentId: string;
  experimentTitle: string;
  lockBasename: string;
  variant: string;
  kind: "test" | "llm" | "cached-llm";
  reportPath: string;
  lockPath: string | null;
  track: Track;
  modelCount: number;
  modelIds: string[];
  policy: string;
  group: string;
  decision: string;
  selection: string;
  oof: NormalizedMetrics | null;
  baseTest: NormalizedMetrics | null;
  test: NormalizedMetrics;
  llm: { decisionCount: number; switchedToControl: number } | null;
}

interface ArtifactGroup {
  experimentId: string;
  title: string;
  files: string[];
}

interface EmbeddingGenerationProvenance {
  timing: {
    startedAt: string;
    completedAt: string;
    durationSeconds: number;
    timeZone: string;
    measurement: string;
  };
  hardware: {
    hostname: string;
    operatingSystem: string;
    kernel: string;
    cpu: string;
    cpuCores: number;
    cpuThreads: number;
    memoryGiB: number;
    gpu: string;
    gpuMemoryMiB: number;
    driverVersion: string;
    cudaVersion: string;
  };
  workload: {
    modelId: string;
    modelRevision: string;
    embeddingDimension: number;
    storageDtype: string;
    batchSize: number;
    trainUsers: number;
    testUsers: number;
    trainTweets: number;
    testTweets: number;
    totalTweets: number;
  };
  sourceEvidence: {
    rawEmbeddingManifestSha256: string;
    rawValidationReportSha256: string;
    jobExitCode: number;
    jobStatus: string;
  };
}

interface AnxietyThesisComparison {
  comparisonId: string;
  predictionTarget: "anxiety";
  source: {
    author: string;
    title: string;
    year: number;
    file: string;
    sha256: string;
    table: number;
    printedPage: number;
    pdfPage: number;
    methodPrintedPages: number[];
    methodPdfPages: number[];
  };
  metricDefinitions: {
    controlF1: string;
    anxietyF1: string;
    macroF1: string;
    macroF1Formula: string;
    precisionRecall: string;
  };
  reportedMethod: {
    timeline: string;
    bertInference: string;
    postTruncationTokens: number;
    relevance: string;
    mentionSignal: string;
  };
  reportedResults: Array<{
    model: string;
    precisionAverage: number;
    recallAverage: number;
    controlF1: number;
    anxietyF1: number;
    macroF1: number;
  }>;
  caveats: string[];
}

const experimentTitles: Record<string, string> = {
  seed42_strict_blind: "Embeddings legados strict-blind (binário)",
  seed42_ternary_strict_blind: "Embeddings legados strict-blind (ternário)",
  seed42_raw_qwen3_binary: "Qwen3 bruto binário (baseline)",
  seed42_relevance_features_qwen3_binary: "Qwen3 bruto binário com canal de relevância",
  seed42_temporal_relevance_qwen3_binary: "Qwen3 bruto binário com relevância temporal",
  seed42_anxiety_temporal_champion_qwen3_binary: "Qwen3 bruto binário com relevância temporal — ansiedade",
  seed42_raw_qwen3_ternary_diagnosed_only: "Qwen3 bruto ternário (apenas diagnosticados)",
  seed42_raw_qwen3_ternary_symmetric: "Qwen3 bruto ternário simétrico",
  seed42_raw_qwen3_embeddings: "Artefatos de embeddings Qwen3 bruto",
};

const methodIdeas: Array<{ title: string; body: string; signal: string }> = [
  {
    title: "Embeddings legados strict-blind",
    signal: "Baseline inicial sem vazamento",
    body:
      "Essas execuções comprovaram o contrato strict-blind e expuseram que a configuração antiga de embeddings/rótulos não generalizava bem para o teste selado.",
  },
  {
    title: "Qwen3 bruto binário",
    signal: "Enquadramento direto diagnosticado/controle",
    body:
      "A trilha binária bruta usa artefatos Qwen3 regenerados, trava de ensemble por OOF de treino e scores de teste sem rótulo. É o principal baseline binário comparável para os experimentos seguintes.",
  },
  {
    title: "Sequências com canal de relevância",
    signal: "Expor a relevância por tweet do GPT-3.5 aos modelos de sequência",
    body:
      "Testou se os modelos de sequência deveriam ver o score de relevância como um canal de entrada. Melhorou levemente o macro F1 OOF em relação à trava binária bruta.",
  },
  {
    title: "Relevância temporal",
    signal: "Separar sinal histórico do risco recente",
    body:
      "Adiciona agregados de recência/tom e troca a exportação de sequência para a janela cronológica mais recente, mirando diretamente as evidências de FP/FN da inspeção das linhas do tempo.",
  },
  {
    title: "Políticas ternárias",
    signal: "Estrutura diagnosticado/controle/sem-evidência",
    body:
      "As trilhas ternárias testam se políticas de rótulo sem-evidência e simétricas tornam a seleção mais robusta. A política simétrica somada à arquitetura legada segue sendo o melhor macro F1 de teste selado do repositório.",
  },
  {
    title: "Desambiguação de FP por LLM",
    signal: "Pós-filtro unidirecional de diagnosticado para controle",
    body:
      "O LLM não é um segundo classificador. Ele apenas rejeita usuários previstos como diagnosticados quando a evidência da linha do tempo parece não clínica, de terceiros ou de cunho lexical/fandom.",
  },
  {
    title: "Campeão temporal transferido para ansiedade",
    signal: "Mesmas cinco famílias, nova trava treinada apenas no OOF de ansiedade",
    body:
      "A composição Focal LogReg, LogReg, CNN e dois stackers foi mantida fixa. Pesos e limiar foram selecionados no OOF de treino de ansiedade, com cross-fitting aninhado dos stackers e teste aberto somente após a trava.",
  },
];

const locks = await collectLocks();
const results = await collectResults(locks);
const artifacts = await collectArtifacts();
const anxietyEmbeddingProvenance = await readEmbeddingGenerationProvenance();
const anxietyThesisComparison = await readAnxietyThesisComparison();
const html = renderPage(locks, results, artifacts, anxietyEmbeddingProvenance, anxietyThesisComparison);
await writeFileEnsured(docsPath, html);
await writeFileEnsured(mirrorPath, html);
console.log(JSON.stringify({ docsPath, mirrorPath, lockCount: locks.length, resultCount: results.length }, null, 2));

async function collectLocks(): Promise<LockEntry[]> {
  const paths = await collectFiles(outputRoot, (path) => /\/ensemble\/.+\.json$/u.test(path));
  const entries = await Promise.all(paths.map(readLockEntry));
  return entries.filter((entry): entry is LockEntry => entry !== null).sort(compareLock);
}

async function readLockEntry(path: string): Promise<LockEntry | null> {
  const json = await readJson(path);
  const oof = normalizeMetrics(recordValue(json, "oofMetrics") ?? recordValue(json, "overallMetrics"));
  if (!oof) return null;
  const experimentId = experimentFromPath(path);
  const lockBasename = basenameWithoutJson(path);
  const modelIds = stringArray(arrayValue(json, "modelIds") ?? arrayValue(json, "selectedModelIds") ?? []);
  return {
    experimentId,
    experimentTitle: titleFor(experimentId),
    lockBasename,
    lockPath: path,
    track: trackFor(experimentId),
    modelCount: modelIds.length,
    modelIds,
    weights: weightMap(recordValue(json, "weights")),
    policy: stringValue(json, "labelPolicyId") ?? binaryPolicy(experimentId),
    group: stringValue(json, "selectionGroupId") ?? stringValue(json, "selectionStrategy") ?? "locked ensemble",
    groupDescription: stringValue(json, "selectionGroupDescription") ?? "",
    decision: decisionFromLock(json),
    selection: stringValue(json, "selectionStrategy") ?? "locked",
    oof,
  };
}

async function collectResults(lockEntries: readonly LockEntry[]): Promise<ResultEntry[]> {
  const paths = await collectFiles(outputRoot, (path) => /\/reports\/.*final-test-report.*\.json$/u.test(path));
  const entries = await Promise.all(paths.map((path) => readResultEntry(path, lockEntries)));
  return entries.filter((entry): entry is ResultEntry => entry !== null).sort(compareResult);
}

async function readResultEntry(path: string, lockEntries: readonly LockEntry[]): Promise<ResultEntry | null> {
  const json = await readJson(path);
  const test = normalizeMetrics(recordValue(json, "testMetrics"));
  if (!test) return null;

  const experimentId = experimentFromPath(path);
  const lockBasename = lockBasenameFromReport(path);
  const embeddedLock = recordValue(json, "lock");
  const lock = lockEntries.find((entry) => entry.experimentId === experimentId && entry.lockBasename === lockBasename);
  const modelIds = embeddedLock ? stringArray(arrayValue(embeddedLock, "modelIds") ?? []) : [];
  const resolvedModelIds = modelIds.length ? modelIds : (lock?.modelIds ?? []);
  const llm = recordValue(json, "llmDisambiguator");
  const kind = path.includes("cached-llm") ? "cached-llm" : llm ? "llm" : "test";
  return {
    experimentId,
    experimentTitle: titleFor(experimentId),
    lockBasename,
    variant: resultVariant(path),
    kind,
    reportPath: path,
    lockPath: lock?.lockPath ?? null,
    track: trackFor(experimentId),
    modelCount: resolvedModelIds.length || lock?.modelCount || 0,
    modelIds: resolvedModelIds,
    policy: (embeddedLock && stringValue(embeddedLock, "labelPolicyId")) ?? lock?.policy ?? binaryPolicy(experimentId),
    group: (embeddedLock && stringValue(embeddedLock, "selectionGroupId")) ?? lock?.group ?? "locked ensemble",
    decision: embeddedLock ? decisionFromLock(embeddedLock) : (lock?.decision ?? "regra travada"),
    selection: (embeddedLock && stringValue(embeddedLock, "selectionStrategy")) ?? lock?.selection ?? "locked",
    oof: embeddedLock ? normalizeMetrics(recordValue(embeddedLock, "oofMetrics")) : (lock?.oof ?? null),
    baseTest: normalizeMetrics(recordValue(json, "baseTestMetrics")),
    test,
    llm: llm
      ? {
          decisionCount: numberValue(llm, "decisionCount") ?? 0,
          switchedToControl: numberValue(llm, "switchedToControl") ?? 0,
        }
      : null,
  };
}

async function collectArtifacts(): Promise<ArtifactGroup[]> {
  const outputPaths = await collectFiles(outputRoot, isDurableResultArtifact);
  const configPaths = [
    ...await collectFiles("configs", (path) => path.endsWith(".json")),
    ...await collectFiles("ternary-classification/configs", (path) => path.endsWith(".json")),
  ];
  const byExperiment = new Map<string, string[]>();
  for (const path of outputPaths) {
    const experimentId = experimentFromPath(path);
    const list = byExperiment.get(experimentId) ?? [];
    list.push(path);
    byExperiment.set(experimentId, list);
  }
  const groups = [...byExperiment.entries()]
    .map(([experimentId, files]) => ({
      experimentId,
      title: titleFor(experimentId),
      files: files.sort(),
    }))
    .sort((left, right) => left.title.localeCompare(right.title));
  groups.unshift({
    experimentId: "setembrobr",
    title: "Proveniência documental de ansiedade",
    files: [anxietyEmbeddingProvenancePath, anxietyThesisComparisonPath],
  });
  groups.unshift({ experimentId: "configs", title: "Arquivos de configuração", files: configPaths.sort() });
  return groups;
}

function isDurableResultArtifact(path: string): boolean {
  if (path.endsWith("lock-results-summary.html")) return true;
  if (/\/ensemble\/.+\.json$/u.test(path)) return true;
  if (/\/reports\/.+\.json$/u.test(path)) return true;
  if (/\/llm-disambiguator\/test_decisions_.+\.csv$/u.test(path)) return true;
  if (/\/llm-disambiguator\/prompt-lock\.json$/u.test(path)) return true;
  if (/\/llm-disambiguator\/cache-seed-manifest\.json$/u.test(path)) return true;
  if (/\/manifest\/.+\.csv$/u.test(path)) return true;
  if (/\/sequences\/.+\/sequence_manifest\.json$/u.test(path)) return true;
  return false;
}

async function collectFiles(dir: string, predicate: (path: string) => boolean): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true }).catch(() => []);
  const out: string[] = [];
  for (const entry of entries) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...await collectFiles(path, predicate));
    else if (entry.isFile() && predicate(path)) out.push(path);
  }
  return out.sort();
}

function renderPage(
  lockEntries: readonly LockEntry[],
  resultEntries: readonly ResultEntry[],
  artifactGroups: readonly ArtifactGroup[],
  anxietyEmbeddingProvenance: EmbeddingGenerationProvenance,
  anxietyThesisComparison: AnxietyThesisComparison,
): string {
  const navLinks: Array<{ id: string; label: string }> = [
    { id: "overview", label: "Visão geral" },
    { id: "architecture", label: "Arquitetura" },
    { id: "binary", label: "Trilha binária" },
    { id: "ternary", label: "Trilha ternária" },
    { id: "anxiety", label: "Ansiedade" },
    { id: "llm", label: "Filtro LLM" },
    { id: "methods", label: "Lições" },
    { id: "temporal", label: "FP / FN" },
    { id: "artifacts", label: "Artefatos" },
  ];
  return `<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SetembroBR — Resumo dos Resultados Travados</title>
  <style>${css()}</style>
</head>
<body>
  <nav>
    <div class="wrap">
      <a class="brand" href="#overview"><span class="dot"></span>SetembroBR · Travas</a>
      <div class="links">
        ${navLinks.map((link) => `<a href="#${link.id}">${escapeHtml(link.label)}</a>`).join("\n        ")}
      </div>
    </div>
  </nav>
  <main>
    ${overviewSection(resultEntries, lockEntries)}
    ${architectureSection()}
    ${trackSection("binary", lockEntries, resultEntries)}
    ${trackSection("ternary", lockEntries, resultEntries)}
    ${trackSection("anxiety", lockEntries, resultEntries, anxietyEmbeddingProvenance, anxietyThesisComparison)}
    ${renderLlmSection(resultEntries)}
    ${methodSection()}
    ${renderTemporalEvidenceSection()}
    ${renderArtifacts(artifactGroups)}
    <section id="regen">
      <p class="kicker">Reproduzir</p>
      <h2>Regeneração</h2>
      <p class="col">Rode <code class="k">make docs-lock-results-summary</code> após produzir novas travas ou relatórios finais. A cópia em docs e o espelho de saída são propositalmente idênticos byte a byte.</p>
    </section>
    <footer>Retreinamento strict-blind do SetembroBR · gerado a partir de <code class="k">outputs/setembrobr</code>.</footer>
  </main>
  ${scrollSpyScript()}
</body>
</html>
`;
}

function overviewSection(resultEntries: readonly ResultEntry[], lockEntries: readonly LockEntry[]): string {
  const finalResults = resultEntries.filter((entry) => entry.kind !== "cached-llm");
  const bestBinary = bestBy(finalResults.filter((entry) => entry.track === "binary"), (entry) => entry.test.macroF1);
  const bestTernary = bestBy(finalResults.filter((entry) => entry.track === "ternary"), (entry) => entry.test.macroF1);
  const bestAnxiety = bestBy(finalResults.filter((entry) => entry.track === "anxiety"), (entry) => entry.test.macroF1);
  const temporalBase = resultEntries.find(
    (entry) => entry.experimentId === "seed42_temporal_relevance_qwen3_binary" && entry.kind === "test",
  );
  const temporalLlm = resultEntries.find(
    (entry) => entry.experimentId === "seed42_temporal_relevance_qwen3_binary" && entry.kind === "llm",
  );
  const raw = resultEntries.find((entry) => entry.experimentId === "seed42_raw_qwen3_binary" && entry.kind === "test");
  const temporalLift = temporalBase && raw ? temporalBase.test.macroF1 - raw.test.macroF1 : null;
  const llmLift = temporalLlm && temporalBase ? temporalLlm.test.macroF1 - temporalBase.test.macroF1 : null;
  return `<section id="overview">
  <div class="col">
    <p class="kicker">SetembroBR · relatório de travas strict-blind</p>
    <h1>Resultados travados do SetembroBR, <span class="muted">depressão e ansiedade reportadas em separado.</span></h1>
    <p class="lead">Gerado a partir dos artefatos do repositório em <code class="k">outputs/setembrobr</code>. As métricas OOF explicam a seleção de modelos; as métricas de teste selado são reportadas separadamente, e ansiedade nunca é ranqueada contra as trilhas de depressão.</p>
  </div>
  <div class="meta">
    <div class="row"><span class="lab">Seed</span><span class="val">42</span></div>
    <div class="row"><span class="lab">Fonte</span><span class="val"><code class="k">outputs/setembrobr</code></span></div>
    <div class="row"><span class="lab">Travas</span><span class="val">${lockEntries.length} travas de ensemble · ${finalResults.length} relatórios de teste selado</span></div>
    <div class="row"><span class="lab">Métrica</span><span class="val">Macro F1 (primária) · F1 / precisão / recall da classe positiva</span></div>
  </div>
  <div class="grid two">
    <div class="scard">
      <div class="head"><span class="nm">Ansiedade</span><span class="tag">strict-blind independente</span></div>
      <p>Macro F1 de teste selado <strong>${metric(bestAnxiety?.test.macroF1)}</strong>${bestAnxiety ? ` — ${escapeHtml(bestAnxiety.experimentTitle)} (${escapeHtml(bestAnxiety.variant)})` : ""}. Este resultado não participa dos rankings de depressão.</p>
    </div>
  </div>
  <div class="grid two">
    <div class="scard">
      <div class="head"><span class="nm">Trilha binária</span><span class="tag">2 classes</span></div>
      <p>Melhor macro F1 de teste selado <strong>${metric(bestBinary?.test.macroF1)}</strong>${bestBinary ? ` — ${escapeHtml(bestBinary.experimentTitle)} (${escapeHtml(bestBinary.variant)})` : ""}.</p>
    </div>
    <div class="scard">
      <div class="head"><span class="nm">Trilha ternária</span><span class="tag">3 classes</span></div>
      <p>Melhor macro F1 de teste selado <strong>${metric(bestTernary?.test.macroF1)}</strong>${bestTernary ? ` — ${escapeHtml(bestTernary.experimentTitle)} (${escapeHtml(bestTernary.variant)})` : ""}.</p>
    </div>
  </div>
  <p>A linha binária bruta evoluiu de <code class="k">${metric(raw?.test.macroF1)}</code> para <code class="k">${metric(
    temporalBase?.test.macroF1,
  )}</code> com as features de relevância temporal (${deltaText(temporalLift)}), e depois para <code class="k">${metric(
    temporalLlm?.test.macroF1,
  )}</code> com a desambiguação completa de falsos positivos por LLM (${deltaText(llmLift)}).</p>
  <div class="note">
    <div class="lab">Strict-blind &amp; comparabilidade</div>
    <p>As linhas desambiguadas por LLM são estimativas pontuais do teste final, não novos sinais de seleção — o LLM apenas troca usuários previstos como diagnosticados para controle. O macro F1 binário é uma média de 2 classes; o macro F1 OOF ternário é uma média de 3 classes sobre diagnosticado / controle / sem-evidência. Ansiedade usa outro alvo e outra prevalência. Essas trilhas <strong>não são diretamente comparáveis</strong>.</p>
  </div>
</section>`;
}

function architectureSection(): string {
  return `<section id="architecture">
  <p class="kicker">Como as travas são construídas</p>
  <h2>Arquitetura do ensemble</h2>
  <p class="col">Toda trava é uma combinação ponderada de modelos-base selecionados pelos scores out-of-fold (OOF) de treino e então congelada. Os scores de teste permanecem sem rótulo; a seleção nunca vê os rótulos de teste.</p>
  <div class="grid">
    <div class="scard">
      <div class="head"><span class="nm">Modelos-base tabulares</span></div>
      <ul>
        <li><b>LogReg</b> / LogReg multinomial</li>
        <li><b>XGBoost</b>, <b>HistGB</b> (gradient boosting)</li>
        <li><b>MLP</b>, <b>ExtraTrees</b></li>
        <li class="con">Especializados: Focal LogReg, LogReg hierárquica (com gating por evidência), baseline de relevância</li>
      </ul>
    </div>
    <div class="scard">
      <div class="head"><span class="nm">Modelos-base de sequência</span></div>
      <ul>
        <li><b>CNN</b> (e CNN de kernel largo)</li>
        <li><b>BiLSTM</b> com pooling de média / máximo</li>
        <li>Encoder <b>Tiny Transformer</b></li>
        <li class="con">Canal opcional de relevância por tweet anexado aos embeddings</li>
      </ul>
    </div>
    <div class="scard">
      <div class="head"><span class="nm">Seleção &amp; fusão</span></div>
      <ul>
        <li>Scores OOF por fold alinhados por usuário</li>
        <li><b>Enumeração de pesos</b> (<code class="k">ranked-prefix-pruned</code> / exaustiva)</li>
        <li>Binário: <b>varredura de limiar</b>; ternário: <b>regra de decisão</b> + refinamento local de pesos</li>
        <li class="con">Segundo nível opcional <code class="k">stacking_logreg</code> sobre o OOF dos modelos-base</li>
      </ul>
    </div>
  </div>
  <div class="grid two">
    <div class="scard mute">
      <div class="head"><span class="nm">ensemble-lock</span><span class="tag">seleção livre</span></div>
      <p>Ensemble escolhido a partir do conjunto completo de candidatos, com pesos reajustados para maximizar o macro F1 OOF.</p>
    </div>
    <div class="scard mute">
      <div class="head"><span class="nm">legacy-cnn-logreg-mlp-lock</span><span class="tag">restrita</span></div>
      <p>Travada na arquitetura histórica do depression-nlp (CNN + LogReg + MLP); pesos reajustados no OOF para um port equivalente.</p>
    </div>
  </div>
  <p class="muted">Fonte: <code class="k">src/ensemble.ts</code> (binário), <code class="k">src/ternary.ts</code> (políticas ternárias, regras de decisão, refinamento local).</p>
</section>`;
}

function methodSection(): string {
  return `<section id="methods">
  <p class="kicker">O que cada experimento ensinou</p>
  <h2>Lições por método</h2>
  <div class="grid">
    ${methodIdeas
      .map(
        (idea) => `<div class="scard">
      <div class="head"><span class="nm">${escapeHtml(idea.title)}</span></div>
      <p class="muted">${escapeHtml(idea.signal)}</p>
      <p>${escapeHtml(idea.body)}</p>
    </div>`,
      )
      .join("\n")}
  </div>
  <table>
    <thead><tr><th>Eixo do método</th><th>Ideia comparada</th><th>Lição observada</th></tr></thead>
    <tbody>
      <tr><td>Binário vs ternário</td><td>Diagnosticado/controle direto vs políticas diagnosticado/controle/sem-evidência</td><td>As travas ternárias simétricas produziram o melhor macro F1 de teste selado do repositório, enquanto as execuções binárias temporais são mais fáceis de interpretar nos trade-offs de FP/FN.</td></tr>
      <tr><td>Embeddings legados vs Qwen3 bruto</td><td>Embeddings strict-blind originais vs artefatos Qwen3 brutos</td><td>As execuções com Qwen3 bruto viraram a base de comparação útil depois que a generalização de teste dos embeddings legados strict-blind foi ruim.</td></tr>
      <tr><td>Canal de relevância vs temporal</td><td>Expor relevância por tweet vs modelar recência/tom explicitamente</td><td>A relevância ajudou o OOF de forma modesta; as features temporais deslocaram o equilíbrio FP/FN e melhoraram o macro F1 binário de teste selado.</td></tr>
      <tr><td>Modelo-base vs pós-filtro por LLM</td><td>Score estatístico do modelo vs rejeição semântica unidirecional de falsos positivos</td><td>A desambiguação por LLM melhorou a precisão e o macro F1 ao remover falsos positivos lexicais, de fandom, de terceiros e de contexto profissional.</td></tr>
    </tbody>
  </table>
</section>`;
}

function trackSection(
  track: Track,
  lockEntries: readonly LockEntry[],
  resultEntries: readonly ResultEntry[],
  embeddingProvenance?: EmbeddingGenerationProvenance,
  thesisComparison?: AnxietyThesisComparison,
): string {
  const finals = resultEntries
    .filter((entry) => entry.track === track && entry.kind !== "cached-llm")
    .sort((left, right) => compareMetric(right.test.macroF1, left.test.macroF1));
  const locks = [...lockEntries]
    .filter((entry) => entry.track === track)
    .sort((left, right) => compareMetric(right.oof?.macroF1, left.oof?.macroF1));
  const baselineMacro =
    track === "binary"
      ? rawBinaryBaselineMacro
      : track === "ternary"
        ? (resultEntries.find(
          (entry) => entry.experimentId === "seed42_raw_qwen3_ternary_diagnosed_only" && entry.lockBasename === "ensemble-lock",
        )?.test.macroF1 ?? null)
        : null;
  const bestMacro = bestBy(finals, (entry) => entry.test.macroF1);
  const bestDiag = bestBy(finals, (entry) => entry.test.diagnosedF1);
  const bestPrecision = bestBy(finals, (entry) => entry.test.precision);
  const bestRecall = bestBy(finals, (entry) => entry.test.recall);

  const heading =
    track === "binary"
      ? { kicker: "Trilha A · 2 classes", title: "Resultados binários", classes: "diagnosticado vs controle" }
      : track === "ternary"
        ? { kicker: "Trilha B · 3 classes", title: "Resultados ternários", classes: "diagnosticado vs controle vs sem-evidência" }
        : { kicker: "Alvo independente · 2 classes", title: "Resultados de ansiedade", classes: "ansiedade vs controle" };
  const intro =
    track === "binary"
      ? `Diagnosticado-vs-controle com limiar varrido. O delta é contra o macro F1 de teste do baseline Qwen3 bruto binário <code class="k">${metric(rawBinaryBaselineMacro)}</code>.`
      : track === "ternary"
        ? `Diagnosticado / controle / sem-evidência com uma política de rótulo (<code class="k">diag_*</code> apenas diagnosticados vs <code class="k">sym_*</code> simétrica) mais uma regra de decisão. O macro F1 OOF é uma média de 3 classes — <strong>não comparável à trilha binária</strong>. O delta é contra o <code class="k">ensemble-lock</code> ternário apenas-diagnosticados.`
        : `Experimento strict-blind exclusivo de ansiedade. A composição do campeão foi fixada em Focal LogReg, LogReg, CNN e Stacking ×2; somente pesos positivos e o limiar foram ajustados no OOF de treino. O teste selado é documentado sem comparação ou ranking contra depressão.`;
  const positiveLabel = track === "anxiety" ? "ansiedade" : "diagnosticado";
  const positiveShort = track === "anxiety" ? "ansiedade" : "diag.";

  const finalRows = finals
    .map((entry, index) => {
      const delta = baselineMacro === null ? null : entry.test.macroF1 - baselineMacro;
      return `<tr>
        <td>${index + 1}</td>
        <td><strong>${escapeHtml(entry.experimentTitle)}</strong><br><code>${escapeHtml(entry.lockBasename)}</code><br>${familyChips(entry.modelIds)}</td>
        <td>${escapeHtml(entry.variant)}</td>
        <td>${metric(entry.oof?.macroF1)}</td>
        <td><strong>${metric(entry.test.macroF1)}</strong></td>
        <td>${metric(entry.test.diagnosedF1)}</td>
        <td>${metric(entry.test.precision)}</td>
        <td>${metric(entry.test.recall)}</td>
        <td>${confusion(entry.test)}</td>
        <td>${deltaCell(delta)}</td>
      </tr>`;
    })
    .join("\n");

  const oofRows = locks
    .map(
      (entry) => `<tr>
      <td><strong>${escapeHtml(entry.experimentTitle)}</strong><br><code>${escapeHtml(entry.lockBasename)}</code><br>${familyChips(entry.modelIds)}</td>
      <td>${metric(entry.oof?.macroF1)}</td>
      <td>${metric(entry.oof?.diagnosedF1)}</td>
      <td>${metric(entry.oof?.precision)}</td>
      <td>${metric(entry.oof?.recall)}</td>
      <td>${confusion(entry.oof)}</td>
      <td><code>${escapeHtml(entry.policy)}</code></td>
      <td><code>${escapeHtml(entry.decision)}</code></td>
    </tr>`,
    )
    .join("\n");

  return `<section id="${track}">
  <p class="kicker">${escapeHtml(heading.kicker)}</p>
  <h2>${escapeHtml(heading.title)} <span class="muted">· ${escapeHtml(heading.classes)}</span></h2>
  <p class="col">${intro}</p>${track === "anxiety" && embeddingProvenance ? `\n  ${embeddingGenerationBlock(embeddingProvenance)}` : ""}
  <div class="kpis">
    ${kpi("Melhor macro F1 (teste)", metric(bestMacro?.test.macroF1), bestMacro?.experimentTitle ?? "n/d")}
    ${kpi(`Melhor F1 ${positiveLabel}`, metric(bestDiag?.test.diagnosedF1), bestDiag?.experimentTitle ?? "n/d")}
    ${kpi(`Melhor precisão ${positiveLabel}`, metric(bestPrecision?.test.precision), bestPrecision?.experimentTitle ?? "n/d")}
    ${kpi(`Melhor recall ${positiveLabel}`, metric(bestRecall?.test.recall), bestRecall?.experimentTitle ?? "n/d")}
  </div>
  <h3>Resultados de teste selado (ranqueados)</h3>
  <table>
    <thead><tr><th>#</th><th>Trilha / trava</th><th>Relatório</th><th>Macro OOF</th><th>Macro teste</th><th>F1 ${positiveShort}</th><th>Precisão ${positiveShort}</th><th>Recall ${positiveShort}</th><th>Confusão</th><th>Delta</th></tr></thead>
    <tbody>${finalRows}</tbody>
  </table>
  <h3>Resultados OOF travados</h3>
  <p class="muted">Métricas OOF de treino usadas para travar modelo, pesos, limiar ou regra de decisão.</p>
  <table>
    <thead><tr><th>Trilha / trava</th><th>Macro OOF</th><th>F1 ${positiveShort}</th><th>Precisão ${positiveShort}</th><th>Recall ${positiveShort}</th><th>Confusão OOF</th><th>Política</th><th>Decisão</th></tr></thead>
    <tbody>${oofRows}</tbody>
  </table>${track === "anxiety" && bestMacro && thesisComparison ? `\n  ${anxietyThesisComparisonBlock(bestMacro, thesisComparison)}` : ""}${track === "ternary" ? `\n  ${ternaryCompositionBlock(locks)}` : ""}
</section>`;
}

function anxietyThesisComparisonBlock(
  champion: ResultEntry,
  comparison: AnxietyThesisComparison,
): string {
  const bestMacro = bestBy(comparison.reportedResults, (entry) => entry.macroF1);
  const bestAnxiety = bestBy(comparison.reportedResults, (entry) => entry.anxietyF1);
  const bestControl = bestBy(comparison.reportedResults, (entry) => entry.controlF1);
  const championControlF1 = champion.test.controlF1;
  if (!bestMacro || !bestAnxiety || !bestControl || championControlF1 === null) return "";
  const bestMacroModels = comparison.reportedResults
    .filter((entry) => entry.macroF1 === bestMacro.macroF1)
    .map((entry) => entry.model)
    .join(" / ");

  const highlighted = comparison.reportedResults.filter(
    (entry) => entry.macroF1 === bestMacro.macroF1 || entry.model === "LSTM.BERT",
  );
  const thesisRows = highlighted
    .map(
      (entry) => `<tr>
        <td>${escapeHtml(comparison.source.author)} (2025)</td>
        <td><code>${escapeHtml(entry.model)}</code></td>
        <td>${entry.controlF1.toFixed(2)}</td>
        <td>${entry.anxietyF1.toFixed(2)}</td>
        <td><strong>${entry.macroF1.toFixed(2)}</strong></td>
      </tr>`,
    )
    .join("\n");

  return `<h3>Comparação com a tese de Santos (2025)</h3>
  <p class="col">A Tabela ${integer(comparison.source.table)} da tese reporta resultados no mesmo alvo de ansiedade do SetembroBR. As três medidas F1 abaixo têm definições compatíveis: <code>${escapeHtml(comparison.metricDefinitions.controlF1)}</code> é o F1 de controle, <code>${escapeHtml(comparison.metricDefinitions.anxietyF1)}</code> é o F1 da classe diagnosticada/ansiedade e <code>${escapeHtml(comparison.metricDefinitions.macroF1)}</code> é a média aritmética das duas classes.</p>
  <table>
    <thead><tr><th>Fonte</th><th>Modelo</th><th>F1 controle</th><th>F1 ansiedade</th><th>Macro F1</th></tr></thead>
    <tbody><tr>
        <td><strong>Esta reprodução strict-blind</strong></td>
        <td><code>ensemble-lock</code></td>
        <td>${metric(championControlF1)}</td>
        <td>${metric(champion.test.diagnosedF1)}</td>
        <td><strong>${metric(champion.test.macroF1)}</strong></td>
      </tr>
      ${thesisRows}</tbody>
  </table>
  <h4>Diferenças de protocolo</h4>
  <table>
    <thead><tr><th>Eixo</th><th>Tese de Santos (2025)</th><th>Esta reprodução strict-blind</th></tr></thead>
    <tbody>
      <tr><td>Linha do tempo</td><td>${escapeHtml(comparison.reportedMethod.timeline)}</td><td>Agregados sobre embeddings brutos e sequência cronológica dos 128 tweets mais recentes para a CNN</td></tr>
      <tr><td>Inferência sequencial</td><td>${escapeHtml(comparison.reportedMethod.bertInference)}; ${integer(comparison.reportedMethod.postTruncationTokens)} tokens por post</td><td>Janela recente-128 determinística, sem amostragem aleatória no teste</td></tr>
      <tr><td>Relevância</td><td>${escapeHtml(comparison.reportedMethod.relevance)}</td><td>Proxy lexical fixo <code>anxiety-lexical-v1</code>, independente de rótulos</td></tr>
      <tr><td>Composição</td><td>${escapeHtml(comparison.reportedMethod.mentionSignal)}</td><td>Focal LogReg, LogReg, CNN e Stacking ×2; sem rede de menções</td></tr>
      <tr><td>Seleção</td><td>Resultados finais publicados na partição de teste do SetembroBR</td><td>Pesos e limiar definidos somente no OOF de treino; teste pontuado sem rótulos e aberto uma vez após a trava</td></tr>
    </tbody>
  </table>
  <div class="kpis">
    ${kpi("Delta vs melhor macro F1 da tese", signedMetric(champion.test.macroF1 - bestMacro.macroF1), `vs ${bestMacroModels} · valor publicado ${bestMacro.macroF1.toFixed(2)}`)}
    ${kpi("Delta vs melhor F1 ansiedade da tese", signedMetric(champion.test.diagnosedF1 - bestAnxiety.anxietyF1), `vs ${bestAnxiety.model} · valor publicado ${bestAnxiety.anxietyF1.toFixed(2)}`)}
    ${kpi("Delta vs melhor F1 controle da tese", signedMetric(championControlF1 - bestControl.controlF1), `vs ${bestControl.model} · valor publicado ${bestControl.controlF1.toFixed(2)}`)}
  </div>
  <div class="note">
    <div class="lab">Leitura correta da comparação</div>
    <p>Os deltas são pontos estimados: <strong>+3,285 p.p.</strong> em macro F1, <strong>+3,416 p.p.</strong> em F1 de ansiedade e <strong>+2,154 p.p.</strong> em F1 de controle. A tese arredonda a duas casas decimais, usa outra representação e outro procedimento de inferência, e não disponibiliza aqui as predições por usuário necessárias para McNemar ou bootstrap pareado. Portanto, isto não demonstra superioridade estatística.</p>
    <p><code>P</code> e <code>R</code> da tese são descritos como precisão e revocação médias do modelo, sem convenção de agregação explicitada; eles não são comparados à precisão e ao recall da classe ansiedade desta reprodução.</p>
    <p>Fonte: <code>${escapeHtml(comparison.source.file)}</code>, Tabela ${integer(comparison.source.table)}, p. ${integer(comparison.source.printedPage)} impressa (p. ${integer(comparison.source.pdfPage)} do PDF), SHA-256 <code>${escapeHtml(comparison.source.sha256)}</code>. Procedimento descrito nas p. ${comparison.source.methodPrintedPages.map(integer).join("–")} impressas.</p>
  </div>`;
}

function embeddingGenerationBlock(provenance: EmbeddingGenerationProvenance): string {
  const { timing, hardware, workload, sourceEvidence } = provenance;
  return `<h3>Geração dos embeddings de ansiedade</h3>
  <div class="grid two">
    <div class="scard">
      <div class="head"><span class="nm">Tempo medido</span><span class="tag">wall-clock</span></div>
      <p><strong>${escapeHtml(durationText(timing.durationSeconds))}</strong> para gerar todos os embeddings de treino e teste.</p>
      <ul>
        <li>Início: <code>${escapeHtml(localTimestamp(timing.startedAt))}</code></li>
        <li>Conclusão: <code>${escapeHtml(localTimestamp(timing.completedAt))}</code></li>
        <li>Carga: ${integer(workload.totalTweets)} tweets de ${integer(workload.trainUsers + workload.testUsers)} usuários</li>
        <li>Modelo: <code>${escapeHtml(workload.modelId)}</code>, ${integer(workload.embeddingDimension)} dimensões, <code>${escapeHtml(workload.storageDtype)}</code>, batch ${integer(workload.batchSize)}</li>
      </ul>
    </div>
    <div class="scard">
      <div class="head"><span class="nm">Hardware Fedora</span><span class="tag">CUDA</span></div>
      <ul>
        <li>GPU: <strong>${escapeHtml(hardware.gpu)}</strong> · ${integer(hardware.gpuMemoryMiB)} MiB</li>
        <li>Driver ${escapeHtml(hardware.driverVersion)} · CUDA ${escapeHtml(hardware.cudaVersion)}</li>
        <li>CPU: ${escapeHtml(hardware.cpu)} · ${integer(hardware.cpuCores)} cores / ${integer(hardware.cpuThreads)} threads</li>
        <li>RAM: ${integer(hardware.memoryGiB)} GiB</li>
        <li>${escapeHtml(hardware.operatingSystem)} · kernel <code>${escapeHtml(hardware.kernel)}</code></li>
      </ul>
    </div>
  </div>
  <div class="note">
    <div class="lab">Proveniência da medição</div>
    <p>Tempo de parede medido no host <code>${escapeHtml(hardware.hostname)}</code> pelo nascimento do PID da execução até o status concluído com exit code ${integer(sourceEvidence.jobExitCode)}. Manifesto bruto <code>${escapeHtml(sourceEvidence.rawEmbeddingManifestSha256)}</code>; revisão Qwen <code>${escapeHtml(workload.modelRevision)}</code>.</p>
  </div>`;
}

function ternaryCompositionBlock(locks: readonly LockEntry[]): string {
  if (locks.length === 0) return "";
  const cards = locks
    .map((lock) => {
      const rows = [...lock.modelIds]
        .sort((left, right) => (lock.weights[right] ?? 0) - (lock.weights[left] ?? 0))
        .map(
          (modelId) => `<tr>
        <td><code>${escapeHtml(modelId)}</code></td>
        <td>${escapeHtml(modelFamily(modelId))}</td>
        <td>${weightText(lock.weights[modelId])}</td>
      </tr>`,
        )
        .join("\n");
      const groupLine = lock.groupDescription
        ? `<p class="muted">Grupo de seleção: <code class="k">${escapeHtml(lock.group)}</code> — ${escapeHtml(lock.groupDescription)}</p>`
        : `<p class="muted">Grupo de seleção: <code class="k">${escapeHtml(lock.group)}</code></p>`;
      return `<div class="scard">
      <div class="head"><span class="nm">${escapeHtml(lock.experimentTitle)}</span><span class="tag">${escapeHtml(lock.lockBasename)}</span></div>
      <p>Split / política de rótulo: <code class="k">${escapeHtml(lock.policy)}</code> · manifesto <code class="k">${escapeHtml(policyManifest(lock.policy))}</code></p>
      <p>Regra de decisão: <code class="k">${escapeHtml(lock.decision)}</code></p>
      ${groupLine}
      <table>
        <thead><tr><th>Modelo</th><th>Família</th><th>Peso</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
    })
    .join("\n");
  return `<h3>Composição por modelo (splits e políticas)</h3>
  <p class="muted">Cada trava ternária aplica uma única política de rótulo (split) e regra de decisão a todos os seus modelos selecionados; os pesos abaixo são a contribuição de cada modelo no ensemble.</p>
  <div class="grid two">${cards}</div>`;
}

function renderTemporalEvidenceSection(): string {
  return `<section id="temporal">
  <p class="kicker">Análise qualitativa de erros</p>
  <h2>Exploração temporal de FP / FN</h2>
  <p class="col">Amostrei FPs e FNs de <code class="k">seed42_relevance_features_qwen3_binary</code> em casos de alta confiança, próximos ao limiar, baixa-evidência/score-alto e alta-evidência/score-baixo. As linhas do tempo foram inspecionadas em ordem cronológica, com ênfase nas postagens recentes.</p>
  <div class="grid two">
    <div class="scard">
      <div class="head"><span class="nm">Grupos de falsos positivos</span></div>
      <ol>
        <li><strong>Controles parecidos com verdadeiros positivos:</strong> linguagem pessoal forte de depressão, incluindo ideação de automutilação, choro, ansiedade, invalidação familiar, terapia, medicação ou busca por ajuda. Exemplos: <code>CD_762</code>, <code>CD_7762</code>, <code>CD_2120</code>.</li>
        <li><strong>Sinal passado, fechamento recente tipo controle:</strong> linguagem anterior de depressão/ansiedade seguida de postagens finais dominadas por fandom, vida cotidiana, escola, família, conversa social ou piadas. Exemplos: <code>CD_1014</code>, <code>CD_7832</code>, em parte <code>CD_156</code>.</li>
        <li><strong>Falsos alarmes lexicais:</strong> termos de morte/automutilação dentro de piadas, discussões, ameaças, gírias sexuais/profanas ou drama exagerado. Exemplos: <code>CD_10886</code>, em parte <code>CD_156</code>.</li>
        <li><strong>Contaminação por terceiros/conta de apoio:</strong> conteúdo de apoio à saúde mental dirigido a outras pessoas, e não autorrevelação. Exemplo: <code>CD_1480</code>.</li>
      </ol>
    </div>
    <div class="scard">
      <div class="head"><span class="nm">Grupos de falsos negativos</span></div>
      <ol>
        <li><strong>Pouca evidência visível de depressão:</strong> linhas do tempo recentes majoritariamente de fandom, TV, política, piadas, interação social ou comentário público. Exemplos: <code>D_504</code>, <code>D_513</code>, <code>D_1485</code>.</li>
        <li><strong>Tom depressivo indireto:</strong> insônia, não comer/dormir, cansaço, desânimo, choro, luto, menções a tratamento/psicólogo, baixa motivação e linguagem de não estar bem. Exemplos: <code>D_393</code>, <code>D_51</code>, <code>D_1064</code>, <code>D_1103</code>.</li>
        <li><strong>Evidência forte anterior, silêncio recente:</strong> há evidência forte de automutilação/autodesvalorização, mas diluída por postagens casuais posteriores. Exemplo: <code>D_1400</code>.</li>
        <li><strong>Alta relevância em contexto não clínico:</strong> a relevância é puxada por mortes ficcionais, TV, luto de fandom ou outras pessoas. Exemplo: <code>D_1672</code>.</li>
      </ol>
    </div>
  </div>
  <div class="note">
    <div class="lab">Conclusão de modelagem</div>
    <p>A seleção de sequência muito dependente de relevância pode dar peso excessivo a tweets antigos de alta relevância e perder sintomas recentes sutis. Por isso o experimento temporal adicionou agregados de janela recente, deltas temporais, posição da última alta relevância, marcadores de tom indireto e exportação cronológica de sequência recente.</p>
  </div>
</section>`;
}

function renderLlmSection(entries: readonly ResultEntry[]): string {
  const llmRows = [...entries]
    .filter((entry) => entry.kind === "llm" || entry.kind === "cached-llm")
    .sort((left, right) => left.track.localeCompare(right.track) || compareMetric(right.test.macroF1, left.test.macroF1))
    .map((entry) => {
      const base = entry.baseTest;
      const fixed = base ? base.fp - entry.test.fp : null;
      const lost = base ? base.tp - entry.test.tp : null;
      const delta = base ? entry.test.macroF1 - base.macroF1 : null;
      return `<tr>
      <td><span class="badge">${escapeHtml(trackLabel(entry.track))}</span> <strong>${escapeHtml(entry.experimentTitle)}</strong><br><code>${escapeHtml(entry.variant)}</code></td>
      <td>${metric(base?.macroF1)}</td>
      <td>${metric(entry.test.macroF1)}</td>
      <td>${deltaCell(delta)}</td>
      <td>${entry.llm?.decisionCount ?? "n/d"}</td>
      <td>${entry.llm?.switchedToControl ?? "n/d"}</td>
      <td>${fixed ?? "n/d"}</td>
      <td>${lost ?? "n/d"}</td>
    </tr>`;
    })
    .join("\n");
  return `<section id="llm">
  <p class="kicker">Filtro unidirecional pós-trava</p>
  <h2>Desambiguação de falsos positivos por LLM</h2>
  <p class="col">O prompt do LLM foi refinado na análise de erros do OOF de treino e usou arquivos de decisão de teste strict-blind. É um filtro pós-trava, unidirecional, que apenas troca usuários previstos como diagnosticados para controle — aplicado nas duas trilhas.</p>
  <table>
    <thead><tr><th>Execução</th><th>Macro base</th><th>Macro LLM</th><th>Delta</th><th>Decisões</th><th>Trocados</th><th>FP corrigidos</th><th>TP perdidos</th></tr></thead>
    <tbody>${llmRows}</tbody>
  </table>
  <div class="note">
    <div class="lab">Ressalva de bootstrap</div>
    <p>Na execução temporal completa do LLM, 289 decisões foram semeadas a partir do cache existente e 41 novas chamadas preencheram usuários faltantes/conflitantes. O delta pareado por bootstrap vs base temporal cruzou o zero: <code>[-0.004882, +0.018171]</code>.</p>
  </div>
</section>`;
}

function renderArtifacts(groups: readonly ArtifactGroup[]): string {
  const body = groups
    .map((group) => {
      const badge =
        group.experimentId === "configs" || group.experimentId === "setembrobr"
          ? ""
          : `<span class="badge">${escapeHtml(trackLabel(trackFor(group.experimentId)))}</span> `;
      return `<details>
      <summary>${badge}${escapeHtml(group.title)} <span class="muted">(${group.files.length} files)</span></summary>
      <ul>${group.files.map((file) => `<li><code>${escapeHtml(file)}</code></li>`).join("\n")}</ul>
    </details>`;
    })
    .join("\n");
  return `<section id="artifacts">
  <p class="kicker">Proveniência</p>
  <h2>Índice de artefatos</h2>
  <p class="col">Travas, relatórios finais, diagnósticos, decisões de LLM e artefatos JSON relevantes descobertos em <code class="k">outputs/setembrobr</code>.</p>
  ${body}
</section>`;
}

function css(): string {
  return `:root{color-scheme:light;--bg:#fbfbfd;--panel:#f5f5f7;--card:#fff;--border:#e4e4e9;--border-2:#d2d2d7;--border-soft:#eeeef1;--ink:#1d1d1f;--ink-2:#6e6e73;--ink-3:#86868b;--accent:#1d1d1f;--accent-soft:#f0f0f2;--pos:#0f766e;--neg:#b42318;--neg-soft:#fdecec;--radius:18px;--glass-hi:inset 0 1px 0 #ffffffe6;--blur:saturate(160%) blur(20px);--sh-1:0 1px 2px #0000000a,0 4px 16px #0000000d;--sh-2:0 2px 8px #0000000f,0 22px 50px #00000017;--mono:"SF Mono","JetBrains Mono",ui-monospace,Menlo,monospace;--sans:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display","Helvetica Neue",Inter,system-ui,sans-serif}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:78px}body{margin:0;font-family:var(--sans);background:var(--bg);color:var(--ink);font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}a{color:var(--ink);text-decoration:none}
nav{position:sticky;top:0;z-index:50;backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);background:#fcfcfdc7;border-bottom:1px solid var(--border)}nav .wrap{display:flex;align-items:center;gap:26px;height:58px;max-width:1180px;margin:0 auto;padding:0 28px}nav .brand{display:flex;align-items:center;gap:9px;font-weight:600;font-size:14px;color:var(--ink)}nav .brand .dot{width:9px;height:9px;border-radius:50%;background:var(--ink);flex:0 0 9px}nav .links{display:flex;flex-wrap:wrap;gap:20px;margin-left:auto;font-size:13.5px}nav .links a{color:var(--ink-3);transition:color .15s}nav .links a:hover,nav .links a.active{color:var(--ink)}
main{max-width:1180px;margin:0 auto;padding:0 28px}section{padding:52px 0;border-bottom:1px solid var(--border-soft);scroll-margin-top:78px}section:last-of-type{border-bottom:0}.col{max-width:760px}.kicker{letter-spacing:.15em;text-transform:uppercase;color:var(--ink-3);font-size:12px;font-weight:600;margin:0 0 14px}
h1{font-size:42px;font-weight:700;line-height:1.07;letter-spacing:-.035em;max-width:22ch;margin:0 0 22px}h1 .muted{color:var(--ink-3)}h2{font-size:27px;font-weight:700;line-height:1.12;letter-spacing:-.025em;margin:0 0 14px}h2 .muted{color:var(--ink-3);font-weight:600}h3{font-size:16px;font-weight:650;letter-spacing:-.01em;margin:30px 0 10px;color:var(--ink)}p{color:var(--ink-2);margin:13px 0}p.lead{font-size:18px;line-height:1.5;max-width:64ch;color:var(--ink-2)}.muted{color:var(--ink-3)}b,strong{color:var(--ink);font-weight:600}
code{font-family:var(--mono);font-size:12px}code.k{background:var(--panel);border:1px solid var(--border);color:var(--ink);border-radius:6px;padding:1px 6px;font-size:12.5px}
.meta{margin:26px 0;border:1px solid var(--border);border-radius:14px;background:var(--card);box-shadow:var(--sh-1);overflow:hidden;max-width:620px}.meta .row{display:flex;align-items:center;gap:14px;padding:12px 18px;border-bottom:1px solid var(--border-soft)}.meta .row:last-child{border-bottom:0}.meta .lab{font-family:var(--mono);font-size:11.5px;text-transform:uppercase;letter-spacing:.1em;color:var(--ink-3);width:74px;flex:0 0 74px}.meta .val{font-size:14px;color:var(--ink);font-weight:500}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:24px}.grid.two{grid-template-columns:repeat(2,1fr)}.scard{border-radius:var(--radius);background:#ffffffb8;border:1px solid #ffffff8c;backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);box-shadow:0 0 0 1px #0000000a,var(--sh-1),var(--glass-hi);padding:22px;transition:transform .2s,box-shadow .2s}.scard:hover{box-shadow:0 0 0 1px #0000000a,var(--sh-2),var(--glass-hi);transform:translateY(-2px)}.scard.mute{background:var(--panel);border-color:var(--border);box-shadow:none}.scard.mute:hover{transform:none;box-shadow:0 0 0 1px #0000000a,var(--sh-1)}.scard .head{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:4px}.scard .nm{font-family:var(--mono);font-size:16px;font-weight:600;letter-spacing:-.02em;color:var(--ink)}.scard .tag{margin-left:auto;font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--ink-3)}.scard p{font-size:14px;margin:8px 0 0}.scard ul{list-style:none;margin:12px 0 0;padding:0;display:flex;flex-direction:column;gap:8px}.scard li{position:relative;padding-left:18px;font-size:14px;line-height:1.45;color:var(--ink-2)}.scard li::before{content:"";position:absolute;left:2px;top:8px;width:5px;height:5px;border-radius:50%;background:var(--ink)}.scard li.con{color:var(--ink-3)}.scard li.con::before{background:var(--border-2)}.scard ol{margin:10px 0 0 18px;padding:0;font-size:14px;color:var(--ink-2)}.scard ol li{margin:7px 0}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:14px;margin:22px 0}.kpi{padding:16px 18px;border:1px solid var(--border);border-radius:14px;background:var(--card);box-shadow:var(--sh-1)}.kpi span{color:var(--ink-3);font-size:12px;font-weight:600;letter-spacing:.04em;text-transform:uppercase}.kpi strong{display:block;margin-top:6px;font-size:26px;font-family:var(--mono);letter-spacing:-.02em}.kpi code{display:block;margin-top:8px;color:var(--ink-3);font-size:12px;background:none;padding:0}
table{width:100%;border-collapse:collapse;margin:14px 0}th,td{padding:10px 12px;border-bottom:1px solid var(--border-soft);text-align:left;vertical-align:top;font-size:13.5px;color:var(--ink-2)}th{color:var(--ink-3);background:var(--panel);font-size:11px;letter-spacing:.06em;text-transform:uppercase;font-weight:600;position:sticky}tbody tr:hover{background:#fafafb}td strong{color:var(--ink)}td code{background:var(--accent-soft);border-radius:4px;padding:1px 5px;color:var(--ink-2)}
.pos{color:var(--pos);font-weight:650}.neg{color:var(--neg);font-weight:650}
.chips{display:inline-flex;flex-wrap:wrap;gap:5px;margin-top:5px}.chip{display:inline-block;font-family:var(--mono);border:1px solid var(--border-2);border-radius:8px;background:var(--panel);color:var(--ink-2);padding:2px 7px;font-size:11.5px}.chip.bad{border-color:#f0c7c2;background:var(--neg-soft);color:var(--neg)}.badge{display:inline-block;font-family:var(--mono);font-size:10.5px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;background:var(--ink);color:#fff;border-radius:6px;padding:2px 7px;vertical-align:middle}
.note{margin-top:22px;border-left:2px solid var(--ink);background:var(--panel);border-radius:0 12px 12px 0;padding:14px 18px}.note .lab{font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);margin-bottom:7px}.note p{margin:0;color:var(--ink-2);font-size:14px}
details{margin:10px 0;padding:12px 16px;border:1px solid var(--border);border-radius:12px;background:var(--card)}summary{cursor:pointer;font-weight:600;color:var(--ink);font-size:14px}details ul{margin:10px 0 2px 18px;padding:0}details li{margin:4px 0;color:var(--ink-2)}details li code{background:var(--panel);border:1px solid var(--border-soft);border-radius:5px;padding:1px 5px}
footer{padding:36px 0 80px;color:var(--ink-3);font-size:13px}
@media(max-width:900px){nav .links{display:none}h1{font-size:32px;max-width:18ch}h2{font-size:23px}.grid,.grid.two,.kpis{grid-template-columns:1fr}table{display:block;overflow-x:auto;white-space:nowrap}}
@media print{body{background:#fff}nav{display:none}.scard,.kpi{box-shadow:none;break-inside:avoid}}`;
}

function scrollSpyScript(): string {
  return `<script>
  (function(){
    var links=[].slice.call(document.querySelectorAll('nav .links a'));
    var byId={}; links.forEach(function(a){byId[a.getAttribute('href').slice(1)]=a;});
    var secs=links.map(function(a){return document.getElementById(a.getAttribute('href').slice(1));}).filter(Boolean);
    if(!secs.length) return; var cur=null, vis={};
    function set(id){ if(id===cur) return; cur=id; links.forEach(function(a){a.classList.remove('active');}); if(byId[id]) byId[id].classList.add('active'); }
    var ob=new IntersectionObserver(function(es){
      es.forEach(function(e){ if(e.isIntersecting) vis[e.target.id]=e.intersectionRatio; else delete vis[e.target.id]; });
      var top=null, y=Infinity; secs.forEach(function(s){ if(!vis[s.id]) return; var t=s.getBoundingClientRect().top; if(t<y){y=t; top=s.id;} });
      if(top){ set(top); return; }
      var passed=null; secs.forEach(function(s){ if(s.getBoundingClientRect().top<120) passed=s.id; }); if(passed) set(passed);
    },{rootMargin:'-10% 0px -70% 0px', threshold:[0,0.25,0.5,1]});
    secs.forEach(function(s){ ob.observe(s); });
  })();
  </script>`;
}

function kpi(label: string, value: string, detail: string): string {
  return `<div class="kpi"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><code>${escapeHtml(detail)}</code></div>`;
}

function normalizeMetrics(value: unknown): NormalizedMetrics | null {
  if (!isRecord(value)) return null;
  const macroF1 = numberValue(value, "macroF1");
  const diagnosedF1 = numberValue(value, "diagnosedF1") ?? perClassNumber(value, "f1");
  const controlF1 = numberValue(value, "controlF1");
  const precision = numberValue(value, "precision") ?? numberValue(value, "diagnosedPrecision") ?? perClassNumber(value, "precision");
  const recall = numberValue(value, "recall") ?? numberValue(value, "diagnosedRecall") ?? perClassNumber(value, "recall");
  const accuracy = numberValue(value, "accuracy");
  if (macroF1 === null || diagnosedF1 === null || precision === null || recall === null || accuracy === null) return null;
  const counts = countsFromMetrics(value);
  return { macroF1, diagnosedF1, controlF1, precision, recall, accuracy, ...counts };
}

function countsFromMetrics(metrics: JsonRecord): { tp: number; fp: number; tn: number; fn: number } {
  const tp = numberValue(metrics, "tp");
  const fp = numberValue(metrics, "fp");
  const tn = numberValue(metrics, "tn");
  const fn = numberValue(metrics, "fn");
  if (tp !== null && fp !== null && tn !== null && fn !== null) return { tp, fp, tn, fn };
  const confusion = recordValue(metrics, "confusion");
  const diagnosed = confusion ? recordValue(confusion, "diagnosed") : null;
  if (!confusion || !diagnosed) return { tp: 0, fp: 0, tn: 0, fn: 0 };
  const labels = Object.keys(confusion);
  const truePositive = numberValue(diagnosed, "diagnosed") ?? 0;
  const falseNegative = labels.filter((label) => label !== "diagnosed").reduce((sum, label) => sum + (numberValue(diagnosed, label) ?? 0), 0);
  const falsePositive = labels
    .filter((label) => label !== "diagnosed")
    .reduce((sum, label) => sum + (numberValue(recordValue(confusion, label), "diagnosed") ?? 0), 0);
  const total = labels.reduce((outer, actual) => {
    const row = recordValue(confusion, actual);
    return outer + labels.reduce((inner, predicted) => inner + (numberValue(row, predicted) ?? 0), 0);
  }, 0);
  return { tp: truePositive, fp: falsePositive, tn: total - truePositive - falsePositive - falseNegative, fn: falseNegative };
}

function perClassNumber(metrics: JsonRecord, key: string): number | null {
  const perClass = recordValue(metrics, "perClass");
  const diagnosed = perClass ? recordValue(perClass, "diagnosed") : null;
  return diagnosed ? numberValue(diagnosed, key) : null;
}

function decisionFromLock(lock: JsonRecord): string {
  const threshold = numberValue(lock, "threshold");
  if (threshold !== null) return `limiar ${threshold.toFixed(6)}`;
  const decisionRule = recordValue(lock, "decisionRule");
  return (decisionRule && stringValue(decisionRule, "ruleId")) ?? "regra travada";
}

function resultVariant(path: string): string {
  const file = path.split("/").pop() ?? path;
  if (file.includes("cached-llm")) return "sonda LLM apenas em cache";
  if (file.includes("llm-disambiguated")) return "teste selado desambiguado por LLM";
  if (file.startsWith("legacy-cnn-logreg-mlp-lock")) return "teste selado com arquitetura legada";
  return "teste selado";
}

function lockBasenameFromReport(path: string): string {
  const file = path.split("/").pop() ?? path;
  if (file.startsWith("legacy-cnn-logreg-mlp-lock")) return "legacy-cnn-logreg-mlp-lock";
  return "ensemble-lock";
}

function binaryPolicy(experimentId: string): string {
  if (experimentId.includes("anxiety")) return "ansiedade-binário";
  return experimentId.includes("ternary") ? "ternário" : "binário";
}

function trackFor(experimentId: string): Track {
  if (experimentId.includes("anxiety")) return "anxiety";
  return experimentId.includes("ternary") ? "ternary" : "binary";
}

function stringArray(value: readonly unknown[]): string[] {
  return value.filter((item): item is string => typeof item === "string");
}

function weightMap(value: JsonRecord | null): Record<string, number> {
  if (!value) return {};
  const out: Record<string, number> = {};
  for (const [key, raw] of Object.entries(value)) {
    if (typeof raw === "number" && Number.isFinite(raw)) out[key] = raw;
  }
  return out;
}

function weightText(value: number | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "n/d";
}

function policyManifest(policy: string): string {
  return `train_manifest_${policy}_seed42.csv`;
}

function modelFamily(modelId: string): string {
  const id = modelId.toLowerCase();
  if (id.includes("stack")) return "Stacking";
  if (id.includes("cnn")) return "CNN";
  if (id.includes("bilstm") || id.includes("lstm")) return "BiLSTM";
  if (id.includes("transformer")) return "Transformer";
  if (id.includes("focal")) return "Focal LogReg";
  if (id.includes("hierarchical")) return "Hierarchical LogReg";
  if (id.includes("relevance_baseline")) return "Relevance baseline";
  if (id.includes("multinomial") || id.includes("logreg")) return "LogReg";
  if (id.includes("xgb")) return "XGBoost";
  if (id.includes("hist_gradient") || id.includes("hgb")) return "HistGB";
  if (id.includes("extra_trees")) return "ExtraTrees";
  if (id.includes("mlp")) return "MLP";
  return "Other";
}

function familyChips(modelIds: readonly string[]): string {
  if (modelIds.length === 0) return `<span class="muted">n/d</span>`;
  const counts = new Map<string, number>();
  for (const id of modelIds) {
    const family = modelFamily(id);
    counts.set(family, (counts.get(family) ?? 0) + 1);
  }
  return `<span class="chips">${[...counts.entries()]
    .map(([family, count]) => `<span class="chip">${escapeHtml(count > 1 ? `${family} ×${count}` : family)}</span>`)
    .join("")}</span>`;
}

function trackLabel(track: Track): string {
  if (track === "anxiety") return "ansiedade";
  return track === "ternary" ? "ternário" : "binário";
}

function experimentFromPath(path: string): string {
  const parts = path.split("/");
  const index = parts.indexOf("setembrobr");
  return index >= 0 ? (parts[index + 1] ?? "setembrobr") : "setembrobr";
}

function titleFor(experimentId: string): string {
  return experimentTitles[experimentId] ?? experimentId;
}

function basenameWithoutJson(path: string): string {
  return (path.split("/").pop() ?? path).replace(/\.json$/u, "");
}

function compareResult(left: ResultEntry, right: ResultEntry): number {
  return compareMetric(right.test.macroF1, left.test.macroF1) || left.experimentTitle.localeCompare(right.experimentTitle) || left.variant.localeCompare(right.variant);
}

function compareLock(left: LockEntry, right: LockEntry): number {
  return left.experimentTitle.localeCompare(right.experimentTitle) || left.lockBasename.localeCompare(right.lockBasename);
}

function compareMetric(left: number | null | undefined, right: number | null | undefined): number {
  return (left ?? -Infinity) - (right ?? -Infinity);
}

function metric(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(6) : "n/d";
}

function signedMetric(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(6)}` : "n/d";
}

function deltaText(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/d";
  return `<span class="${value >= 0 ? "pos" : "neg"}">${signedMetric(value)}</span>`;
}

function deltaCell(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return `<span class="muted">n/d</span>`;
  return `<span class="${value >= 0 ? "pos" : "neg"}">${signedMetric(value)}</span>`;
}

function bestBy<T>(entries: readonly T[], score: (entry: T) => number | null | undefined): T | undefined {
  return [...entries].sort((left, right) => compareMetric(score(right), score(left)))[0];
}

function confusion(metricsValue: NormalizedMetrics | null | undefined): string {
  if (!metricsValue) return "n/d";
  return `<span class="chips"><span class="chip">VP ${metricsValue.tp}</span><span class="chip bad">FP ${metricsValue.fp}</span><span class="chip">VN ${metricsValue.tn}</span><span class="chip bad">FN ${metricsValue.fn}</span></span>`;
}

async function readJson(path: string): Promise<JsonRecord> {
  return JSON.parse(await readFile(path, "utf8")) as JsonRecord;
}

async function readEmbeddingGenerationProvenance(): Promise<EmbeddingGenerationProvenance> {
  const provenance = JSON.parse(await readFile(anxietyEmbeddingProvenancePath, "utf8")) as EmbeddingGenerationProvenance;
  const measuredSeconds = Math.round(
    (Date.parse(provenance.timing.completedAt) - Date.parse(provenance.timing.startedAt)) / 1000,
  );
  if (!Number.isFinite(measuredSeconds) || measuredSeconds !== provenance.timing.durationSeconds) {
    throw new Error(`Invalid anxiety embedding duration in ${anxietyEmbeddingProvenancePath}`);
  }
  if (provenance.workload.trainTweets + provenance.workload.testTweets !== provenance.workload.totalTweets) {
    throw new Error(`Invalid anxiety embedding tweet count in ${anxietyEmbeddingProvenancePath}`);
  }
  return provenance;
}

async function readAnxietyThesisComparison(): Promise<AnxietyThesisComparison> {
  const comparison = JSON.parse(await readFile(anxietyThesisComparisonPath, "utf8")) as AnxietyThesisComparison;
  if (comparison.predictionTarget !== "anxiety" || comparison.reportedResults.length === 0) {
    throw new Error(`Invalid anxiety thesis comparison in ${anxietyThesisComparisonPath}`);
  }
  for (const result of comparison.reportedResults) {
    const recomputedMacro = (result.controlF1 + result.anxietyF1) / 2;
    if (Math.abs(recomputedMacro - result.macroF1) > 0.005_001) {
      throw new Error(`Invalid rounded macro F1 for ${result.model} in ${anxietyThesisComparisonPath}`);
    }
  }
  return comparison;
}

function durationText(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return `${hours}h ${minutes}min ${seconds}s`;
}

function localTimestamp(value: string): string {
  return `${value.slice(0, 10)} ${value.slice(11, 19)} BRT`;
}

function integer(value: number): string {
  return new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 0 }).format(value);
}

async function writeFileEnsured(path: string, text: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, text);
}

function recordValue(value: unknown, key: string): JsonRecord | null {
  if (!isRecord(value)) return null;
  const nested = value[key];
  return isRecord(nested) ? nested : null;
}

function arrayValue(value: unknown, key: string): unknown[] | null {
  if (!isRecord(value)) return null;
  const nested = value[key];
  return Array.isArray(nested) ? nested : null;
}

function stringValue(value: unknown, key: string): string | null {
  if (!isRecord(value)) return null;
  const nested = value[key];
  return typeof nested === "string" ? nested : null;
}

function numberValue(value: unknown, key: string): number | null {
  if (!isRecord(value)) return null;
  const nested = value[key];
  return typeof nested === "number" && Number.isFinite(nested) ? nested : null;
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function escapeHtml(value: unknown): string {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
