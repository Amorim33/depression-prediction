import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { writeJson } from "../src/artifacts.ts";
import { writeCsv } from "../src/csv.ts";
import { assertSafeIdentifier, createDatabaseClient } from "../src/db.ts";
import { loadDatabaseUrl } from "../src/config.ts";
import { manifestHash, readManifest } from "../src/manifest.ts";
import { loadTernaryConfig, resolveSourceOutputPath, resolveTernaryOutputPath } from "../src/ternary-config.ts";
import { computeEvidenceScore, evidenceMarkerToCsvRecord } from "../src/ternary.ts";
import type { EvidenceMarker, SplitName } from "../src/types.ts";

const config = await loadTernaryConfig();
const sourceManifestPath = resolveSourceOutputPath(config, "manifest", `split_manifest_seed${config.seed}.csv`);
const manifestRows = await readManifest(sourceManifestPath);
const sourceManifestHash = manifestHash(manifestRows);
const client = createDatabaseClient(await loadDatabaseUrl());

try {
  for (const split of ["train", "test"] as const satisfies readonly SplitName[]) {
    const table = config.database.tables[split === "train" ? "trainEmbeddings" : "testEmbeddings"];
    const markers = await loadMarkers(table);
    const splitRows = manifestRows.filter((row) => row.split === split);
    const aligned = splitRows.map((row) => {
      const marker = markers.get(row.userId);
      if (!marker) throw new Error(`${split} evidence markers missing manifest user ${row.userId}`);
      return marker;
    });
    const outputPath = resolveTernaryOutputPath(config, "evidence-markers", `${split}_markers.csv`);
    await mkdir(dirname(outputPath), { recursive: true });
    await writeFile(
      outputPath,
      writeCsv(
        [
          "user_id",
          "total_tweets",
          "max_relevance",
          "rel3_count",
          "rel5_count",
          "rel6_count",
          "rel7_count",
          "rel3_ratio",
          "rel5_ratio",
          "rel6_ratio",
          "rel7_ratio",
          "top10_avg_relevance",
          "evidence_score",
        ],
        aligned.map(evidenceMarkerToCsvRecord),
      ),
    );
    console.log(`wrote ${outputPath}`);
  }

  await writeJson(resolveTernaryOutputPath(config, "evidence-markers", "evidence-marker-meta.json"), {
    dataset: "setembrobr",
    seed: config.seed,
    sourceManifestHash,
    evidenceFormulaVersion: "v1",
    rawTextIncluded: false,
    dbTables: {
      trainEmbeddings: config.database.tables.trainEmbeddings,
      testEmbeddings: config.database.tables.testEmbeddings,
    },
    createdAt: new Date(0).toISOString(),
  });
} finally {
  await client.end();
}

async function loadMarkers(table: string): Promise<Map<string, EvidenceMarker>> {
  assertSafeIdentifier(table);
  const rows = await client.sql.unsafe(`
    with ranked as (
      select user_id,
             coalesce(gpt_3_5_relevance, 0)::float8 as rel,
             row_number() over (
               partition by user_id
               order by coalesce(gpt_3_5_relevance, 0) desc, tweet_index desc
             ) as rel_rank
      from ${table}
    )
    select user_id,
           count(*)::float8 as total_tweets,
           coalesce(max(rel), 0)::float8 as max_relevance,
           count(*) filter (where rel >= 3)::float8 as rel3_count,
           count(*) filter (where rel >= 5)::float8 as rel5_count,
           count(*) filter (where rel >= 6)::float8 as rel6_count,
           count(*) filter (where rel >= 7)::float8 as rel7_count,
           coalesce(avg(rel) filter (where rel_rank <= 10), 0)::float8 as top10_avg_relevance
    from ranked
    group by user_id
    order by user_id
  `);
  const out = new Map<string, EvidenceMarker>();
  for (const row of rows) {
    const totalTweets = Number(row.total_tweets);
    const denom = Math.max(totalTweets, 1);
    const partial = {
      userId: String(row.user_id),
      totalTweets,
      maxRelevance: Number(row.max_relevance),
      rel3Count: Number(row.rel3_count),
      rel5Count: Number(row.rel5_count),
      rel6Count: Number(row.rel6_count),
      rel7Count: Number(row.rel7_count),
      rel3Ratio: Number(row.rel3_count) / denom,
      rel5Ratio: Number(row.rel5_count) / denom,
      rel6Ratio: Number(row.rel6_count) / denom,
      rel7Ratio: Number(row.rel7_count) / denom,
      top10AvgRelevance: Number(row.top10_avg_relevance),
    };
    out.set(partial.userId, { ...partial, evidenceScore: computeEvidenceScore(partial) });
  }
  return out;
}
