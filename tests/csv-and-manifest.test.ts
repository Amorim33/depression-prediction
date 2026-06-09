import { describe, expect, test } from "bun:test";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseCsv, writeCsv } from "../src/csv.ts";
import { manifestHash, readManifest, writeManifest } from "../src/manifest.ts";
import type { ManifestRow } from "../src/types.ts";

describe("csv and manifest", () => {
  test("round-trips quoted CSV cells", () => {
    const text = writeCsv(["a", "b"], [{ a: "x,y", b: 'quote " ok' }]);
    expect(parseCsv(text)).toEqual([{ a: "x,y", b: 'quote " ok' }]);
  });

  test("writes deterministic manifest bytes and hash", async () => {
    const rows: ManifestRow[] = [
      { dataset: "setembrobr", split: "train", label: "diagnosed", userId: "D1", rowHash: "h1", fold: 1 },
      { dataset: "setembrobr", split: "test", label: "control", userId: "C1", rowHash: "h2", fold: null },
    ];
    const dir = await mkdtemp(join(tmpdir(), "manifest-"));
    const a = join(dir, "a.csv");
    const b = join(dir, "b.csv");
    await writeManifest(a, rows);
    await writeManifest(b, rows);
    expect(await readFile(a, "utf8")).toBe(await readFile(b, "utf8"));
    expect(manifestHash(await readManifest(a))).toBe(manifestHash(await readManifest(b)));
  });
});

