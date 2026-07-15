import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";

describe("Fedora SetembroBR archive tool", () => {
  test("requires verified, resumable deletion and supports full restoration", async () => {
    const source = await readFile("scripts/archive_setembrobr_fedora.py", "utf8");

    expect(source).toContain('choices=("archive", "verify", "restore")');
    expect(source).toContain("archive mode requires explicit --delete-after-verify");
    expect(source).toContain("restored payload mismatch before deletion");
    expect(source).toContain("append_jsonl(manifest_path, record)");
    expect(source.indexOf("append_jsonl(manifest_path, record)")).toBeLessThan(
      source.indexOf("source_path.unlink()", source.indexOf("append_jsonl(manifest_path, record)")),
    );
    expect(source).toContain("restored file SHA-256 mismatch");
    expect(source).toContain("symbolic links are not supported");
    expect(source).toContain("restricted research archive");
  });

  test("records the completed restricted archive and independent verification", async () => {
    const record = JSON.parse(
      await readFile("docs/fedora-setembrobr-archive.json", "utf8"),
    ) as {
      archive: {
        fileCount: number;
        originalBytes: number;
        archivedPayloadBytes: number;
        spaceSavedBytes: number;
        codecCounts: { zstd: number; none: number };
      };
      storage: { windowsNtfsPartitionUsed: boolean };
      integrity: {
        independentFullVerification: {
          ok: boolean;
          verifiedFiles: number;
          recordedAt: string;
          method: string;
        };
        manifestJsonlSha256: string;
        sha256SumsSha256: string;
      };
      access: { publicReleaseAllowed: boolean };
    };

    expect(record.archive.fileCount).toBe(1_036);
    expect(record.archive.codecCounts.zstd + record.archive.codecCounts.none).toBe(
      record.archive.fileCount,
    );
    expect(record.archive.originalBytes - record.archive.archivedPayloadBytes).toBe(
      record.archive.spaceSavedBytes,
    );
    expect(record.integrity.independentFullVerification).toEqual({
      ok: true,
      verifiedFiles: 1_036,
      recordedAt: "2026-07-15T10:27:30-03:00",
      method:
        "Reread every archived object, verify its archived size and SHA-256, stream-decompress each Zstandard object, and recompute the original payload SHA-256.",
    });
    expect(record.integrity.manifestJsonlSha256).toMatch(/^[a-f0-9]{64}$/);
    expect(record.integrity.sha256SumsSha256).toMatch(/^[a-f0-9]{64}$/);
    expect(record.storage.windowsNtfsPartitionUsed).toBe(false);
    expect(record.access.publicReleaseAllowed).toBe(false);
  });
});
