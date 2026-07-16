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
      storage: {
        windowsNtfsPartitionUsed: boolean;
        freeBytesAfterExternalMove: number;
        externalSsd: {
          diskById: string;
          diskBytes: number;
          filesystem: string;
          filesystemUuid: string;
          label: string;
          stateAtCompletion: string;
        };
        externalMove: {
          verifiedFiles: number;
          sourceAfterVerification: string;
          serviceResult: string;
          unmountMonitorResult: string;
        };
      };
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
    expect(record.storage.freeBytesAfterExternalMove).toBe(310_130_343_936);
    expect(record.storage.externalSsd).toMatchObject({
      diskById:
        "/dev/disk/by-id/usb-SanDisk_Portable_SSD_323534304551343035333837-0:0",
      diskBytes: 1_000_204_886_016,
      filesystem: "exfat",
      filesystemUuid: "6A57-8F8E",
      label: "SETEMBROBR",
      stateAtCompletion: "verified and unmounted",
    });
    expect(record.storage.externalMove).toMatchObject({
      verifiedFiles: 1_036,
      sourceAfterVerification: "removed",
      serviceResult: "success",
      unmountMonitorResult: "success",
    });
    expect(record.access.publicReleaseAllowed).toBe(false);
  });

  test("guards the external move and deletes the source only after full SSD verification", async () => {
    const move = await readFile("scripts/move_setembrobr_archive_to_ssd.sh", "utf8");
    const unmount = await readFile(
      "scripts/unmount_setembrobr_ssd_after_move.sh",
      "utf8",
    );

    expect(move).toContain(
      'readonly DISK_BY_ID="/dev/disk/by-id/usb-SanDisk_Portable_SSD_323534304551343035333837-0:0"',
    );
    expect(move).toContain('readonly EXPECTED_DISK_BYTES="1000204886016"');
    expect(move).toContain('readonly EXPECTED_VOLUME_UUID="6A57-8F8E"');
    expect(move).toContain('cd "$DESTINATION"');
    expect(move).toContain("sha256sum -c --quiet SHA256SUMS");
    expect(
      move.indexOf('log "SSD_HASH_VERIFICATION_COMPLETE files=${EXPECTED_FILE_COUNT}"'),
    ).toBeLessThan(move.indexOf('rm -rf --one-file-system "$SOURCE"'));
    expect(move).toContain('log "SOURCE_REMOVED path=${SOURCE}"');
    expect(unmount).toContain('if [[ $move_status != "0" ]]');
    expect(unmount).toContain("ssd_left_mounted=true");
    expect(unmount).toContain("SSD_UNMOUNTED_AFTER_VERIFIED_MOVE");
  });
});
