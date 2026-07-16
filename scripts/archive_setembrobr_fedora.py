#!/usr/bin/env python3
"""Create, verify, and restore resumable lossless SetembroBR research archives.

The archive is deliberately file-chunked. Large Parquet/CSV/Pickle files are
wrapped in independent Zstandard frames, while already compressed binary files
are stored byte-for-byte. Each source is deleted only after both its archive
hash and restored-byte hash have been verified and the durable manifest record
has been flushed to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


COMPRESS_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".parquet",
    ".pkl",
    ".tsv",
    ".txt",
}
BUFFER_BYTES = 8 * 1024 * 1024


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(BUFFER_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_zstd_payload(path: Path) -> str:
    process = subprocess.Popen(
        ["zstd", "-q", "-d", "-c", str(path)],
        stdout=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError(f"unable to read zstd output for {path}")
    digest = sha256_stream(process.stdout)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"zstd decompression failed for {path} with exit code {return_code}")
    return digest


def fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fsync_file(temporary)
    os.replace(temporary, path)
    fsync_dir(path.parent)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def parse_source(value: str) -> tuple[str, Path]:
    source_id, separator, raw_path = value.partition("=")
    if not separator or not source_id or not raw_path:
        raise argparse.ArgumentTypeError("source must use SOURCE_ID=/absolute/path")
    if source_id.startswith("/") or ".." in Path(source_id).parts:
        raise argparse.ArgumentTypeError(f"unsafe source id: {source_id}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"source path is not absolute: {path}")
    return source_id, path


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def record_key(record: dict[str, Any]) -> str:
    return f"{record['sourceId']}::{record['relativePath']}"


def source_key(source_id: str, relative_path: Path) -> str:
    return f"{source_id}::{relative_path.as_posix()}"


def archive_relative_path(source_id: str, relative_path: Path, codec: str) -> Path:
    payload_path = Path("payload") / source_id / relative_path
    if codec == "zstd":
        return payload_path.with_name(f"{payload_path.name}.zst")
    return payload_path


def verify_record(archive_root: Path, record: dict[str, Any], *, payload: bool) -> None:
    archive_path = archive_root / str(record["archivePath"])
    if not archive_path.is_file():
        raise RuntimeError(f"archive payload is missing: {archive_path}")
    if archive_path.stat().st_size != int(record["archiveSize"]):
        raise RuntimeError(f"archive size mismatch: {archive_path}")
    if sha256_file(archive_path) != record["archiveSha256"]:
        raise RuntimeError(f"archive SHA-256 mismatch: {archive_path}")
    if payload:
        restored_hash = (
            sha256_zstd_payload(archive_path)
            if record["codec"] == "zstd"
            else record["archiveSha256"]
        )
        if restored_hash != record["originalSha256"]:
            raise RuntimeError(f"restored payload SHA-256 mismatch: {archive_path}")


def choose_codec(source_path: Path) -> str:
    return "zstd" if source_path.suffix.lower() in COMPRESS_SUFFIXES else "none"


def create_zstd_archive(source_path: Path, destination: Path, level: int) -> tuple[str, int, str]:
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    subprocess.run(
        ["zstd", f"-{level}", "-T0", "-q", "-f", str(source_path), "-o", str(temporary)],
        check=True,
    )
    subprocess.run(["zstd", "-q", "-t", str(temporary)], check=True)
    restored_hash = sha256_zstd_payload(temporary)
    archive_hash = sha256_file(temporary)
    archive_size = temporary.stat().st_size
    fsync_file(temporary)
    os.replace(temporary, destination)
    fsync_dir(destination.parent)
    return archive_hash, archive_size, restored_hash


def create_raw_archive(source_path: Path, destination: Path) -> tuple[str, int, str]:
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    shutil.copy2(source_path, temporary)
    archive_hash = sha256_file(temporary)
    archive_size = temporary.stat().st_size
    fsync_file(temporary)
    os.replace(temporary, destination)
    fsync_dir(destination.parent)
    return archive_hash, archive_size, archive_hash


def remove_empty_directories(root: Path) -> None:
    if not root.exists():
        return
    directories = sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True)
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass


def validate_sources(sources: list[tuple[str, Path]], archive_root: Path) -> None:
    source_ids = [source_id for source_id, _ in sources]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("duplicate source id")
    for source_id, path in sources:
        if not path.is_dir():
            raise RuntimeError(f"source directory does not exist for {source_id}: {path}")
        if archive_root == path or archive_root.is_relative_to(path):
            raise RuntimeError(f"archive root must not be inside source directory {path}")
        for candidate in path.rglob("*"):
            if candidate.is_symlink():
                raise RuntimeError(f"symbolic links are not supported: {candidate}")


def build_summary(
    archive_root: Path,
    records: list[dict[str, Any]],
    sources: list[tuple[str, Path]],
    started_at: str,
    free_bytes_before: int,
) -> dict[str, Any]:
    original_bytes = sum(int(record["originalSize"]) for record in records)
    archive_bytes = sum(int(record["archiveSize"]) for record in records)
    return {
        "archiveFormat": "setembrobr-file-chunks-v1",
        "archiveRoot": str(archive_root),
        "startedAt": started_at,
        "completedAt": now_iso(),
        "sourceRoots": {source_id: str(path) for source_id, path in sources},
        "fileCount": len(records),
        "codecCounts": dict(sorted(Counter(str(record["codec"]) for record in records).items())),
        "originalBytes": original_bytes,
        "archiveBytes": archive_bytes,
        "spaceSavedBytes": original_bytes - archive_bytes,
        "archiveRatio": archive_bytes / original_bytes if original_bytes else 0,
        "freeBytesBefore": free_bytes_before,
        "freeBytesAfter": shutil.disk_usage(archive_root).free,
        "hashAlgorithm": "sha256",
        "sourceDeletionPolicy": "delete only after archive hash and restored-byte hash pass",
        "distribution": "restricted research archive; see README.md",
    }


def write_archive_indexes(archive_root: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_json(archive_root / "archive-summary.json", summary)
    checksum_lines = [
        f"{record['archiveSha256']}  {record['archivePath']}"
        for record in sorted(records, key=lambda item: str(item["archivePath"]))
    ]
    checksum_path = archive_root / "SHA256SUMS"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    fsync_file(checksum_path)
    readme = f"""# SetembroBR restricted research archive

This directory is a lossless, resumable archive of the depression and anxiety
SetembroBR v6 datasets and Qwen3 tweet-level embedding artifacts.

- Format: independent file chunks (`.zst` where compression helps; otherwise raw bytes)
- Files: {summary['fileCount']}
- Original bytes: {summary['originalBytes']}
- Archived bytes: {summary['archiveBytes']}
- SHA-256: `SHA256SUMS` and `.archive-state/manifest.jsonl`
- Restore/verify tool: `archive_setembrobr_fedora.py`

Verify all archive and restored-payload hashes:

```bash
python3 archive_setembrobr_fedora.py --mode verify --archive-root {archive_root}
```

Restore files to their recorded original absolute paths:

```bash
python3 archive_setembrobr_fedora.py --mode restore --archive-root {archive_root}
```

## Distribution restriction

This archive contains raw X/Twitter text, sensitive mental-health labels, and
derived embeddings. It is prepared for controlled preservation and authorized
research transfer, not public publication. Current X Developer Policy generally
limits third-party redistribution to Post/User IDs. Confirm institutional ethics,
the corpus access agreement, and current platform terms before any transfer.
Policy: https://docs.x.com/developer-terms/policy
"""
    readme_path = archive_root / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    fsync_file(readme_path)
    tool_copy = archive_root / "archive_setembrobr_fedora.py"
    shutil.copy2(Path(__file__).resolve(), tool_copy)
    fsync_file(tool_copy)


def archive(args: argparse.Namespace) -> None:
    if not args.source:
        raise RuntimeError("archive mode requires at least one --source")
    if not args.delete_after_verify:
        raise RuntimeError("archive mode requires explicit --delete-after-verify")
    archive_root = args.archive_root.expanduser().resolve()
    sources = [parse_source(value) for value in args.source]
    archive_root.mkdir(parents=True, exist_ok=True)
    validate_sources(sources, archive_root)
    state_dir = archive_root / ".archive-state"
    manifest_path = state_dir / "manifest.jsonl"
    records = load_records(manifest_path)
    by_key = {record_key(record): record for record in records}
    started_at = now_iso()
    free_bytes_before = shutil.disk_usage(archive_root).free
    candidates: list[tuple[str, Path, Path]] = []
    for source_id, root in sources:
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            candidates.append((source_id, root, path))

    for index, (source_id, source_root, source_path) in enumerate(candidates, start=1):
        relative_path = source_path.relative_to(source_root)
        key = source_key(source_id, relative_path)
        existing = by_key.get(key)
        if existing:
            archive_path = archive_root / str(existing["archivePath"])
            if not archive_path.is_file() or archive_path.stat().st_size != int(existing["archiveSize"]):
                raise RuntimeError(f"incomplete existing archive record: {key}")
            if sha256_file(source_path) != existing["originalSha256"]:
                raise RuntimeError(f"source changed after archive record was written: {source_path}")
            source_path.unlink()
            continue

        source_stat = source_path.stat()
        if not stat.S_ISREG(source_stat.st_mode):
            raise RuntimeError(f"source is not a regular file: {source_path}")
        original_hash = sha256_file(source_path)
        original_size = source_stat.st_size
        codec = choose_codec(source_path)
        destination_relative = archive_relative_path(source_id, relative_path, codec)
        destination = archive_root / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"untracked archive payload already exists: {destination}")

        if codec == "zstd":
            archive_hash, archive_size, restored_hash = create_zstd_archive(
                source_path,
                destination,
                args.zstd_level,
            )
            if archive_size >= original_size:
                destination.unlink()
                codec = "none"
                destination_relative = archive_relative_path(source_id, relative_path, codec)
                destination = archive_root / destination_relative
                archive_hash, archive_size, restored_hash = create_raw_archive(source_path, destination)
        else:
            archive_hash, archive_size, restored_hash = create_raw_archive(source_path, destination)

        if restored_hash != original_hash:
            raise RuntimeError(f"restored payload mismatch before deletion: {source_path}")
        record = {
            "sourceId": source_id,
            "sourceRoot": str(source_root),
            "sourcePath": str(source_path),
            "relativePath": relative_path.as_posix(),
            "archivePath": destination_relative.as_posix(),
            "codec": codec,
            "originalSize": original_size,
            "archiveSize": archive_size,
            "originalSha256": original_hash,
            "archiveSha256": archive_hash,
            "mode": stat.S_IMODE(source_stat.st_mode),
            "mtimeNs": source_stat.st_mtime_ns,
            "completedAt": now_iso(),
        }
        append_jsonl(manifest_path, record)
        records.append(record)
        by_key[key] = record
        source_path.unlink()
        if index % args.progress_every == 0 or index == len(candidates):
            print(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(candidates),
                        "source": str(source_path),
                        "freeGiB": round(shutil.disk_usage(archive_root).free / (1024**3), 2),
                    }
                ),
                flush=True,
            )

    for _, source_root in sources:
        remove_empty_directories(source_root)
    summary = build_summary(archive_root, records, sources, started_at, free_bytes_before)
    write_archive_indexes(archive_root, records, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def verify(args: argparse.Namespace) -> None:
    archive_root = args.archive_root.expanduser().resolve()
    records = load_records(archive_root / ".archive-state" / "manifest.jsonl")
    if not records:
        raise RuntimeError(f"archive manifest is empty: {archive_root}")
    for index, record in enumerate(records, start=1):
        verify_record(archive_root, record, payload=True)
        if index % args.progress_every == 0 or index == len(records):
            print(json.dumps({"verified": index, "total": len(records)}), flush=True)
    print(json.dumps({"ok": True, "verifiedFiles": len(records), "archiveRoot": str(archive_root)}, indent=2))


def restore(args: argparse.Namespace) -> None:
    archive_root = args.archive_root.expanduser().resolve()
    records = load_records(archive_root / ".archive-state" / "manifest.jsonl")
    if not records:
        raise RuntimeError(f"archive manifest is empty: {archive_root}")
    for index, record in enumerate(records, start=1):
        destination = Path(str(record["sourcePath"]))
        if destination.exists():
            if sha256_file(destination) != record["originalSha256"]:
                raise RuntimeError(f"restore destination already exists with different bytes: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.restore-partial")
        temporary.unlink(missing_ok=True)
        archive_path = archive_root / str(record["archivePath"])
        if record["codec"] == "zstd":
            subprocess.run(["zstd", "-q", "-d", "-f", str(archive_path), "-o", str(temporary)], check=True)
        else:
            shutil.copy2(archive_path, temporary)
        if sha256_file(temporary) != record["originalSha256"]:
            raise RuntimeError(f"restored file SHA-256 mismatch: {destination}")
        os.chmod(temporary, int(record["mode"]))
        os.utime(temporary, ns=(int(record["mtimeNs"]), int(record["mtimeNs"])))
        fsync_file(temporary)
        os.replace(temporary, destination)
        fsync_dir(destination.parent)
        if index % args.progress_every == 0 or index == len(records):
            print(json.dumps({"restored": index, "total": len(records)}), flush=True)
    print(json.dumps({"ok": True, "restoredFiles": len(records)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("archive", "verify", "restore"), required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], help="SOURCE_ID=/absolute/path")
    parser.add_argument("--delete-after-verify", action="store_true")
    parser.add_argument("--zstd-level", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.zstd_level < 1 or args.zstd_level > 19:
        raise RuntimeError("zstd level must be between 1 and 19")
    if args.progress_every < 1:
        raise RuntimeError("progress interval must be positive")
    if shutil.which("zstd") is None:
        raise RuntimeError("zstd executable was not found")
    if args.mode == "archive":
        archive(args)
    elif args.mode == "verify":
        verify(args)
    else:
        restore(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"archive error: {error}", file=sys.stderr)
        raise
