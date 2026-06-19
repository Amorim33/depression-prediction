#!/usr/bin/env python3
"""Prepare strict-blind raw Qwen3 binary experiment artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from raw_ternary_prepare_setembrobr import (  # noqa: E402
    export_sequences,
    prepare_features,
    read_config,
    read_raw_manifest,
    resolve_path,
    sha256_file,
    write_json,
    write_strict_blind_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/setembrobr.seed42.raw-qwen3-binary.json")
    parser.add_argument("--mode", choices=["prepare", "export-sequences", "all"], default="all")
    parser.add_argument("--raw-artifacts-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--top-n", type=int, default=128)
    parser.add_argument("--sequence-order", choices=["relevance_desc", "recent_chronological"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = Path.cwd()
    config_path = resolve_path(args.config, repo_dir)
    config = read_config(config_path)
    raw_artifacts_dir = resolve_path(args.raw_artifacts_dir or config["rawArtifactsDir"], repo_dir)
    output_dir = resolve_path(args.output_dir or config["outputDir"], repo_dir)
    seed = int(config["seed"])
    sequence_export = config.get("sequenceExport", {})
    sequence_order = args.sequence_order or sequence_export.get("order", "relevance_desc")
    raw_rows = read_raw_manifest(raw_artifacts_dir, seed)

    manifest_report = write_strict_blind_manifests(output_dir, raw_rows, seed)
    report: dict[str, object] = {
        "dataset": "setembrobr",
        "seed": seed,
        "config": str(config_path),
        "rawArtifactsDir": str(raw_artifacts_dir),
        "outputDir": str(output_dir),
        "mode": args.mode,
        "manifest": manifest_report,
        "rawEmbeddingManifestHash": sha256_file(raw_artifacts_dir / "reports" / "embedding_generation_manifest.json"),
    }
    if args.mode in {"prepare", "all"}:
        report.update(prepare_features(raw_artifacts_dir, output_dir, raw_rows, seed))
    if args.mode in {"export-sequences", "all"}:
        report.update(export_sequences(raw_artifacts_dir, output_dir, raw_rows, seed, args.top_n, sequence_order))

    report_path = output_dir / "reports" / "raw_binary_prepare_manifest.json"
    write_json(report_path, report)
    print(json.dumps({"status": "ok", "report": str(report_path)}))


if __name__ == "__main__":
    main()
