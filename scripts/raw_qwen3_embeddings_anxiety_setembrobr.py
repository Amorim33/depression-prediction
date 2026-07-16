#!/usr/bin/env python3
"""Validate raw SetembroBR anxiety CSVs and generate Qwen3 embedding artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from raw_qwen3_embeddings_setembrobr import (
    UserRecord,
    command_output,
    ensure_dirs,
    generate_embeddings,
    normalize_label,
    normalize_user_id,
    package_versions,
    resolve_path,
    selected_records,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/setembrobr.seed42.raw-qwen3-anxiety-embeddings.json")
    parser.add_argument("--mode", choices=["validate", "embed"], default="validate")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--smoke-users", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shard-users", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def label_code(label_name: str) -> int:
    return 1 if label_name == "diagnosed" else 0


def validate_split(
    dataset_dir: Path,
    split: str,
    filenames: list[str],
    expected_users: int,
    expected_tweets: int,
    tweet_delimiter: str,
    redact_labels: bool,
) -> tuple[list[UserRecord], dict[str, Any], dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    records: list[UserRecord] = []
    seen: set[str] = set()
    hashes: dict[str, str] = {}
    label_counts: dict[str, int] = {}
    total_tweets = 0
    empty_segments = 0
    min_tweets: int | None = None
    max_tweets = 0

    for filename in filenames:
        path = dataset_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[f"{split}/{filename}"] = sha256_file(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            required = {"User_ID", "Diagnosed_YN", "Text"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"{path} is missing required columns {sorted(required)}")
            for row in reader:
                user_id = normalize_user_id(row["User_ID"])
                if user_id in seen:
                    raise ValueError(f"duplicate user {user_id} in raw CSV files for {split}")
                seen.add(user_id)
                label_name = normalize_label(row["Diagnosed_YN"])
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
                texts = str(row["Text"]).split(tweet_delimiter)
                tweet_count = len(texts)
                total_tweets += tweet_count
                empty_segments += sum(not text.strip() for text in texts)
                min_tweets = tweet_count if min_tweets is None else min(min_tweets, tweet_count)
                max_tweets = max(max_tweets, tweet_count)
                records.append(
                    UserRecord(
                        split=split,
                        user_id=user_id,
                        label_name=label_name,
                        label_code=label_code(label_name),
                        texts=texts,
                        relevance_raw=[None] * tweet_count,
                        relevance_score=np.zeros((tweet_count,), dtype=np.int16),
                    )
                )

    if len(records) != expected_users:
        raise ValueError(f"{split} users: expected {expected_users}, got {len(records)}")
    if total_tweets != expected_tweets:
        raise ValueError(f"{split} tweets: expected {expected_tweets}, got {total_tweets}")

    report: dict[str, Any] = {
        "split": split,
        "users": len(records),
        "expectedUsers": expected_users,
        "tweetCount": total_tweets,
        "expectedTweets": expected_tweets,
        "minTweets": min_tweets or 0,
        "maxTweets": max_tweets,
        "emptyTweetSegments": empty_segments,
        "relevanceAvailable": False,
    }
    if not redact_labels:
        report["labelCounts"] = dict(sorted(label_counts.items()))
    return records, report, hashes


def stratified_folds(records: list[UserRecord], seed: int, folds: int) -> dict[str, int]:
    by_label: dict[int, list[UserRecord]] = {}
    for record in records:
        by_label.setdefault(record.label_code, []).append(record)
    assignments: dict[str, int] = {}
    for label, label_records in sorted(by_label.items()):
        shuffled = list(label_records)
        random.Random(seed + label).shuffle(shuffled)
        for index, record in enumerate(shuffled):
            assignments[record.user_id] = index % folds
    return assignments


def row_hash(dataset_id: str, split: str, user_id: str, label: str) -> str:
    value = f"{dataset_id}|raw-qwen3|{split}|{label}|{user_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_manifest_files(
    output_dir: Path,
    records_by_split: dict[str, list[UserRecord]],
    dataset_id: str,
    seed: int,
    folds: int,
) -> str:
    manifest_path = output_dir / "manifests" / f"raw_split_manifest_seed{seed}.csv"
    train_folds = stratified_folds(records_by_split["train"], seed, folds)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "split", "user_id", "label", "fold", "row_hash"])
        writer.writeheader()
        for split in ["train", "test"]:
            for record in records_by_split[split]:
                redacted = split == "test"
                writer.writerow(
                    {
                        "dataset": dataset_id,
                        "split": split,
                        "user_id": record.user_id,
                        "label": "" if redacted else record.label_code,
                        "fold": train_folds[record.user_id] if split == "train" else "",
                        "row_hash": row_hash(
                            dataset_id,
                            split,
                            record.user_id,
                            "redacted" if redacted else str(record.label_code),
                        ),
                    }
                )

    test_users_path = output_dir / "manifests" / f"raw_test_users_seed{seed}.csv"
    with test_users_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id"])
        writer.writeheader()
        for record in records_by_split["test"]:
            writer.writerow({"user_id": record.user_id})
    return sha256_file(manifest_path)


def main() -> None:
    args = parse_args()
    repo_dir = Path.cwd()
    config_path = resolve_path(args.config, repo_dir)
    config = load_config(config_path)
    dataset_dir = resolve_path(args.dataset_dir or config["datasetDir"], repo_dir)
    output_dir = resolve_path(args.output_dir or config["outputDir"], repo_dir)
    ensure_dirs(output_dir)

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    invalid_splits = [split for split in splits if split not in {"train", "test"}]
    if invalid_splits:
        raise ValueError(f"unsupported splits: {invalid_splits}")

    records_by_split: dict[str, list[UserRecord]] = {}
    split_reports: dict[str, dict[str, Any]] = {}
    dataset_hashes: dict[str, str] = {}
    for split in ["train", "test"]:
        records, report, hashes = validate_split(
            dataset_dir=dataset_dir,
            split=split,
            filenames=list(config["rawCsvFiles"][split]),
            expected_users=int(config["expectedUsers"][split]),
            expected_tweets=int(config["expectedTweets"][split]),
            tweet_delimiter=str(config["tweetDelimiter"]),
            redact_labels=split == "test" and bool(config.get("redactTestLabels", True)),
        )
        records_by_split[split] = records
        split_reports[split] = report
        dataset_hashes.update(hashes)

    selected = selected_records(records_by_split, args.smoke_users)
    seed = int(config["seed"])
    manifest_hash = write_manifest_files(
        output_dir,
        selected,
        str(config["datasetId"]),
        seed,
        int(config.get("folds", 5)),
    )
    validation_payload = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "mode": "validate",
        "config": config,
        "datasetDir": str(dataset_dir),
        "outputDir": str(output_dir),
        "splitReports": split_reports,
        "selectedUsers": {split: len(records) for split, records in selected.items()},
        "datasetHashes": dataset_hashes,
        "relevanceHashes": {},
        "splitManifestHash": manifest_hash,
        "command": sys.argv,
    }
    write_json(output_dir / "reports" / "raw_validation_report.json", validation_payload)
    if args.mode == "validate":
        print(json.dumps({"status": "ok", "report": str(output_dir / "reports" / "raw_validation_report.json")}))
        return

    generation = generate_embeddings(
        output_dir=output_dir,
        records_by_split={split: selected[split] for split in splits},
        config=config,
        splits=splits,
        batch_size=int(args.batch_size or config.get("batchSize", 16)),
        shard_users=int(args.shard_users or config.get("shardUsers", 64)),
        device_request=args.device,
        force=args.force,
    )
    manifest_payload = {
        **validation_payload,
        "mode": "embed",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "gpu": command_output(["nvidia-smi"]),
        "packageVersions": package_versions(),
        "embedding": generation,
        "command": sys.argv,
    }
    manifest_path = output_dir / "reports" / "embedding_generation_manifest.json"
    write_json(manifest_path, manifest_payload)
    print(json.dumps({"status": "ok", "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
