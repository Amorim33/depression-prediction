#!/usr/bin/env python3
"""Export selected tweet timelines for the LLM disambiguator."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/setembrobr.seed42.raw-qwen3-ternary-symmetric.json")
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--users-file", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--raw-artifacts-dir")
    parser.add_argument("--max-evidence-tweets", type=int, default=256)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_config(path: Path) -> dict[str, Any]:
    parsed = read_json(path)
    parent_ref = parsed.get("extends")
    if not isinstance(parent_ref, str):
        return parsed
    parent_path = Path(parent_ref).expanduser()
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()
    child = {key: value for key, value in parsed.items() if key != "extends"}
    return deep_merge(read_config(parent_path), child)


def deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for key, value in child.items():
        parent_value = merged.get(key)
        if isinstance(parent_value, dict) and isinstance(value, dict):
            merged[key] = deep_merge(parent_value, value)
        else:
            merged[key] = value
    return merged


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def read_users(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    users = [str(row["user_id"]).strip() for row in rows if str(row.get("user_id", "")).strip()]
    if not users:
        raise ValueError(f"{path} did not contain users")
    if len(set(users)) != len(users):
        raise ValueError(f"{path} contains duplicate users")
    return users


def relevance_value(raw: Any) -> int:
    if raw is None:
        return 0
    try:
        if isinstance(raw, float) and math.isnan(raw):
            return 0
        numeric = int(raw)
    except (TypeError, ValueError):
        return 0
    return min(max(numeric, 0), 10)


def selected_tweets(rows: list[dict[str, Any]], max_tweets: int) -> list[dict[str, Any]]:
    high = [row for row in rows if int(row["gpt5_relevance"]) >= 7]
    low = [row for row in rows if int(row["gpt5_relevance"]) < 7]
    key = lambda row: (int(row["gpt5_relevance"]), int(row["tweet_index"]))
    selected = sorted(high, key=key, reverse=True) + sorted(low, key=key, reverse=True)
    selected = selected[:max_tweets]
    return sorted(selected, key=lambda row: int(row["tweet_index"]))


def main() -> None:
    args = parse_args()
    repo_dir = Path.cwd()
    config_path = resolve_path(args.config, repo_dir)
    config = read_config(config_path)
    raw_artifacts_dir = resolve_path(args.raw_artifacts_dir or config["rawArtifactsDir"], repo_dir)
    users = read_users(resolve_path(args.users_file, repo_dir))
    wanted = set(users)
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)

    dataset_path = raw_artifacts_dir / "tweet_embeddings" / args.split
    dataset = ds.dataset(dataset_path, format="parquet")
    for batch in dataset.to_batches(columns=["user_id", "tweet_index", "tweet_text", "gpt5_relevance"], batch_size=65536):
        batch_users = batch.column("user_id").to_pylist()
        indexes = batch.column("tweet_index").to_pylist()
        texts = batch.column("tweet_text").to_pylist()
        relevances = batch.column("gpt5_relevance").to_pylist()
        for uid, tweet_index, text, raw_rel in zip(batch_users, indexes, texts, relevances):
            uid_text = str(uid)
            if uid_text not in wanted:
                continue
            by_user[uid_text].append(
                {
                    "tweet_index": int(tweet_index),
                    "tweet_text": "" if text is None else str(text),
                    "gpt5_relevance": relevance_value(raw_rel),
                }
            )

    missing = [user_id for user_id in users if user_id not in by_user]
    if missing:
        raise ValueError(f"{args.split} timeline pack missing users: {missing[:5]}")

    out_path = resolve_path(args.output_jsonl, repo_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for user_id in users:
            tweets = selected_tweets(by_user[user_id], args.max_evidence_tweets)
            handle.write(
                json.dumps(
                    {
                        "user_id": user_id,
                        "split": args.split,
                        "total_selected": len(tweets),
                        "selected_tweets": tweets,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    print(json.dumps({"status": "ok", "users": len(users), "output": str(out_path)}))


if __name__ == "__main__":
    main()
