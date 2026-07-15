#!/usr/bin/env python3
"""Prepare strict-blind anxiety manifests, features, and chronological sequences."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from anxiety_champion_relevance import alignment_hash, load_config, require_post_lock, resolve, sha256_file, write_json
from raw_ternary_prepare_setembrobr import (
    EMBEDDING_BLOCKS,
    EVIDENCE_COLUMNS,
    TEMPORAL_COLUMNS,
    UserAgg,
    evidence_feature_row,
    select_sequence_indexes,
    sequence_sort_order,
    write_marker_csv,
)


def label_name(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"1", "diagnosed"}:
        return "diagnosed"
    if normalized in {"0", "control"}:
        return "control"
    raise ValueError(f"unsupported anxiety train label {value!r}")


def read_raw_rows(config: dict[str, Any], raw_dir: Path) -> list[dict[str, Any]]:
    path = raw_dir / "manifests" / f"raw_split_manifest_seed{config['seed']}.csv"
    rows = list(csv.DictReader(path.read_text().splitlines()))
    out = []
    for row in rows:
        split = row["split"]
        label = label_name(row["label"]) if split == "train" else None
        out.append(
            {
                "dataset": "setembrobr",
                "split": split,
                "user_id": row["user_id"],
                "label": label,
                "label_code": 1 if label == "diagnosed" else (0 if label == "control" else -1),
                "fold": int(row["fold"]) + 1 if split == "train" else None,
                "row_hash": row["row_hash"],
            }
        )
    for split in ["train", "test"]:
        count = sum(row["split"] == split for row in out)
        if count != int(config["expectedUsers"][split]):
            raise RuntimeError(f"{split} manifest users expected {config['expectedUsers'][split]}, got {count}")
    return out


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows({header: row.get(header, "") for header in headers} for row in rows)
    return sha256_file(path)


def write_manifests(config: dict[str, Any], output_dir: Path, raw_rows: list[dict[str, Any]]) -> None:
    seed = int(config["seed"])
    train_rows = [row for row in raw_rows if row["split"] == "train"]
    test_rows = [row for row in raw_rows if row["split"] == "test"]
    manifest_dir = output_dir / "manifest"
    strict_hash = write_csv(
        manifest_dir / f"strict_blind_split_manifest_seed{seed}.csv",
        ["dataset", "prediction_target", "split", "user_id", "label", "fold", "row_hash"],
        [
            {
                "dataset": "setembrobr",
                "prediction_target": "anxiety",
                "split": row["split"],
                "user_id": row["user_id"],
                "label": row["label"] if row["split"] == "train" else -1,
                "fold": row["fold"] if row["split"] == "train" else "",
                "row_hash": row["row_hash"],
            }
            for row in raw_rows
        ],
    )
    train_hash = write_csv(
        manifest_dir / f"train_binary_manifest_seed{seed}.csv",
        ["dataset", "prediction_target", "split", "label", "user_id", "row_hash", "fold"],
        [
            {
                "dataset": "setembrobr",
                "prediction_target": "anxiety",
                "split": "train",
                "label": row["label"],
                "user_id": row["user_id"],
                "row_hash": row["row_hash"],
                "fold": row["fold"],
            }
            for row in train_rows
        ],
    )
    test_hash = write_csv(
        manifest_dir / f"test_inference_manifest_seed{seed}.csv",
        ["dataset", "prediction_target", "split", "user_id", "label", "fold", "row_hash"],
        [
            {
                "dataset": "setembrobr",
                "prediction_target": "anxiety",
                "split": "test",
                "user_id": row["user_id"],
                "label": -1,
                "fold": "",
                "row_hash": row["row_hash"],
            }
            for row in test_rows
        ],
    )
    write_json(
        output_dir / "reports" / "anxiety-manifest-report.json",
        {
            "dataset": "setembrobr",
            "predictionTarget": "anxiety",
            "seed": seed,
            "strictBlindManifestHash": strict_hash,
            "trainManifestHash": train_hash,
            "testInferenceManifestHash": test_hash,
            "rawSplitManifestSha256": config["rawSplitManifestSha256"],
            "trainUsers": len(train_rows),
            "testUsers": len(test_rows),
            "sealedLabelsCreated": False,
        },
    )


def proxy_paths(config: dict[str, Any], repo: Path, split: str, source_path: Path) -> tuple[Path, Path]:
    work_dir = resolve(config.get("workDir", config["outputDir"]), repo)
    proxy_dir = resolve(config["relevanceProxy"]["artifactDir"], repo)
    sidecar = proxy_dir / "sidecars" / split / f"{source_path.stem}.npz"
    return sidecar, work_dir / "relevance-proxy" / "pooled"


def load_sidecar(config: dict[str, Any], repo: Path, split: str, source_path: Path, users: list[str], indexes: list[int]) -> np.ndarray:
    sidecar_path, _pooled_dir = proxy_paths(config, repo, split, source_path)
    data = np.load(sidecar_path, allow_pickle=False)
    try:
        scores = data["scores"].astype(np.int16)
        expected_alignment = str(data["alignment_sha256"])
    finally:
        data.close()
    if len(scores) != len(users) or expected_alignment != alignment_hash(users, indexes):
        raise RuntimeError(f"proxy alignment mismatch for {source_path}")
    return scores


def collect_aggs(config: dict[str, Any], repo: Path, raw_dir: Path, split: str) -> dict[str, UserAgg]:
    aggs: dict[str, UserAgg] = defaultdict(UserAgg)
    paths = sorted((raw_dir / "tweet_embeddings" / split).glob("*.parquet"))
    for temporal_pass in [False, True]:
        for path in paths:
            table = pq.read_table(path, columns=["user_id", "tweet_index", "tweet_text"])
            users = [str(value) for value in table.column("user_id").to_pylist()]
            indexes = [int(value) for value in table.column("tweet_index").to_pylist()]
            texts = ["" if value is None else str(value) for value in table.column("tweet_text").to_pylist()]
            scores = load_sidecar(config, repo, split, path, users, indexes)
            for user_id, tweet_index, text, score in zip(users, indexes, texts, scores):
                if temporal_pass:
                    aggs[user_id].add_temporal(text, int(score), tweet_index)
                else:
                    aggs[user_id].add(text, int(score), tweet_index)
    return aggs


def load_pool(path: Path, ordered_user_ids: list[str]) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    try:
        user_ids = data["user_ids"].astype(str)
        embeddings = data["embeddings"].astype(np.float32)
        indexes = {user_id: index for index, user_id in enumerate(user_ids)}
        missing = [user_id for user_id in ordered_user_ids if user_id not in indexes]
        if missing:
            raise RuntimeError(f"pooled artifact missing users: {missing[:5]}")
        return embeddings[[indexes[user_id] for user_id in ordered_user_ids]]
    finally:
        data.close()


def write_features(config: dict[str, Any], repo: Path, raw_dir: Path, output_dir: Path, split: str, rows: list[dict[str, Any]]) -> str:
    work_dir = resolve(config.get("workDir", config["outputDir"]), repo)
    user_ids = [row["user_id"] for row in rows]
    aggs = collect_aggs(config, repo, raw_dir, split)
    missing = [user_id for user_id in user_ids if user_id not in aggs]
    if missing:
        raise RuntimeError(f"{split} proxy aggregation missing users: {missing[:5]}")
    markers = [aggs[user_id].marker(user_id) for user_id in user_ids]
    marker_hash = write_marker_csv(output_dir / "relevance-proxy" / f"{split}-markers.csv", markers)
    arrays: dict[str, Any] = {
        "user_ids": np.asarray(user_ids, dtype=object),
        "labels": np.asarray([row["label_code"] if split == "train" else -1 for row in rows], dtype=np.int16),
        "folds": np.asarray([row["fold"] if split == "train" else -1 for row in rows], dtype=np.int16),
        "evidence_markers": np.asarray([evidence_feature_row(marker) for marker in markers], dtype=np.float32),
        "stylistic": np.asarray([aggs[user_id].stylistic() for user_id in user_ids], dtype=np.float32),
        "relevance_counts": np.asarray([aggs[user_id].relevance_counts() for user_id in user_ids], dtype=np.float32),
        "temporal_markers": np.asarray([aggs[user_id].temporal_markers() for user_id in user_ids], dtype=np.float32),
        "mean": load_pool(raw_dir / "pooled" / f"{split}_user_mean.npz", user_ids),
    }
    work_dir = resolve(config.get("workDir", config["outputDir"]), repo)
    pooled_dir = work_dir / "relevance-proxy" / "pooled"
    for block in EMBEDDING_BLOCKS:
        if block == "mean":
            continue
        arrays[block] = load_pool(pooled_dir / f"{split}_user_{block}.npz", user_ids)
    feature_path = work_dir / "features" / f"{split}_raw_features.npz"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_path, **arrays)
    write_json(
        work_dir / "features" / f"{split}-feature-manifest.json",
        {
            "predictionTarget": "anxiety",
            "split": split,
            "featureSha256": sha256_file(feature_path),
            "markerSha256": marker_hash,
            "evidenceColumns": EVIDENCE_COLUMNS,
            "temporalColumns": TEMPORAL_COLUMNS,
            "proxyDefinitionSha256": sha256_file(output_dir / "relevance-proxy" / "proxy-definition.json"),
            "labelsValue": "train labels" if split == "train" else -1,
        },
    )
    return sha256_file(feature_path)


def write_sequences(config: dict[str, Any], repo: Path, raw_dir: Path, split: str, rows: list[dict[str, Any]]) -> str:
    work_dir = resolve(config.get("workDir", config["outputDir"]), repo)
    top_n = 128
    dimension = int(config["database"]["embeddingDimension"])
    user_ids = [row["user_id"] for row in rows]
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    sequences = np.zeros((len(user_ids), top_n, dimension), dtype=np.float16)
    relevances = np.zeros((len(user_ids), top_n), dtype=np.int16)
    lengths = np.zeros(len(user_ids), dtype=np.int32)
    for path in sorted((raw_dir / "tweet_embeddings" / split).glob("*.parquet")):
        table = pq.read_table(path, columns=["user_id", "tweet_index", "embedding"])
        users = [str(value) for value in table.column("user_id").to_pylist()]
        indexes = [int(value) for value in table.column("tweet_index").to_pylist()]
        scores = load_sidecar(config, repo, split, path, users, indexes)
        score_values = scores.tolist()
        by_user: dict[str, list[int]] = defaultdict(list)
        for row_index, user_id in enumerate(users):
            by_user[user_id].append(row_index)
        embedding_column = table.column("embedding").combine_chunks()
        for user_id, row_indexes in by_user.items():
            selected = select_sequence_indexes(row_indexes, score_values, indexes, top_n, "recent_chronological")
            taken = embedding_column.take(pa.array(selected, type=pa.int64()))
            vectors = taken.values.to_numpy(zero_copy_only=False).reshape(len(selected), dimension).astype(np.float16)
            target = user_index[user_id]
            sequences[target, : len(selected), :] = vectors
            relevances[target, : len(selected)] = scores[selected]
            lengths[target] = len(selected)
    path = work_dir / "sequences" / "top128" / f"{split}_seq.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        user_ids=np.asarray(user_ids, dtype=object),
        labels=np.asarray([row["label_code"] if split == "train" else -1 for row in rows], dtype=np.int32),
        sequences=sequences,
        lengths=lengths,
        relevances=relevances,
    )
    write_json(
        path.parent / f"{split}-sequence-manifest.json",
        {
            "predictionTarget": "anxiety",
            "split": split,
            "topN": top_n,
            "embeddingDimension": dimension,
            "sequenceOrder": "recent_chronological",
            "sortOrder": sequence_sort_order("recent_chronological"),
            "testLabelsValue": -1,
            "sequenceSha256": sha256_file(path),
        },
    )
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setembrobr.seed42.anxiety-temporal-champion-qwen3-binary.json")
    parser.add_argument("--stage", choices=["manifests", "train", "test"], required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    config = load_config(resolve(args.config, repo))
    raw_dir = resolve(config["rawArtifactsDir"], repo)
    output_dir = resolve(config["outputDir"], repo)
    raw_rows = read_raw_rows(config, raw_dir)
    if args.stage == "manifests":
        write_manifests(config, output_dir, raw_rows)
        return
    if args.stage == "test":
        require_post_lock(output_dir)
    split = args.stage
    rows = [row for row in raw_rows if row["split"] == split]
    feature_hash = write_features(config, repo, raw_dir, output_dir, split, rows)
    sequence_hash = write_sequences(config, repo, raw_dir, split, rows)
    write_json(
        output_dir / "reports" / f"{split}-prepare-manifest.json",
        {
            "predictionTarget": "anxiety",
            "split": split,
            "strictBlindManifestSha256": sha256_file(
                output_dir / "manifest" / f"strict_blind_split_manifest_seed{config['seed']}.csv"
            ),
            "featureSha256": feature_hash,
            "sequenceSha256": sequence_hash,
            "sealedLabelsRead": False,
        },
    )


if __name__ == "__main__":
    main()
