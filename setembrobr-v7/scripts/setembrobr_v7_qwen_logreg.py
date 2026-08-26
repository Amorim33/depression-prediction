#!/usr/bin/env python3
"""Strict-blind Qwen3 mean-embedding logistic baseline for SetembroBR v7."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SHARD_RE = re.compile(r"part-(\d{6})-(\d{6})\.parquet\.zst$")
SOURCE_COLUMNS = ["User_ID", "Diagnosed_YN", "TextLists", "Split"]
SANITIZED_COLUMNS = ["User_ID", "TextLists", "Split"]
LABEL_TO_CODE = {"no": 0, "yes": 1}
CODE_TO_LABEL = {0: "no", 1: "yes"}


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def sanitized(self) -> Path:
        return self.root / "sanitized"

    @property
    def sealed(self) -> Path:
        return self.root / "sealed"

    @property
    def pool_shards(self) -> Path:
        return self.root / "pool-shards"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def ensemble(self) -> Path:
        return self.root / "ensemble"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def corpus(self) -> Path:
        return self.root / "corpus"

    def ensure(self) -> None:
        for path in [
            self.manifests,
            self.sanitized,
            self.sealed,
            self.pool_shards,
            self.embeddings,
            self.scores,
            self.models,
            self.ensemble,
            self.reports,
            self.corpus,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_sha256(path: Path, expected: str, description: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{description} hash mismatch: expected {expected}, got {actual}")


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json(payload)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        tmp.write(data)
        temporary = Path(tmp.name)
    os.replace(temporary, path)
    return sha256_bytes(data)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(tmp.name)
    os.replace(temporary, path)
    return sha256_file(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    config["_configPath"] = str(path.resolve())
    config["_configSha256"] = sha256_file(path)
    return config


def normalized_source_frame(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    expected_hash = config["source"]["sha256"]
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"v7 source hash mismatch: expected {expected_hash}, got {actual_hash}")
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("v7 source pickle must contain a pandas DataFrame")
    if list(frame.columns) != config["source"]["schema"]:
        raise RuntimeError(f"v7 schema mismatch: {list(frame.columns)}")
    if len(frame) != int(config["source"]["expectedUsers"]):
        raise RuntimeError(f"v7 user count mismatch: {len(frame)}")

    result = frame.copy()
    result["User_ID"] = result["User_ID"].astype(str)
    result["Diagnosed_YN"] = result["Diagnosed_YN"].astype(str).str.strip().str.lower()
    result["Split"] = result["Split"].astype(str).str.strip().str.lower()
    if result["User_ID"].duplicated().any():
        raise RuntimeError("v7 source contains duplicate User_ID values")
    if set(result["Diagnosed_YN"]) != set(LABEL_TO_CODE):
        raise RuntimeError(f"v7 source contains invalid labels: {sorted(set(result['Diagnosed_YN']))}")
    if set(result["Split"]) != {"train", "test"}:
        raise RuntimeError(f"v7 source contains invalid splits: {sorted(set(result['Split']))}")
    for index, texts in enumerate(result["TextLists"]):
        if not isinstance(texts, list) or not texts:
            raise RuntimeError(f"v7 TextLists row {index} must be a non-empty list")
        if any(not isinstance(text, str) for text in texts):
            raise RuntimeError(f"v7 TextLists row {index} contains a non-string post")

    expected_posts = int(config["source"]["expectedPosts"])
    actual_posts = int(result["TextLists"].map(len).sum())
    if actual_posts != expected_posts:
        raise RuntimeError(f"v7 post count mismatch: expected {expected_posts}, got {actual_posts}")
    for split, expected in config["source"]["splits"].items():
        selected = result[result["Split"] == split]
        positives = int((selected["Diagnosed_YN"] == "yes").sum())
        negatives = int((selected["Diagnosed_YN"] == "no").sum())
        observed = {"users": len(selected), "positive": positives, "negative": negatives}
        if observed != expected:
            raise RuntimeError(f"v7 {split} distribution mismatch: expected {expected}, got {observed}")
    return result


def read_zstd_json(path: Path) -> tuple[dict[str, Any], str, str]:
    archive_hash = hashlib.sha256()
    original_hash = hashlib.sha256()

    class HashingReader(io.RawIOBase):
        def __init__(self, raw: Any) -> None:
            self.raw = raw

        def readable(self) -> bool:
            return True

        def readinto(self, buffer: Any) -> int:
            data = self.raw.read(len(buffer))
            if not data:
                return 0
            archive_hash.update(data)
            buffer[: len(data)] = data
            return len(data)

    output = bytearray()
    with path.open("rb") as raw:
        hashed = io.BufferedReader(HashingReader(raw), buffer_size=1024 * 1024)
        with zstd.ZstdDecompressor().stream_reader(hashed) as reader:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                original_hash.update(chunk)
                output.extend(chunk)
    return json.loads(output), archive_hash.hexdigest(), original_hash.hexdigest()


def archive_records(archive_root: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    manifest_path = archive_root / config["embeddings"]["archiveStateManifest"]
    records: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("sourceId") == config["embeddings"]["archiveSourceId"]:
                records[str(record["archivePath"])] = record
    if not records:
        raise RuntimeError("embedding archive state contains no depression artifact records")
    return records


def validate_embedding_provenance(archive_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    embedding_config = config["embeddings"]
    manifest_path = archive_root / embedding_config["generationManifest"]
    payload, archive_hash, original_hash = read_zstd_json(manifest_path)
    if archive_hash != embedding_config["generationManifestArchiveSha256"]:
        raise RuntimeError(f"embedding generation archive hash mismatch: {archive_hash}")
    if original_hash != embedding_config["generationManifestSha256"]:
        raise RuntimeError(f"embedding generation manifest hash mismatch: {original_hash}")
    generated = payload["embedding"]
    expected_pairs = {
        "modelId": embedding_config["modelId"],
        "modelRevision": embedding_config["modelRevision"],
        "embeddingDimension": int(embedding_config["dimension"]),
        "embeddingStorageDtype": embedding_config["storageDtype"],
    }
    for key, expected in expected_pairs.items():
        if generated.get(key) != expected:
            raise RuntimeError(f"embedding provenance mismatch for {key}: {generated.get(key)!r}")
    if payload.get("splitManifestHash") != embedding_config["sourceSplitManifestSha256"]:
        raise RuntimeError("embedding generation uses an unexpected SetembroBR v6 split manifest")
    return {
        "generationManifestArchiveSha256": archive_hash,
        "generationManifestSha256": original_hash,
        "modelId": generated["modelId"],
        "modelRevision": generated["modelRevision"],
        "embeddingDimension": generated["embeddingDimension"],
        "embeddingStorageDtype": generated["embeddingStorageDtype"],
        "sourceSplitManifestSha256": payload["splitManifestHash"],
    }


def validate_command(source_path: Path, archive_root: Path, config: dict[str, Any], output: ArtifactPaths) -> None:
    output.ensure()
    frame = normalized_source_frame(source_path, config)
    provenance = validate_embedding_provenance(archive_root, config)
    records = archive_records(archive_root, config)
    pooled_relative = config["embeddings"]["pooledUserIndex"]
    pooled_record = records.get(pooled_relative)
    if not pooled_record:
        raise RuntimeError(f"archive state is missing {pooled_relative}")
    source_user_ids = load_source_user_index(archive_root / pooled_relative, pooled_record)
    expected_source_users = int(config["embeddings"]["sourceUserCount"])
    if len(source_user_ids) != expected_source_users or len(set(source_user_ids)) != expected_source_users:
        raise RuntimeError(
            f"archived source user index expected {expected_source_users} unique users, "
            f"found {len(source_user_ids)}"
        )
    shard_records = [
        record
        for path, record in records.items()
        if path.startswith("payload/depression/artifacts/tweet_embeddings/train/")
        and path.endswith(".parquet.zst")
    ]
    expected_shards = int(config["embeddings"]["expectedTrainShards"])
    if len(shard_records) != expected_shards:
        raise RuntimeError(
            f"expected {expected_shards} archived train embedding shards, found {len(shard_records)}"
        )
    report = {
        "ok": True,
        "experimentId": config["experimentId"],
        "configSha256": config["_configSha256"],
        "source": {
            "path": str(source_path),
            "sha256": config["source"]["sha256"],
            "users": len(frame),
            "posts": int(frame["TextLists"].map(len).sum()),
            "trainUsers": int((frame["Split"] == "train").sum()),
            "testUsers": int((frame["Split"] == "test").sum()),
        },
        "embeddingProvenance": provenance,
        "archive": {
            "path": str(archive_root),
            "trainShardCount": len(shard_records),
            "sourceUsers": len(source_user_ids),
            "archiveStateManifestSha256": sha256_file(
                archive_root / config["embeddings"]["archiveStateManifest"]
            ),
        },
    }
    write_json(output.reports / "source-audit.json", report)


def assign_folds(user_ids: Sequence[str], labels: Sequence[int], fold_count: int, seed: int) -> np.ndarray:
    if len(user_ids) != len(labels):
        raise ValueError("user_ids and labels have different lengths")
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    folds = np.full(len(user_ids), -1, dtype=np.int16)
    y = np.asarray(labels, dtype=np.int8)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(y)), y)):
        folds[validation] = fold
    if np.any(folds < 0):
        raise RuntimeError("fold assignment is incomplete")
    return folds


def prepare_command(source_path: Path, config: dict[str, Any], output: ArtifactPaths) -> None:
    output.ensure()
    frame = normalized_source_frame(source_path, config)
    train = frame[frame["Split"] == "train"].sort_values("User_ID").reset_index(drop=True)
    test = frame[frame["Split"] == "test"].sort_values("User_ID").reset_index(drop=True)
    if set(train["User_ID"]) & set(test["User_ID"]):
        raise RuntimeError("v7 train and test users overlap")

    labels = train["Diagnosed_YN"].map(LABEL_TO_CODE).to_numpy(dtype=np.int8)
    folds = assign_folds(
        train["User_ID"].tolist(),
        labels.tolist(),
        int(config["validation"]["foldCount"]),
        int(config["validation"]["seed"]),
    )
    train_manifest_hash = write_csv(
        output.manifests / "train_manifest.csv",
        ["user_id", "label", "fold"],
        (
            {"user_id": user_id, "label": label, "fold": int(fold)}
            for user_id, label, fold in zip(train["User_ID"], train["Diagnosed_YN"], folds)
        ),
    )
    test_inference_hash = write_csv(
        output.manifests / "test_inference.csv",
        ["user_id"],
        ({"user_id": user_id} for user_id in test["User_ID"]),
    )
    sealed_hash = write_csv(
        output.sealed / "test_labels.csv",
        ["user_id", "label"],
        ({"user_id": user_id, "label": label} for user_id, label in zip(test["User_ID"], test["Diagnosed_YN"])),
    )

    train_sanitized = train[SANITIZED_COLUMNS].copy()
    test_sanitized = test[SANITIZED_COLUMNS].copy()
    train_sanitized.to_pickle(output.sanitized / "train.pkl")
    test_sanitized.to_pickle(output.sanitized / "test.pkl")
    prepared = {
        "ok": True,
        "experimentId": config["experimentId"],
        "sourceSha256": config["source"]["sha256"],
        "configSha256": config["_configSha256"],
        "trainUsers": len(train),
        "testUsers": len(test),
        "trainPosts": int(train["TextLists"].map(len).sum()),
        "testPosts": int(test["TextLists"].map(len).sum()),
        "foldCounts": {str(fold): int((folds == fold).sum()) for fold in sorted(set(folds.tolist()))},
        "trainManifestSha256": train_manifest_hash,
        "testInferenceSha256": test_inference_hash,
        "sealedTestLabelsSha256": sealed_hash,
        "sanitizedTrainSha256": sha256_file(output.sanitized / "train.pkl"),
        "sanitizedTestSha256": sha256_file(output.sanitized / "test.pkl"),
        "testLabelsSealedBeforeTraining": True,
    }
    write_json(output.manifests / "prepared-manifest.json", prepared)


def text_alignment_hash(user_ids: Sequence[str], targets: dict[str, list[str]]) -> str:
    digest = hashlib.sha256()
    for user_id in user_ids:
        digest.update(user_id.encode("utf-8"))
        digest.update(b"\0")
        texts = targets[user_id]
        digest.update(str(len(texts)).encode())
        digest.update(b"\0")
        for text in texts:
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def match_ordered_posts(source_texts: Sequence[str], target_texts: Sequence[str]) -> np.ndarray:
    matched: list[int] = []
    target_index = 0
    for source_index, source_text in enumerate(source_texts):
        if target_index < len(target_texts) and source_text == target_texts[target_index]:
            matched.append(source_index)
            target_index += 1
    if target_index != len(target_texts):
        missing = len(target_texts) - target_index
        raise RuntimeError(
            f"target posts are not an ordered source subsequence: matched {target_index}, missing {missing}"
        )
    return np.asarray(matched, dtype=np.int64)


class HashingReader(io.RawIOBase):
    def __init__(self, raw: Any, digest: Any) -> None:
        self.raw = raw
        self.digest = digest

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        data = self.raw.read(len(buffer))
        if not data:
            return 0
        self.digest.update(data)
        buffer[: len(data)] = data
        return len(data)


def decompress_verified(source: Path, record: dict[str, Any], destination: Any) -> tuple[str, str, int]:
    archive_digest = hashlib.sha256()
    original_digest = hashlib.sha256()
    original_size = 0
    with source.open("rb") as raw:
        hashed = io.BufferedReader(HashingReader(raw, archive_digest), buffer_size=1024 * 1024)
        with zstd.ZstdDecompressor().stream_reader(hashed) as reader:
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                original_digest.update(chunk)
                original_size += len(chunk)
                destination.write(chunk)
    archive_hash = archive_digest.hexdigest()
    original_hash = original_digest.hexdigest()
    if archive_hash != record["archiveSha256"]:
        raise RuntimeError(f"archive hash mismatch for {source.name}: {archive_hash}")
    if original_hash != record["originalSha256"]:
        raise RuntimeError(f"reconstructed hash mismatch for {source.name}: {original_hash}")
    if original_size != int(record["originalSize"]):
        raise RuntimeError(f"reconstructed size mismatch for {source.name}: {original_size}")
    return archive_hash, original_hash, original_size


def fixed_size_embeddings(table: pa.Table, dimension: int) -> np.ndarray:
    column = table.column("embedding").combine_chunks()
    if not pa.types.is_fixed_size_list(column.type) or column.type.list_size != dimension:
        raise RuntimeError(f"unexpected embedding column type: {column.type}")
    values = column.values.to_numpy(zero_copy_only=False)
    return values.reshape(len(table), dimension).astype(np.float16, copy=False)


def pool_one_shard(
    source_path: Path,
    archive_record: dict[str, Any],
    relevant_user_ids: Sequence[str],
    targets: dict[str, list[str]],
    dimension: int,
    destination: Path,
    config_sha256: str,
    temporary_dir: Path | None = None,
) -> dict[str, Any]:
    expected_alignment = text_alignment_hash(relevant_user_ids, targets)
    if temporary_dir is not None:
        temporary_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            with np.load(destination, allow_pickle=True) as cached:
                cached_ids = cached["user_ids"].astype(str).tolist()
                if (
                    cached_ids == list(relevant_user_ids)
                    and str(cached["source_archive_sha256"]) == archive_record["archiveSha256"]
                    and str(cached["target_alignment_sha256"]) == expected_alignment
                    and str(cached["config_sha256"]) == config_sha256
                    and cached["embeddings"].shape == (len(relevant_user_ids), dimension)
                    and cached["counts"].astype(int).tolist()
                    == [len(targets[user_id]) for user_id in relevant_user_ids]
                ):
                    return {
                        "source": source_path.name,
                        "resumed": True,
                        "users": len(relevant_user_ids),
                        "matchedPosts": int(cached["counts"].sum()),
                        "sourceArchiveSha256": archive_record["archiveSha256"],
                        "sourceOriginalSha256": archive_record["originalSha256"],
                        "targetAlignmentSha256": expected_alignment,
                        "poolShardSha256": sha256_file(destination),
                    }
        except Exception:
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", dir=temporary_dir) as reconstructed:
        archive_hash, original_hash, _size = decompress_verified(source_path, archive_record, reconstructed)
        reconstructed.flush()
        table = pq.read_table(
            reconstructed.name,
            columns=["user_id", "tweet_index", "tweet_text", "embedding"],
            filters=[("user_id", "in", list(relevant_user_ids))],
        )

    if len(table) == 0:
        raise RuntimeError(f"embedding shard {source_path.name} has no relevant users")
    user_column = np.asarray([str(value) for value in table.column("user_id").to_pylist()], dtype=object)
    index_column = np.asarray(table.column("tweet_index").to_numpy(), dtype=np.int64)
    text_column = np.asarray([str(value) for value in table.column("tweet_text").to_pylist()], dtype=object)
    embeddings = fixed_size_embeddings(table, dimension)

    pooled: list[np.ndarray] = []
    counts: list[int] = []
    for user_id in relevant_user_ids:
        positions = np.flatnonzero(user_column == user_id)
        if positions.size == 0:
            raise RuntimeError(f"{source_path.name} is missing source user {user_id}")
        order = np.argsort(index_column[positions], kind="stable")
        positions = positions[order]
        indexes = index_column[positions]
        if len(np.unique(indexes)) != len(indexes):
            raise RuntimeError(f"{source_path.name}:{user_id} contains duplicate tweet indexes")
        source_texts = text_column[positions].tolist()
        target_texts = targets[user_id]
        selected_relative = match_ordered_posts(source_texts, target_texts)
        selected = positions[selected_relative]
        vector = (embeddings[selected].astype(np.float64).sum(axis=0) / len(selected)).astype(np.float32)
        if not np.isfinite(vector).all():
            raise RuntimeError(f"{source_path.name}:{user_id} produced a non-finite pooled vector")
        pooled.append(vector)
        counts.append(len(selected))

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
    np.savez(
        temporary,
        user_ids=np.asarray(relevant_user_ids, dtype=object),
        embeddings=np.stack(pooled).astype(np.float32),
        counts=np.asarray(counts, dtype=np.int32),
        source_archive_sha256=np.asarray(archive_hash),
        source_original_sha256=np.asarray(original_hash),
        target_alignment_sha256=np.asarray(expected_alignment),
        config_sha256=np.asarray(config_sha256),
    )
    os.replace(temporary, destination)
    return {
        "source": source_path.name,
        "resumed": False,
        "users": len(relevant_user_ids),
        "matchedPosts": int(sum(counts)),
        "sourceArchiveSha256": archive_hash,
        "sourceOriginalSha256": original_hash,
        "targetAlignmentSha256": expected_alignment,
        "poolShardSha256": sha256_file(destination),
    }


def load_sanitized_targets(output: ArtifactPaths) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    train = pd.read_pickle(output.sanitized / "train.pkl")
    test = pd.read_pickle(output.sanitized / "test.pkl")
    for split, frame in [("train", train), ("test", test)]:
        if list(frame.columns) != SANITIZED_COLUMNS:
            raise RuntimeError(f"sanitized {split} schema is invalid: {list(frame.columns)}")
        if "Diagnosed_YN" in frame.columns:
            raise RuntimeError(f"sanitized {split} unexpectedly contains diagnosis labels")
    combined = pd.concat([train, test], ignore_index=True)
    targets = {str(row.User_ID): list(row.TextLists) for row in combined.itertuples(index=False)}
    if len(targets) != len(combined):
        raise RuntimeError("sanitized inputs contain duplicate users")
    return train, test, targets


def load_source_user_index(path: Path, record: dict[str, Any]) -> list[str]:
    if sha256_file(path) != record["archiveSha256"]:
        raise RuntimeError("archived pooled user index hash mismatch")
    with np.load(path, allow_pickle=True) as payload:
        return payload["user_ids"].astype(str).tolist()


def save_combined_pool(
    split: str,
    frame: pd.DataFrame,
    pooled_by_user: dict[str, tuple[np.ndarray, int]],
    destination: Path,
    dimension: int,
) -> str:
    user_ids = frame["User_ID"].astype(str).tolist()
    missing = [user_id for user_id in user_ids if user_id not in pooled_by_user]
    if missing:
        raise RuntimeError(f"{split} pool is missing {len(missing)} users")
    vectors = np.stack([pooled_by_user[user_id][0] for user_id in user_ids]).astype(np.float32)
    counts = np.asarray([pooled_by_user[user_id][1] for user_id in user_ids], dtype=np.int32)
    if vectors.shape != (len(user_ids), dimension):
        raise RuntimeError(f"{split} combined pool has wrong shape {vectors.shape}")
    np.savez(destination, user_ids=np.asarray(user_ids, dtype=object), embeddings=vectors, counts=counts)
    return sha256_file(destination)


def pool_command(archive_root: Path, config: dict[str, Any], output: ArtifactPaths, temporary_dir: Path | None) -> None:
    output.ensure()
    train, test, targets = load_sanitized_targets(output)
    records = archive_records(archive_root, config)
    embedding_config = config["embeddings"]
    pooled_relative = embedding_config["pooledUserIndex"]
    pooled_record = records.get(pooled_relative)
    if not pooled_record:
        raise RuntimeError(f"archive state is missing {pooled_relative}")
    source_user_ids = load_source_user_index(archive_root / pooled_relative, pooled_record)
    expected_source_users = int(embedding_config["sourceUserCount"])
    if len(source_user_ids) != expected_source_users or len(set(source_user_ids)) != expected_source_users:
        raise RuntimeError("archived pooled user index has invalid source-user coverage")
    source_positions = {user_id: index for index, user_id in enumerate(source_user_ids)}
    missing_users = sorted(set(targets) - set(source_positions))
    if missing_users:
        raise RuntimeError(f"archived source embeddings are missing {len(missing_users)} v7 users")

    shard_files: dict[tuple[int, int], Path] = {}
    for path in sorted(archive_root.glob(embedding_config["shardGlob"])):
        match = SHARD_RE.match(path.name)
        if not match:
            continue
        shard_files[(int(match.group(1)), int(match.group(2)))] = path

    required: list[tuple[Path, dict[str, Any], list[str]]] = []
    for (start, end), source_path in sorted(shard_files.items()):
        relevant = [user_id for user_id in source_user_ids[start:end] if user_id in targets]
        if not relevant:
            continue
        relative = str(source_path.relative_to(archive_root))
        record = records.get(relative)
        if not record:
            raise RuntimeError(f"archive state has no record for {relative}")
        required.append((source_path, record, relevant))
    covered = {user_id for _path, _record, users in required for user_id in users}
    if covered != set(targets):
        raise RuntimeError(f"shard index covers {len(covered)} of {len(targets)} v7 users")

    shard_reports: list[dict[str, Any]] = []
    for shard_number, (source_path, record, relevant) in enumerate(required, start=1):
        destination = output.pool_shards / source_path.name.replace(".parquet.zst", ".npz")
        report = pool_one_shard(
            source_path,
            record,
            relevant,
            targets,
            int(embedding_config["dimension"]),
            destination,
            config["_configSha256"],
            temporary_dir,
        )
        shard_reports.append(report)
        state = "resume" if report["resumed"] else "pooled"
        print(f"{state} shard {shard_number}/{len(required)} {source_path.name} users={len(relevant)}", flush=True)

    pooled_by_user: dict[str, tuple[np.ndarray, int]] = {}
    for report in shard_reports:
        path = output.pool_shards / report["source"].replace(".parquet.zst", ".npz")
        with np.load(path, allow_pickle=True) as payload:
            for user_id, vector, count in zip(
                payload["user_ids"].astype(str), payload["embeddings"], payload["counts"].astype(int)
            ):
                if user_id in pooled_by_user:
                    raise RuntimeError(f"pooled user {user_id} appears in multiple shards")
                pooled_by_user[user_id] = (np.asarray(vector, dtype=np.float32), int(count))
    if set(pooled_by_user) != set(targets):
        raise RuntimeError("combined shard pools do not cover the sanitized corpus exactly")
    for user_id, (_vector, count) in pooled_by_user.items():
        if count != len(targets[user_id]):
            raise RuntimeError(f"pooled count mismatch for {user_id}: {count}")

    dimension = int(embedding_config["dimension"])
    train_hash = save_combined_pool(
        "train", train, pooled_by_user, output.embeddings / "train_user_mean.npz", dimension
    )
    test_hash = save_combined_pool(
        "test", test, pooled_by_user, output.embeddings / "test_user_mean.npz", dimension
    )
    matched_posts = int(sum(report["matchedPosts"] for report in shard_reports))
    if matched_posts != int(config["source"]["expectedPosts"]):
        raise RuntimeError(f"pooled post coverage mismatch: {matched_posts}")
    report = {
        "ok": True,
        "experimentId": config["experimentId"],
        "configSha256": config["_configSha256"],
        "sourceCorpusSha256": config["source"]["sha256"],
        "embeddingGenerationManifestSha256": embedding_config["generationManifestSha256"],
        "modelId": embedding_config["modelId"],
        "modelRevision": embedding_config["modelRevision"],
        "dimension": dimension,
        "pooling": embedding_config["pooling"],
        "users": len(pooled_by_user),
        "matchedPosts": matched_posts,
        "sourceShardCount": len(shard_reports),
        "resumedShardCount": int(sum(bool(item["resumed"]) for item in shard_reports)),
        "trainPoolSha256": train_hash,
        "testPoolSha256": test_hash,
        "shards": shard_reports,
    }
    write_json(output.reports / "embedding-pool-audit.json", report)


def load_pool(path: Path, expected_dimension: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        if set(payload.files) != {"user_ids", "embeddings", "counts"}:
            raise RuntimeError(f"pool {path} contains unexpected fields: {payload.files}")
        user_ids = payload["user_ids"].astype(str).tolist()
        embeddings = payload["embeddings"].astype(np.float32)
        counts = payload["counts"].astype(np.int32)
    if embeddings.shape != (len(user_ids), expected_dimension):
        raise RuntimeError(f"pool {path} has invalid shape {embeddings.shape}")
    if counts.shape != (len(user_ids),) or np.any(counts <= 0):
        raise RuntimeError(f"pool {path} has invalid post counts")
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"pool {path} has non-finite embeddings")
    if len(set(user_ids)) != len(user_ids):
        raise RuntimeError(f"pool {path} has duplicate users")
    return user_ids, embeddings, counts


def aligned_embeddings(rows: Sequence[dict[str, str]], pool_path: Path, dimension: int) -> np.ndarray:
    pool_ids, embeddings, _counts = load_pool(pool_path, dimension)
    index = {user_id: position for position, user_id in enumerate(pool_ids)}
    requested = [row["user_id"] for row in rows]
    if set(requested) != set(pool_ids):
        raise RuntimeError(f"pool {pool_path} user set does not match its manifest")
    return embeddings[np.asarray([index[user_id] for user_id in requested], dtype=np.int64)]


def build_classifier(config: dict[str, Any]) -> Pipeline:
    classifier = config["classifier"]
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=float(classifier["C"]),
                    penalty=classifier["penalty"],
                    solver=classifier["solver"],
                    max_iter=int(classifier["maxIter"]),
                    class_weight=classifier["classWeight"],
                    random_state=int(classifier["randomState"]),
                ),
            ),
        ]
    )


def positive_probability(model: Pipeline, features: np.ndarray) -> np.ndarray:
    classes = list(model.named_steps["logreg"].classes_)
    if 1 not in classes:
        return np.ones(len(features), dtype=np.float64) if classes == [1] else np.zeros(len(features))
    return model.predict_proba(features)[:, classes.index(1)].astype(np.float64)


def train_oof_command(config: dict[str, Any], output: ArtifactPaths) -> None:
    output.ensure()
    rows = read_csv(output.manifests / "train_manifest.csv")
    expected = int(config["source"]["splits"]["train"]["users"])
    if len(rows) != expected or len({row["user_id"] for row in rows}) != expected:
        raise RuntimeError("train manifest coverage is invalid")
    dimension = int(config["embeddings"]["dimension"])
    features = aligned_embeddings(rows, output.embeddings / "train_user_mean.npz", dimension)
    labels = np.asarray([LABEL_TO_CODE[row["label"]] for row in rows], dtype=np.int8)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int16)
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    fold_artifacts = []
    for fold in range(int(config["validation"]["foldCount"])):
        validation = folds == fold
        training = ~validation
        model = build_classifier(config)
        model.fit(features[training], labels[training])
        scores[validation] = positive_probability(model, features[validation])
        model_path = output.models / f"fold_{fold}.joblib"
        joblib.dump(model, model_path)
        fold_artifacts.append(
            {
                "fold": fold,
                "trainUsers": int(training.sum()),
                "validationUsers": int(validation.sum()),
                "modelSha256": sha256_file(model_path),
            }
        )
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise RuntimeError("OOF scoring produced invalid probabilities")
    score_path = output.scores / f"train_oof_{config['modelId']}.csv"
    score_hash = write_csv(
        score_path,
        ["user_id", "label", "fold", "score", "model_id"],
        (
            {
                "user_id": row["user_id"],
                "label": row["label"],
                "fold": row["fold"],
                "score": repr(float(score)),
                "model_id": config["modelId"],
            }
            for row, score in zip(rows, scores)
        ),
    )
    write_json(
        output.manifests / "oof-model-manifest.json",
        {
            "experimentId": config["experimentId"],
            "modelId": config["modelId"],
            "configSha256": config["_configSha256"],
            "trainManifestSha256": sha256_file(output.manifests / "train_manifest.csv"),
            "trainPoolSha256": sha256_file(output.embeddings / "train_user_mean.npz"),
            "scoreSha256": score_hash,
            "folds": fold_artifacts,
        },
    )


def metrics_from_codes(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.int8)
    predicted = np.asarray(predicted, dtype=np.int8)
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    diagnosed_f1 = ratio(2 * precision * recall, precision + recall)
    control_precision = ratio(tn, tn + fn)
    control_recall = ratio(tn, tn + fp)
    control_f1 = ratio(2 * control_precision * control_recall, control_precision + control_recall)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "confusionMatrix": {
            "labels": ["no", "yes"],
            "matrix": [[tn, fp], [fn, tp]],
        },
        "macroF1": (diagnosed_f1 + control_f1) / 2,
        "diagnosedF1": diagnosed_f1,
        "precision": precision,
        "recall": recall,
        "controlF1": control_f1,
        "accuracy": ratio(tp + tn, len(actual)),
    }


def metric_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["macroF1"]),
        float(metrics["diagnosedF1"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
    )


def select_threshold(actual: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    unique = sorted(set(float(score) for score in scores))
    candidates = {-np.finfo(float).eps, 1.0 + np.finfo(float).eps}
    for index, score in enumerate(unique):
        candidates.add(score)
        if index + 1 < len(unique):
            candidates.add((score + unique[index + 1]) / 2.0)
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    for threshold in sorted(candidates):
        metrics = metrics_from_codes(actual, (scores >= threshold).astype(np.int8))
        if best_metrics is None or metric_key(metrics) > metric_key(best_metrics):
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("threshold sweep had no candidates")
    return best_threshold, best_metrics


def audit_oof_command(config: dict[str, Any], output: ArtifactPaths) -> None:
    manifest_rows = read_csv(output.manifests / "train_manifest.csv")
    score_path = output.scores / f"train_oof_{config['modelId']}.csv"
    score_rows = read_csv(score_path)
    expected_fields = {"user_id", "label", "fold", "score", "model_id"}
    if not score_rows or set(score_rows[0]) != expected_fields:
        raise RuntimeError("OOF score schema is invalid")
    manifest = {row["user_id"]: row for row in manifest_rows}
    seen: set[str] = set()
    fold_counts: dict[str, int] = {}
    for row in score_rows:
        user_id = row["user_id"]
        if user_id in seen:
            raise RuntimeError(f"OOF score contains duplicate user {user_id}")
        seen.add(user_id)
        expected = manifest.get(user_id)
        if not expected or row["label"] != expected["label"] or row["fold"] != expected["fold"]:
            raise RuntimeError(f"OOF row does not match train manifest for {user_id}")
        if row["model_id"] != config["modelId"]:
            raise RuntimeError(f"OOF row has wrong model id for {user_id}")
        score = float(row["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError(f"OOF row has invalid score for {user_id}")
        fold_counts[row["fold"]] = fold_counts.get(row["fold"], 0) + 1
    if seen != set(manifest):
        raise RuntimeError("OOF user coverage differs from the train manifest")
    test_users = {row["user_id"] for row in read_csv(output.manifests / "test_inference.csv")}
    if seen & test_users:
        raise RuntimeError("OOF scores contain held-out test users")
    write_json(
        output.reports / "oof-audit.json",
        {
            "ok": True,
            "experimentId": config["experimentId"],
            "modelId": config["modelId"],
            "users": len(seen),
            "foldCounts": dict(sorted(fold_counts.items())),
            "trainManifestSha256": sha256_file(output.manifests / "train_manifest.csv"),
            "oofScoreSha256": sha256_file(score_path),
            "oofModelManifestSha256": sha256_file(output.manifests / "oof-model-manifest.json"),
            "testUsersExcluded": True,
        },
    )


def lock_command(config: dict[str, Any], output: ArtifactPaths) -> None:
    audit = json.loads((output.reports / "oof-audit.json").read_text())
    if not audit.get("ok"):
        raise RuntimeError("OOF audit did not pass")
    source_audit_path = output.reports / "source-audit.json"
    pool_audit_path = output.reports / "embedding-pool-audit.json"
    source_audit = json.loads(source_audit_path.read_text())
    pool_audit = json.loads(pool_audit_path.read_text())
    if not source_audit.get("ok") or not pool_audit.get("ok"):
        raise RuntimeError("source and embedding-pool audits must pass before locking")
    score_path = output.scores / f"train_oof_{config['modelId']}.csv"
    rows = read_csv(score_path)
    actual = np.asarray([LABEL_TO_CODE[row["label"]] for row in rows], dtype=np.int8)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    threshold, metrics = select_threshold(actual, scores)
    lock = {
        "experimentId": config["experimentId"],
        "modelId": config["modelId"],
        "predictionColumn": config["predictionColumn"],
        "threshold": threshold,
        "thresholdRule": "score >= threshold",
        "thresholdSelection": {
            "primaryMetric": config["validation"]["thresholdMetric"],
            "tieBreakers": config["validation"]["thresholdTieBreakers"],
            "finalDeterministicTieBreaker": "lowest numeric threshold candidate",
            "source": "five-fold train OOF only",
        },
        "oofMetrics": metrics,
        "classifier": config["classifier"],
        "embedding": {
            "modelId": config["embeddings"]["modelId"],
            "modelRevision": config["embeddings"]["modelRevision"],
            "dimension": config["embeddings"]["dimension"],
            "pooling": config["embeddings"]["pooling"],
        },
        "hashChain": {
            "configSha256": config["_configSha256"],
            "sourceCorpusSha256": config["source"]["sha256"],
            "sourceAuditSha256": sha256_file(source_audit_path),
            "preparedManifestSha256": sha256_file(output.manifests / "prepared-manifest.json"),
            "embeddingPoolAuditSha256": sha256_file(pool_audit_path),
            "trainPoolSha256": sha256_file(output.embeddings / "train_user_mean.npz"),
            "trainManifestSha256": sha256_file(output.manifests / "train_manifest.csv"),
            "oofScoreSha256": sha256_file(score_path),
            "oofAuditSha256": sha256_file(output.reports / "oof-audit.json"),
        },
    }
    write_json(output.ensemble / "model-lock.json", lock)


def fit_full_command(config: dict[str, Any], output: ArtifactPaths) -> None:
    lock_path = output.ensemble / "model-lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("experimentId") != config["experimentId"]:
        raise RuntimeError("model lock belongs to another experiment")
    if lock["hashChain"]["configSha256"] != config["_configSha256"]:
        raise RuntimeError("model lock was created with a different config")
    locked_inputs = {
        "sourceAuditSha256": output.reports / "source-audit.json",
        "preparedManifestSha256": output.manifests / "prepared-manifest.json",
        "embeddingPoolAuditSha256": output.reports / "embedding-pool-audit.json",
        "trainPoolSha256": output.embeddings / "train_user_mean.npz",
        "trainManifestSha256": output.manifests / "train_manifest.csv",
        "oofScoreSha256": output.scores / f"train_oof_{config['modelId']}.csv",
        "oofAuditSha256": output.reports / "oof-audit.json",
    }
    for name, path in locked_inputs.items():
        require_sha256(path, lock["hashChain"][name], name)
    rows = read_csv(output.manifests / "train_manifest.csv")
    features = aligned_embeddings(
        rows, output.embeddings / "train_user_mean.npz", int(config["embeddings"]["dimension"])
    )
    labels = np.asarray([LABEL_TO_CODE[row["label"]] for row in rows], dtype=np.int8)
    model = build_classifier(config)
    model.fit(features, labels)
    model_path = output.models / "full_fit.joblib"
    joblib.dump(model, model_path)
    write_json(
        output.manifests / "full-fit-model-manifest.json",
        {
            "experimentId": config["experimentId"],
            "modelId": config["modelId"],
            "configSha256": config["_configSha256"],
            "users": len(rows),
            "positive": int(labels.sum()),
            "negative": int((labels == 0).sum()),
            "modelLockSha256": sha256_file(lock_path),
            "trainManifestSha256": sha256_file(output.manifests / "train_manifest.csv"),
            "trainPoolSha256": sha256_file(output.embeddings / "train_user_mean.npz"),
            "modelSha256": sha256_file(model_path),
        },
    )


def score_test_command(config: dict[str, Any], output: ArtifactPaths) -> None:
    lock_path = output.ensemble / "model-lock.json"
    full_manifest = json.loads((output.manifests / "full-fit-model-manifest.json").read_text())
    if full_manifest.get("configSha256") != config["_configSha256"]:
        raise RuntimeError("full-fit model was created with a different config")
    require_sha256(lock_path, full_manifest["modelLockSha256"], "full-fit model lock")
    model_path = output.models / "full_fit.joblib"
    if sha256_file(model_path) != full_manifest["modelSha256"]:
        raise RuntimeError("full-fit model hash mismatch")
    rows = read_csv(output.manifests / "test_inference.csv")
    if not rows or set(rows[0]) != {"user_id"}:
        raise RuntimeError("test inference manifest is not label-free")
    features = aligned_embeddings(
        rows, output.embeddings / "test_user_mean.npz", int(config["embeddings"]["dimension"])
    )
    model = joblib.load(model_path)
    scores = positive_probability(model, features)
    destination = output.scores / f"test_{config['modelId']}.csv"
    write_csv(
        destination,
        ["user_id", "score", "model_id"],
        (
            {
                "user_id": row["user_id"],
                "score": repr(float(score)),
                "model_id": config["modelId"],
            }
            for row, score in zip(rows, scores)
        ),
    )


def audit_test_command(config: dict[str, Any], output: ArtifactPaths) -> None:
    inference_rows = read_csv(output.manifests / "test_inference.csv")
    score_path = output.scores / f"test_{config['modelId']}.csv"
    score_rows = read_csv(score_path)
    expected_fields = {"user_id", "score", "model_id"}
    if not score_rows or set(score_rows[0]) != expected_fields:
        raise RuntimeError(f"test score schema is invalid: {set(score_rows[0]) if score_rows else set()}")
    forbidden = {"label", "true_label", "fold", "Diagnosed_YN"}
    if expected_fields & forbidden:
        raise RuntimeError("test score schema leaks test labels or folds")
    expected_ids = [row["user_id"] for row in inference_rows]
    observed_ids = [row["user_id"] for row in score_rows]
    if observed_ids != expected_ids or len(set(observed_ids)) != len(observed_ids):
        raise RuntimeError("test score coverage/order differs from the inference manifest")
    if len(observed_ids) != int(config["source"]["splits"]["test"]["users"]):
        raise RuntimeError("test score has the wrong number of users")
    for row in score_rows:
        score = float(row["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError(f"invalid test score for {row['user_id']}")
        if row["model_id"] != config["modelId"]:
            raise RuntimeError(f"invalid model id for {row['user_id']}")
    write_json(
        output.reports / "label-free-test-score-manifest.json",
        {
            "ok": True,
            "experimentId": config["experimentId"],
            "modelId": config["modelId"],
            "users": len(score_rows),
            "schema": list(score_rows[0]),
            "forbiddenFieldsAbsent": sorted(forbidden),
            "testInferenceSha256": sha256_file(output.manifests / "test_inference.csv"),
            "testPoolSha256": sha256_file(output.embeddings / "test_user_mean.npz"),
            "modelLockSha256": sha256_file(output.ensemble / "model-lock.json"),
            "fullFitModelManifestSha256": sha256_file(output.manifests / "full-fit-model-manifest.json"),
            "testScoreSha256": sha256_file(score_path),
        },
    )


def bootstrap_intervals(
    actual: np.ndarray, predicted: np.ndarray, samples: int, seed: int
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(actual == 0)
    positive = np.flatnonzero(actual == 1)
    keys = ["macroF1", "diagnosedF1", "precision", "recall", "accuracy"]
    values = {key: np.empty(samples, dtype=np.float64) for key in keys}
    for sample in range(samples):
        indexes = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        metrics = metrics_from_codes(actual[indexes], predicted[indexes])
        for key in keys:
            values[key][sample] = metrics[key]
    return {
        key: [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))]
        for key, distribution in values.items()
    }


def evaluate_command(config: dict[str, Any], output: ArtifactPaths) -> None:
    score_manifest_path = output.reports / "label-free-test-score-manifest.json"
    score_manifest = json.loads(score_manifest_path.read_text())
    if not score_manifest.get("ok"):
        raise RuntimeError("label-free test score audit did not pass")
    score_path = output.scores / f"test_{config['modelId']}.csv"
    lock_path = output.ensemble / "model-lock.json"
    require_sha256(score_path, score_manifest["testScoreSha256"], "audited test score")
    require_sha256(lock_path, score_manifest["modelLockSha256"], "audited model lock")
    prepared = json.loads((output.manifests / "prepared-manifest.json").read_text())
    require_sha256(
        output.sealed / "test_labels.csv",
        prepared["sealedTestLabelsSha256"],
        "sealed test labels",
    )
    label_rows = read_csv(output.sealed / "test_labels.csv")
    score_rows = read_csv(score_path)
    score_ids = [row["user_id"] for row in score_rows]
    label_ids = [row["user_id"] for row in label_rows]
    if len(set(label_ids)) != len(label_ids) or set(label_ids) != set(score_ids):
        raise RuntimeError("sealed test-label coverage differs from the audited test scores")
    labels = {row["user_id"]: LABEL_TO_CODE[row["label"]] for row in label_rows}
    actual = np.asarray([labels[row["user_id"]] for row in score_rows], dtype=np.int8)
    scores = np.asarray([float(row["score"]) for row in score_rows], dtype=np.float64)
    lock = json.loads(lock_path.read_text())
    predicted = (scores >= float(lock["threshold"])).astype(np.int8)
    metrics = metrics_from_codes(actual, predicted)
    intervals = bootstrap_intervals(
        actual,
        predicted,
        int(config["validation"]["bootstrapSamples"]),
        int(config["validation"]["seed"]),
    )
    write_json(
        output.reports / "final-test-report.json",
        {
            "experimentId": config["experimentId"],
            "modelId": config["modelId"],
            "split": "test",
            "users": len(actual),
            "metrics": metrics,
            "stratifiedBootstrap95": intervals,
            "bootstrapSamples": int(config["validation"]["bootstrapSamples"]),
            "bootstrapSeed": int(config["validation"]["seed"]),
            "modelLockSha256": sha256_file(lock_path),
            "labelFreeTestScoreManifestSha256": sha256_file(score_manifest_path),
            "testScoreSha256": sha256_file(score_path),
            "sealedTestLabelsSha256": sha256_file(output.sealed / "test_labels.csv"),
        },
    )


def materialize_command(source_path: Path, config: dict[str, Any], output: ArtifactPaths) -> None:
    final_report_path = output.reports / "final-test-report.json"
    if not final_report_path.exists():
        raise RuntimeError("final test report must exist before materializing the derived corpus")
    final_report = json.loads(final_report_path.read_text())
    score_path = output.scores / f"test_{config['modelId']}.csv"
    require_sha256(score_path, final_report["testScoreSha256"], "evaluated test score")
    normalized = normalized_source_frame(source_path, config)
    original = pd.read_pickle(source_path)
    score_rows = read_csv(score_path)
    lock = json.loads((output.ensemble / "model-lock.json").read_text())
    threshold = float(lock["threshold"])
    predictions = {
        row["user_id"]: CODE_TO_LABEL[int(float(row["score"]) >= threshold)] for row in score_rows
    }
    derived = original.copy()
    prediction_column = config["predictionColumn"]
    derived[prediction_column] = pd.Series(pd.NA, index=derived.index, dtype="string")
    test_mask = normalized["Split"] == "test"
    normalized_predictions = normalized.loc[test_mask, "User_ID"].map(predictions).astype("string")
    derived.loc[test_mask, prediction_column] = normalized_predictions.to_numpy()
    if int(derived[prediction_column].notna().sum()) != int(config["source"]["splits"]["test"]["users"]):
        raise RuntimeError("derived corpus does not contain exactly 400 predictions")
    if derived.loc[normalized["Split"] == "train", prediction_column].notna().any():
        raise RuntimeError("derived corpus populated predictions on train rows")
    if derived.loc[test_mask, prediction_column].isna().any():
        raise RuntimeError("derived corpus is missing test predictions")
    if not derived[SOURCE_COLUMNS].equals(original[SOURCE_COLUMNS]):
        raise RuntimeError("derived corpus changed an original source column")
    destination = output.corpus / "SetembroBR-v7-min-qwen-logreg.pkl"
    derived.to_pickle(destination)
    write_json(
        output.reports / "derived-corpus-manifest.json",
        {
            "experimentId": config["experimentId"],
            "sourceFilename": source_path.name,
            "sourceSha256": config["source"]["sha256"],
            "derivedFilename": destination.name,
            "derivedSha256": sha256_file(destination),
            "rows": len(derived),
            "schema": list(derived.columns),
            "predictionColumn": prediction_column,
            "trainNonNullPredictions": int(
                derived.loc[normalized["Split"] == "train", prediction_column].notna().sum()
            ),
            "testNonNullPredictions": int(derived.loc[test_mask, prediction_column].notna().sum()),
            "predictionValues": sorted(derived.loc[test_mask, prediction_column].dropna().unique().tolist()),
            "testScoreSha256": sha256_file(score_path),
            "finalTestReportSha256": sha256_file(final_report_path),
        },
    )


def audit_final_command(source_path: Path, config: dict[str, Any], output: ArtifactPaths) -> None:
    required = {
        "sourceAuditSha256": output.reports / "source-audit.json",
        "preparedManifestSha256": output.manifests / "prepared-manifest.json",
        "embeddingPoolAuditSha256": output.reports / "embedding-pool-audit.json",
        "oofModelManifestSha256": output.manifests / "oof-model-manifest.json",
        "oofAuditSha256": output.reports / "oof-audit.json",
        "modelLockSha256": output.ensemble / "model-lock.json",
        "fullFitModelManifestSha256": output.manifests / "full-fit-model-manifest.json",
        "labelFreeTestScoreManifestSha256": output.reports / "label-free-test-score-manifest.json",
        "finalTestReportSha256": output.reports / "final-test-report.json",
        "derivedCorpusManifestSha256": output.reports / "derived-corpus-manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"final audit is missing required artifacts: {missing}")
    if sha256_file(source_path) != config["source"]["sha256"]:
        raise RuntimeError("original v7 source changed before final audit")
    for name in ["source-audit.json", "embedding-pool-audit.json", "oof-audit.json", "label-free-test-score-manifest.json"]:
        if not json.loads((output.reports / name).read_text()).get("ok"):
            raise RuntimeError(f"required audit did not pass: {name}")
    derived_manifest = json.loads((output.reports / "derived-corpus-manifest.json").read_text())
    derived_path = output.corpus / derived_manifest["derivedFilename"]
    if sha256_file(derived_path) != derived_manifest["derivedSha256"]:
        raise RuntimeError("derived corpus hash mismatch")
    source = pd.read_pickle(source_path)
    derived = pd.read_pickle(derived_path)
    prediction_column = config["predictionColumn"]
    if list(derived.columns) != [*SOURCE_COLUMNS, prediction_column]:
        raise RuntimeError("derived corpus schema is invalid")
    if not derived[SOURCE_COLUMNS].equals(source[SOURCE_COLUMNS]):
        raise RuntimeError("derived corpus changed an original source value")
    normalized = normalized_source_frame(source_path, config)
    train_mask = normalized["Split"] == "train"
    test_mask = normalized["Split"] == "test"
    if derived.loc[train_mask, prediction_column].notna().any():
        raise RuntimeError("derived corpus has non-null train predictions")
    test_predictions = derived.loc[test_mask, prediction_column]
    if test_predictions.isna().any() or set(test_predictions.astype(str)) - set(LABEL_TO_CODE):
        raise RuntimeError("derived corpus has incomplete or invalid test predictions")
    if int(test_predictions.notna().sum()) != int(config["source"]["splits"]["test"]["users"]):
        raise RuntimeError("derived corpus has the wrong number of test predictions")
    final = {
        "ok": True,
        "experimentId": config["experimentId"],
        "modelId": config["modelId"],
        "sourceCorpusSha256": config["source"]["sha256"],
        "sourceCorpusUnchanged": True,
        "expectedUsers": int(config["source"]["expectedUsers"]),
        "expectedPosts": int(config["source"]["expectedPosts"]),
        "oofUsers": int(config["source"]["splits"]["train"]["users"]),
        "testUsers": int(config["source"]["splits"]["test"]["users"]),
        "embeddingRegenerationPerformed": False,
        "testPredictionsOnly": True,
        "hashChain": {name: sha256_file(path) for name, path in required.items()},
        "derivedCorpusSha256": derived_manifest["derivedSha256"],
    }
    write_json(output.reports / "final-audit.json", final)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "validate",
            "seal",
            "pool-embeddings",
            "train-oof",
            "audit-oof",
            "lock",
            "fit-full",
            "score-test",
            "audit-test",
            "evaluate-test",
            "materialize-corpus",
            "audit-final",
        ],
    )
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--source-pkl", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(".work/output"))
    parser.add_argument("--temporary-dir", type=Path)
    return parser.parse_args()


def require_path(path: Path | None, flag: str) -> Path:
    if path is None:
        raise SystemExit(f"{flag} is required for this command")
    return path.expanduser().resolve()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.expanduser().resolve())
    output = ArtifactPaths(args.output_dir.expanduser().resolve())
    output.ensure()
    source_commands = {"validate", "seal", "materialize-corpus", "audit-final"}
    archive_commands = {"validate", "pool-embeddings"}
    source = require_path(args.source_pkl, "--source-pkl") if args.command in source_commands else None
    archive = require_path(args.archive_root, "--archive-root") if args.command in archive_commands else None
    temporary = args.temporary_dir.expanduser().resolve() if args.temporary_dir else None
    if temporary:
        temporary.mkdir(parents=True, exist_ok=True)

    commands = {
        "validate": lambda: validate_command(source, archive, config, output),
        "seal": lambda: prepare_command(source, config, output),
        "pool-embeddings": lambda: pool_command(archive, config, output, temporary),
        "train-oof": lambda: train_oof_command(config, output),
        "audit-oof": lambda: audit_oof_command(config, output),
        "lock": lambda: lock_command(config, output),
        "fit-full": lambda: fit_full_command(config, output),
        "score-test": lambda: score_test_command(config, output),
        "audit-test": lambda: audit_test_command(config, output),
        "evaluate-test": lambda: evaluate_command(config, output),
        "materialize-corpus": lambda: materialize_command(source, config, output),
        "audit-final": lambda: audit_final_command(source, config, output),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
