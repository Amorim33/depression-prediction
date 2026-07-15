#!/usr/bin/env python3
"""Validate raw SetembroBR data and generate local Qwen3 embedding artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import random
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_USERS = {"train": 10776, "test": 2696}
RAW_CSV_FILES = {
    "train": ["train_D_SetembroBR_v6.csv", "train_D_c_SetembroBR_v6.csv"],
    "test": ["test_D_SetembroBR_v6.csv", "test_D_c_SetembroBR_v6.csv"],
}
RELEVANCE_FILES = {
    "train": "train_D_relevancia_1to10_all.pkl",
    "test": "teste_D_relevancia_1to10.pkl",
}
REQUIRED_RELEVANCE_COLUMNS = ["User_ID", "Diagnosed_YN", "Text", "label"]
THRESHOLDS = [3, 6, 7]


@dataclass(frozen=True)
class UserRecord:
    split: str
    user_id: str
    label_name: str
    label_code: int
    texts: list[str]
    relevance_raw: list[int | None]
    relevance_score: np.ndarray


@dataclass(frozen=True)
class ValidationResult:
    records_by_split: dict[str, list[UserRecord]]
    split_reports: dict[str, dict[str, Any]]
    dataset_hashes: dict[str, str]
    relevance_hashes: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/setembrobr.seed42.raw-qwen3-embeddings.json")
    parser.add_argument("--mode", choices=["validate", "embed"], default="validate")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--relevance-dir")
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


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def ensure_dirs(output_dir: Path) -> None:
    for relative in [
        "manifests",
        "shards",
        "tweet_embeddings/train",
        "tweet_embeddings/test",
        "pooled",
        "reports",
    ]:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_user_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "diagnosed", "d", "depressed"}:
        return "diagnosed"
    if text in {"0", "false", "no", "n", "control", "c", "not diagnosed"}:
        return "control"
    raise ValueError(f"unsupported Diagnosed_YN value: {value!r}")


def label_code(label_name: str) -> int:
    return 1 if label_name == "diagnosed" else 0


def load_csv_labels(dataset_dir: Path, split: str) -> dict[str, str]:
    csv.field_size_limit(sys.maxsize)
    labels: dict[str, str] = {}
    for filename in RAW_CSV_FILES[split]:
        path = dataset_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            required = {"User_ID", "Diagnosed_YN", "Text"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"{path} is missing required columns {sorted(required)}")
            for row in reader:
                user_id = normalize_user_id(row["User_ID"])
                current = normalize_label(row["Diagnosed_YN"])
                previous = labels.get(user_id)
                if previous is not None:
                    raise ValueError(f"duplicate user {user_id} in raw CSV files for {split}")
                labels[user_id] = current
    return labels


def load_relevance_frame(relevance_dir: Path, split: str) -> pd.DataFrame:
    path = relevance_dir / RELEVANCE_FILES[split]
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        frame = pickle.load(handle)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{path} did not contain a pandas DataFrame")
    missing = [column for column in REQUIRED_RELEVANCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def parse_relevance_values(values: Any, split: str, user_id: str) -> tuple[list[int | None], np.ndarray, dict[str, Any]]:
    if not isinstance(values, list):
        raise TypeError(f"{split}:{user_id} label column must be a list")
    raw: list[int | None] = []
    clipped: list[int] = []
    non_digit_count = 0
    out_of_range: dict[str, int] = {}
    digit_counts: dict[str, int] = {}
    for value in values:
        text = str(value).strip()
        if text.isdigit():
            numeric = int(text)
            raw.append(numeric)
            clipped_score = min(max(numeric, 0), 10)
            clipped.append(clipped_score)
            digit_counts[str(numeric)] = digit_counts.get(str(numeric), 0) + 1
            if numeric < 0 or numeric > 10:
                out_of_range[str(numeric)] = out_of_range.get(str(numeric), 0) + 1
        else:
            raw.append(None)
            clipped.append(0)
            non_digit_count += 1
    report = {
        "digit_counts": digit_counts,
        "non_digit_count": non_digit_count,
        "out_of_range_numeric": out_of_range,
    }
    return raw, np.asarray(clipped, dtype=np.int16), report


def merge_counts(left: dict[str, int], right: dict[str, int]) -> None:
    for key, value in right.items():
        left[key] = left.get(key, 0) + int(value)


def validate_split(dataset_dir: Path, relevance_dir: Path, split: str) -> tuple[list[UserRecord], dict[str, Any]]:
    csv_labels = load_csv_labels(dataset_dir, split)
    frame = load_relevance_frame(relevance_dir, split)
    if len(frame) != EXPECTED_USERS[split]:
        raise ValueError(f"{split} relevance rows: expected {EXPECTED_USERS[split]}, got {len(frame)}")
    if len(csv_labels) != EXPECTED_USERS[split]:
        raise ValueError(f"{split} raw CSV users: expected {EXPECTED_USERS[split]}, got {len(csv_labels)}")

    records: list[UserRecord] = []
    seen: set[str] = set()
    total_tweets = 0
    total_digit_scores = 0
    total_non_digit_scores = 0
    out_of_range_numeric: dict[str, int] = {}
    digit_counts: dict[str, int] = {}
    users_with_out_of_range = 0
    users_with_non_digit = 0

    for row_index, row in frame.iterrows():
        user_id = normalize_user_id(row["User_ID"])
        if user_id in seen:
            raise ValueError(f"duplicate user {user_id} in {split} relevance pickle")
        seen.add(user_id)
        if user_id not in csv_labels:
            raise ValueError(f"{split} relevance user {user_id} is missing from raw CSV files")

        relevance_label = normalize_label(row["Diagnosed_YN"])
        if relevance_label != csv_labels[user_id]:
            raise ValueError(
                f"{split}:{user_id} label mismatch, CSV={csv_labels[user_id]} relevance={relevance_label}"
            )

        texts = row["Text"]
        labels = row["label"]
        if not isinstance(texts, list):
            raise TypeError(f"{split}:{user_id} Text column must be a list")
        if not isinstance(labels, list):
            raise TypeError(f"{split}:{user_id} label column must be a list")
        if len(texts) != len(labels):
            raise ValueError(f"{split}:{user_id} Text and label lengths differ")

        relevance_raw, relevance_score, relevance_report = parse_relevance_values(labels, split, user_id)
        merge_counts(digit_counts, relevance_report["digit_counts"])
        merge_counts(out_of_range_numeric, relevance_report["out_of_range_numeric"])
        non_digit_count = int(relevance_report["non_digit_count"])
        total_tweets += len(texts)
        total_non_digit_scores += non_digit_count
        total_digit_scores += len(texts) - non_digit_count
        if relevance_report["out_of_range_numeric"]:
            users_with_out_of_range += 1
        if non_digit_count > 0:
            users_with_non_digit += 1

        records.append(
            UserRecord(
                split=split,
                user_id=user_id,
                label_name=relevance_label,
                label_code=label_code(relevance_label),
                texts=[str(text) for text in texts],
                relevance_raw=relevance_raw,
                relevance_score=relevance_score,
            )
        )

    missing_from_pickle = sorted(set(csv_labels) - seen)
    if missing_from_pickle:
        sample = ", ".join(missing_from_pickle[:5])
        raise ValueError(f"{split} raw CSV users missing from relevance pickle: {sample}")

    label_counts = {
        "control": sum(1 for record in records if record.label_name == "control"),
        "diagnosed": sum(1 for record in records if record.label_name == "diagnosed"),
    }
    report = {
        "split": split,
        "users": len(records),
        "expectedUsers": EXPECTED_USERS[split],
        "labelCounts": label_counts,
        "tweetCount": total_tweets,
        "digitRelevanceCount": total_digit_scores,
        "nonDigitRelevanceCount": total_non_digit_scores,
        "usersWithNonDigitRelevance": users_with_non_digit,
        "usersWithOutOfRangeNumericRelevance": users_with_out_of_range,
        "digitRelevanceCounts": dict(sorted(digit_counts.items(), key=lambda item: int(item[0]))),
        "outOfRangeNumericRelevance": dict(sorted(out_of_range_numeric.items(), key=lambda item: int(item[0]))),
    }
    return records, report


def validate_inputs(dataset_dir: Path, relevance_dir: Path) -> ValidationResult:
    dataset_hashes: dict[str, str] = {}
    relevance_hashes: dict[str, str] = {}
    for split, filenames in RAW_CSV_FILES.items():
        for filename in filenames:
            path = dataset_dir / filename
            dataset_hashes[f"{split}/{filename}"] = sha256_file(path)
    for split, filename in RELEVANCE_FILES.items():
        path = relevance_dir / filename
        relevance_hashes[f"{split}/{filename}"] = sha256_file(path)

    records_by_split: dict[str, list[UserRecord]] = {}
    split_reports: dict[str, dict[str, Any]] = {}
    for split in ["train", "test"]:
        records, report = validate_split(dataset_dir, relevance_dir, split)
        records_by_split[split] = records
        split_reports[split] = report
    return ValidationResult(records_by_split, split_reports, dataset_hashes, relevance_hashes)


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


def row_hash(split: str, label_code_value: int, user_id: str) -> str:
    value = f"setembrobr|raw-qwen3|{split}|{label_code_value}|{user_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def selected_records(records_by_split: dict[str, list[UserRecord]], smoke_users: int) -> dict[str, list[UserRecord]]:
    if smoke_users <= 0:
        return records_by_split
    return {split: records[:smoke_users] for split, records in records_by_split.items()}


def write_manifest_files(output_dir: Path, records_by_split: dict[str, list[UserRecord]], seed: int, folds: int) -> str:
    manifest_path = output_dir / "manifests" / f"raw_split_manifest_seed{seed}.csv"
    train_folds = stratified_folds(records_by_split["train"], seed, folds)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "split", "user_id", "label", "fold", "row_hash"],
        )
        writer.writeheader()
        for split in ["train", "test"]:
            for record in records_by_split[split]:
                writer.writerow(
                    {
                        "dataset": "setembrobr",
                        "split": split,
                        "user_id": record.user_id,
                        "label": record.label_code,
                        "fold": train_folds[record.user_id] if split == "train" else "",
                        "row_hash": row_hash(split, record.label_code, record.user_id),
                    }
                )

    test_users_path = output_dir / "manifests" / f"raw_test_users_seed{seed}.csv"
    with test_users_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id"])
        writer.writeheader()
        for record in records_by_split["test"]:
            writer.writerow({"user_id": record.user_id})

    test_labels_path = output_dir / "manifests" / f"raw_test_labels_seed{seed}.csv"
    with test_labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id", "label"])
        writer.writeheader()
        for record in records_by_split["test"]:
            writer.writerow({"user_id": record.user_id, "label": record.label_code})

    return sha256_file(manifest_path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validation_report_payload(
    config: dict[str, Any],
    dataset_dir: Path,
    relevance_dir: Path,
    output_dir: Path,
    validation: ValidationResult,
    selected: dict[str, list[UserRecord]],
    manifest_hash: str,
    command: list[str],
) -> dict[str, Any]:
    return {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "mode": "validate",
        "config": config,
        "datasetDir": str(dataset_dir),
        "relevanceDir": str(relevance_dir),
        "outputDir": str(output_dir),
        "splitReports": validation.split_reports,
        "selectedUsers": {split: len(records) for split, records in selected.items()},
        "datasetHashes": validation.dataset_hashes,
        "relevanceHashes": validation.relevance_hashes,
        "splitManifestHash": manifest_hash,
        "command": command,
    }


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for package_name in ["torch", "transformers", "sentence_transformers", "pyarrow", "safetensors"]:
        try:
            module = __import__(package_name)
            versions[package_name] = str(getattr(module, "__version__", "unknown"))
        except Exception as error:  # noqa: BLE001 - report environment issues in artifact manifest.
            versions[package_name] = f"unavailable: {error}"
    return versions


def command_output(args: list[str]) -> str:
    try:
        completed = subprocess.run(args, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return "unavailable"
    output = (completed.stdout + completed.stderr).strip()
    return output[:4000]


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except Exception:
        if requested == "cuda":
            raise
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedding_model(model_id: str, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    model_kwargs: dict[str, Any] = {}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    try:
        return SentenceTransformer(
            model_id,
            device=device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            processor_kwargs={"padding_side": "left"},
        )
    except TypeError:
        try:
            return SentenceTransformer(
                model_id,
                device=device,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
                tokenizer_kwargs={"padding_side": "left"},
            )
        except TypeError:
            return SentenceTransformer(model_id, device=device, model_kwargs=model_kwargs)


def model_revision(model: Any) -> str:
    candidates: list[Any] = []
    model_card_data = getattr(model, "model_card_data", None)
    if model_card_data is not None:
        for attribute in ["base_model_revision", "model_revision", "revision"]:
            candidates.append(getattr(model_card_data, attribute, None))
    try:
        first_module = model._first_module()
        auto_model = getattr(first_module, "auto_model", None)
        config = getattr(auto_model, "config", None)
        candidates.append(getattr(config, "_commit_hash", None))
    except Exception:
        pass
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return "unknown"


def encode_texts(model: Any, texts: list[str], batch_size: int, embedding_dimension: int) -> np.ndarray:
    safe_texts = [text if text.strip() else " " for text in texts]
    embeddings = model.encode(
        safe_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.shape[1] > embedding_dimension:
        embeddings = embeddings[:, :embedding_dimension]
    if embeddings.shape[1] != embedding_dimension:
        raise ValueError(f"expected embedding dim {embedding_dimension}, got {embeddings.shape[1]}")
    return embeddings


class TweetEmbeddingWriter:
    def __init__(self, path: Path, embedding_dimension: int, storage_dtype: str) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if storage_dtype not in {"float16", "float32"}:
            raise ValueError(f"unsupported embeddingStorageDtype: {storage_dtype}")
        self.path = path
        self.embedding_dimension = embedding_dimension
        self.storage_dtype = storage_dtype
        self.pa = pa
        self.value_type = pa.float16() if storage_dtype == "float16" else pa.float32()
        self.numpy_dtype = np.float16 if storage_dtype == "float16" else np.float32
        self.schema = pa.schema(
            [
                ("user_id", pa.string()),
                ("tweet_index", pa.int32()),
                ("tweet_text", pa.string()),
                ("gpt5_relevance", pa.int16()),
                ("embedding", pa.list_(self.value_type, list_size=embedding_dimension)),
            ]
        )
        self.writer = pq.ParquetWriter(path, self.schema, compression="zstd")

    def write_batch(
        self,
        record: UserRecord,
        start_index: int,
        texts: list[str],
        gpt5_relevance: list[int | None],
        embeddings: np.ndarray,
    ) -> None:
        pa = self.pa
        vector_values = pa.array(
            embeddings.astype(self.numpy_dtype, copy=False).reshape(-1),
            type=self.value_type,
        )
        embedding_array = pa.FixedSizeListArray.from_arrays(vector_values, self.embedding_dimension)
        table = pa.Table.from_arrays(
            [
                pa.array([record.user_id] * len(texts), type=pa.string()),
                pa.array(list(range(start_index, start_index + len(texts))), type=pa.int32()),
                pa.array(texts, type=pa.string()),
                pa.array(gpt5_relevance, type=pa.int16()),
                embedding_array,
            ],
            schema=self.schema,
        )
        self.writer.write_table(table)

    def close(self) -> None:
        self.writer.close()


def empty_embedding(embedding_dimension: int) -> np.ndarray:
    return np.zeros((embedding_dimension,), dtype=np.float32)


def process_user(
    model: Any,
    record: UserRecord,
    batch_size: int,
    embedding_dimension: int,
    tweet_writer: TweetEmbeddingWriter | None,
) -> dict[str, Any]:
    mean_sum = np.zeros((embedding_dimension,), dtype=np.float64)
    threshold_sums = {threshold: np.zeros((embedding_dimension,), dtype=np.float64) for threshold in THRESHOLDS}
    threshold_counts = {threshold: 0 for threshold in THRESHOLDS}
    total_count = 0

    for start in range(0, len(record.texts), batch_size):
        end = min(start + batch_size, len(record.texts))
        embeddings = encode_texts(model, record.texts[start:end], batch_size, embedding_dimension)
        if tweet_writer is not None:
            tweet_writer.write_batch(
                record,
                start,
                record.texts[start:end],
                record.relevance_raw[start:end],
                embeddings,
            )
        for offset, embedding in enumerate(embeddings):
            tweet_index = start + offset
            relevance = int(record.relevance_score[tweet_index])
            mean_sum += embedding
            total_count += 1
            for threshold in THRESHOLDS:
                if relevance >= threshold:
                    threshold_sums[threshold] += embedding
                    threshold_counts[threshold] += 1

    if total_count == 0:
        mean_embedding = empty_embedding(embedding_dimension)
    else:
        mean_embedding = (mean_sum / total_count).astype(np.float32)

    threshold_embeddings: dict[int, np.ndarray] = {}
    for threshold in THRESHOLDS:
        count = threshold_counts[threshold]
        if count == 0:
            threshold_embeddings[threshold] = empty_embedding(embedding_dimension)
        else:
            threshold_embeddings[threshold] = (threshold_sums[threshold] / count).astype(np.float32)

    return {
        "tweet_count": total_count,
        "mean": mean_embedding,
        "threshold_embeddings": threshold_embeddings,
        "threshold_counts": threshold_counts,
    }


def save_shard(
    path: Path,
    split: str,
    records: list[UserRecord],
    processed: list[dict[str, Any]],
    include_relevance_pools: bool = True,
    redact_test_labels: bool = False,
) -> None:
    artifact_labels = np.asarray(
        [record.label_code if split == "train" else -1 for record in records],
        dtype=np.int16,
    )
    payload: dict[str, np.ndarray] = {
        "user_ids": np.asarray([record.user_id for record in records], dtype=object),
        "labels": artifact_labels,
        "tweet_counts": np.asarray([item["tweet_count"] for item in processed], dtype=np.int32),
        "mean_embeddings": np.stack([item["mean"] for item in processed]).astype(np.float32),
    }
    if not (redact_test_labels and split == "test"):
        payload["true_labels"] = np.asarray([record.label_code for record in records], dtype=np.int16)
    if include_relevance_pools:
        for threshold in THRESHOLDS:
            payload[f"rel{threshold}_embeddings"] = np.stack(
                [item["threshold_embeddings"][threshold] for item in processed]
            ).astype(np.float32)
            payload[f"rel{threshold}_counts"] = np.asarray(
                [item["threshold_counts"][threshold] for item in processed], dtype=np.int32
            )
    np.savez(path, **payload)


def reduce_split(
    output_dir: Path,
    split: str,
    shard_paths: list[Path],
    include_relevance_pools: bool = True,
) -> dict[str, Any]:
    shards = [np.load(path, allow_pickle=True) for path in shard_paths]
    user_ids = np.concatenate([shard["user_ids"] for shard in shards])
    labels = np.concatenate([shard["labels"] for shard in shards])
    tweet_counts = np.concatenate([shard["tweet_counts"] for shard in shards])

    pooled_specs = [("mean", "mean_embeddings", "tweet_counts")]
    if include_relevance_pools:
        pooled_specs.extend(
            (f"rel{threshold}", f"rel{threshold}_embeddings", f"rel{threshold}_counts")
            for threshold in THRESHOLDS
        )
    for suffix, embedding_key, count_key in pooled_specs:
        embeddings = np.concatenate([shard[embedding_key] for shard in shards]).astype(np.float32)
        counts = np.concatenate([shard[count_key] for shard in shards]).astype(np.int32)
        np.savez(
            output_dir / "pooled" / f"{split}_user_{suffix}.npz",
            user_ids=user_ids,
            labels=labels,
            embeddings=embeddings,
            counts=counts,
        )

    for shard in shards:
        shard.close()
    return {
        "split": split,
        "users": int(user_ids.shape[0]),
        "totalTweets": int(tweet_counts.sum()),
        "minTweets": int(tweet_counts.min()) if tweet_counts.size else 0,
        "maxTweets": int(tweet_counts.max()) if tweet_counts.size else 0,
    }


def generate_embeddings(
    output_dir: Path,
    records_by_split: dict[str, list[UserRecord]],
    config: dict[str, Any],
    splits: list[str],
    batch_size: int,
    shard_users: int,
    device_request: str,
    force: bool,
) -> dict[str, Any]:
    model_id = str(config["embeddingModelId"])
    embedding_dimension = int(config["embeddingDimension"])
    storage_dtype = str(config.get("embeddingStorageDtype", "float16"))
    include_relevance_pools = bool(config.get("includeRelevancePools", True))
    redact_test_labels = bool(config.get("redactTestLabels", False))
    min_free_gib = float(config.get("minFreeGiB", 0))
    device = resolve_device(device_request)
    model = load_embedding_model(model_id, device)
    revision = model_revision(model)

    split_summaries: dict[str, Any] = {}
    for split in splits:
        records = records_by_split[split]
        shard_dir = output_dir / "shards" / split
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_paths: list[Path] = []
        for start in range(0, len(records), shard_users):
            end = min(start + shard_users, len(records))
            shard_path = shard_dir / f"{split}_shard_{start:06d}_{end:06d}.npz"
            tweet_path = output_dir / "tweet_embeddings" / split / f"part-{start:06d}-{end:06d}.parquet"
            shard_paths.append(shard_path)
            if shard_path.exists() and tweet_path.exists() and not force:
                continue
            free_gib = shutil.disk_usage(output_dir).free / (1024**3)
            if min_free_gib > 0 and free_gib < min_free_gib:
                raise RuntimeError(
                    f"refusing to start {split} shard {start}:{end}: "
                    f"{free_gib:.2f} GiB free is below minFreeGiB={min_free_gib:.2f}"
                )
            if force:
                for path in [shard_path, tweet_path]:
                    if path.exists():
                        path.unlink()
            writer = TweetEmbeddingWriter(tweet_path, embedding_dimension, storage_dtype)
            try:
                processed = [
                    process_user(model, record, batch_size, embedding_dimension, writer)
                    for record in records[start:end]
                ]
            finally:
                writer.close()
            save_shard(
                shard_path,
                split,
                records[start:end],
                processed,
                include_relevance_pools=include_relevance_pools,
                redact_test_labels=redact_test_labels,
            )
        split_summaries[split] = reduce_split(
            output_dir,
            split,
            shard_paths,
            include_relevance_pools=include_relevance_pools,
        )
    return {
        "device": device,
        "modelId": model_id,
        "modelRevision": revision,
        "embeddingDimension": embedding_dimension,
        "embeddingStorageDtype": storage_dtype,
        "batchSize": batch_size,
        "shardUsers": shard_users,
        "includeRelevancePools": include_relevance_pools,
        "redactTestLabels": redact_test_labels,
        "minFreeGiB": min_free_gib,
        "tweetLevelDataset": {
            split: {
                "path": str(output_dir / "tweet_embeddings" / split),
                "columns": ["user_id", "tweet_index", "tweet_text", "gpt5_relevance", "embedding"],
            }
            for split in splits
        },
        "splits": split_summaries,
    }


def main() -> None:
    args = parse_args()
    repo_dir = Path.cwd()
    config_path = resolve_path(args.config, repo_dir)
    config = load_config(config_path)
    dataset_dir = resolve_path(args.dataset_dir or config["datasetDir"], repo_dir)
    relevance_dir = resolve_path(args.relevance_dir or config["relevanceDir"], repo_dir)
    output_dir = resolve_path(args.output_dir or config["outputDir"], repo_dir)
    ensure_dirs(output_dir)

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    invalid_splits = [split for split in splits if split not in {"train", "test"}]
    if invalid_splits:
        raise ValueError(f"unsupported splits: {invalid_splits}")

    validation = validate_inputs(dataset_dir, relevance_dir)
    selected = selected_records(validation.records_by_split, args.smoke_users)
    manifest_hash = write_manifest_files(output_dir, selected, int(config["seed"]), int(config.get("folds", 5)))
    validation_payload = validation_report_payload(
        config,
        dataset_dir,
        relevance_dir,
        output_dir,
        validation,
        selected,
        manifest_hash,
        sys.argv,
    )
    write_json(output_dir / "reports" / "raw_validation_report.json", validation_payload)

    if args.mode == "validate":
        print(json.dumps({"status": "ok", "report": str(output_dir / "reports" / "raw_validation_report.json")}))
        return

    batch_size = int(args.batch_size or config.get("batchSize", 16))
    shard_users = int(args.shard_users or config.get("shardUsers", 64))
    generation = generate_embeddings(
        output_dir=output_dir,
        records_by_split={split: selected[split] for split in splits},
        config=config,
        splits=splits,
        batch_size=batch_size,
        shard_users=shard_users,
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
    write_json(output_dir / "reports" / "embedding_generation_manifest.json", manifest_payload)
    print(json.dumps({"status": "ok", "manifest": str(output_dir / "reports" / "embedding_generation_manifest.json")}))


if __name__ == "__main__":
    main()
