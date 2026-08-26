#!/usr/bin/env python3
"""Strict-blind transfer of the fixed SetembroBR depression champion to v7."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import importlib.metadata
import itertools
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from setembrobr_v7_qwen_logreg import (
    SHARD_RE,
    ArtifactPaths,
    archive_records,
    decompress_verified,
    fixed_size_embeddings,
    load_source_user_index,
    match_ordered_posts,
    read_csv,
    sha256_file,
    text_alignment_hash,
    write_csv,
    write_json,
)


LABEL_NAMES = {0: "control", 1: "diagnosed"}
EMBEDDING_BLOCKS = {"mean_pca": "mean", "rel3_pca": "rel3", "rel6_pca": "rel6", "rel7_pca": "rel7"}
FEATURE_NAMES = [
    "evidence_markers",
    "stylistic",
    "relevance_counts",
    "temporal_markers",
    "mean",
    "rel3",
    "rel6",
    "rel7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare-train",
            "train-tabular-oof",
            "train-sequence-oof",
            "train-stack-oof",
            "audit-oof",
            "lock",
            "prepare-test",
            "score-test",
            "audit-test",
            "evaluate",
            "audit-feature-support",
            "run-oof",
            "run-test",
        ],
    )
    parser.add_argument("--config", type=Path, default=Path("ensemble-config.json"))
    parser.add_argument("--baseline-config", type=Path, default=Path("config.json"))
    parser.add_argument("--baseline-output", type=Path, default=Path(".work/output"))
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(".work/ensemble"))
    parser.add_argument("--temporary-dir", type=Path)
    parser.add_argument("--feature-helper", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path, baseline_path: Path, helper_path: Path) -> dict[str, Any]:
    config = load_json(path)
    checks = [
        (baseline_path, config["sourceConfigSha256"], "baseline config"),
        (helper_path, config["featureHelperSha256"], "feature helper"),
    ]
    for input_path, expected, description in checks:
        observed = sha256_file(input_path)
        if observed != expected:
            raise RuntimeError(f"{description} hash mismatch: expected {expected}, got {observed}")
    config["_configPath"] = str(path.resolve())
    config["_configSha256"] = sha256_file(path)
    config["_baselineConfig"] = load_json(baseline_path)
    return config


def load_feature_helper(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("setembrobr_v7_feature_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load feature helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected(config: dict[str, Any], key: str) -> int:
    return int(config["expected"][key])


def all_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = config["candidateModels"]
    return [*models["tabular"], *models["sequence"], *models["stacking"]]


def manifest_path(baseline: ArtifactPaths, split: str) -> Path:
    return baseline.manifests / ("train_manifest.csv" if split == "train" else "test_inference.csv")


def score_path(output: Path, model_id: str, test: bool = False) -> Path:
    prefix = "test_score" if test else "train_oof"
    return output / "scores" / f"{prefix}_{model_id}.csv"


def model_manifest(output: Path, candidate: dict[str, Any], payload: dict[str, Any]) -> None:
    write_json(
        output / "model-manifests" / f"{candidate['modelId']}.json",
        {
            "modelId": candidate["modelId"],
            "family": candidate["family"],
            "createdAt": "1970-01-01T00:00:00.000Z",
            **payload,
        },
    )


def require_lock(output: Path) -> dict[str, Any]:
    path = output / "ensemble" / "ensemble-lock.json"
    provenance = output / "reports" / "lock-provenance.json"
    if not path.is_file() or not provenance.is_file():
        raise RuntimeError("post-lock stage requires an immutable OOF lock")
    record = load_json(provenance)
    if record["lockSha256"] != sha256_file(path):
        raise RuntimeError("ensemble lock changed after OOF selection")
    return load_json(path)


def validate_baseline(config: dict[str, Any], baseline: ArtifactPaths) -> None:
    prepared_path = baseline.manifests / "prepared-manifest.json"
    if not prepared_path.is_file():
        raise RuntimeError("completed baseline preparation is required")
    prepared = load_json(prepared_path)
    if prepared["sourceSha256"] != config["sourcePickleSha256"]:
        raise RuntimeError("baseline source pickle hash differs from the ensemble pin")
    train_rows = read_csv(manifest_path(baseline, "train"))
    test_rows = read_csv(manifest_path(baseline, "test"))
    if len(train_rows) != expected(config, "trainUsers") or len(test_rows) != expected(config, "testUsers"):
        raise RuntimeError("baseline split sizes differ from the ensemble contract")
    if set(train_rows[0]) != {"user_id", "label", "fold"} or set(test_rows[0]) != {"user_id"}:
        raise RuntimeError("baseline manifests have unexpected schemas")


def pool_vector(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask]
    if len(selected) == 0:
        return np.zeros(values.shape[1], dtype=np.float32)
    return (selected.astype(np.float64).sum(axis=0) / len(selected)).astype(np.float32)


def shard_cache_valid(
    path: Path,
    user_ids: Sequence[str],
    archive_hash: str,
    alignment_hash: str,
    config_hash: str,
    dimension: int,
    top_n: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            return bool(
                data["user_ids"].astype(str).tolist() == list(user_ids)
                and str(data["source_archive_sha256"]) == archive_hash
                and str(data["target_alignment_sha256"]) == alignment_hash
                and str(data["config_sha256"]) == config_hash
                and data["mean"].shape == (len(user_ids), dimension)
                and data["sequences"].shape == (len(user_ids), top_n, dimension)
            )
    except Exception:
        return False


def prepare_one_shard(
    source: Path,
    record: dict[str, Any],
    user_ids: list[str],
    targets: dict[str, list[str]],
    destination: Path,
    config: dict[str, Any],
    temporary_dir: Path,
    helper: Any,
) -> dict[str, Any]:
    dimension = expected(config, "embeddingDimension")
    top_n = int(config["sequenceExport"]["topN"])
    alignment_hash = text_alignment_hash(user_ids, targets)
    if shard_cache_valid(
        destination,
        user_ids,
        record["archiveSha256"],
        alignment_hash,
        config["_configSha256"],
        dimension,
        top_n,
    ):
        with np.load(destination, allow_pickle=True) as data:
            return {"users": len(user_ids), "posts": int(data["counts"].sum()), "resumed": True}

    temporary_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", dir=temporary_dir) as reconstructed:
        archive_hash, original_hash, _ = decompress_verified(source, record, reconstructed)
        reconstructed.flush()
        table = pq.read_table(
            reconstructed.name,
            columns=["user_id", "tweet_index", "tweet_text", "gpt5_relevance", "embedding"],
            filters=[("user_id", "in", user_ids)],
        )
    source_users = np.asarray([str(value) for value in table.column("user_id").to_pylist()], dtype=object)
    source_indexes = np.asarray(table.column("tweet_index").to_numpy(), dtype=np.int64)
    source_texts = np.asarray([str(value) for value in table.column("tweet_text").to_pylist()], dtype=object)
    source_relevance = table.column("gpt5_relevance").to_pylist()
    source_embeddings = fixed_size_embeddings(table, dimension)

    result: dict[str, list[Any]] = {name: [] for name in FEATURE_NAMES}
    sequences: list[np.ndarray] = []
    sequence_relevance: list[np.ndarray] = []
    lengths: list[int] = []
    counts: list[int] = []
    for user_id in user_ids:
        positions = np.flatnonzero(source_users == user_id)
        if len(positions) == 0:
            raise RuntimeError(f"{source.name} is missing source user {user_id}")
        positions = positions[np.argsort(source_indexes[positions], kind="stable")]
        if len(np.unique(source_indexes[positions])) != len(positions):
            raise RuntimeError(f"{source.name}:{user_id} has duplicate tweet indexes")
        relative = match_ordered_posts(source_texts[positions].tolist(), targets[user_id])
        selected = positions[relative]
        texts = source_texts[selected].tolist()
        relevance = np.asarray([helper.relevance_value(source_relevance[index]) for index in selected], dtype=np.int16)
        embeddings = source_embeddings[selected]
        if len(texts) != len(targets[user_id]):
            raise RuntimeError(f"{user_id}: retained post count changed during exact matching")

        agg = helper.UserAgg()
        for tweet_index, (text, rel) in enumerate(zip(texts, relevance)):
            agg.add(text, int(rel), tweet_index)
        for tweet_index, (text, rel) in enumerate(zip(texts, relevance)):
            agg.add_temporal(text, int(rel), tweet_index)
        marker = agg.marker(user_id)
        result["evidence_markers"].append(helper.evidence_feature_row(marker))
        result["stylistic"].append(agg.stylistic())
        result["relevance_counts"].append(agg.relevance_counts())
        result["temporal_markers"].append(agg.temporal_markers())
        result["mean"].append(pool_vector(embeddings, np.ones(len(embeddings), dtype=bool)))
        result["rel3"].append(pool_vector(embeddings, relevance >= 3))
        result["rel6"].append(pool_vector(embeddings, relevance >= 6))
        result["rel7"].append(pool_vector(embeddings, relevance >= 7))

        start = max(len(embeddings) - top_n, 0)
        chosen = embeddings[start:]
        chosen_relevance = relevance[start:]
        sequence = np.zeros((top_n, dimension), dtype=np.float16)
        rel_sequence = np.zeros(top_n, dtype=np.int16)
        sequence[: len(chosen)] = chosen
        rel_sequence[: len(chosen)] = chosen_relevance
        sequences.append(sequence)
        sequence_relevance.append(rel_sequence)
        lengths.append(len(chosen))
        counts.append(len(embeddings))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
    np.savez(
        temporary,
        user_ids=np.asarray(user_ids, dtype=object),
        **{name: np.asarray(values, dtype=np.float32) for name, values in result.items()},
        sequences=np.stack(sequences).astype(np.float16),
        sequence_relevances=np.stack(sequence_relevance).astype(np.int16),
        lengths=np.asarray(lengths, dtype=np.int32),
        counts=np.asarray(counts, dtype=np.int32),
        source_archive_sha256=np.asarray(archive_hash),
        source_original_sha256=np.asarray(original_hash),
        target_alignment_sha256=np.asarray(alignment_hash),
        config_sha256=np.asarray(config["_configSha256"]),
    )
    os.replace(temporary, destination)
    return {"users": len(user_ids), "posts": int(sum(counts)), "resumed": False}


def combine_split(config: dict[str, Any], output: Path, split: str, rows: list[dict[str, str]], shards: list[Path]) -> None:
    count = len(rows)
    dimension = expected(config, "embeddingDimension")
    top_n = int(config["sequenceExport"]["topN"])
    positions = {row["user_id"]: index for index, row in enumerate(rows)}
    arrays = {
        "evidence_markers": np.zeros((count, 12), dtype=np.float32),
        "stylistic": np.zeros((count, 7), dtype=np.float32),
        "relevance_counts": np.zeros((count, 8), dtype=np.float32),
        "temporal_markers": np.zeros((count, 77), dtype=np.float32),
        **{name: np.zeros((count, dimension), dtype=np.float32) for name in ["mean", "rel3", "rel6", "rel7"]},
    }
    sequence_dir = output / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    sequence_path = sequence_dir / f"{split}_sequences.npy"
    relevance_path = sequence_dir / f"{split}_relevances.npy"
    lengths_path = sequence_dir / f"{split}_lengths.npy"
    sequence = np.lib.format.open_memmap(sequence_path, mode="w+", dtype=np.float16, shape=(count, top_n, dimension))
    relevance = np.lib.format.open_memmap(relevance_path, mode="w+", dtype=np.int16, shape=(count, top_n))
    lengths = np.lib.format.open_memmap(lengths_path, mode="w+", dtype=np.int32, shape=(count,))
    seen: set[str] = set()
    posts = 0
    for shard in shards:
        with np.load(shard, allow_pickle=True) as data:
            for source_index, user_id in enumerate(data["user_ids"].astype(str)):
                if user_id not in positions or user_id in seen:
                    raise RuntimeError(f"invalid or duplicate combined user {user_id}")
                target_index = positions[user_id]
                for name in arrays:
                    arrays[name][target_index] = data[name][source_index]
                sequence[target_index] = data["sequences"][source_index]
                relevance[target_index] = data["sequence_relevances"][source_index]
                lengths[target_index] = data["lengths"][source_index]
                posts += int(data["counts"][source_index])
                seen.add(user_id)
    if seen != set(positions):
        raise RuntimeError(f"{split} combined artifacts are missing {len(set(positions) - seen)} users")
    sequence.flush()
    relevance.flush()
    lengths.flush()
    del sequence, relevance, lengths
    feature_dir = output / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.savez(feature_dir / f"{split}_raw_features.npz", user_ids=np.asarray([row["user_id"] for row in rows]), **arrays)
    np.save(sequence_dir / f"{split}_user_ids.npy", np.asarray([row["user_id"] for row in rows]))
    write_json(
        output / "reports" / f"{split}-feature-preparation.json",
        {
            "ok": True,
            "split": split,
            "users": count,
            "matchedPosts": posts,
            "featureSha256": sha256_file(feature_dir / f"{split}_raw_features.npz"),
            "sequenceHashes": {
                "embeddings": sha256_file(sequence_path),
                "relevances": sha256_file(relevance_path),
                "lengths": sha256_file(lengths_path),
            },
            "labelFree": split == "test",
            "sourcePickleSha256": config["sourcePickleSha256"],
            "configSha256": config["_configSha256"],
        },
    )


def audit_feature_support(config: dict[str, Any], output: Path) -> None:
    reports = {}
    for split in ["train", "test"]:
        path = output / "features" / f"{split}_raw_features.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            pool_rows = {
                name: int(np.any(data[name] != 0, axis=1).sum())
                for name in ["mean", "rel3", "rel6", "rel7"]
            }
            maximum_relevance = float(data["evidence_markers"][:, 1].max())
            users = int(len(data["user_ids"]))
        report = {
            "ok": True,
            "split": split,
            "users": users,
            "maximumRetainedPostRelevance": maximum_relevance,
            "nonzeroPoolRows": pool_rows,
            "collapsedPools": [name for name, count in pool_rows.items() if count == 0],
            "diagnosisLabelsRead": False,
            "configSha256": config["_configSha256"],
        }
        write_json(output / "reports" / f"{split}-feature-support-audit.json", report)
        reports[split] = report
    if not reports:
        raise RuntimeError("no prepared feature artifacts are available to audit")
    print(json.dumps({"status": "ok", "splits": reports}), flush=True)


def prepare_split(
    config: dict[str, Any],
    baseline: ArtifactPaths,
    archive_root: Path,
    output: Path,
    temporary_dir: Path,
    split: str,
    helper: Any,
) -> None:
    validate_baseline(config, baseline)
    if split == "test":
        require_lock(output)
    sanitized_path = baseline.sanitized / f"{split}.pkl"
    frame = pd.read_pickle(sanitized_path)
    if list(frame.columns) != ["User_ID", "TextLists", "Split"] or "Diagnosed_YN" in frame:
        raise RuntimeError(f"sanitized {split} input is not label-free")
    rows = read_csv(manifest_path(baseline, split))
    targets = {str(row.User_ID): list(row.TextLists) for row in frame.itertuples(index=False)}
    if set(targets) != {row["user_id"] for row in rows}:
        raise RuntimeError(f"sanitized {split} users differ from immutable manifest")

    baseline_config = config["_baselineConfig"]
    records = archive_records(archive_root, baseline_config)
    pooled_relative = baseline_config["embeddings"]["pooledUserIndex"]
    source_user_ids = load_source_user_index(archive_root / pooled_relative, records[pooled_relative])
    source_set = set(source_user_ids)
    if not set(targets).issubset(source_set):
        raise RuntimeError(f"archive is missing {len(set(targets) - source_set)} v7 users")
    shard_files: dict[tuple[int, int], Path] = {}
    for path in archive_root.glob(baseline_config["embeddings"]["shardGlob"]):
        match = SHARD_RE.match(path.name)
        if match:
            shard_files[(int(match.group(1)), int(match.group(2)))] = path

    shard_outputs: list[Path] = []
    total_posts = 0
    for (start, end), source in sorted(shard_files.items()):
        relevant = [user_id for user_id in source_user_ids[start:end] if user_id in targets]
        if not relevant:
            continue
        relative = str(source.relative_to(archive_root))
        if relative not in records:
            raise RuntimeError(f"archive state is missing {relative}")
        destination = output / "pool-shards" / split / f"{source.stem}.ensemble.npz"
        status = prepare_one_shard(
            source, records[relative], relevant, targets, destination, config, temporary_dir, helper
        )
        shard_outputs.append(destination)
        total_posts += int(status["posts"])
        print(json.dumps({"event": "prepared-shard", "split": split, "source": source.name, **status}), flush=True)
    combine_split(config, output, split, rows, shard_outputs)
    audit_feature_support(config, output)
    report = load_json(output / "reports" / f"{split}-feature-preparation.json")
    if report["matchedPosts"] != total_posts:
        raise RuntimeError(f"{split} shard and combined post counts disagree")
    print(json.dumps({"status": "ok", "split": split, "users": len(rows), "posts": total_posts}), flush=True)


def load_features(output: Path, split: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    with np.load(output / "features" / f"{split}_raw_features.npz", allow_pickle=True) as data:
        user_ids = data["user_ids"].astype(str)
        blocks = {name: data[name].astype(np.float32) for name in FEATURE_NAMES}
    expected_ids = [row["user_id"] for row in rows]
    if user_ids.tolist() != expected_ids:
        raise RuntimeError(f"{split} feature order differs from manifest")
    result: dict[str, Any] = {"user_ids": user_ids, "blocks": blocks}
    if split == "train":
        result["labels"] = np.asarray([1 if row["label"] == "yes" else 0 for row in rows], dtype=np.int64)
        result["folds"] = np.asarray([int(row["fold"]) for row in rows], dtype=np.int32)
    return result


def subset(split: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    result = {key: split[key][mask] for key in ["user_ids", "labels", "folds"] if key in split}
    result["blocks"] = {name: values[mask] for name, values in split["blocks"].items()}
    return result


def fit_transform(train: dict[str, Any], candidate: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    fitted_blocks = []
    pieces = []
    components = int(candidate.get("pcaComponents", 80))
    for offset, block_name in enumerate(candidate["featureBlocks"]):
        source = EMBEDDING_BLOCKS.get(block_name, block_name)
        values = train["blocks"][source]
        if block_name in EMBEDDING_BLOCKS:
            count = min(components, len(train["labels"]) - 1, values.shape[1])
            pca = PCA(n_components=count, random_state=int(candidate["seed"]) + offset * 101)
            pieces.append(pca.fit_transform(values))
        else:
            pca = None
            pieces.append(values)
        fitted_blocks.append({"name": block_name, "source": source, "pca": pca})
    scaler = StandardScaler()
    transformed = scaler.fit_transform(np.nan_to_num(np.hstack(pieces), copy=False)).astype(np.float32)
    return {"blocks": fitted_blocks, "scaler": scaler}, transformed


def apply_transform(split: dict[str, Any], transform: dict[str, Any]) -> np.ndarray:
    pieces = []
    for block in transform["blocks"]:
        values = split["blocks"][block["source"]]
        pieces.append(block["pca"].transform(values) if block["pca"] is not None else values)
    return transform["scaler"].transform(np.nan_to_num(np.hstack(pieces), copy=False)).astype(np.float32)


def balanced_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = len(labels) / (2 * np.maximum(counts, 1.0))
    return weights[labels].astype(np.float32)


def fit_focal(features: np.ndarray, labels: np.ndarray, gamma: float, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels.astype(np.float32))
    model = nn.Linear(features.shape[1], 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    positives = max(float((labels == 1).sum()), 1.0)
    negatives = max(float((labels == 0).sum()), 1.0)
    positive_weight = len(labels) / (2 * positives)
    negative_weight = len(labels) / (2 * negatives)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(250):
        optimizer.zero_grad()
        logits = model(x).squeeze(1).clamp(-30, 30)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        probability = torch.exp(-bce)
        weights = torch.where(
            y > 0.5,
            torch.full_like(y, positive_weight),
            torch.full_like(y, negative_weight),
        )
        loss = (((1 - probability) ** gamma) * bce * weights).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("focal linear model produced non-finite loss")
        loss.backward()
        optimizer.step()
    return {
        "kind": "focal_linear",
        "weight": model.weight.detach().numpy().astype(np.float32),
        "bias": model.bias.detach().numpy().astype(np.float32),
    }


def fit_estimator(candidate: dict[str, Any], features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    family = candidate["family"]
    seed = int(candidate["seed"])
    if family in {"logreg", "hierarchical_logreg"}:
        estimator = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs"
        )
        estimator.fit(features, labels)
    elif family == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=(int(candidate["hiddenSize"]),),
            alpha=float(candidate["alpha"]),
            random_state=seed,
            max_iter=500,
            early_stopping=False,
        )
        estimator.fit(features, labels)
    elif family == "xgboost":
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=int(candidate["nEstimators"]),
            max_depth=int(candidate["maxDepth"]),
            learning_rate=float(candidate["learningRate"]),
            subsample=float(candidate["subsample"]),
            colsample_bytree=float(candidate["colsampleBytree"]),
            reg_lambda=float(candidate["regLambda"]),
            min_child_weight=float(candidate["minChildWeight"]),
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        estimator.fit(features, labels, sample_weight=balanced_weights(labels))
    elif family == "focal_linear":
        return fit_focal(features, labels, float(candidate["gamma"]), seed)
    else:
        raise RuntimeError(f"unsupported tabular family {family}")
    return {"kind": "sklearn", "estimator": estimator}


def predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    if model["kind"] == "focal_linear":
        logits = features @ model["weight"].reshape(-1) + float(model["bias"][0])
        return np.clip(1 / (1 + np.exp(-np.clip(logits, -30, 30))), 0, 1)
    estimator = model["estimator"]
    classes = list(estimator.classes_)
    return estimator.predict_proba(features)[:, classes.index(1)]


def write_scores(path: Path, rows: list[dict[str, Any]], labels: bool) -> str:
    fields = ["user_id", "label", "fold", "score", "model_id"] if labels else ["user_id", "score", "model_id"]
    return write_csv(path, fields, sorted(rows, key=lambda row: row["user_id"]))


def train_tabular_oof(config: dict[str, Any], baseline: ArtifactPaths, output: Path) -> None:
    rows = read_csv(manifest_path(baseline, "train"))
    train = load_features(output, "train", rows)
    required = set(config["ensemble"]["requiredModelIds"])
    for candidate in config["candidateModels"]["tabular"]:
        model_id = candidate["modelId"]
        output_rows = []
        checkpoint_hashes = {}
        for fold in sorted(set(train["folds"].tolist())):
            fold_candidate = {**candidate, "seed": int(candidate["seed"]) + int(fold)}
            training = subset(train, train["folds"] != fold)
            validation = subset(train, train["folds"] == fold)
            checkpoint = output / "checkpoints" / "tabular" / model_id / f"fold-{fold}.joblib"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint.is_file():
                payload = joblib.load(checkpoint)
                transform, model = payload["transform"], payload["model"]
            else:
                transform, x_train = fit_transform(training, fold_candidate)
                model = fit_estimator(fold_candidate, x_train, training["labels"])
                joblib.dump({"candidate": fold_candidate, "transform": transform, "model": model}, checkpoint)
            probabilities = predict(model, apply_transform(validation, transform))
            checkpoint_hashes[str(fold)] = sha256_file(checkpoint)
            output_rows.extend(
                {
                    "user_id": user_id,
                    "label": LABEL_NAMES[int(label)],
                    "fold": int(fold),
                    "score": f"{float(probability):.10f}",
                    "model_id": model_id,
                }
                for user_id, label, probability in zip(
                    validation["user_ids"], validation["labels"], probabilities
                )
            )
            print(json.dumps({"event": "tabular-fold", "modelId": model_id, "fold": int(fold)}), flush=True)
        score_hash = write_scores(score_path(output, model_id), output_rows, True)
        model_manifest(
            output,
            candidate,
            {
                "candidate": model_id in required,
                "supportOnly": model_id not in required,
                "sourceChampionLockSha256": config["sourceChampion"]["lockSha256"],
                "trainManifestSha256": sha256_file(manifest_path(baseline, "train")),
                "scoreSha256": score_hash,
                "checkpointHashes": checkpoint_hashes,
                "testArtifactsReadDuringOof": False,
            },
        )
    del train
    gc.collect()


class Cnn1D(nn.Module):
    def __init__(self, input_dim: int, num_filters: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(input_dim, num_filters, kernel, padding=kernel // 2) for kernel in (3, 5, 7)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * 3, 2)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            hidden = F.relu(conv(values))
            hidden = hidden.masked_fill(mask.unsqueeze(1).expand_as(hidden) < 0.5, float("-inf"))
            pooled.append(hidden.max(dim=2).values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


def load_sequences(config: dict[str, Any], output: Path, split: str) -> dict[str, np.ndarray]:
    directory = output / "sequences"
    relevance = np.load(directory / f"{split}_relevances.npy", mmap_mode="r")
    divisor = np.float16(config["sequenceExport"]["relevanceChannelDivisor"])
    return {
        "user_ids": np.load(directory / f"{split}_user_ids.npy"),
        "sequences": np.load(directory / f"{split}_sequences.npy", mmap_mode="r"),
        "relevances": np.clip(relevance.astype(np.float16) / divisor, 0, 1),
        "lengths": np.load(directory / f"{split}_lengths.npy", mmap_mode="r"),
    }


def sequence_masks(lengths: np.ndarray, size: int) -> np.ndarray:
    return (np.arange(size)[None, :] < lengths[:, None]).astype(np.float32)


def indexed_sequence(split: dict[str, np.ndarray], indexes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indexes_numpy = indexes.numpy()
    values = torch.from_numpy(np.asarray(split["sequences"][indexes_numpy]))
    relevance = torch.from_numpy(np.asarray(split["relevances"][indexes_numpy])).unsqueeze(2)
    values = torch.cat([values, relevance], dim=2)
    mask = torch.from_numpy(sequence_masks(np.asarray(split["lengths"][indexes_numpy]), values.shape[1]))
    return values, mask


def infer_sequence(
    model: nn.Module,
    device: torch.device,
    split: dict[str, np.ndarray],
    batch_size: int,
    indexes: np.ndarray | None = None,
) -> np.ndarray:
    selected = np.arange(len(split["user_ids"]), dtype=np.int64) if indexes is None else indexes
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch_indexes = torch.from_numpy(selected[start : start + batch_size])
            values, mask = indexed_sequence(split, batch_indexes)
            logits = model(values.to(device).float(), mask.to(device))
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.clip(np.nan_to_num(np.concatenate(probabilities), nan=0.5, posinf=1, neginf=0), 0, 1)


def build_sequence_model(candidate: dict[str, Any], input_dim: int) -> Cnn1D:
    return Cnn1D(input_dim, int(candidate["numFilters"]), float(candidate["dropout"]))


def train_sequence_fold(
    training: dict[str, np.ndarray], indexes: np.ndarray, candidate: dict[str, Any], seed: int
) -> tuple[nn.Module, torch.device]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sequence_model(candidate, training["sequences"].shape[2] + 1).to(device)
    labels = torch.from_numpy(training["labels"])
    index_tensor = torch.from_numpy(indexes)
    loader = DataLoader(
        TensorDataset(index_tensor),
        batch_size=int(candidate["batchSize"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    selected_labels = labels[index_tensor]
    positives = max(int((selected_labels == 1).sum()), 1)
    negatives = max(int((selected_labels == 0).sum()), 1)
    weights = torch.tensor(
        [len(selected_labels) / (2 * negatives), len(selected_labels) / (2 * positives)],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    for _ in range(int(candidate["epochs"])):
        model.train()
        for (batch_indexes,) in loader:
            values, mask = indexed_sequence(training, batch_indexes)
            optimizer.zero_grad()
            loss = loss_function(
                model(values.to(device).float(), mask.to(device)), labels[batch_indexes].to(device)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("CNN produced non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
    return model, device


def sequence_checkpoint(model: nn.Module, candidate: dict[str, Any], input_dim: int) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "inputDim": input_dim,
        "stateDict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def train_sequence_oof(config: dict[str, Any], baseline: ArtifactPaths, output: Path) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    rows = read_csv(manifest_path(baseline, "train"))
    train = load_sequences(config, output, "train")
    if train["user_ids"].astype(str).tolist() != [row["user_id"] for row in rows]:
        raise RuntimeError("train sequence order differs from manifest")
    train["labels"] = np.asarray([1 if row["label"] == "yes" else 0 for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int32)
    for candidate in config["candidateModels"]["sequence"]:
        model_id = candidate["modelId"]
        output_rows = []
        checkpoint_hashes = {}
        for fold in sorted(set(folds.tolist())):
            train_indexes = np.flatnonzero(folds != fold).astype(np.int64)
            validation_indexes = np.flatnonzero(folds == fold).astype(np.int64)
            checkpoint = output / "checkpoints" / "sequence" / model_id / f"fold-{fold}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if checkpoint.is_file():
                payload = torch.load(checkpoint, map_location=device, weights_only=True)
                model = build_sequence_model(payload["candidate"], int(payload["inputDim"])).to(device)
                model.load_state_dict(payload["stateDict"])
            else:
                model, device = train_sequence_fold(
                    train, train_indexes, candidate, int(candidate["seed"]) + fold
                )
                torch.save(sequence_checkpoint(model, candidate, train["sequences"].shape[2] + 1), checkpoint)
            probabilities = infer_sequence(
                model, device, train, int(candidate["batchSize"]), validation_indexes
            )
            checkpoint_hashes[str(fold)] = sha256_file(checkpoint)
            output_rows.extend(
                {
                    "user_id": user_id,
                    "label": LABEL_NAMES[int(label)],
                    "fold": fold,
                    "score": f"{float(probability):.10f}",
                    "model_id": model_id,
                }
                for user_id, label, probability in zip(
                    train["user_ids"][validation_indexes], train["labels"][validation_indexes], probabilities
                )
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(json.dumps({"event": "sequence-fold", "modelId": model_id, "fold": fold}), flush=True)
        score_hash = write_scores(score_path(output, model_id), output_rows, True)
        model_manifest(
            output,
            candidate,
            {
                "candidate": True,
                "supportOnly": False,
                "scoreSha256": score_hash,
                "checkpointHashes": checkpoint_hashes,
                "fixedEpochs": True,
                "deviceName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "host": platform.node(),
                "testArtifactsReadDuringOof": False,
            },
        )
    del train
    gc.collect()


def fit_meta(candidate: dict[str, Any], features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        C=float(candidate.get("c", 1)),
        max_iter=int(candidate.get("maxIter", 1000)),
        class_weight="balanced",
        random_state=int(candidate["seed"]),
        solver="lbfgs",
    )
    model.fit(features, labels)
    return model


def meta_probability(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(features)[:, list(model.classes_).index(1)]


def nested_base_predictions(
    candidate: dict[str, Any], train: dict[str, Any], outer_fold: int, checkpoint_root: Path
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    outer_training = subset(train, train["folds"] != outer_fold)
    outer_validation = subset(train, train["folds"] == outer_fold)
    inner_folds = sorted(set(outer_training["folds"].tolist()))
    inner_oof = np.full(len(outer_training["labels"]), np.nan, dtype=np.float64)
    outer_sum = np.zeros(len(outer_validation["labels"]), dtype=np.float64)
    provenance = []
    for inner_fold in inner_folds:
        inner_training = subset(outer_training, outer_training["folds"] != inner_fold)
        inner_validation = subset(outer_training, outer_training["folds"] == inner_fold)
        fold_candidate = {
            **candidate,
            "seed": int(candidate["seed"]) + outer_fold * 100 + int(inner_fold),
        }
        checkpoint = (
            checkpoint_root
            / candidate["modelId"]
            / f"outer-{outer_fold}"
            / f"inner-{inner_fold}.joblib"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint.is_file():
            payload = joblib.load(checkpoint)
            transform, model = payload["transform"], payload["model"]
        else:
            transform, x_train = fit_transform(inner_training, fold_candidate)
            model = fit_estimator(fold_candidate, x_train, inner_training["labels"])
            joblib.dump({"candidate": fold_candidate, "transform": transform, "model": model}, checkpoint)
        inner_oof[outer_training["folds"] == inner_fold] = predict(
            model, apply_transform(inner_validation, transform)
        )
        outer_sum += predict(model, apply_transform(outer_validation, transform))
        fit_folds = sorted(set(inner_training["folds"].tolist()))
        provenance.append(
            {
                "outerFold": outer_fold,
                "innerValidationFold": int(inner_fold),
                "baseModelId": candidate["modelId"],
                "fitFolds": fit_folds,
                "outerFoldExcluded": outer_fold not in fit_folds,
                "innerValidationFoldExcluded": int(inner_fold) not in fit_folds,
                "checkpointSha256": sha256_file(checkpoint),
            }
        )
    if np.isnan(inner_oof).any():
        raise RuntimeError(f"nested OOF gap for {candidate['modelId']} outer fold {outer_fold}")
    return inner_oof, outer_sum / len(inner_folds), provenance


def train_stack_oof(config: dict[str, Any], baseline: ArtifactPaths, output: Path) -> None:
    rows = read_csv(manifest_path(baseline, "train"))
    train = load_features(output, "train", rows)
    tabular = {candidate["modelId"]: candidate for candidate in config["candidateModels"]["tabular"]}
    stackers = config["candidateModels"]["stacking"]
    base_ids = sorted({base for candidate in stackers for base in candidate["baseModelIds"]})
    if any(base not in tabular for base in base_ids):
        raise RuntimeError("nested stack refers to an unavailable tabular base model")
    outer_payloads = {}
    provenance = []
    for outer_fold in sorted(set(train["folds"].tolist())):
        inner_columns = {}
        outer_columns = {}
        for base_id in base_ids:
            inner, outer, records = nested_base_predictions(
                tabular[base_id], train, int(outer_fold), output / "checkpoints" / "nested-stack-bases"
            )
            inner_columns[base_id] = inner
            outer_columns[base_id] = outer
            provenance.extend(records)
            print(json.dumps({"event": "nested-base", "outerFold": int(outer_fold), "modelId": base_id}), flush=True)
        outer_payloads[int(outer_fold)] = {
            "training": subset(train, train["folds"] != outer_fold),
            "validation": subset(train, train["folds"] == outer_fold),
            "inner": inner_columns,
            "outer": outer_columns,
        }
    provenance_path = output / "reports" / "nested-stacking-provenance.json"
    write_json(
        provenance_path,
        {
            "records": provenance,
            "allOuterFoldsExcluded": all(record["outerFoldExcluded"] for record in provenance),
            "allInnerValidationFoldsExcluded": all(
                record["innerValidationFoldExcluded"] for record in provenance
            ),
        },
    )
    for candidate in stackers:
        output_rows = []
        checkpoint_hashes = {}
        for outer_fold, payload in outer_payloads.items():
            x_train = np.column_stack([payload["inner"][base] for base in candidate["baseModelIds"]])
            x_validation = np.column_stack([payload["outer"][base] for base in candidate["baseModelIds"]])
            fold_candidate = {**candidate, "seed": int(candidate["seed"]) + outer_fold}
            checkpoint = output / "checkpoints" / "stacking" / candidate["modelId"] / f"outer-{outer_fold}.joblib"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if checkpoint.is_file():
                model = joblib.load(checkpoint)["model"]
            else:
                model = fit_meta(fold_candidate, x_train, payload["training"]["labels"])
                joblib.dump({"candidate": fold_candidate, "model": model}, checkpoint)
            probabilities = meta_probability(model, x_validation)
            checkpoint_hashes[str(outer_fold)] = sha256_file(checkpoint)
            output_rows.extend(
                {
                    "user_id": user_id,
                    "label": LABEL_NAMES[int(label)],
                    "fold": outer_fold,
                    "score": f"{float(probability):.10f}",
                    "model_id": candidate["modelId"],
                }
                for user_id, label, probability in zip(
                    payload["validation"]["user_ids"],
                    payload["validation"]["labels"],
                    probabilities,
                )
            )
        score_hash = write_scores(score_path(output, candidate["modelId"]), output_rows, True)
        model_manifest(
            output,
            candidate,
            {
                "candidate": True,
                "supportOnly": False,
                "baseModelIds": candidate["baseModelIds"],
                "nestedCrossFitting": True,
                "nestedStackingProvenanceSha256": sha256_file(provenance_path),
                "scoreSha256": score_hash,
                "checkpointHashes": checkpoint_hashes,
                "testArtifactsReadDuringOof": False,
            },
        )
    del train, outer_payloads
    gc.collect()


def read_score(path: Path, model_id: str, labelled: bool) -> list[dict[str, str]]:
    rows = read_csv(path)
    expected_fields = {"user_id", "label", "fold", "score", "model_id"} if labelled else {
        "user_id",
        "score",
        "model_id",
    }
    if not rows or set(rows[0]) != expected_fields or any(row["model_id"] != model_id for row in rows):
        raise RuntimeError(f"unexpected score artifact {path}")
    return rows


def audit_oof(config: dict[str, Any], baseline: ArtifactPaths, output: Path) -> None:
    forbidden = [
        output / "reports" / "test-feature-preparation.json",
        output / "reports" / "label-free-test-score-manifest.json",
        output / "reports" / "final-test-report.json",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("OOF audit must precede all ensemble test artifacts")
    manifest = read_csv(manifest_path(baseline, "train"))
    by_user = {row["user_id"]: row for row in manifest}
    findings = []
    models = {}
    for candidate in all_candidates(config):
        model_id = candidate["modelId"]
        try:
            rows = read_score(score_path(output, model_id), model_id, True)
        except Exception as error:
            findings.append(str(error))
            continue
        user_ids = [row["user_id"] for row in rows]
        if len(rows) != len(manifest) or len(set(user_ids)) != len(rows) or set(user_ids) != set(by_user):
            findings.append(f"{model_id}: OOF coverage mismatch")
        for row in rows:
            source = by_user.get(row["user_id"])
            expected_label = "diagnosed" if source and source["label"] == "yes" else "control"
            if source is None or row["label"] != expected_label or row["fold"] != source["fold"]:
                findings.append(f"{model_id}: OOF label/fold mismatch")
                break
            score = float(row["score"])
            if not np.isfinite(score) or not 0 <= score <= 1:
                findings.append(f"{model_id}: invalid OOF score")
                break
        models[model_id] = {"rows": len(rows), "sha256": sha256_file(score_path(output, model_id))}
    provenance_path = output / "reports" / "nested-stacking-provenance.json"
    if not provenance_path.is_file():
        findings.append("nested stacking provenance is missing")
        nested = {}
    else:
        nested = load_json(provenance_path)
        if not nested.get("allOuterFoldsExcluded") or not nested.get("allInnerValidationFoldsExcluded"):
            findings.append("nested stacking exclusion audit failed")
    report = {
        "ok": not findings,
        "experimentId": config["experimentId"],
        "trainUsers": len(manifest),
        "models": models,
        "nestedRecords": len(nested.get("records", [])),
        "sourceChampionLockSha256": config["sourceChampion"]["lockSha256"],
        "testEmbeddingsRead": False,
        "testLabelsRead": False,
        "findings": findings,
    }
    path = output / "reports" / "oof-audit.json"
    write_json(path, report)
    if findings:
        raise RuntimeError("OOF audit failed: " + "; ".join(findings[:3]))
    print(json.dumps({"status": "ok", "trainUsers": len(manifest), "models": len(models)}), flush=True)


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))

    def ratio(numerator: float, denominator: float) -> float:
        return 0 if denominator == 0 else numerator / denominator

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
        "precision": precision,
        "recall": recall,
        "diagnosedF1": diagnosed_f1,
        "controlF1": control_f1,
        "macroF1": (diagnosed_f1 + control_f1) / 2,
        "accuracy": ratio(tp + tn, len(labels)),
        "confusionMatrix": [[tn, fp], [fn, tp]],
    }


def metric_key(metrics: dict[str, Any], threshold: float) -> tuple[float, float, float, float]:
    return (
        float(metrics["macroF1"]),
        float(metrics["diagnosedF1"]),
        float(metrics["precision"]),
        -threshold,
    )


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    total_positive = int(labels.sum())
    total_negative = len(labels) - total_positive
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(1 - sorted_labels)
    candidate_ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    empty_threshold = float(np.nextafter(float(sorted_scores.max()), math.inf))
    empty_metrics = binary_metrics(labels, np.zeros(len(labels), dtype=np.int64))
    best = (metric_key(empty_metrics, empty_threshold), empty_threshold, empty_metrics)
    for end in candidate_ends:
        tp = int(cumulative_tp[end])
        fp = int(cumulative_fp[end])
        fn = total_positive - tp
        tn = total_negative - fp
        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        diagnosed_f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        control_precision = 0 if tn + fn == 0 else tn / (tn + fn)
        control_recall = 0 if tn + fp == 0 else tn / (tn + fp)
        control_f1 = (
            0
            if control_precision + control_recall == 0
            else 2 * control_precision * control_recall / (control_precision + control_recall)
        )
        metrics = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "diagnosedF1": diagnosed_f1,
            "controlF1": control_f1,
            "macroF1": (diagnosed_f1 + control_f1) / 2,
            "accuracy": (tp + tn) / len(labels),
            "confusionMatrix": [[tn, fp], [fn, tp]],
        }
        threshold = float(sorted_scores[end])
        candidate = (metric_key(metrics, threshold), threshold, metrics)
        if candidate[0] > best[0]:
            best = candidate
    return best[1], best[2]


def enumerate_weights(model_count: int, step: float, minimum: float) -> Iterable[tuple[float, ...]]:
    units = round(1 / step)
    minimum_units = round(minimum / step)
    remaining = units - model_count * minimum_units
    for bars in itertools.combinations(range(remaining + model_count - 1), model_count - 1):
        cuts = (-1, *bars, remaining + model_count - 1)
        extras = [cuts[index + 1] - cuts[index] - 1 for index in range(model_count)]
        yield tuple((extra + minimum_units) * step for extra in extras)


def load_score_matrix(
    output: Path, model_ids: list[str], labelled: bool
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    reference: list[str] | None = None
    labels = None
    columns = []
    for model_id in model_ids:
        rows = read_score(score_path(output, model_id, not labelled), model_id, labelled)
        by_user = {row["user_id"]: row for row in rows}
        if reference is None:
            reference = sorted(by_user)
            if labelled:
                labels = np.asarray(
                    [1 if by_user[user_id]["label"] == "diagnosed" else 0 for user_id in reference],
                    dtype=np.int64,
                )
        if set(by_user) != set(reference):
            raise RuntimeError(f"score user mismatch for {model_id}")
        columns.append(np.asarray([float(by_user[user_id]["score"]) for user_id in reference]))
    return reference or [], np.column_stack(columns), labels


def lock_ensemble(config: dict[str, Any], output: Path) -> None:
    audit_path = output / "reports" / "oof-audit.json"
    if not audit_path.is_file() or not load_json(audit_path).get("ok"):
        raise RuntimeError("passing OOF audit is required before lock")
    model_ids = config["ensemble"]["requiredModelIds"]
    users, matrix, labels = load_score_matrix(output, model_ids, True)
    assert labels is not None
    best = None
    evaluated = 0
    for weights in enumerate_weights(
        len(model_ids), float(config["ensemble"]["weightStep"]), float(config["ensemble"]["minimumWeight"])
    ):
        threshold, metrics = best_threshold(labels, matrix @ np.asarray(weights))
        candidate = (metric_key(metrics, threshold), weights, threshold, metrics)
        if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and weights < best[1]):
            best = candidate
        evaluated += 1
    assert best is not None
    source_weights = np.asarray([config["sourceChampion"]["weights"][model] for model in model_ids])
    source_threshold, source_metrics = best_threshold(labels, matrix @ source_weights)
    lock = {
        "schemaVersion": 1,
        "experimentId": config["experimentId"],
        "dataset": config["dataset"],
        "selectionDataset": "SetembroBR v7 train OOF only",
        "trainUsers": len(users),
        "modelIds": model_ids,
        "weights": {model: weight for model, weight in zip(model_ids, best[1])},
        "threshold": best[2],
        "oofMetrics": best[3],
        "evaluatedWeightVectors": evaluated,
        "sourceEqualWeightReference": {
            "weights": config["sourceChampion"]["weights"],
            "oofSelectedThreshold": source_threshold,
            "oofMetrics": source_metrics,
        },
        "sourceChampionLockSha256": config["sourceChampion"]["lockSha256"],
        "oofAuditSha256": sha256_file(audit_path),
        "testEmbeddingsRead": False,
        "testScoresRead": False,
        "testLabelsRead": False,
        "createdAt": "1970-01-01T00:00:00.000Z",
    }
    path = output / "ensemble" / "ensemble-lock.json"
    write_json(path, lock)
    write_json(
        output / "reports" / "lock-provenance.json",
        {"lockSha256": sha256_file(path), "oofAuditSha256": sha256_file(audit_path), "models": model_ids},
    )
    print(json.dumps({"status": "ok", "oofMetrics": lock["oofMetrics"], "threshold": lock["threshold"]}), flush=True)


def score_tabular_test(config: dict[str, Any], baseline: ArtifactPaths, output: Path) -> None:
    require_lock(output)
    rows = read_csv(manifest_path(baseline, "test"))
    test = load_features(output, "test", rows)
    for candidate in config["candidateModels"]["tabular"]:
        model_id = candidate["modelId"]
        checkpoints = sorted((output / "checkpoints" / "tabular" / model_id).glob("fold-*.joblib"))
        if len(checkpoints) != int(config["foldCount"]):
            raise RuntimeError(f"{model_id}: expected five tabular checkpoints")
        total = np.zeros(len(rows), dtype=np.float64)
        for checkpoint in checkpoints:
            payload = joblib.load(checkpoint)
            total += predict(payload["model"], apply_transform(test, payload["transform"]))
        write_scores(
            score_path(output, model_id, True),
            [
                {"user_id": row["user_id"], "score": f"{float(score):.10f}", "model_id": model_id}
                for row, score in zip(rows, total / len(checkpoints))
            ],
            False,
        )
        print(json.dumps({"event": "test-score", "modelId": model_id}), flush=True)
    del test
    gc.collect()


def score_sequence_test(config: dict[str, Any], baseline: ArtifactPaths, output: Path) -> None:
    require_lock(output)
    rows = read_csv(manifest_path(baseline, "test"))
    test = load_sequences(config, output, "test")
    if test["user_ids"].astype(str).tolist() != [row["user_id"] for row in rows]:
        raise RuntimeError("test sequence order differs from redacted manifest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for candidate in config["candidateModels"]["sequence"]:
        model_id = candidate["modelId"]
        checkpoints = sorted((output / "checkpoints" / "sequence" / model_id).glob("fold-*.pt"))
        if len(checkpoints) != int(config["foldCount"]):
            raise RuntimeError(f"{model_id}: expected five sequence checkpoints")
        total = np.zeros(len(rows), dtype=np.float64)
        for checkpoint in checkpoints:
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
            model = build_sequence_model(payload["candidate"], int(payload["inputDim"])).to(device)
            model.load_state_dict(payload["stateDict"])
            total += infer_sequence(model, device, test, int(candidate["batchSize"]))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        write_scores(
            score_path(output, model_id, True),
            [
                {"user_id": row["user_id"], "score": f"{float(score):.10f}", "model_id": model_id}
                for row, score in zip(rows, total / len(checkpoints))
            ],
            False,
        )
        print(json.dumps({"event": "test-score", "modelId": model_id}), flush=True)
    del test
    gc.collect()


def score_stack_test(config: dict[str, Any], baseline: ArtifactPaths, output: Path) -> None:
    require_lock(output)
    train_ids_expected = sorted(row["user_id"] for row in read_csv(manifest_path(baseline, "train")))
    test_ids_expected = sorted(row["user_id"] for row in read_csv(manifest_path(baseline, "test")))
    for candidate in config["candidateModels"]["stacking"]:
        train_ids, x_train, labels = load_score_matrix(output, candidate["baseModelIds"], True)
        test_ids, x_test, _ = load_score_matrix(output, candidate["baseModelIds"], False)
        if train_ids != train_ids_expected or test_ids != test_ids_expected:
            raise RuntimeError("stack score users differ from immutable manifests")
        assert labels is not None
        model = fit_meta(candidate, x_train, labels)
        checkpoint = output / "checkpoints" / "stacking" / candidate["modelId"] / "final-from-train-oof.joblib"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"candidate": candidate, "model": model}, checkpoint)
        probabilities = meta_probability(model, x_test)
        write_scores(
            score_path(output, candidate["modelId"], True),
            [
                {"user_id": user_id, "score": f"{float(score):.10f}", "model_id": candidate["modelId"]}
                for user_id, score in zip(test_ids, probabilities)
            ],
            False,
        )
        print(json.dumps({"event": "test-score", "modelId": candidate["modelId"]}), flush=True)


def score_test(config: dict[str, Any], baseline: ArtifactPaths, output: Path) -> None:
    score_tabular_test(config, baseline, output)
    score_sequence_test(config, baseline, output)
    score_stack_test(config, baseline, output)


def audit_test(config: dict[str, Any], baseline: ArtifactPaths, output: Path) -> None:
    lock = require_lock(output)
    expected_users = {row["user_id"] for row in read_csv(manifest_path(baseline, "test"))}
    models = {}
    for candidate in all_candidates(config):
        model_id = candidate["modelId"]
        path = score_path(output, model_id, True)
        rows = read_score(path, model_id, False)
        user_ids = [row["user_id"] for row in rows]
        if (
            len(rows) != expected(config, "testUsers")
            or len(set(user_ids)) != len(rows)
            or set(user_ids) != expected_users
        ):
            raise RuntimeError(f"label-free test score coverage mismatch: {model_id}")
        if any(not np.isfinite(float(row["score"])) or not 0 <= float(row["score"]) <= 1 for row in rows):
            raise RuntimeError(f"invalid label-free test score: {model_id}")
        models[model_id] = {"rows": len(rows), "sha256": sha256_file(path)}
    report = {
        "ok": True,
        "experimentId": config["experimentId"],
        "users": len(expected_users),
        "models": models,
        "lockSha256": sha256_file(output / "ensemble" / "ensemble-lock.json"),
        "thresholdUnusedForScoring": lock["threshold"],
        "testScoresLabelFree": True,
        "testLabelsRead": False,
    }
    write_json(output / "reports" / "label-free-test-score-manifest.json", report)
    print(json.dumps({"status": "ok", "users": len(expected_users), "models": len(models)}), flush=True)


def bootstrap_intervals(
    labels: np.ndarray, predictions: np.ndarray, seed: int, samples: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values = {name: np.empty(samples, dtype=np.float64) for name in ["macroF1", "diagnosedF1", "precision", "recall", "accuracy"]}
    for index in range(samples):
        selected = np.concatenate(
            [rng.choice(positive, len(positive), replace=True), rng.choice(negative, len(negative), replace=True)]
        )
        metrics = binary_metrics(labels[selected], predictions[selected])
        for name in values:
            values[name][index] = metrics[name]
    return {
        "method": "deterministic class-stratified percentile bootstrap",
        "samples": samples,
        "seed": seed,
        **{name: [float(value) for value in np.quantile(array, [0.025, 0.975])] for name, array in values.items()},
    }


def evaluate(config: dict[str, Any], baseline: ArtifactPaths, output: Path) -> None:
    lock = require_lock(output)
    score_manifest_path = output / "reports" / "label-free-test-score-manifest.json"
    if not score_manifest_path.is_file() or not load_json(score_manifest_path).get("ok"):
        raise RuntimeError("passing label-free score audit is required before opening labels")
    score_manifest = load_json(score_manifest_path)
    lock_path = output / "ensemble" / "ensemble-lock.json"
    if score_manifest["lockSha256"] != sha256_file(lock_path):
        raise RuntimeError("test score audit is bound to another lock")
    user_ids, matrix, _ = load_score_matrix(output, lock["modelIds"], False)
    weights = np.asarray([lock["weights"][model] for model in lock["modelIds"]])
    scores = matrix @ weights
    predictions = (scores >= float(lock["threshold"])).astype(np.int64)

    sealed_path = baseline.sealed / "test_labels.csv"
    sealed_rows = read_csv(sealed_path)
    sealed = {row["user_id"]: row["label"] for row in sealed_rows}
    if len(sealed_rows) != expected(config, "testUsers") or set(sealed) != set(user_ids):
        raise RuntimeError("sealed test labels differ from label-free score users")
    labels = np.asarray([1 if sealed[user_id] == "yes" else 0 for user_id in user_ids], dtype=np.int64)
    metrics = binary_metrics(labels, predictions)
    intervals = bootstrap_intervals(
        labels,
        predictions,
        int(config["seed"]),
        int(config["evaluation"]["bootstrapSamples"]),
    )
    report = {
        "ok": True,
        "experimentId": config["experimentId"],
        "dataset": config["dataset"],
        "predictionTarget": config["predictionTarget"],
        "evaluationType": "strict-blind fixed champion architecture retrained on v7 train",
        "testUsers": len(user_ids),
        "testMetrics": metrics,
        "confidenceIntervals95": intervals,
        "threshold": lock["threshold"],
        "weights": lock["weights"],
        "oofMetrics": lock["oofMetrics"],
        "lockSha256": sha256_file(lock_path),
        "scoreManifestSha256": sha256_file(score_manifest_path),
        "sealedLabelSha256": sha256_file(sealed_path),
        "sourcePickleSha256": config["sourcePickleSha256"],
        "runtime": {
            "host": platform.node(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": importlib.metadata.version("torch"),
            "xgboost": importlib.metadata.version("xgboost"),
            "numpy": importlib.metadata.version("numpy"),
            "scikitLearn": importlib.metadata.version("scikit-learn"),
        },
        "evaluatedOnce": True,
    }
    write_json(
        output / "sealed" / "test-label-provenance.json",
        {
            "sealedLabelSha256": sha256_file(sealed_path),
            "lockSha256": sha256_file(lock_path),
            "scoreManifestSha256": sha256_file(score_manifest_path),
            "labelsCopied": False,
        },
    )
    report_path = output / "reports" / "final-test-report.json"
    write_json(report_path, report)
    write_json(
        output / "reports" / "final-artifact-chain.json",
        {
            "ok": True,
            "configSha256": config["_configSha256"],
            "sourcePickleSha256": config["sourcePickleSha256"],
            "sourceChampionLockSha256": config["sourceChampion"]["lockSha256"],
            "oofAuditSha256": sha256_file(output / "reports" / "oof-audit.json"),
            "ensembleLockSha256": sha256_file(lock_path),
            "labelFreeTestScoreManifestSha256": sha256_file(score_manifest_path),
            "finalTestReportSha256": sha256_file(report_path),
        },
    )
    print(json.dumps({"status": "ok", "testMetrics": metrics}), flush=True)


def run_oof(
    config: dict[str, Any],
    baseline: ArtifactPaths,
    archive: Path,
    output: Path,
    temporary: Path,
    helper: Any,
) -> None:
    prepare_split(config, baseline, archive, output, temporary, "train", helper)
    train_tabular_oof(config, baseline, output)
    train_sequence_oof(config, baseline, output)
    train_stack_oof(config, baseline, output)
    audit_oof(config, baseline, output)
    lock_ensemble(config, output)


def run_test(
    config: dict[str, Any],
    baseline: ArtifactPaths,
    archive: Path,
    output: Path,
    temporary: Path,
    helper: Any,
) -> None:
    prepare_split(config, baseline, archive, output, temporary, "test", helper)
    score_test(config, baseline, output)
    audit_test(config, baseline, output)
    evaluate(config, baseline, output)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    baseline_config = args.baseline_config.expanduser().resolve()
    helper_path = args.feature_helper.expanduser().resolve()
    config = load_config(config_path, baseline_config, helper_path)
    baseline = ArtifactPaths(args.baseline_output.expanduser().resolve())
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    helper = load_feature_helper(helper_path)
    if args.command in {"prepare-train", "prepare-test", "run-oof", "run-test"}:
        if args.archive_root is None or args.temporary_dir is None:
            raise RuntimeError("preparation commands require --archive-root and --temporary-dir")
        archive = args.archive_root.expanduser().resolve()
        temporary = args.temporary_dir.expanduser().resolve()
    else:
        archive = Path(".")
        temporary = Path(".")
    dispatch = {
        "prepare-train": lambda: prepare_split(config, baseline, archive, output, temporary, "train", helper),
        "train-tabular-oof": lambda: train_tabular_oof(config, baseline, output),
        "train-sequence-oof": lambda: train_sequence_oof(config, baseline, output),
        "train-stack-oof": lambda: train_stack_oof(config, baseline, output),
        "audit-oof": lambda: audit_oof(config, baseline, output),
        "lock": lambda: lock_ensemble(config, output),
        "prepare-test": lambda: prepare_split(config, baseline, archive, output, temporary, "test", helper),
        "score-test": lambda: score_test(config, baseline, output),
        "audit-test": lambda: audit_test(config, baseline, output),
        "evaluate": lambda: evaluate(config, baseline, output),
        "audit-feature-support": lambda: audit_feature_support(config, output),
        "run-oof": lambda: run_oof(config, baseline, archive, output, temporary, helper),
        "run-test": lambda: run_test(config, baseline, archive, output, temporary, helper),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
