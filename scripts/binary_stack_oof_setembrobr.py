#!/usr/bin/env python3
"""Train strict-blind binary stacking models from train OOF scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    parent_ref = cfg.get("extends")
    if not isinstance(parent_ref, str):
        return cfg
    parent_path = Path(parent_ref).expanduser()
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()
    child = {key: value for key, value in cfg.items() if key != "extends"}
    return deep_merge(load_config(parent_path), child)


def deep_merge(parent: dict, child: dict) -> dict:
    merged = dict(parent)
    for key, value in child.items():
        parent_value = merged.get(key)
        if isinstance(parent_value, dict) and isinstance(value, dict):
            merged[key] = deep_merge(parent_value, value)
        else:
            merged[key] = value
    return merged


def read_csv_with_hash(path: Path):
    text = path.read_text()
    return list(csv.DictReader(text.splitlines())), hashlib.sha256(text.encode()).hexdigest()


def strict_blind_manifest_hash(cfg) -> str:
    path = Path(cfg["outputDir"]) / "manifest" / f"strict_blind_split_manifest_seed{cfg['seed']}.csv"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_train_manifest_hash(cfg) -> str:
    path = Path(cfg["outputDir"]) / "manifest" / f"train_binary_manifest_seed{cfg['seed']}.csv"
    _rows, digest = read_csv_with_hash(path)
    return digest


def read_score_rows(path: Path):
    rows, digest = read_csv_with_hash(path)
    for row in rows:
        row["score"] = float(row["score"])
    return rows, digest


def score_path(cfg, prefix: str, model_id: str) -> Path:
    return Path(cfg["outputDir"]) / "scores" / f"{prefix}_{model_id}.csv"


def align_train_scores(cfg, base_model_ids: list[str]):
    by_model = {}
    hashes = {}
    for model_id in base_model_ids:
        path = score_path(cfg, "train_oof", model_id)
        rows, digest = read_score_rows(path)
        hashes[model_id] = digest
        by_model[model_id] = {row["user_id"]: row for row in rows}

    expected_user_ids = set(next(iter(by_model.values())).keys())
    for model_id, rows_by_user in by_model.items():
        if set(rows_by_user.keys()) != expected_user_ids:
            raise RuntimeError(f"{model_id}: train OOF user set mismatch")
    user_ids = sorted(expected_user_ids)
    features = []
    labels = []
    folds = []
    for user_id in user_ids:
        base = by_model[base_model_ids[0]][user_id]
        label = base["label"]
        fold = int(base["fold"])
        row_features = []
        for model_id in base_model_ids:
            row = by_model[model_id].get(user_id)
            if row is None:
                raise RuntimeError(f"{model_id}: missing train OOF user {user_id}")
            if row.get("label") != label or int(row.get("fold", -1)) != fold:
                raise RuntimeError(f"{model_id}: train OOF alignment mismatch for {user_id}")
            row_features.append(row["score"])
        features.append(row_features)
        labels.append(1 if label == "diagnosed" else 0)
        folds.append(fold)
    return user_ids, np.array(features, dtype=np.float64), np.array(labels, dtype=np.int64), np.array(folds, dtype=np.int32), hashes


def align_score_features(cfg, base_model_ids: list[str]):
    by_model = {}
    hashes = {}
    for model_id in base_model_ids:
        path = score_path(cfg, "test_score", model_id)
        rows, digest = read_score_rows(path)
        hashes[model_id] = digest
        by_model[model_id] = {row["user_id"]: row for row in rows}

    expected_user_ids = set(next(iter(by_model.values())).keys())
    for model_id, rows_by_user in by_model.items():
        if set(rows_by_user.keys()) != expected_user_ids:
            raise RuntimeError(f"{model_id}: label-free score user set mismatch")
    user_ids = sorted(expected_user_ids)
    features = []
    for user_id in user_ids:
        row_features = []
        for model_id in base_model_ids:
            row = by_model[model_id].get(user_id)
            if row is None:
                raise RuntimeError(f"{model_id}: missing label-free score user {user_id}")
            if "label" in row or "fold" in row:
                raise RuntimeError(f"{model_id}: label-free score contains forbidden columns")
            row_features.append(row["score"])
        features.append(row_features)
    return user_ids, np.array(features, dtype=np.float64), hashes


def balanced_sample_weights(y):
    counts = np.bincount(y, minlength=2).astype(np.float64)
    class_weights = len(y) / (2.0 * np.maximum(counts, 1.0))
    return class_weights[y]


def fit_model(candidate, x_train, y_train):
    model = LogisticRegression(
        C=float(candidate.get("c", 1.0)),
        max_iter=int(candidate.get("maxIter", 1000)),
        class_weight="balanced",
        random_state=int(candidate["seed"]),
        solver="lbfgs",
    )
    model.fit(x_train, y_train, sample_weight=balanced_sample_weights(y_train))
    return model


def positive_prob(model, probs, count: int):
    classes = list(model.classes_)
    if 1 not in classes:
        return np.ones(count, dtype=np.float64) if classes == [1] else np.zeros(count, dtype=np.float64)
    return probs[:, classes.index(1)]


def write_scores(path: Path, rows, include_labels: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["user_id", "label", "fold", "score", "model_id"] if include_labels else ["user_id", "score", "model_id"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in headers})


def train_candidate(cfg, candidate) -> None:
    model_id = candidate["modelId"]
    base_model_ids = list(candidate["baseModelIds"])
    train_users, x_train_all, y_all, folds, train_hashes = align_train_scores(cfg, base_model_ids)
    score_users, x_score, score_hashes = align_score_features(cfg, base_model_ids)

    oof_rows = []
    for fold in sorted(set(int(value) for value in folds)):
        train_mask = folds != fold
        validation_mask = folds == fold
        model = fit_model(candidate, x_train_all[train_mask], y_all[train_mask])
        probs = positive_prob(model, model.predict_proba(x_train_all[validation_mask]), int(validation_mask.sum()))
        for user_id, label_code, prob in zip(np.array(train_users)[validation_mask], y_all[validation_mask], probs):
            oof_rows.append(
                {
                    "user_id": user_id,
                    "label": "diagnosed" if int(label_code) == 1 else "control",
                    "fold": fold,
                    "score": f"{float(prob):.8f}",
                    "model_id": model_id,
                }
            )

    model = fit_model(candidate, x_train_all, y_all)
    score_probs = positive_prob(model, model.predict_proba(x_score), len(x_score))
    score_rows = [
        {"user_id": user_id, "score": f"{float(prob):.8f}", "model_id": model_id}
        for user_id, prob in zip(score_users, score_probs)
    ]

    out_dir = Path(cfg["outputDir"])
    write_scores(out_dir / "scores" / f"train_oof_{model_id}.csv", sorted(oof_rows, key=lambda row: row["user_id"]), True)
    write_scores(out_dir / "scores" / f"test_score_{model_id}.csv", score_rows, False)
    manifest = {
        "modelId": model_id,
        "candidate": True,
        "family": candidate["family"],
        "seed": candidate["seed"],
        "manifestHash": strict_blind_manifest_hash(cfg),
        "trainManifestHash": read_train_manifest_hash(cfg),
        "featureSource": cfg.get("featureSource", "raw_artifacts"),
        "rawArtifactsDir": cfg.get("rawArtifactsDir"),
        "dbTables": {},
        "baseModelIds": base_model_ids,
        "sourceScoreHashes": {"trainOof": train_hashes, "labelFreeScore": score_hashes},
        "featureBlocks": ["train_oof_scores"],
        "hyperparameters": {key: value for key, value in candidate.items() if key not in {"modelId", "family", "baseModelIds"}},
        "scoreSchema": "binary-score-v1",
        "usesTestLabelsForTraining": False,
        "usesTestScoresForTraining": False,
        "gpuUsed": False,
        "createdAt": "1970-01-01T00:00:00.000Z",
    }
    manifest_path = out_dir / "model-manifests" / f"{model_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {model_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setembrobr.seed42.raw-qwen3-binary.json")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    candidates = list(cfg.get("candidateModels", {}).get("stacking", []))
    if args.only:
        wanted = set(args.only)
        candidates = [candidate for candidate in candidates if candidate["modelId"] in wanted]
    if not candidates:
        raise SystemExit("No binary stacking candidates selected")

    for candidate in candidates:
        train_candidate(cfg, candidate)


if __name__ == "__main__":
    main()
