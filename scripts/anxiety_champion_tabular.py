#!/usr/bin/env python3
"""Train anxiety tabular OOF checkpoints or score the label-free test split after lock."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from anxiety_champion_relevance import load_config, require_post_lock, resolve, sha256_file, write_json

EMBEDDING_BLOCKS = {"mean_pca": "mean", "rel3_pca": "rel3", "rel6_pca": "rel6", "rel7_pca": "rel7"}


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text().splitlines()))


def load_features(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    try:
        blocks = {}
        for block in ["evidence_markers", "stylistic", "relevance_counts", "temporal_markers", "mean", "rel3", "rel6", "rel7"]:
            if block in data.files:
                blocks[block] = data[block].astype(np.float32)
        return {
            "user_ids": data["user_ids"].astype(str),
            "labels": data["labels"].astype(np.int64),
            "folds": data["folds"].astype(np.int32),
            "blocks": blocks,
        }
    finally:
        data.close()


def reorder(split: dict[str, Any], user_ids: list[str], labels: list[int] | None = None, folds: list[int] | None = None) -> dict[str, Any]:
    by_user = {user_id: index for index, user_id in enumerate(split["user_ids"])}
    missing = [user_id for user_id in user_ids if user_id not in by_user]
    if missing:
        raise RuntimeError(f"feature artifact missing users: {missing[:5]}")
    indexes = np.asarray([by_user[user_id] for user_id in user_ids], dtype=np.int64)
    out = {"user_ids": np.asarray(user_ids), "blocks": {name: values[indexes] for name, values in split["blocks"].items()}}
    if labels is not None:
        out["labels"] = np.asarray(labels, dtype=np.int64)
    if folds is not None:
        out["folds"] = np.asarray(folds, dtype=np.int32)
    return out


def subset(split: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    out = {key: value[mask] for key, value in split.items() if key in {"user_ids", "labels", "folds"}}
    out["blocks"] = {name: value[mask] for name, value in split["blocks"].items()}
    return out


def fit_transform(train: dict[str, Any], candidate: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    fitted_blocks = []
    pieces = []
    pca_components = int(candidate.get("pcaComponents", 80))
    for offset, block_name in enumerate(candidate["featureBlocks"]):
        source = EMBEDDING_BLOCKS.get(block_name, block_name)
        values = train["blocks"][source]
        if block_name in EMBEDDING_BLOCKS:
            count = min(pca_components, len(train["labels"]) - 1, values.shape[1])
            pca = PCA(n_components=count, random_state=int(candidate["seed"]) + offset * 101)
            pieces.append(pca.fit_transform(values))
            fitted_blocks.append({"name": block_name, "source": source, "pca": pca})
        else:
            pieces.append(values)
            fitted_blocks.append({"name": block_name, "source": source, "pca": None})
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
    model = torch.nn.Linear(features.shape[1], 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    positives = max(float((labels == 1).sum()), 1.0)
    negatives = max(float((labels == 0).sum()), 1.0)
    pos_weight = len(labels) / (2.0 * positives)
    neg_weight = len(labels) / (2.0 * negatives)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for _epoch in range(250):
        optimizer.zero_grad()
        logits = model(x).squeeze(1).clamp(-30.0, 30.0)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        point_prob = torch.exp(-bce)
        class_weights = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.full_like(y, neg_weight))
        loss = (((1.0 - point_prob) ** gamma) * bce * class_weights).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("anxiety focal linear produced a non-finite loss")
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
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs")
        estimator.fit(features, labels)
        return {"kind": "sklearn", "estimator": estimator}
    if family == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=(int(candidate["hiddenSize"]),),
            alpha=float(candidate["alpha"]),
            random_state=seed,
            max_iter=500,
            early_stopping=False,
        )
        estimator.fit(features, labels)
        return {"kind": "sklearn", "estimator": estimator}
    if family == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is required for the anxiety champion")
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=int(candidate.get("nEstimators", 250)),
            max_depth=int(candidate.get("maxDepth", 3)),
            learning_rate=float(candidate.get("learningRate", 0.05)),
            subsample=float(candidate.get("subsample", 0.85)),
            colsample_bytree=float(candidate.get("colsampleBytree", 0.85)),
            reg_lambda=float(candidate.get("regLambda", 3.0)),
            min_child_weight=float(candidate.get("minChildWeight", 2.0)),
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        estimator.fit(features, labels, sample_weight=balanced_weights(labels))
        return {"kind": "sklearn", "estimator": estimator}
    if family in {"focal_linear", "focal_logreg"}:
        return fit_focal(features, labels, float(candidate.get("gamma", 1.0)), seed)
    raise RuntimeError(f"unsupported anxiety tabular family {family}")


def predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    if model["kind"] == "focal_linear":
        logits = features @ model["weight"].reshape(-1) + float(model["bias"][0])
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30))), 0, 1)
    estimator = model["estimator"]
    classes = list(estimator.classes_)
    probabilities = estimator.predict_proba(features)
    return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(features), dtype=np.float64)


def write_scores(path: Path, rows: list[dict[str, Any]], include_labels: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["user_id", "label", "fold", "score", "model_id"] if include_labels else ["user_id", "score", "model_id"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows({header: row[header] for header in headers} for row in rows)


def train_oof(config: dict[str, Any], repo: Path) -> None:
    output_dir = resolve(config["outputDir"], repo)
    work_dir = resolve(config.get("workDir", config["outputDir"]), repo)
    manifest_path = output_dir / "manifest" / f"train_binary_manifest_seed{config['seed']}.csv"
    manifest_rows = read_csv(manifest_path)
    user_ids = [row["user_id"] for row in manifest_rows]
    labels = [1 if row["label"] == "diagnosed" else 0 for row in manifest_rows]
    folds = [int(row["fold"]) for row in manifest_rows]
    train = reorder(load_features(work_dir / "features" / "train_raw_features.npz"), user_ids, labels, folds)
    strict_hash = sha256_file(output_dir / "manifest" / f"strict_blind_split_manifest_seed{config['seed']}.csv")
    proxy_hash = sha256_file(output_dir / "relevance-proxy" / "proxy-definition.json")
    for candidate in config["candidateModels"]["tabular"]:
        model_id = candidate["modelId"]
        rows = []
        checkpoint_hashes = {}
        for fold in sorted(set(folds)):
            fold_candidate = {**candidate, "seed": int(candidate["seed"]) + fold}
            training = subset(train, train["folds"] != fold)
            validation = subset(train, train["folds"] == fold)
            transform, x_train = fit_transform(training, fold_candidate)
            model = fit_estimator(fold_candidate, x_train, training["labels"])
            probabilities = predict(model, apply_transform(validation, transform))
            checkpoint = output_dir / "checkpoints" / "tabular" / model_id / f"fold-{fold}.joblib"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"candidate": fold_candidate, "transform": transform, "model": model}, checkpoint)
            checkpoint_hashes[str(fold)] = sha256_file(checkpoint)
            for user_id, label, probability in zip(validation["user_ids"], validation["labels"], probabilities):
                rows.append(
                    {
                        "user_id": user_id,
                        "label": "diagnosed" if int(label) == 1 else "control",
                        "fold": fold,
                        "score": f"{float(probability):.8f}",
                        "model_id": model_id,
                    }
                )
        write_scores(output_dir / "scores" / f"train_oof_{model_id}.csv", sorted(rows, key=lambda row: row["user_id"]), True)
        write_json(
            output_dir / "model-manifests" / f"{model_id}.json",
            {
                "modelId": model_id,
                "candidate": model_id in config["ensemble"]["requiredModelIds"],
                "supportOnly": model_id not in config["ensemble"]["requiredModelIds"],
                "family": candidate["family"],
                "seed": candidate["seed"],
                "predictionTarget": "anxiety",
                "featureSource": "raw_artifacts",
                "manifestHash": strict_hash,
                "trainManifestHash": sha256_file(manifest_path),
                "relevanceProxyKind": config["relevanceProxy"]["kind"],
                "relevanceProxyDefinitionHash": proxy_hash,
                "checkpointHashes": checkpoint_hashes,
                "artifactHashes": {
                    "strictBlindManifestSha256": strict_hash,
                    "trainManifestSha256": sha256_file(manifest_path),
                    "trainFeatureManifestSha256": sha256_file(work_dir / "features" / "train-feature-manifest.json"),
                    "relevanceProxyDefinitionSha256": proxy_hash,
                },
                "featureBlocks": candidate["featureBlocks"],
                "hyperparameters": {key: value for key, value in candidate.items() if key not in {"modelId", "family", "featureBlocks"}},
                "scoreSchema": "binary-score-v1",
                "usesTestLabelsForTraining": False,
                "usesTestScoresForTraining": False,
                "testArtifactsReadDuringOof": False,
                "createdAt": "1970-01-01T00:00:00.000Z",
            },
        )
        print(f"wrote train OOF {model_id}")


def score_test(config: dict[str, Any], repo: Path) -> None:
    output_dir = resolve(config["outputDir"], repo)
    work_dir = resolve(config.get("workDir", config["outputDir"]), repo)
    require_post_lock(output_dir)
    test_manifest = read_csv(output_dir / "manifest" / f"test_inference_manifest_seed{config['seed']}.csv")
    user_ids = [row["user_id"] for row in test_manifest]
    test = reorder(load_features(work_dir / "features" / "test_raw_features.npz"), user_ids)
    for candidate in config["candidateModels"]["tabular"]:
        model_id = candidate["modelId"]
        total = np.zeros(len(user_ids), dtype=np.float64)
        count = 0
        checkpoint_dir = output_dir / "checkpoints" / "tabular" / model_id
        for checkpoint in sorted(checkpoint_dir.glob("fold-*.joblib")):
            payload = joblib.load(checkpoint)
            total += predict(payload["model"], apply_transform(test, payload["transform"]))
            count += 1
        if count != int(config["foldCount"]):
            raise RuntimeError(f"{model_id}: expected {config['foldCount']} checkpoints, got {count}")
        rows = [
            {"user_id": user_id, "score": f"{float(probability):.8f}", "model_id": model_id}
            for user_id, probability in zip(user_ids, total / count)
        ]
        write_scores(output_dir / "scores" / f"test_score_{model_id}.csv", rows, False)
        print(f"wrote label-free test score {model_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setembrobr.seed42.anxiety-temporal-champion-qwen3-binary.json")
    parser.add_argument("--stage", choices=["oof", "score-test"], required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    config = load_config(resolve(args.config, repo))
    train_oof(config, repo) if args.stage == "oof" else score_test(config, repo)


if __name__ == "__main__":
    main()
