#!/usr/bin/env python3
"""Train strict-blind raw binary tabular OOF models from raw Qwen3 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

EMBEDDING_BLOCKS = {
    "mean_pca": "mean",
    "rel3_pca": "rel3",
    "rel6_pca": "rel6",
    "rel7_pca": "rel7",
}

EVIDENCE_SCORE_INDEX = 11


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


def load_raw_feature_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    out = {
        "user_ids": data["user_ids"].astype(str),
        "labels": data["labels"].astype(np.int64),
        "folds": data["folds"].astype(np.int32),
        "blocks": {},
    }
    for block in ["evidence_markers", "stylistic", "relevance_counts", "temporal_markers", "mean", "rel3", "rel6", "rel7"]:
        if block in data.files:
            out["blocks"][block] = data[block].astype(np.float32)
    data.close()
    return out


def reorder_raw_split(split, user_ids: list[str], labels=None, folds=None):
    by_user = {uid: index for index, uid in enumerate(split["user_ids"])}
    missing = [uid for uid in user_ids if uid not in by_user]
    if missing:
        raise RuntimeError(f"raw feature NPZ missing users: {missing[:5]}")
    indexes = np.asarray([by_user[uid] for uid in user_ids], dtype=np.int64)
    out = {
        "user_ids": np.asarray(user_ids),
        "blocks": {name: values[indexes] for name, values in split["blocks"].items()},
    }
    if labels is not None:
        out["labels"] = np.asarray(labels, dtype=np.int64)
    if folds is not None:
        out["folds"] = np.asarray(folds, dtype=np.int32)
    return out


def load_all_raw_features(cfg, train_manifest_rows, required_blocks: set[str]):
    unsupported = sorted(
        block for block in required_blocks if block not in {"evidence_markers", "stylistic", "relevance_counts", "temporal_markers", *EMBEDDING_BLOCKS}
    )
    if unsupported:
        raise RuntimeError(f"raw_artifacts featureSource does not support blocks: {unsupported}")
    required_sources = {EMBEDDING_BLOCKS.get(block, block) for block in required_blocks}
    feature_dir = Path(cfg["outputDir"]) / "features"
    raw_train = load_raw_feature_npz(feature_dir / "train_raw_features.npz")
    raw_test = load_raw_feature_npz(feature_dir / "test_raw_features.npz")
    train_ids = [row["user_id"] for row in train_manifest_rows]
    train_labels = [1 if row["label"] == "diagnosed" else 0 for row in train_manifest_rows]
    train_folds = [int(row["fold"]) for row in train_manifest_rows]
    test_rows, _hash = read_csv_with_hash(Path(cfg["outputDir"]) / "manifest" / f"test_inference_manifest_seed{cfg['seed']}.csv")
    test_ids = [row["user_id"] for row in test_rows]
    train = reorder_raw_split(raw_train, train_ids, train_labels, train_folds)
    split_test = reorder_raw_split(raw_test, test_ids)
    if required_sources - set(train["blocks"]):
        raise RuntimeError(f"raw train features missing blocks: {sorted(required_sources - set(train['blocks']))}")
    if required_sources - set(split_test["blocks"]):
        raise RuntimeError(f"raw test features missing blocks: {sorted(required_sources - set(split_test['blocks']))}")
    return train, split_test


def subset(split, mask):
    out = {key: value[mask] for key, value in split.items() if key in {"user_ids", "labels", "folds"}}
    out["blocks"] = {name: value[mask] for name, value in split["blocks"].items()}
    return out


def build_features(train, val, split_test, candidate):
    train_pieces = []
    val_pieces = []
    test_pieces = []
    pca_components = int(candidate.get("pcaComponents", 80))
    for offset, block_name in enumerate(candidate["featureBlocks"]):
        if block_name in EMBEDDING_BLOCKS:
            source = EMBEDDING_BLOCKS[block_name]
            n_components = min(pca_components, len(train["labels"]) - 1, train["blocks"][source].shape[1])
            if n_components < 1:
                raise RuntimeError(f"{candidate['modelId']} cannot fit PCA for {block_name}")
            pca = PCA(n_components=n_components, random_state=int(candidate["seed"]) + offset * 101)
            train_pieces.append(pca.fit_transform(train["blocks"][source]))
            val_pieces.append(pca.transform(val["blocks"][source]))
            test_pieces.append(pca.transform(split_test["blocks"][source]))
        else:
            train_pieces.append(train["blocks"][block_name])
            val_pieces.append(val["blocks"][block_name])
            test_pieces.append(split_test["blocks"][block_name])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.nan_to_num(np.hstack(train_pieces), copy=False))
    x_val = scaler.transform(np.nan_to_num(np.hstack(val_pieces), copy=False))
    x_test = scaler.transform(np.nan_to_num(np.hstack(test_pieces), copy=False))
    return x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(np.float32)


def positive_prob(model, probs, count: int):
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        return np.ones(count, dtype=np.float64) if classes == [1] else np.zeros(count, dtype=np.float64)
    return probs[:, classes.index(1)]


def balanced_sample_weights(y_train):
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    class_weights = len(y_train) / (2.0 * np.maximum(counts, 1.0))
    return class_weights[y_train].astype(np.float32)


def fit_predict(candidate, x_train, y_train, x_val, x_test):
    family = candidate["family"]
    seed = int(candidate["seed"])
    if family in {"logreg", "hierarchical_logreg"}:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs")
        model.fit(x_train, y_train)
        return positive_prob(model, model.predict_proba(x_val), len(x_val)), positive_prob(model, model.predict_proba(x_test), len(x_test))
    if family == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(int(candidate["hiddenSize"]),),
            alpha=float(candidate["alpha"]),
            random_state=seed,
            max_iter=500,
            early_stopping=False,
        )
        model.fit(x_train, y_train)
        return positive_prob(model, model.predict_proba(x_val), len(x_val)), positive_prob(model, model.predict_proba(x_test), len(x_test))
    if family == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=int(candidate.get("nEstimators", 400)),
            max_depth=int(candidate.get("maxDepth", 5)),
            min_samples_leaf=int(candidate.get("minSamplesLeaf", 10)),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return positive_prob(model, model.predict_proba(x_val), len(x_val)), positive_prob(model, model.predict_proba(x_test), len(x_test))
    if family == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is required for binary xgboost candidates. Install requirements.txt.")
        model = XGBClassifier(
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
        model.fit(x_train, y_train, sample_weight=balanced_sample_weights(y_train))
        return positive_prob(model, model.predict_proba(x_val), len(x_val)), positive_prob(model, model.predict_proba(x_test), len(x_test))
    if family == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=int(candidate.get("maxIter", 220)),
            learning_rate=float(candidate.get("learningRate", 0.04)),
            max_leaf_nodes=int(candidate.get("maxLeafNodes", 15)),
            max_depth=int(candidate["maxDepth"]) if candidate.get("maxDepth") is not None else None,
            l2_regularization=float(candidate.get("l2Regularization", 0.1)),
            early_stopping=False,
            random_state=seed,
        )
        model.fit(x_train, y_train, sample_weight=balanced_sample_weights(y_train))
        return positive_prob(model, model.predict_proba(x_val), len(x_val)), positive_prob(model, model.predict_proba(x_test), len(x_test))
    if family in {"focal_linear", "focal_logreg"}:
        return fit_focal_predict(x_train, y_train, x_val, x_test, float(candidate.get("gamma", 1.0)), seed)
    if family == "relevance_baseline":
        return relevance_baseline_score(candidate, x_val), relevance_baseline_score(candidate, x_test)
    raise RuntimeError(f"Unsupported binary tabular family: {family}")


def fit_focal_predict(x_train, y_train, x_val, x_test, gamma: float, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    x = torch.from_numpy(x_train).to(device)
    y = torch.from_numpy(y_train.astype(np.float32)).to(device)
    model = torch.nn.Linear(x_train.shape[1], 1).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    positives = max(float((y_train == 1).sum()), 1.0)
    negatives = max(float((y_train == 0).sum()), 1.0)
    pos_weight = len(y_train) / (2.0 * positives)
    neg_weight = len(y_train) / (2.0 * negatives)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for _epoch in range(250):
        opt.zero_grad()
        logits = model(x).squeeze(1).clamp(-30.0, 30.0)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt = torch.exp(-bce)
        weights = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.full_like(y, neg_weight))
        loss = (((1.0 - pt) ** gamma) * bce * weights).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("binary focal linear produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(x_val).to(device)).squeeze(1).clamp(-30.0, 30.0)
        test_logits = model(torch.from_numpy(x_test).to(device)).squeeze(1).clamp(-30.0, 30.0)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
    return clean_probs(val_probs), clean_probs(test_probs)


def relevance_baseline_score(_candidate, features):
    return clean_probs(features[:, EVIDENCE_SCORE_INDEX])


def clean_probs(values):
    return np.clip(np.nan_to_num(values.astype(np.float64), nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)


def write_scores(path: Path, rows, include_labels: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["user_id", "label", "fold", "score", "model_id"] if include_labels else ["user_id", "score", "model_id"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in headers})


def feature_cache_key(candidate, fold: int) -> tuple:
    return (tuple(candidate["featureBlocks"]), int(candidate.get("pcaComponents", 80)), fold)


def train_candidate(cfg, manifest_hash: str, train_manifest_hash: str, train_all, split_test, candidate, feature_cache) -> None:
    model_id = candidate["modelId"]
    folds = sorted(set(int(fold) for fold in train_all["folds"]))
    oof_rows = []
    test_sum = np.zeros(len(split_test["user_ids"]), dtype=np.float64)
    test_count = 0
    for fold in folds:
        train_mask = train_all["folds"] != fold
        val_mask = train_all["folds"] == fold
        train = subset(train_all, train_mask)
        val = subset(train_all, val_mask)
        fold_candidate = {**candidate, "seed": int(candidate["seed"]) + fold}
        if candidate["family"] == "relevance_baseline":
            val_probs = clean_probs(val["blocks"]["evidence_markers"][:, EVIDENCE_SCORE_INDEX])
            test_probs = clean_probs(split_test["blocks"]["evidence_markers"][:, EVIDENCE_SCORE_INDEX])
        else:
            cache_key = feature_cache_key(fold_candidate, fold)
            if cache_key not in feature_cache:
                feature_cache[cache_key] = build_features(train, val, split_test, fold_candidate)
            x_train, x_val, x_test = feature_cache[cache_key]
            val_probs, test_probs = fit_predict(fold_candidate, x_train, train["labels"], x_val, x_test)
        for uid, label, prob in zip(val["user_ids"], val["labels"], val_probs):
            oof_rows.append(
                {
                    "user_id": uid,
                    "label": "diagnosed" if label == 1 else "control",
                    "fold": fold,
                    "score": f"{float(prob):.8f}",
                    "model_id": model_id,
                }
            )
        test_sum += test_probs
        test_count += 1

    oof_rows.sort(key=lambda row: row["user_id"])
    test_avg = clean_probs(test_sum / max(test_count, 1))
    test_rows = [
        {"user_id": uid, "score": f"{float(prob):.8f}", "model_id": model_id}
        for uid, prob in zip(split_test["user_ids"], test_avg)
    ]
    out_dir = Path(cfg["outputDir"])
    write_scores(out_dir / "scores" / f"train_oof_{model_id}.csv", oof_rows, True)
    write_scores(out_dir / "scores" / f"test_score_{model_id}.csv", test_rows, False)
    model_manifest = {
        "modelId": model_id,
        "candidate": True,
        "family": candidate["family"],
        "seed": candidate["seed"],
        "manifestHash": manifest_hash,
        "trainManifestHash": train_manifest_hash,
        "featureSource": cfg.get("featureSource", "raw_artifacts"),
        "rawArtifactsDir": cfg.get("rawArtifactsDir"),
        "dbTables": {},
        "featureBlocks": candidate["featureBlocks"],
        "hyperparameters": {key: value for key, value in candidate.items() if key not in {"modelId", "family", "featureBlocks"}},
        "scoreSchema": "binary-score-v1",
        "usesTestLabelsForTraining": False,
        "gpuUsed": False,
        "createdAt": "1970-01-01T00:00:00.000Z",
    }
    manifest_path = out_dir / "model-manifests" / f"{model_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(model_manifest, indent=2) + "\n")
    print(f"wrote {model_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setembrobr.seed42.raw-qwen3-binary.json")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    if cfg.get("featureSource") != "raw_artifacts":
        raise SystemExit("binary_train_tabular_oof_setembrobr.py is scoped to raw_artifacts configs")

    candidates = list(cfg.get("candidateModels", {}).get("tabular", []))
    if args.only:
        wanted = set(args.only)
        candidates = [candidate for candidate in candidates if candidate["modelId"] in wanted]
    if not candidates:
        raise SystemExit("No binary tabular candidates selected")

    train_manifest_path = Path(cfg["outputDir"]) / "manifest" / f"train_binary_manifest_seed{cfg['seed']}.csv"
    train_manifest_rows, train_manifest_hash = read_csv_with_hash(train_manifest_path)
    required_blocks = {block for candidate in candidates for block in candidate["featureBlocks"]}
    train_all, split_test = load_all_raw_features(cfg, train_manifest_rows, required_blocks)
    feature_cache = {}
    for candidate in candidates:
        train_candidate(cfg, strict_blind_manifest_hash(cfg), train_manifest_hash, train_all, split_test, candidate, feature_cache)


if __name__ == "__main__":
    main()
