#!/usr/bin/env python3
"""Train strict-blind ternary stacking models from train OOF probabilities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

LABELS = ["diagnosed", "control", "no-evidence"]
LABEL_TO_CODE = {label: index for index, label in enumerate(LABELS)}


def read_csv_with_hash(path: Path):
    text = path.read_text()
    return list(csv.DictReader(text.splitlines())), hashlib.sha256(text.encode()).hexdigest()


def read_policy_lock(cfg, policy_id: str):
    return json.loads((Path(cfg["outputDir"]) / "label-policies" / f"{policy_id}.json").read_text())


def read_train_manifest_hash(cfg, policy_id: str) -> str:
    path = Path(cfg["outputDir"]) / "manifest" / f"train_manifest_{policy_id}_seed{cfg['seed']}.csv"
    _rows, digest = read_csv_with_hash(path)
    return digest


def read_score_rows(path: Path):
    rows, digest = read_csv_with_hash(path)
    for row in rows:
        for key in ("prob_diagnosed", "prob_control", "prob_no_evidence"):
            row[key] = float(row[key])
    return rows, digest


def score_path(cfg, prefix: str, policy_id: str, model_id: str) -> Path:
    return Path(cfg["outputDir"]) / "scores" / f"{prefix}_{policy_id}_{model_id}.csv"


def align_train_scores(cfg, policy_id: str, base_model_ids: list[str]):
    by_model = {}
    hashes = {}
    for model_id in base_model_ids:
        path = score_path(cfg, "train_oof", policy_id, model_id)
        rows, digest = read_score_rows(path)
        hashes[model_id] = digest
        by_model[model_id] = {row["user_id"]: row for row in rows}

    expected_user_ids = set(next(iter(by_model.values())).keys())
    for model_id, rows_by_user in by_model.items():
        if set(rows_by_user.keys()) != expected_user_ids:
            raise RuntimeError(f"{policy_id}/{model_id}: train OOF user set mismatch")
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
                raise RuntimeError(f"{policy_id}/{model_id}: missing train OOF user {user_id}")
            if row.get("label") != label or int(row.get("fold", -1)) != fold:
                raise RuntimeError(f"{policy_id}/{model_id}: train OOF alignment mismatch for {user_id}")
            row_features.extend([row["prob_diagnosed"], row["prob_control"], row["prob_no_evidence"]])
        features.append(row_features)
        labels.append(LABEL_TO_CODE[label])
        folds.append(fold)
    return user_ids, np.array(features, dtype=np.float64), np.array(labels, dtype=np.int64), np.array(folds, dtype=np.int32), hashes


def align_score_features(cfg, policy_id: str, base_model_ids: list[str]):
    by_model = {}
    hashes = {}
    for model_id in base_model_ids:
        path = score_path(cfg, "test_score", policy_id, model_id)
        rows, digest = read_score_rows(path)
        hashes[model_id] = digest
        by_model[model_id] = {row["user_id"]: row for row in rows}

    expected_user_ids = set(next(iter(by_model.values())).keys())
    for model_id, rows_by_user in by_model.items():
        if set(rows_by_user.keys()) != expected_user_ids:
            raise RuntimeError(f"{policy_id}/{model_id}: label-free score user set mismatch")
    user_ids = sorted(expected_user_ids)
    features = []
    for user_id in user_ids:
        row_features = []
        for model_id in base_model_ids:
            row = by_model[model_id].get(user_id)
            if row is None:
                raise RuntimeError(f"{policy_id}/{model_id}: missing label-free score user {user_id}")
            if "label" in row or "fold" in row:
                raise RuntimeError(f"{policy_id}/{model_id}: label-free score contains forbidden columns")
            row_features.extend([row["prob_diagnosed"], row["prob_control"], row["prob_no_evidence"]])
        features.append(row_features)
    return user_ids, np.array(features, dtype=np.float64), hashes


def balanced_sample_weights(y):
    counts = np.bincount(y, minlength=3).astype(np.float64)
    class_weights = len(y) / (3.0 * np.maximum(counts, 1.0))
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


def map_probs(model, probs):
    out = np.zeros((probs.shape[0], 3), dtype=np.float64)
    for column, code in enumerate(model.classes_):
        out[:, int(code)] = probs[:, column]
    out = np.clip(out, 1e-9, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def write_scores(path: Path, rows, include_labels: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = (
        ["user_id", "label", "fold", "prob_diagnosed", "prob_control", "prob_no_evidence", "model_id", "label_policy_id"]
        if include_labels
        else ["user_id", "prob_diagnosed", "prob_control", "prob_no_evidence", "model_id", "label_policy_id"]
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in headers})


def train_candidate(cfg, policy_id: str, candidate) -> None:
    model_id = candidate["modelId"]
    base_model_ids = list(candidate["baseModelIds"])
    train_users, x_train_all, y_all, folds, train_hashes = align_train_scores(cfg, policy_id, base_model_ids)
    score_users, x_score, score_hashes = align_score_features(cfg, policy_id, base_model_ids)

    oof_rows = []
    for fold in sorted(set(int(value) for value in folds)):
        train_mask = folds != fold
        validation_mask = folds == fold
        model = fit_model(candidate, x_train_all[train_mask], y_all[train_mask])
        probs = map_probs(model, model.predict_proba(x_train_all[validation_mask]))
        for user_id, label_code, prob in zip(np.array(train_users)[validation_mask], y_all[validation_mask], probs):
            oof_rows.append(
                {
                    "user_id": user_id,
                    "label": LABELS[int(label_code)],
                    "fold": fold,
                    "prob_diagnosed": f"{float(prob[LABEL_TO_CODE['diagnosed']]):.8f}",
                    "prob_control": f"{float(prob[LABEL_TO_CODE['control']]):.8f}",
                    "prob_no_evidence": f"{float(prob[LABEL_TO_CODE['no-evidence']]):.8f}",
                    "model_id": model_id,
                    "label_policy_id": policy_id,
                }
            )

    model = fit_model(candidate, x_train_all, y_all)
    score_probs = map_probs(model, model.predict_proba(x_score))
    score_rows = [
        {
            "user_id": user_id,
            "prob_diagnosed": f"{float(prob[LABEL_TO_CODE['diagnosed']]):.8f}",
            "prob_control": f"{float(prob[LABEL_TO_CODE['control']]):.8f}",
            "prob_no_evidence": f"{float(prob[LABEL_TO_CODE['no-evidence']]):.8f}",
            "model_id": model_id,
            "label_policy_id": policy_id,
        }
        for user_id, prob in zip(score_users, score_probs)
    ]

    policy_lock = read_policy_lock(cfg, policy_id)
    artifact_id = f"{policy_id}_{model_id}"
    out_dir = Path(cfg["outputDir"])
    write_scores(out_dir / "scores" / f"train_oof_{artifact_id}.csv", sorted(oof_rows, key=lambda row: row["user_id"]), True)
    write_scores(out_dir / "scores" / f"test_score_{artifact_id}.csv", score_rows, False)
    manifest = {
        "modelId": model_id,
        "artifactId": artifact_id,
        "candidate": True,
        "family": candidate["family"],
        "seed": candidate["seed"],
        "originalManifestHash": policy_lock["originalManifestHash"],
        "trainManifestHash": read_train_manifest_hash(cfg, policy_id),
        "labelPolicyId": policy_id,
        "labelPolicyHash": policy_lock["policyHash"],
        "baseModelIds": base_model_ids,
        "sourceScoreHashes": {"trainOof": train_hashes, "labelFreeScore": score_hashes},
        "featureBlocks": ["train_oof_probabilities"],
        "hyperparameters": {key: value for key, value in candidate.items() if key not in {"modelId", "family", "baseModelIds"}},
        "scoreSchema": "ternary-probability-v1",
        "usesTestLabelsForTraining": False,
        "usesTestScoresForTraining": False,
        "gpuUsed": False,
        "createdAt": "1970-01-01T00:00:00.000Z",
    }
    manifest_path = out_dir / "model-manifests" / f"{artifact_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {artifact_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ternary-classification/configs/setembrobr.seed42.ternary-strict-blind.json")
    parser.add_argument("--policy", nargs="*", default=None)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    candidates = list(cfg.get("candidateModels", {}).get("stacking", []))
    if args.only:
        wanted = set(args.only)
        candidates = [candidate for candidate in candidates if candidate["modelId"] in wanted]
    if not candidates:
        raise SystemExit("No ternary stacking candidates selected")

    policies = [policy["policyId"] for policy in cfg["labelPolicies"]]
    if args.policy:
        wanted_policies = set(args.policy)
        policies = [policy for policy in policies if policy in wanted_policies]
    if not policies:
        raise SystemExit("No ternary label policies selected")

    for policy_id in policies:
        for candidate in candidates:
            train_candidate(cfg, policy_id, candidate)


if __name__ == "__main__":
    main()
