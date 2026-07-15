#!/usr/bin/env python3
"""Create strictly nested stack OOF scores and post-lock label-free test scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from anxiety_champion_relevance import load_config, require_post_lock, resolve, sha256_file, write_json
from anxiety_champion_tabular import (
    apply_transform,
    fit_estimator,
    fit_transform,
    load_features,
    predict,
    reorder,
    subset,
    write_scores,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text().splitlines()))


def fit_meta(candidate: dict[str, Any], features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        C=float(candidate.get("c", 1.0)),
        max_iter=int(candidate.get("maxIter", 1000)),
        class_weight="balanced",
        random_state=int(candidate["seed"]),
        solver="lbfgs",
    )
    model.fit(features, labels)
    return model


def positive_probability(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    probabilities = model.predict_proba(features)
    return probabilities[:, classes.index(1)]


def nested_base_predictions(
    base_candidate: dict[str, Any],
    train: dict[str, Any],
    outer_fold: int,
    checkpoint_root: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    outer_train_mask = train["folds"] != outer_fold
    outer_validation_mask = train["folds"] == outer_fold
    outer_training = subset(train, outer_train_mask)
    outer_validation = subset(train, outer_validation_mask)
    inner_folds = sorted(set(outer_training["folds"].tolist()))
    inner_oof = np.full(len(outer_training["labels"]), np.nan, dtype=np.float64)
    outer_sum = np.zeros(len(outer_validation["labels"]), dtype=np.float64)
    provenance = []
    for inner_fold in inner_folds:
        fit_mask = outer_training["folds"] != inner_fold
        validation_mask = outer_training["folds"] == inner_fold
        inner_training = subset(outer_training, fit_mask)
        inner_validation = subset(outer_training, validation_mask)
        fold_candidate = {
            **base_candidate,
            "seed": int(base_candidate["seed"]) + outer_fold * 100 + inner_fold,
        }
        transform, x_train = fit_transform(inner_training, fold_candidate)
        model = fit_estimator(fold_candidate, x_train, inner_training["labels"])
        inner_oof[validation_mask] = predict(model, apply_transform(inner_validation, transform))
        outer_sum += predict(model, apply_transform(outer_validation, transform))
        checkpoint = (
            checkpoint_root
            / base_candidate["modelId"]
            / f"outer-{outer_fold}"
            / f"inner-{inner_fold}.joblib"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"candidate": fold_candidate, "transform": transform, "model": model}, checkpoint)
        fit_folds = sorted(set(inner_training["folds"].tolist()))
        provenance.append(
            {
                "outerFold": outer_fold,
                "innerValidationFold": inner_fold,
                "baseModelId": base_candidate["modelId"],
                "fitFolds": fit_folds,
                "outerFoldExcluded": outer_fold not in fit_folds,
                "innerValidationFoldExcluded": inner_fold not in fit_folds,
                "checkpointSha256": sha256_file(checkpoint),
            }
        )
    if np.isnan(inner_oof).any():
        raise RuntimeError(f"nested OOF gap for {base_candidate['modelId']} outer fold {outer_fold}")
    return inner_oof, outer_sum / len(inner_folds), provenance


def train_oof(config: dict[str, Any], repo: Path) -> None:
    output_dir = resolve(config["outputDir"], repo)
    work_dir = resolve(config.get("workDir", config["outputDir"]), repo)
    manifest_path = output_dir / "manifest" / f"train_binary_manifest_seed{config['seed']}.csv"
    manifest = read_csv(manifest_path)
    user_ids = [row["user_id"] for row in manifest]
    labels = [1 if row["label"] == "diagnosed" else 0 for row in manifest]
    folds = [int(row["fold"]) for row in manifest]
    train = reorder(load_features(work_dir / "features" / "train_raw_features.npz"), user_ids, labels, folds)
    tabular_by_id = {candidate["modelId"]: candidate for candidate in config["candidateModels"]["tabular"]}
    stack_candidates = config["candidateModels"]["stacking"]
    all_base_ids = sorted({model_id for candidate in stack_candidates for model_id in candidate["baseModelIds"]})
    missing = [model_id for model_id in all_base_ids if model_id not in tabular_by_id]
    if missing:
        raise RuntimeError(f"nested stack base configuration missing: {missing}")

    strict_hash = sha256_file(output_dir / "manifest" / f"strict_blind_split_manifest_seed{config['seed']}.csv")
    proxy_hash = sha256_file(output_dir / "relevance-proxy" / "proxy-definition.json")
    checkpoint_root = output_dir / "checkpoints" / "nested-stack-bases"
    outer_payloads: dict[int, dict[str, Any]] = {}
    all_provenance = []
    for outer_fold in sorted(set(folds)):
        outer_training = subset(train, train["folds"] != outer_fold)
        outer_validation = subset(train, train["folds"] == outer_fold)
        inner_columns = {}
        validation_columns = {}
        for base_id in all_base_ids:
            inner_oof, validation_average, provenance = nested_base_predictions(
                tabular_by_id[base_id], train, outer_fold, checkpoint_root
            )
            inner_columns[base_id] = inner_oof
            validation_columns[base_id] = validation_average
            all_provenance.extend(provenance)
        outer_payloads[outer_fold] = {
            "training": outer_training,
            "validation": outer_validation,
            "innerColumns": inner_columns,
            "validationColumns": validation_columns,
        }
        print(f"built nested base features for outer fold {outer_fold}")

    provenance_path = output_dir / "reports" / "nested-stacking-provenance.json"
    write_json(
        provenance_path,
        {
            "predictionTarget": "anxiety",
            "outerFolds": sorted(set(folds)),
            "records": all_provenance,
            "allOuterFoldsExcluded": all(record["outerFoldExcluded"] for record in all_provenance),
            "allInnerValidationFoldsExcluded": all(
                record["innerValidationFoldExcluded"] for record in all_provenance
            ),
        },
    )

    for candidate in stack_candidates:
        model_id = candidate["modelId"]
        rows = []
        checkpoint_hashes = {}
        for outer_fold, payload in outer_payloads.items():
            base_ids = candidate["baseModelIds"]
            x_train = np.column_stack([payload["innerColumns"][base_id] for base_id in base_ids])
            x_validation = np.column_stack([payload["validationColumns"][base_id] for base_id in base_ids])
            model = fit_meta({**candidate, "seed": int(candidate["seed"]) + outer_fold}, x_train, payload["training"]["labels"])
            probabilities = positive_probability(model, x_validation)
            checkpoint = output_dir / "checkpoints" / "stacking" / model_id / f"outer-{outer_fold}.joblib"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"candidate": candidate, "baseModelIds": base_ids, "model": model}, checkpoint)
            checkpoint_hashes[str(outer_fold)] = sha256_file(checkpoint)
            for user_id, label, probability in zip(
                payload["validation"]["user_ids"], payload["validation"]["labels"], probabilities
            ):
                rows.append(
                    {
                        "user_id": user_id,
                        "label": "diagnosed" if int(label) == 1 else "control",
                        "fold": outer_fold,
                        "score": f"{float(probability):.8f}",
                        "model_id": model_id,
                    }
                )
        write_scores(output_dir / "scores" / f"train_oof_{model_id}.csv", sorted(rows, key=lambda row: row["user_id"]), True)
        write_json(
            output_dir / "model-manifests" / f"{model_id}.json",
            {
                "modelId": model_id,
                "candidate": True,
                "supportOnly": False,
                "family": candidate["family"],
                "seed": candidate["seed"],
                "predictionTarget": "anxiety",
                "featureSource": "raw_artifacts",
                "manifestHash": strict_hash,
                "trainManifestHash": sha256_file(manifest_path),
                "relevanceProxyKind": config["relevanceProxy"]["kind"],
                "relevanceProxyDefinitionHash": proxy_hash,
                "baseModelIds": candidate["baseModelIds"],
                "checkpointHashes": checkpoint_hashes,
                "artifactHashes": {
                    "strictBlindManifestSha256": strict_hash,
                    "trainManifestSha256": sha256_file(manifest_path),
                    "trainFeatureManifestSha256": sha256_file(work_dir / "features" / "train-feature-manifest.json"),
                    "relevanceProxyDefinitionSha256": proxy_hash,
                    "nestedStackingProvenanceSha256": sha256_file(provenance_path),
                },
                "nestedStackingProvenanceHash": sha256_file(provenance_path),
                "nestedCrossFitting": True,
                "scoreSchema": "binary-score-v1",
                "usesTestLabelsForTraining": False,
                "usesTestScoresForTraining": False,
                "testArtifactsReadDuringOof": False,
                "createdAt": "1970-01-01T00:00:00.000Z",
            },
        )
        print(f"wrote nested train OOF {model_id}")


def read_score_vector(path: Path, user_ids: list[str], expected_model_id: str, train: bool) -> np.ndarray:
    rows = read_csv(path)
    expected_headers = {"user_id", "label", "fold", "score", "model_id"} if train else {"user_id", "score", "model_id"}
    if not rows or set(rows[0]) != expected_headers:
        raise RuntimeError(f"unexpected score schema: {path}")
    by_user = {row["user_id"]: row for row in rows}
    if len(by_user) != len(user_ids) or set(by_user) != set(user_ids):
        raise RuntimeError(f"score users do not match manifest: {path}")
    if any(row["model_id"] != expected_model_id for row in rows):
        raise RuntimeError(f"model id mismatch in {path}")
    return np.asarray([float(by_user[user_id]["score"]) for user_id in user_ids], dtype=np.float64)


def score_test(config: dict[str, Any], repo: Path) -> None:
    output_dir = resolve(config["outputDir"], repo)
    require_post_lock(output_dir)
    train_manifest = read_csv(output_dir / "manifest" / f"train_binary_manifest_seed{config['seed']}.csv")
    test_manifest = read_csv(output_dir / "manifest" / f"test_inference_manifest_seed{config['seed']}.csv")
    train_user_ids = [row["user_id"] for row in train_manifest]
    test_user_ids = [row["user_id"] for row in test_manifest]
    labels = np.asarray([1 if row["label"] == "diagnosed" else 0 for row in train_manifest], dtype=np.int64)
    for candidate in config["candidateModels"]["stacking"]:
        base_ids = candidate["baseModelIds"]
        train_features = np.column_stack(
            [
                read_score_vector(output_dir / "scores" / f"train_oof_{model_id}.csv", train_user_ids, model_id, True)
                for model_id in base_ids
            ]
        )
        test_features = np.column_stack(
            [
                read_score_vector(output_dir / "scores" / f"test_score_{model_id}.csv", test_user_ids, model_id, False)
                for model_id in base_ids
            ]
        )
        model = fit_meta(candidate, train_features, labels)
        probabilities = positive_probability(model, test_features)
        rows = [
            {"user_id": user_id, "score": f"{float(probability):.8f}", "model_id": candidate["modelId"]}
            for user_id, probability in zip(test_user_ids, probabilities)
        ]
        write_scores(output_dir / "scores" / f"test_score_{candidate['modelId']}.csv", rows, False)
        print(f"wrote label-free stack test score {candidate['modelId']}")


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
