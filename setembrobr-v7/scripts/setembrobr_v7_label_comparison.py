#!/usr/bin/env python3
"""Paired original-diagnosis versus specialist-signal LogReg comparison on v7."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from setembrobr_v7_qwen_logreg import (
    aligned_embeddings,
    bootstrap_intervals,
    build_classifier,
    metrics_from_codes,
    positive_probability,
    read_csv,
    require_sha256,
    select_threshold,
    sha256_file,
    write_csv,
    write_json,
)


TARGETS = ("original_diagnosis", "specialist_signal")
TARGET_MODEL_IDS = {
    "original_diagnosis": "qwen3_mean_logreg_s42_original_diagnosis",
    "specialist_signal": "qwen3_mean_logreg_s42_specialist_signal",
}
GENERIC_LABEL_TO_CODE = {"no": 0, "yes": 1}


@dataclass(frozen=True)
class ComparisonPaths:
    root: Path

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def sealed(self) -> Path:
        return self.root / "sealed"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def locks(self) -> Path:
        return self.root / "locks"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> None:
        for path in [
            self.manifests,
            self.sealed,
            self.scores,
            self.models,
            self.locks,
            self.reports,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    config["_configPath"] = str(path.resolve())
    config["_configSha256"] = sha256_file(path)
    return config


def target_train_path(output: ComparisonPaths, target: str) -> Path:
    return output.manifests / f"train_{target}.csv"


def target_sealed_path(output: ComparisonPaths, target: str) -> Path:
    return output.sealed / f"test_labels_{target}.csv"


def target_oof_path(output: ComparisonPaths, target: str) -> Path:
    return output.scores / f"train_oof_{TARGET_MODEL_IDS[target]}.csv"


def target_test_score_path(output: ComparisonPaths, target: str) -> Path:
    return output.scores / f"test_{TARGET_MODEL_IDS[target]}.csv"


def target_lock_path(output: ComparisonPaths, target: str) -> Path:
    return output.locks / f"{target}.json"


def normalize_original_label(value: Any) -> str:
    label = str(value).strip().lower()
    if label == "yes":
        return "yes"
    if label == "no":
        return "no"
    raise RuntimeError(f"unsupported original diagnosis label: {value!r}")


def normalize_specialist_label(value: Any) -> str:
    label = str(value).strip().lower()
    if label == "signal":
        return "yes"
    if label == "no_signal":
        return "no"
    raise RuntimeError(f"unsupported specialist label: {value!r}")


def generic_metrics(metrics: dict[str, Any], positive_name: str) -> dict[str, Any]:
    result = dict(metrics)
    result["positiveF1"] = result.pop("diagnosedF1")
    result["positiveLabel"] = positive_name
    result["confusionMatrix"] = {
        "labels": [f"not_{positive_name}", positive_name],
        "matrix": metrics["confusionMatrix"]["matrix"],
    }
    return result


def generic_intervals(intervals: dict[str, list[float]]) -> dict[str, list[float]]:
    result = dict(intervals)
    result["positiveF1"] = result.pop("diagnosedF1")
    return result


def validate_count_contract(
    config: dict[str, Any], target: str, split: str, labels: list[str]
) -> None:
    expected = config["expectedTargets"][target]
    positive = labels.count("yes")
    negative = labels.count("no")
    if positive != int(expected[f"{split}Positive"]):
        raise RuntimeError(f"{target} {split} positive count mismatch")
    if negative != int(expected[f"{split}Negative"]):
        raise RuntimeError(f"{target} {split} negative count mismatch")


def prepare_command(
    source_path: Path,
    specialist_path: Path,
    baseline_output: Path,
    config: dict[str, Any],
    output: ComparisonPaths,
) -> None:
    output.ensure()
    require_sha256(source_path, config["source"]["sha256"], "v7 source corpus")
    require_sha256(
        specialist_path,
        config["specialistLabels"]["sha256"],
        "specialist relabeled corpus",
    )
    pinned = config["baselineArtifacts"]
    baseline_files = {
        "preparedManifestSha256": baseline_output / "manifests/prepared-manifest.json",
        "trainManifestSha256": baseline_output / "manifests/train_manifest.csv",
        "testInferenceSha256": baseline_output / "manifests/test_inference.csv",
        "trainPoolSha256": baseline_output / "embeddings/train_user_mean.npz",
        "testPoolSha256": baseline_output / "embeddings/test_user_mean.npz",
    }
    for name, path in baseline_files.items():
        require_sha256(path, pinned[name], name)

    source = pd.read_pickle(source_path)
    expected_columns = ["User_ID", "Diagnosed_YN", "TextLists", "Split"]
    if list(source.columns) != expected_columns:
        raise RuntimeError(f"v7 source schema mismatch: {list(source.columns)}")
    if len(source) != int(config["source"]["expectedUsers"]):
        raise RuntimeError("v7 source user count mismatch")
    if source["User_ID"].duplicated().any():
        raise RuntimeError("v7 source contains duplicate users")
    source_by_user = source.set_index("User_ID", drop=False)

    specialist = pd.read_csv(specialist_path, dtype=str)
    user_column = config["specialistLabels"]["userColumn"]
    label_column = config["specialistLabels"]["labelColumn"]
    if user_column not in specialist or label_column not in specialist:
        raise RuntimeError("specialist corpus is missing its user or label column")
    if len(specialist) != int(config["specialistLabels"]["expectedRows"]):
        raise RuntimeError("specialist corpus row count mismatch")
    if specialist[user_column].duplicated().any():
        raise RuntimeError("specialist corpus contains duplicate users")
    specialist_counts = specialist[label_column].value_counts().to_dict()
    expected_specialist_counts = {
        "signal": int(config["specialistLabels"]["expectedSignal"]),
        "no_signal": int(config["specialistLabels"]["expectedNoSignal"]),
    }
    if specialist_counts != expected_specialist_counts:
        raise RuntimeError(f"specialist corpus label counts mismatch: {specialist_counts}")
    specialist_by_user = specialist.set_index(user_column)[label_column].to_dict()
    missing_specialist = sorted(set(source_by_user.index) - set(specialist_by_user))
    if missing_specialist:
        raise RuntimeError(f"specialist corpus is missing {len(missing_specialist)} v7 users")

    baseline_train = read_csv(baseline_files["trainManifestSha256"])
    baseline_test = read_csv(baseline_files["testInferenceSha256"])
    if not baseline_train or set(baseline_train[0]) != {"user_id", "label", "fold"}:
        raise RuntimeError("baseline train manifest schema mismatch")
    if not baseline_test or set(baseline_test[0]) != {"user_id"}:
        raise RuntimeError("baseline test inference schema mismatch")
    train_ids = [row["user_id"] for row in baseline_train]
    test_ids = [row["user_id"] for row in baseline_test]
    if len(train_ids) != int(config["source"]["trainUsers"]) or len(set(train_ids)) != len(
        train_ids
    ):
        raise RuntimeError("baseline train user coverage mismatch")
    if len(test_ids) != int(config["source"]["testUsers"]) or len(set(test_ids)) != len(test_ids):
        raise RuntimeError("baseline test user coverage mismatch")
    if set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != set(source_by_user.index):
        raise RuntimeError("baseline manifests do not partition v7 exactly")

    labels_by_target: dict[str, dict[str, str]] = {
        "original_diagnosis": {
            user_id: normalize_original_label(source_by_user.at[user_id, "Diagnosed_YN"])
            for user_id in source_by_user.index
        },
        "specialist_signal": {
            user_id: normalize_specialist_label(specialist_by_user[user_id])
            for user_id in source_by_user.index
        },
    }
    for row in baseline_train:
        if row["label"] != labels_by_target["original_diagnosis"][row["user_id"]]:
            raise RuntimeError("baseline train labels do not reproduce original Diagnosed_YN")

    fold_counts: dict[str, dict[str, dict[str, int]]] = {}
    sealed_hashes: dict[str, str] = {}
    train_hashes: dict[str, str] = {}
    for target in TARGETS:
        train_labels = [labels_by_target[target][user_id] for user_id in train_ids]
        test_labels = [labels_by_target[target][user_id] for user_id in test_ids]
        validate_count_contract(config, target, "train", train_labels)
        validate_count_contract(config, target, "test", test_labels)
        train_hashes[target] = write_csv(
            target_train_path(output, target),
            ["user_id", "label", "fold"],
            (
                {
                    "user_id": row["user_id"],
                    "label": labels_by_target[target][row["user_id"]],
                    "fold": row["fold"],
                }
                for row in baseline_train
            ),
        )
        sealed_hashes[target] = write_csv(
            target_sealed_path(output, target),
            ["user_id", "label"],
            (
                {"user_id": user_id, "label": labels_by_target[target][user_id]}
                for user_id in test_ids
            ),
        )
        target_folds: dict[str, dict[str, int]] = {}
        for fold in range(int(config["validation"]["foldCount"])):
            selected = [
                labels_by_target[target][row["user_id"]]
                for row in baseline_train
                if int(row["fold"]) == fold
            ]
            target_folds[str(fold)] = {
                "users": len(selected),
                "positive": selected.count("yes"),
                "negative": selected.count("no"),
            }
            if not target_folds[str(fold)]["positive"] or not target_folds[str(fold)]["negative"]:
                raise RuntimeError(f"{target} fold {fold} does not contain both classes")
        fold_counts[target] = target_folds

    test_inference_hash = write_csv(
        output.manifests / "test_inference.csv",
        ["user_id"],
        ({"user_id": user_id} for user_id in test_ids),
    )
    train_pairs = [
        (
            labels_by_target["original_diagnosis"][user_id],
            labels_by_target["specialist_signal"][user_id],
        )
        for user_id in train_ids
    ]
    test_pairs = [
        (
            labels_by_target["original_diagnosis"][user_id],
            labels_by_target["specialist_signal"][user_id],
        )
        for user_id in test_ids
    ]

    def pair_counts(pairs: list[tuple[str, str]]) -> dict[str, int]:
        return {
            f"original_{original}__specialist_{specialist_label}": pairs.count(
                (original, specialist_label)
            )
            for original in ("no", "yes")
            for specialist_label in ("no", "yes")
        }

    write_json(
        output.reports / "prepare-report.json",
        {
            "ok": True,
            "experimentId": config["experimentId"],
            "comparisonDesign": "same users, v7 timelines, split, folds, pooled vectors, and classifier; target labels differ",
            "configSha256": config["_configSha256"],
            "sourceCorpusSha256": sha256_file(source_path),
            "specialistCorpusSha256": sha256_file(specialist_path),
            "baselineArtifactHashes": {
                name: sha256_file(path) for name, path in baseline_files.items()
            },
            "trainManifestHashes": train_hashes,
            "testInferenceSha256": test_inference_hash,
            "sealedTestLabelHashes": sealed_hashes,
            "foldCounts": fold_counts,
            "labelPairCounts": {"train": pair_counts(train_pairs), "test": pair_counts(test_pairs)},
            "testLabelsSealedBeforeTraining": True,
        },
    )


def validate_prepared_inputs(
    baseline_output: Path, config: dict[str, Any], output: ComparisonPaths
) -> dict[str, Any]:
    report_path = output.reports / "prepare-report.json"
    report = json.loads(report_path.read_text())
    if not report.get("ok") or report.get("configSha256") != config["_configSha256"]:
        raise RuntimeError("label-comparison preparation did not pass for this config")
    require_sha256(
        baseline_output / "embeddings/train_user_mean.npz",
        config["baselineArtifacts"]["trainPoolSha256"],
        "baseline train pool",
    )
    require_sha256(
        baseline_output / "embeddings/test_user_mean.npz",
        config["baselineArtifacts"]["testPoolSha256"],
        "baseline test pool",
    )
    for target in TARGETS:
        require_sha256(
            target_train_path(output, target),
            report["trainManifestHashes"][target],
            f"{target} train manifest",
        )
    require_sha256(
        output.manifests / "test_inference.csv",
        report["testInferenceSha256"],
        "comparison test inference manifest",
    )
    return report


def train_oof_command(
    baseline_output: Path, config: dict[str, Any], output: ComparisonPaths
) -> None:
    output.ensure()
    validate_prepared_inputs(baseline_output, config, output)
    features_path = baseline_output / "embeddings/train_user_mean.npz"
    model_manifests: dict[str, Any] = {}
    for target in TARGETS:
        rows = read_csv(target_train_path(output, target))
        features = aligned_embeddings(rows, features_path, int(config["embedding"]["dimension"]))
        labels = np.asarray([GENERIC_LABEL_TO_CODE[row["label"]] for row in rows], dtype=np.int8)
        folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int16)
        scores = np.full(len(rows), np.nan, dtype=np.float64)
        artifacts = []
        target_model_dir = output.models / target
        target_model_dir.mkdir(parents=True, exist_ok=True)
        for fold in range(int(config["validation"]["foldCount"])):
            validation = folds == fold
            training = ~validation
            model = build_classifier(config)
            model.fit(features[training], labels[training])
            scores[validation] = positive_probability(model, features[validation])
            model_path = target_model_dir / f"fold_{fold}.joblib"
            joblib.dump(model, model_path)
            artifacts.append(
                {
                    "fold": fold,
                    "trainUsers": int(training.sum()),
                    "validationUsers": int(validation.sum()),
                    "modelSha256": sha256_file(model_path),
                }
            )
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise RuntimeError(f"{target} OOF scoring produced invalid probabilities")
        score_hash = write_csv(
            target_oof_path(output, target),
            ["user_id", "label", "fold", "score", "model_id"],
            (
                {
                    "user_id": row["user_id"],
                    "label": row["label"],
                    "fold": row["fold"],
                    "score": repr(float(score)),
                    "model_id": TARGET_MODEL_IDS[target],
                }
                for row, score in zip(rows, scores)
            ),
        )
        model_manifests[target] = {
            "modelId": TARGET_MODEL_IDS[target],
            "trainManifestSha256": sha256_file(target_train_path(output, target)),
            "trainPoolSha256": sha256_file(features_path),
            "oofScoreSha256": score_hash,
            "folds": artifacts,
        }
    write_json(
        output.manifests / "oof-model-manifest.json",
        {
            "experimentId": config["experimentId"],
            "configSha256": config["_configSha256"],
            "targets": model_manifests,
        },
    )


def audit_oof_command(
    baseline_output: Path, config: dict[str, Any], output: ComparisonPaths
) -> None:
    prepare = validate_prepared_inputs(baseline_output, config, output)
    model_manifest_path = output.manifests / "oof-model-manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    if model_manifest.get("configSha256") != config["_configSha256"]:
        raise RuntimeError("OOF models belong to another comparison config")
    target_audits: dict[str, Any] = {}
    test_users = {row["user_id"] for row in read_csv(output.manifests / "test_inference.csv")}
    shared_order: list[str] | None = None
    shared_folds: list[str] | None = None
    for target in TARGETS:
        manifest_rows = read_csv(target_train_path(output, target))
        score_rows = read_csv(target_oof_path(output, target))
        expected_fields = {"user_id", "label", "fold", "score", "model_id"}
        if not score_rows or set(score_rows[0]) != expected_fields:
            raise RuntimeError(f"{target} OOF score schema mismatch")
        manifest = {row["user_id"]: row for row in manifest_rows}
        observed_order = [row["user_id"] for row in score_rows]
        observed_folds = [row["fold"] for row in score_rows]
        if len(set(observed_order)) != len(observed_order) or set(observed_order) != set(manifest):
            raise RuntimeError(f"{target} OOF coverage mismatch")
        if set(observed_order) & test_users:
            raise RuntimeError(f"{target} OOF contains held-out users")
        for row in score_rows:
            expected = manifest[row["user_id"]]
            if row["label"] != expected["label"] or row["fold"] != expected["fold"]:
                raise RuntimeError(f"{target} OOF label or fold mismatch")
            score = float(row["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError(f"{target} OOF score is invalid")
        if shared_order is None:
            shared_order, shared_folds = observed_order, observed_folds
        elif observed_order != shared_order or observed_folds != shared_folds:
            raise RuntimeError("target arms do not use identical users, order, and folds")
        expected_hash = model_manifest["targets"][target]["oofScoreSha256"]
        require_sha256(target_oof_path(output, target), expected_hash, f"{target} OOF score")
        target_audits[target] = {
            "users": len(score_rows),
            "oofScoreSha256": expected_hash,
            "trainManifestSha256": sha256_file(target_train_path(output, target)),
            "testUsersExcluded": True,
        }
    write_json(
        output.reports / "oof-audit.json",
        {
            "ok": True,
            "experimentId": config["experimentId"],
            "sameUsersOrderAndFolds": True,
            "prepareReportSha256": sha256_file(output.reports / "prepare-report.json"),
            "oofModelManifestSha256": sha256_file(model_manifest_path),
            "targets": target_audits,
            "testLabelsRead": False,
            "sealedLabelHashes": prepare["sealedTestLabelHashes"],
        },
    )


def lock_command(config: dict[str, Any], output: ComparisonPaths) -> None:
    audit_path = output.reports / "oof-audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit.get("ok") or not audit.get("sameUsersOrderAndFolds"):
        raise RuntimeError("OOF audit must pass before locking")
    oof_results: dict[str, Any] = {}
    for target in TARGETS:
        rows = read_csv(target_oof_path(output, target))
        actual = np.asarray([GENERIC_LABEL_TO_CODE[row["label"]] for row in rows], dtype=np.int8)
        scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        threshold, raw_metrics = select_threshold(actual, scores)
        metrics = generic_metrics(
            raw_metrics, config["expectedTargets"][target]["positiveName"]
        )
        lock = {
            "experimentId": config["experimentId"],
            "target": target,
            "positiveLabel": config["expectedTargets"][target]["positiveName"],
            "modelId": TARGET_MODEL_IDS[target],
            "threshold": threshold,
            "thresholdRule": "score >= threshold",
            "selectionSource": "same fixed v7 train folds; this target's OOF scores only",
            "oofMetrics": metrics,
            "classifier": config["classifier"],
            "embedding": config["embedding"],
            "hashChain": {
                "configSha256": config["_configSha256"],
                "trainManifestSha256": sha256_file(target_train_path(output, target)),
                "oofScoreSha256": sha256_file(target_oof_path(output, target)),
                "oofAuditSha256": sha256_file(audit_path),
            },
        }
        write_json(target_lock_path(output, target), lock)
        oof_results[target] = {
            "threshold": threshold,
            "metrics": metrics,
            "lockSha256": sha256_file(target_lock_path(output, target)),
        }
    write_json(
        output.reports / "oof-report.json",
        {
            "experimentId": config["experimentId"],
            "comparison": "identical features, users, folds, and classifier; target-specific OOF thresholds",
            "targets": oof_results,
            "macroF1DeltaSpecialistMinusOriginal": (
                oof_results["specialist_signal"]["metrics"]["macroF1"]
                - oof_results["original_diagnosis"]["metrics"]["macroF1"]
            ),
        },
    )


def fit_full_command(
    baseline_output: Path, config: dict[str, Any], output: ComparisonPaths
) -> None:
    validate_prepared_inputs(baseline_output, config, output)
    features_path = baseline_output / "embeddings/train_user_mean.npz"
    manifests: dict[str, Any] = {}
    for target in TARGETS:
        lock_path = target_lock_path(output, target)
        lock = json.loads(lock_path.read_text())
        if lock.get("target") != target or lock["hashChain"]["configSha256"] != config["_configSha256"]:
            raise RuntimeError(f"{target} lock mismatch")
        require_sha256(
            target_train_path(output, target),
            lock["hashChain"]["trainManifestSha256"],
            f"{target} locked train manifest",
        )
        rows = read_csv(target_train_path(output, target))
        features = aligned_embeddings(rows, features_path, int(config["embedding"]["dimension"]))
        labels = np.asarray([GENERIC_LABEL_TO_CODE[row["label"]] for row in rows], dtype=np.int8)
        model = build_classifier(config)
        model.fit(features, labels)
        model_path = output.models / target / "full_fit.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        manifests[target] = {
            "modelId": TARGET_MODEL_IDS[target],
            "users": len(rows),
            "positive": int(labels.sum()),
            "negative": int((labels == 0).sum()),
            "lockSha256": sha256_file(lock_path),
            "trainManifestSha256": sha256_file(target_train_path(output, target)),
            "trainPoolSha256": sha256_file(features_path),
            "modelSha256": sha256_file(model_path),
        }
    write_json(
        output.manifests / "full-fit-model-manifest.json",
        {
            "experimentId": config["experimentId"],
            "configSha256": config["_configSha256"],
            "targets": manifests,
        },
    )


def score_test_command(
    baseline_output: Path, config: dict[str, Any], output: ComparisonPaths
) -> None:
    validate_prepared_inputs(baseline_output, config, output)
    full_manifest = json.loads((output.manifests / "full-fit-model-manifest.json").read_text())
    if full_manifest.get("configSha256") != config["_configSha256"]:
        raise RuntimeError("full-fit models belong to another comparison config")
    rows = read_csv(output.manifests / "test_inference.csv")
    if not rows or set(rows[0]) != {"user_id"}:
        raise RuntimeError("test inference manifest is not label-free")
    features_path = baseline_output / "embeddings/test_user_mean.npz"
    features = aligned_embeddings(rows, features_path, int(config["embedding"]["dimension"]))
    for target in TARGETS:
        model_path = output.models / target / "full_fit.joblib"
        manifest = full_manifest["targets"][target]
        require_sha256(model_path, manifest["modelSha256"], f"{target} full-fit model")
        require_sha256(
            target_lock_path(output, target), manifest["lockSha256"], f"{target} lock"
        )
        scores = positive_probability(joblib.load(model_path), features)
        write_csv(
            target_test_score_path(output, target),
            ["user_id", "score", "model_id"],
            (
                {
                    "user_id": row["user_id"],
                    "score": repr(float(score)),
                    "model_id": TARGET_MODEL_IDS[target],
                }
                for row, score in zip(rows, scores)
            ),
        )


def audit_test_command(
    baseline_output: Path, config: dict[str, Any], output: ComparisonPaths
) -> None:
    validate_prepared_inputs(baseline_output, config, output)
    inference_path = output.manifests / "test_inference.csv"
    expected_ids = [row["user_id"] for row in read_csv(inference_path)]
    target_hashes: dict[str, Any] = {}
    for target in TARGETS:
        rows = read_csv(target_test_score_path(output, target))
        if not rows or set(rows[0]) != {"user_id", "score", "model_id"}:
            raise RuntimeError(f"{target} test score schema is not label-free")
        if [row["user_id"] for row in rows] != expected_ids:
            raise RuntimeError(f"{target} test score coverage or order mismatch")
        if len(set(expected_ids)) != len(expected_ids):
            raise RuntimeError("test inference manifest contains duplicate users")
        for row in rows:
            score = float(row["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError(f"{target} test score is invalid")
            if row["model_id"] != TARGET_MODEL_IDS[target]:
                raise RuntimeError(f"{target} test score has wrong model id")
        target_hashes[target] = {
            "testScoreSha256": sha256_file(target_test_score_path(output, target)),
            "lockSha256": sha256_file(target_lock_path(output, target)),
        }
    write_json(
        output.reports / "test-score-audit.json",
        {
            "ok": True,
            "experimentId": config["experimentId"],
            "users": len(expected_ids),
            "schema": ["user_id", "score", "model_id"],
            "forbiddenFieldsAbsent": ["Diagnosed_YN", "fold", "label", "true_label"],
            "testInferenceSha256": sha256_file(inference_path),
            "testPoolSha256": sha256_file(
                baseline_output / "embeddings/test_user_mean.npz"
            ),
            "targets": target_hashes,
            "testLabelsRead": False,
        },
    )


def paired_bootstrap_delta(
    original_actual: np.ndarray,
    original_predicted: np.ndarray,
    specialist_actual: np.ndarray,
    specialist_predicted: np.ndarray,
    samples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    joint = original_actual.astype(np.int8) * 2 + specialist_actual.astype(np.int8)
    strata = [np.flatnonzero(joint == code) for code in sorted(set(joint.tolist()))]
    deltas = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        indexes = np.concatenate(
            [rng.choice(stratum, size=len(stratum), replace=True) for stratum in strata]
        )
        original = metrics_from_codes(
            original_actual[indexes], original_predicted[indexes]
        )["macroF1"]
        specialist = metrics_from_codes(
            specialist_actual[indexes], specialist_predicted[indexes]
        )["macroF1"]
        deltas[sample] = specialist - original
    return [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]


def evaluate_command(
    baseline_output: Path, config: dict[str, Any], output: ComparisonPaths
) -> None:
    audit_path = output.reports / "test-score-audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit.get("ok") or audit.get("testLabelsRead"):
        raise RuntimeError("label-free test-score audit must pass before evaluation")
    prepare = json.loads((output.reports / "prepare-report.json").read_text())
    target_results: dict[str, Any] = {}
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for target in TARGETS:
        score_path = target_test_score_path(output, target)
        lock_path = target_lock_path(output, target)
        require_sha256(
            score_path, audit["targets"][target]["testScoreSha256"], f"{target} test score"
        )
        require_sha256(
            lock_path, audit["targets"][target]["lockSha256"], f"{target} lock"
        )
        sealed_path = target_sealed_path(output, target)
        require_sha256(
            sealed_path,
            prepare["sealedTestLabelHashes"][target],
            f"{target} sealed labels",
        )
        score_rows = read_csv(score_path)
        label_rows = read_csv(sealed_path)
        labels = {row["user_id"]: GENERIC_LABEL_TO_CODE[row["label"]] for row in label_rows}
        score_ids = [row["user_id"] for row in score_rows]
        if len(labels) != len(score_ids) or set(labels) != set(score_ids):
            raise RuntimeError(f"{target} sealed-label coverage mismatch")
        actual = np.asarray([labels[user_id] for user_id in score_ids], dtype=np.int8)
        scores = np.asarray([float(row["score"]) for row in score_rows], dtype=np.float64)
        lock = json.loads(lock_path.read_text())
        predicted = (scores >= float(lock["threshold"])).astype(np.int8)
        raw_metrics = metrics_from_codes(actual, predicted)
        positive_name = config["expectedTargets"][target]["positiveName"]
        metrics = generic_metrics(raw_metrics, positive_name)
        intervals = generic_intervals(
            bootstrap_intervals(
                actual,
                predicted,
                int(config["validation"]["bootstrapSamples"]),
                int(config["validation"]["seed"]),
            )
        )
        target_results[target] = {
            "positiveLabel": positive_name,
            "threshold": float(lock["threshold"]),
            "metrics": metrics,
            "stratifiedBootstrap95": intervals,
            "lockSha256": sha256_file(lock_path),
            "testScoreSha256": sha256_file(score_path),
            "sealedTestLabelsSha256": sha256_file(sealed_path),
        }
        arrays[target] = (actual, predicted)

    original_actual, original_predicted = arrays["original_diagnosis"]
    specialist_actual, specialist_predicted = arrays["specialist_signal"]
    delta = (
        target_results["specialist_signal"]["metrics"]["macroF1"]
        - target_results["original_diagnosis"]["metrics"]["macroF1"]
    )
    delta_interval = paired_bootstrap_delta(
        original_actual,
        original_predicted,
        specialist_actual,
        specialist_predicted,
        int(config["validation"]["bootstrapSamples"]),
        int(config["validation"]["seed"]),
    )

    baseline_lock_path = baseline_output / "ensemble/model-lock.json"
    baseline_score_path = baseline_output / "scores/test_qwen3_mean_logreg_s42.csv"
    baseline_report_path = baseline_output / "reports/final-test-report.json"
    require_sha256(
        baseline_lock_path,
        config["baselineArtifacts"]["originalLockSha256"],
        "baseline original lock",
    )
    require_sha256(
        baseline_score_path,
        config["baselineArtifacts"]["originalTestScoreSha256"],
        "baseline original test score",
    )
    require_sha256(
        baseline_report_path,
        config["baselineArtifacts"]["originalFinalReportSha256"],
        "baseline original final report",
    )
    baseline_score_rows = read_csv(baseline_score_path)
    reproduced_score_rows = read_csv(target_test_score_path(output, "original_diagnosis"))
    if [row["user_id"] for row in baseline_score_rows] != [
        row["user_id"] for row in reproduced_score_rows
    ]:
        raise RuntimeError("reproduced original score user order differs from baseline")
    baseline_scores = np.asarray([float(row["score"]) for row in baseline_score_rows])
    reproduced_scores = np.asarray([float(row["score"]) for row in reproduced_score_rows])
    max_score_delta = float(np.max(np.abs(baseline_scores - reproduced_scores)))
    baseline_lock = json.loads(baseline_lock_path.read_text())
    reproduced_lock = json.loads(target_lock_path(output, "original_diagnosis").read_text())
    baseline_report = json.loads(baseline_report_path.read_text())
    if max_score_delta != 0.0:
        raise RuntimeError(f"original arm did not reproduce baseline scores: {max_score_delta}")
    if float(baseline_lock["threshold"]) != float(reproduced_lock["threshold"]):
        raise RuntimeError("original arm did not reproduce baseline OOF threshold")
    if float(baseline_report["metrics"]["macroF1"]) != float(
        target_results["original_diagnosis"]["metrics"]["macroF1"]
    ):
        raise RuntimeError("original arm did not reproduce baseline test Macro F1")

    write_json(
        output.reports / "final-comparison-report.json",
        {
            "ok": True,
            "experimentId": config["experimentId"],
            "question": "On identical v7 users and timelines, are specialist-signal labels easier or harder than original diagnosis labels for the same Qwen-mean LogReg?",
            "controlledFactors": [
                "v7 user set",
                "v7 retained TextLists",
                "v7 train/test assignment",
                "original baseline fold assignment",
                "Qwen3 mean pooled vectors",
                "StandardScaler plus balanced L2 LogisticRegression",
                "seed 42",
                "OOF-only target-specific threshold selection",
            ],
            "changedFactor": "training and evaluation target labels only",
            "targets": target_results,
            "comparison": {
                "macroF1DeltaSpecialistMinusOriginal": delta,
                "pairedJointStratifiedBootstrap95": delta_interval,
                "bootstrapSamples": int(config["validation"]["bootstrapSamples"]),
                "bootstrapSeed": int(config["validation"]["seed"]),
                "labelPairCounts": prepare["labelPairCounts"]["test"],
            },
            "originalBaselineReproduction": {
                "ok": True,
                "maximumAbsoluteTestScoreDelta": max_score_delta,
                "threshold": float(reproduced_lock["threshold"]),
                "macroF1": target_results["original_diagnosis"]["metrics"]["macroF1"],
                "baselineLockSha256": sha256_file(baseline_lock_path),
                "baselineTestScoreSha256": sha256_file(baseline_score_path),
                "baselineFinalReportSha256": sha256_file(baseline_report_path),
            },
            "interpretationLimit": "The v7 cohort was curated using the specialist criterion and its test split was balanced on original diagnosis, so this measures target difficulty inside v7 rather than population-level label quality.",
            "testScoreAuditSha256": sha256_file(audit_path),
        },
    )


def audit_final_command(
    source_path: Path,
    specialist_path: Path,
    baseline_output: Path,
    config: dict[str, Any],
    output: ComparisonPaths,
) -> None:
    required = {
        "prepareReportSha256": output.reports / "prepare-report.json",
        "oofModelManifestSha256": output.manifests / "oof-model-manifest.json",
        "oofAuditSha256": output.reports / "oof-audit.json",
        "oofReportSha256": output.reports / "oof-report.json",
        "fullFitModelManifestSha256": output.manifests / "full-fit-model-manifest.json",
        "testScoreAuditSha256": output.reports / "test-score-audit.json",
        "finalComparisonReportSha256": output.reports / "final-comparison-report.json",
    }
    for target in TARGETS:
        required[f"{target}LockSha256"] = target_lock_path(output, target)
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"final comparison audit is missing artifacts: {missing}")
    require_sha256(source_path, config["source"]["sha256"], "v7 source corpus")
    require_sha256(
        specialist_path,
        config["specialistLabels"]["sha256"],
        "specialist relabeled corpus",
    )
    require_sha256(
        baseline_output / "embeddings/train_user_mean.npz",
        config["baselineArtifacts"]["trainPoolSha256"],
        "baseline train pool",
    )
    require_sha256(
        baseline_output / "embeddings/test_user_mean.npz",
        config["baselineArtifacts"]["testPoolSha256"],
        "baseline test pool",
    )
    for name in ["prepare-report.json", "oof-audit.json", "test-score-audit.json", "final-comparison-report.json"]:
        if not json.loads((output.reports / name).read_text()).get("ok"):
            raise RuntimeError(f"required comparison report did not pass: {name}")
    test_audit = json.loads((output.reports / "test-score-audit.json").read_text())
    for target in TARGETS:
        rows = read_csv(target_test_score_path(output, target))
        if not rows or set(rows[0]) != {"user_id", "score", "model_id"}:
            raise RuntimeError(f"{target} final score schema leaks labels")
        require_sha256(
            target_test_score_path(output, target),
            test_audit["targets"][target]["testScoreSha256"],
            f"{target} final test score",
        )
    write_json(
        output.reports / "final-audit.json",
        {
            "ok": True,
            "experimentId": config["experimentId"],
            "sourceCorpusSha256": sha256_file(source_path),
            "sourceCorpusUnchanged": True,
            "specialistCorpusSha256": sha256_file(specialist_path),
            "specialistCorpusUnchanged": True,
            "embeddingRegenerationPerformed": False,
            "sameUsersTimelinesSplitFoldsFeaturesClassifier": True,
            "onlyTargetLabelsDiffer": True,
            "testScoresLabelFree": True,
            "hashChain": {name: sha256_file(path) for name, path in required.items()},
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "train-oof",
            "audit-oof",
            "lock",
            "fit-full",
            "score-test",
            "audit-test",
            "evaluate",
            "audit-final",
        ],
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-pkl", type=Path)
    parser.add_argument("--specialist-labels", type=Path)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require_path(path: Path | None, flag: str) -> Path:
    if path is None:
        raise RuntimeError(f"{flag} is required for this command")
    return path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output = ComparisonPaths(args.output_dir)
    source = args.source_pkl
    specialist = args.specialist_labels
    commands = {
        "prepare": lambda: prepare_command(
            require_path(source, "--source-pkl"),
            require_path(specialist, "--specialist-labels"),
            args.baseline_output,
            config,
            output,
        ),
        "train-oof": lambda: train_oof_command(args.baseline_output, config, output),
        "audit-oof": lambda: audit_oof_command(args.baseline_output, config, output),
        "lock": lambda: lock_command(config, output),
        "fit-full": lambda: fit_full_command(args.baseline_output, config, output),
        "score-test": lambda: score_test_command(args.baseline_output, config, output),
        "audit-test": lambda: audit_test_command(args.baseline_output, config, output),
        "evaluate": lambda: evaluate_command(args.baseline_output, config, output),
        "audit-final": lambda: audit_final_command(
            require_path(source, "--source-pkl"),
            require_path(specialist, "--specialist-labels"),
            args.baseline_output,
            config,
            output,
        ),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
