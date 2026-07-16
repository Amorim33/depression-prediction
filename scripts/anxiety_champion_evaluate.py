#!/usr/bin/env python3
"""Unseal anxiety test labels after the lock, then perform the single final evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from anxiety_champion_relevance import load_config, resolve, sha256_file, write_json

csv.field_size_limit(100_000_000)


def read_csv(path: Path, delimiter: str = ",", encoding: str = "utf-8") -> list[dict[str, str]]:
    with path.open(newline="", encoding=encoding) as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def normalize_label(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"yes", "sim", "1", "diagnosed"}:
        return "diagnosed"
    if normalized in {"no", "nao", "não", "0", "control"}:
        return "control"
    raise RuntimeError(f"unexpected sealed anxiety label {value!r}")


def metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    tp = sum(a == "diagnosed" and p == "diagnosed" for a, p in zip(actual, predicted))
    fp = sum(a == "control" and p == "diagnosed" for a, p in zip(actual, predicted))
    tn = sum(a == "control" and p == "control" for a, p in zip(actual, predicted))
    fn = sum(a == "diagnosed" and p == "control" for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    diagnosed_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    control_precision = tn / (tn + fn) if tn + fn else 0.0
    control_recall = tn / (tn + fp) if tn + fp else 0.0
    control_f1 = (
        2 * control_precision * control_recall / (control_precision + control_recall)
        if control_precision + control_recall
        else 0.0
    )
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
        "accuracy": (tp + tn) / len(actual) if actual else 0.0,
    }


def write_sealed_labels(path: Path, ordered_users: list[str], labels: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["user_id", "binary_label"])
        writer.writeheader()
        writer.writerows({"user_id": user_id, "binary_label": labels[user_id]} for user_id in ordered_users)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setembrobr.seed42.anxiety-temporal-champion-qwen3-binary.json")
    args = parser.parse_args()
    repo = Path.cwd()
    config = load_config(resolve(args.config, repo))
    output_dir = resolve(config["outputDir"], repo)
    lock_path = output_dir / "ensemble" / "ensemble-lock.json"
    score_manifest_path = output_dir / "reports" / "label-free-test-score-manifest.json"
    report_path = output_dir / "reports" / "final-test-report.json"
    lock_hash = sha256_file(lock_path)
    score_manifest_hash = sha256_file(score_manifest_path)
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        hashes = existing.get("artifactHashes", {})
        if hashes.get("lockSha256") != lock_hash or hashes.get("labelFreeTestScoreManifestSha256") != score_manifest_hash:
            raise RuntimeError("existing anxiety final report is bound to different post-lock artifacts")
        print(json.dumps(existing["testMetrics"], indent=2))
        return

    lock = json.loads(lock_path.read_text())
    score_manifest = json.loads(score_manifest_path.read_text())
    oof_audit = json.loads((output_dir / "reports" / "oof-audit.json").read_text())
    if not oof_audit.get("ok") or not score_manifest.get("ok"):
        raise RuntimeError("anxiety evaluation requires passing OOF and label-free test-score audits")
    if score_manifest.get("lockSha256") != lock_hash:
        raise RuntimeError("ensemble lock changed after label-free test scoring")
    if lock.get("predictionTarget") != "anxiety" or set(lock.get("modelIds", [])) != set(config["ensemble"]["requiredModelIds"]):
        raise RuntimeError("anxiety lock does not contain the exact champion model set")

    raw_dir = resolve(config["rawArtifactsDir"], repo)
    dataset_dir = resolve(config["rawDatasetDir"], repo)
    validation_path = raw_dir / "reports" / "raw_validation_report.json"
    validation = json.loads(validation_path.read_text())
    expected_dataset_hashes = validation.get("datasetHashes", {})
    source_hashes = {}
    labels_by_user: dict[str, str] = {}
    for filename in config["rawCsvFiles"]["test"]:
        source_path = dataset_dir / filename
        actual_hash = sha256_file(source_path)
        expected_hash = expected_dataset_hashes.get(f"test/{filename}")
        if actual_hash != expected_hash:
            raise RuntimeError(f"sealed source hash mismatch for {filename}")
        source_hashes[filename] = actual_hash
        for row in read_csv(source_path, delimiter=";", encoding="utf-8-sig"):
            user_id = row.get("User_ID", "").strip()
            if not user_id or user_id in labels_by_user:
                raise RuntimeError(f"missing or duplicate sealed user {user_id!r}")
            labels_by_user[user_id] = normalize_label(row.get("Diagnosed_YN", ""))

    test_manifest_path = output_dir / "manifest" / f"test_inference_manifest_seed{config['seed']}.csv"
    ordered_users = [row["user_id"] for row in read_csv(test_manifest_path)]
    if len(ordered_users) != int(config["expectedUsers"]["test"]) or set(ordered_users) != set(labels_by_user):
        raise RuntimeError("sealed anxiety labels do not match the redacted test user manifest")

    sealed_path = resolve(config["sealedTestLabelsPath"], repo)
    if sealed_path.exists():
        raise RuntimeError("sealed anxiety labels already exist without a final report; refusing ambiguous re-evaluation")
    write_sealed_labels(sealed_path, ordered_users, labels_by_user)
    sealed_manifest_path = output_dir / "sealed" / "sealed-label-manifest.json"
    write_json(
        sealed_manifest_path,
        {
            "predictionTarget": "anxiety",
            "lockSha256": lock_hash,
            "labelFreeTestScoreManifestSha256": score_manifest_hash,
            "testInferenceManifestSha256": sha256_file(test_manifest_path),
            "rawValidationReportSha256": sha256_file(validation_path),
            "sourceHashes": source_hashes,
            "sealedLabelsSha256": sha256_file(sealed_path),
            "users": len(ordered_users),
            "createdAfterLock": True,
        },
    )

    scores_by_model: dict[str, dict[str, float]] = {}
    for model_id in lock["modelIds"]:
        score_path = output_dir / "scores" / f"test_score_{model_id}.csv"
        if sha256_file(score_path) != score_manifest["sourceHashes"].get(model_id):
            raise RuntimeError(f"test score changed after audit: {model_id}")
        rows = read_csv(score_path)
        scores_by_model[model_id] = {row["user_id"]: float(row["score"]) for row in rows}
        if set(scores_by_model[model_id]) != set(ordered_users):
            raise RuntimeError(f"test score user mismatch: {model_id}")
    combined = [
        sum(float(lock["weights"][model_id]) * scores_by_model[model_id][user_id] for model_id in lock["modelIds"])
        for user_id in ordered_users
    ]
    predicted = ["diagnosed" if score > float(lock["threshold"]) else "control" for score in combined]
    actual = [labels_by_user[user_id] for user_id in ordered_users]
    result = metrics(actual, predicted)
    artifact_hashes = {
        "lockSha256": lock_hash,
        "labelFreeTestScoreManifestSha256": score_manifest_hash,
        "sealedLabelsSha256": sha256_file(sealed_path),
        "sealedLabelManifestSha256": sha256_file(sealed_manifest_path),
        "strictBlindManifestSha256": sha256_file(
            output_dir / "manifest" / f"strict_blind_split_manifest_seed{config['seed']}.csv"
        ),
        "rawEmbeddingManifestSha256": config["rawEmbeddingManifestSha256"],
        "rawSplitManifestSha256": config["rawSplitManifestSha256"],
        "relevanceProxyDefinitionSha256": sha256_file(output_dir / "relevance-proxy" / "proxy-definition.json"),
    }
    report = {
        "dataset": "setembrobr",
        "predictionTarget": "anxiety",
        "positiveClass": "anxiety",
        "seed": config["seed"],
        "manifestHash": lock["manifestHash"],
        "lock": lock,
        "oofMetrics": lock["oofMetrics"],
        "testMetrics": result,
        "anxietyMetrics": {
            "f1": result["diagnosedF1"],
            "precision": result["precision"],
            "recall": result["recall"],
        },
        "confusionMatrix": {
            "anxiety": {"anxiety": result["tp"], "control": result["fn"]},
            "control": {"anxiety": result["fp"], "control": result["tn"]},
        },
        "testUsers": len(ordered_users),
        "testEvaluationCount": 1,
        "artifactHashes": artifact_hashes,
        "evaluatedAt": "1970-01-01T00:00:00.000Z",
    }
    write_json(report_path, report)
    write_json(
        output_dir / "reports" / "reproduction-manifest.json",
        {
            "predictionTarget": "anxiety",
            "experimentId": "seed42_anxiety_temporal_champion_qwen3_binary",
            "artifactHashes": {**artifact_hashes, "finalTestReportSha256": sha256_file(report_path)},
            "rawEmbeddingsCopied": False,
            "rawEmbeddingsReferencedInPlace": True,
            "workArtifactsRegenerable": True,
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
