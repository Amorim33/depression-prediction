#!/usr/bin/env python3
"""Sealed future-validation evaluation for the locked SetembroBR v7 fold-3 GPT prompt."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

import setembrobr_v7_gpt_oof as oof


PROMPT_FOLD = 3
PROMPT_SHA256 = "94a2871ff61f98b8a3b9827b6de3c0f6cf4efcd461209a3e3412b12fe46567d9"
VALIDATION_USERS = 400
EVALUATION_EXPERIMENT = "seed42_setembrobr_v7_gpt56_fold3_future_validation"
SCORE_FILENAME = "test_score_gpt-5.6-sol-high-fold-3.csv"


def evaluation_config(config_path: Path) -> dict[str, Any]:
    config = oof.load_config(config_path)
    config["experimentId"] = EVALUATION_EXPERIMENT
    return config


def selected_prompt(
    config: dict[str, Any], paths: oof.ArtifactPaths
) -> tuple[dict[str, Any], dict[str, Any], str]:
    lock = oof.load_prompt_lock(config, paths)
    selected = oof.fold_lock(lock, PROMPT_FOLD)
    if selected["classifierPromptSha256"] != PROMPT_SHA256:
        raise RuntimeError("fold-3 prompt hash differs from the user-selected OOF winner")
    prompt_path = paths.tracked_prompts / "generated" / f"fold-{PROMPT_FOLD}.md"
    prompt = prompt_path.read_text(encoding="utf-8")
    if oof.sha256_text(prompt) != PROMPT_SHA256:
        raise RuntimeError("fold-3 prompt contents no longer match the locked hash")
    return lock, selected, prompt


def validation_request_id(user_id: str) -> str:
    return "v_" + oof.sha256_text(f"{EVALUATION_EXPERIMENT}\0{user_id}")[:24]


def normalized_validation_frame(source_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    actual_hash = oof.sha256_file(source_path)
    expected_hash = str(config["source"]["sha256"])
    if actual_hash != expected_hash:
        raise RuntimeError(f"source hash mismatch: expected {expected_hash}, got {actual_hash}")
    frame = pd.read_pickle(source_path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("v7 source pickle must contain a pandas DataFrame")
    if list(frame.columns) != config["source"]["schema"]:
        raise RuntimeError(f"v7 source schema mismatch: {list(frame.columns)}")
    split = frame["Split"].astype(str).str.strip().str.lower()
    if set(split) != {"train", "test"}:
        raise RuntimeError("v7 source must contain only the immutable train and test splits")
    validation = frame.loc[split.eq("test"), oof.SOURCE_COLUMNS].copy()
    del frame, split
    validation["User_ID"] = validation["User_ID"].astype(str)
    validation["Diagnosed_YN"] = validation["Diagnosed_YN"].astype(str).str.strip().str.lower()
    validation["Split"] = "test"
    validation = validation.sort_values("User_ID").reset_index(drop=True)
    if len(validation) != VALIDATION_USERS:
        raise RuntimeError(f"expected {VALIDATION_USERS} validation users, got {len(validation)}")
    if validation["User_ID"].duplicated().any():
        raise RuntimeError("validation source contains duplicate user IDs")
    if set(validation["Diagnosed_YN"]) != set(oof.LABEL_TO_CODE):
        raise RuntimeError("validation source contains invalid diagnosis labels")
    for row_index, posts in enumerate(validation["TextLists"]):
        if not isinstance(posts, list) or not posts:
            raise RuntimeError(f"validation row {row_index} has an invalid timeline")
        if any(not isinstance(post, str) for post in posts):
            raise RuntimeError(f"validation row {row_index} contains a non-string post")
    return validation


def build_batch_files(
    config: dict[str, Any], paths: oof.ArtifactPaths, selected: dict[str, Any], prompt: str
) -> dict[str, Any]:
    mapping_rows = oof.read_csv(paths.mappings / "validation-users.csv")
    by_request = {row["request_id"]: row for row in mapping_rows}
    rows = list(oof.read_gzip_jsonl(paths.heldout / "fold-3-future-validation-unlabeled.jsonl.gz"))
    if {str(row["request_id"]) for row in rows} != set(by_request):
        raise RuntimeError("validation timeline file differs from its label-free mapping")

    encoding = oof.tokenizer(str(config["classification"]["model"]))
    context_limit = int(config["classification"]["contextWindowTokens"])
    safety = int(config["classification"]["inputTokenSafetyMargin"])
    max_bytes = int(config["batch"]["maxFileBytes"])
    max_requests = int(config["batch"]["maxRequestsPerFile"])
    shards: list[dict[str, Any]] = []
    request_manifest_rows: list[dict[str, Any]] = []
    shard_lines: list[bytes] = []
    shard_tokens = 0
    shard_bytes = 0
    shard_index = 0

    def flush() -> None:
        nonlocal shard_lines, shard_tokens, shard_bytes, shard_index
        if not shard_lines:
            return
        shard_path = paths.batches / "future-validation" / f"input-{shard_index:03d}.jsonl"
        payload = b"".join(shard_lines)
        shards.append(
            {
                "shardId": f"validation-fold-3-{shard_index:03d}",
                "fold": PROMPT_FOLD,
                "path": str(shard_path.resolve()),
                "sha256": oof.atomic_write_bytes(shard_path, payload),
                "requests": len(shard_lines),
                "estimatedInputTokens": shard_tokens,
                "promptSha256": PROMPT_SHA256,
                "status": "pending",
                "superseded": False,
            }
        )
        shard_lines = []
        shard_tokens = 0
        shard_bytes = 0
        shard_index += 1

    for row in rows:
        request_id = str(row["request_id"])
        body = oof.classifier_request_body(
            config, prompt, PROMPT_SHA256, selected["evidenceCodes"], row["posts"]
        )
        tokens = oof.estimate_request_tokens(
            encoding,
            prompt,
            str(body["input"]),
            oof.classification_schema(selected["evidenceCodes"]),
        )
        if tokens + int(config["classification"]["maxOutputTokens"]) > context_limit - safety:
            raise RuntimeError(f"full validation timeline exceeds context limit for {request_id}")
        request_hash = oof.sha256_bytes(oof.canonical_json(body))
        batch_row = {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
        encoded = (oof.compact_json(batch_row) + "\n").encode("utf-8")
        if shard_lines and (
            len(shard_lines) >= max_requests or shard_bytes + len(encoded) > max_bytes
        ):
            flush()
        shard_lines.append(encoded)
        shard_tokens += tokens
        shard_bytes += len(encoded)
        request_manifest_rows.append(
            {
                "request_id": request_id,
                "fold": PROMPT_FOLD,
                "request_sha256": request_hash,
                "prompt_sha256": PROMPT_SHA256,
                "timeline_sha256": by_request[request_id]["timeline_sha256"],
            }
        )
    flush()
    oof.write_csv(
        paths.batches / "request-manifest.csv",
        ["request_id", "fold", "request_sha256", "prompt_sha256", "timeline_sha256"],
        request_manifest_rows,
    )
    manifest = {
        "experimentId": EVALUATION_EXPERIMENT,
        "configSha256": config["_configSha256"],
        "promptLockSha256": oof.sha256_file(paths.tracked_manifests / "prompt-lock.json"),
        "selectedOofFold": PROMPT_FOLD,
        "promptSha256": PROMPT_SHA256,
        "endpoint": "/v1/responses",
        "model": config["classification"]["model"],
        "reasoningEffort": config["classification"]["reasoningEffort"],
        "requests": len(request_manifest_rows),
        "shards": shards,
        "builtAt": oof.utc_now(),
    }
    oof.write_json(paths.batches / "batch-manifest.json", manifest)
    return manifest


def prepare_command(source_path: Path, config: dict[str, Any], paths: oof.ArtifactPaths) -> None:
    paths.ensure()
    existing_manifest = paths.batches / "batch-manifest.json"
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if any(item.get("submittedAt") for item in existing.get("shards", [])):
            raise RuntimeError("cannot re-prepare validation data after an inference submission")
    lock, selected, prompt = selected_prompt(config, paths)
    validation = normalized_validation_frame(source_path, config)
    identity_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    for row in validation.itertuples(index=False):
        user_id = str(row.User_ID)
        request_id = validation_request_id(user_id)
        posts = list(row.TextLists)
        identity_rows.append({"user_id": user_id, "request_id": request_id})
        label_rows.append({"user_id": user_id, "label": str(row.Diagnosed_YN)})
        mapping_rows.append(
            {
                "request_id": request_id,
                "timeline_sha256": oof.timeline_hash(posts),
                "post_count": len(posts),
            }
        )
        timeline_rows.append({"request_id": request_id, "posts": posts})
    del validation
    identity_hash = oof.write_csv(
        paths.mappings / "validation-identities.csv", ["user_id", "request_id"], identity_rows
    )
    labels_hash = oof.write_csv(
        paths.work / "sealed-labels" / "validation-labels.csv", ["user_id", "label"], label_rows
    )
    mapping_hash = oof.write_csv(
        paths.mappings / "validation-users.csv",
        ["request_id", "timeline_sha256", "post_count"],
        mapping_rows,
    )
    timeline_hash = oof.write_gzip_jsonl(
        paths.heldout / "fold-3-future-validation-unlabeled.jsonl.gz", timeline_rows
    )
    manifest = build_batch_files(config, paths, selected, prompt)
    report = {
        "ok": True,
        "experimentId": EVALUATION_EXPERIMENT,
        "users": VALIDATION_USERS,
        "selectedOofFold": PROMPT_FOLD,
        "promptSha256": PROMPT_SHA256,
        "promptLockSha256": oof.sha256_file(paths.tracked_manifests / "prompt-lock.json"),
        "promptLockedAt": lock["lockedAt"],
        "predictionThreshold": config["validation"]["predictionThreshold"],
        "labelsExcludedFromRequests": True,
        "identityMapSha256": identity_hash,
        "sealedLabelsSha256": labels_hash,
        "labelFreeMappingSha256": mapping_hash,
        "labelFreeTimelinesSha256": timeline_hash,
        "batchManifestSha256": oof.sha256_file(paths.batches / "batch-manifest.json"),
        "estimatedInputTokens": sum(int(item["estimatedInputTokens"]) for item in manifest["shards"]),
        "sourceSha256": config["source"]["sha256"],
    }
    oof.write_json(paths.reports / "validation-prepare-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def completed_output_rows(manifest: dict[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    ordered = sorted(
        [item for item in manifest["shards"] if not item.get("superseded") and item.get("fetchedAt")],
        key=lambda item: (str(item.get("submittedAt", "")), item["shardId"]),
    )
    for shard in ordered:
        output_path = Path(shard["outputPath"])
        if oof.sha256_file(output_path) != shard["outputSha256"]:
            raise RuntimeError(f"batch output hash mismatch: {output_path}")
        for row in oof.read_jsonl(output_path):
            yield shard, row


def audit_command(config: dict[str, Any], paths: oof.ArtifactPaths) -> None:
    lock, selected, prompt = selected_prompt(config, paths)
    manifest_path = paths.batches / "batch-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    incomplete = [
        item["shardId"]
        for item in manifest["shards"]
        if not item.get("superseded")
        and (item.get("status") != "completed" or not item.get("fetchedAt"))
    ]
    if incomplete:
        raise RuntimeError(f"all validation shards must be completed and fetched: {incomplete}")
    mapping_rows = oof.read_csv(paths.mappings / "validation-users.csv")
    identity_rows = oof.read_csv(paths.mappings / "validation-identities.csv")
    expected_ids = {row["request_id"] for row in mapping_rows}
    if len(expected_ids) != VALIDATION_USERS:
        raise RuntimeError("validation label-free mapping does not contain 400 unique requests")
    if {row["request_id"] for row in identity_rows} != expected_ids:
        raise RuntimeError("validation identity map differs from label-free mapping")
    by_mapping = {row["request_id"]: row for row in mapping_rows}
    by_identity = {row["request_id"]: row for row in identity_rows}
    request_manifest = {
        row["request_id"]: row for row in oof.read_csv(paths.batches / "request-manifest.csv")
    }
    if set(request_manifest) != expected_ids:
        raise RuntimeError("validation request manifest does not cover exactly the evaluation users")

    request_rows: dict[str, dict[str, Any]] = {}
    request_occurrences: Counter[str] = Counter()
    for shard, row in oof.leaf_batch_rows(manifest):
        oof.require_fields(row, {"custom_id", "method", "url", "body"}, "validation batch request")
        request_id = str(row["custom_id"])
        if request_id not in expected_ids:
            raise RuntimeError("validation batch contains an unknown request ID")
        if row["method"] != "POST" or row["url"] != "/v1/responses":
            raise RuntimeError("validation batch endpoint is invalid")
        body = row["body"]
        forbidden = oof.nested_forbidden_keys(
            body, {"label", "fold", "user_id", "userid", "diagnosed_yn", "split"}
        )
        if forbidden:
            raise RuntimeError(f"validation request contains forbidden fields: {sorted(forbidden)}")
        logical = request_manifest[request_id]
        if oof.sha256_bytes(oof.canonical_json(body)) != logical["request_sha256"]:
            raise RuntimeError("validation request body changed after preparation")
        if logical["prompt_sha256"] != PROMPT_SHA256 or shard["promptSha256"] != PROMPT_SHA256:
            raise RuntimeError("validation request does not use the selected fold-3 prompt")
        if oof.sha256_text(str(body["instructions"])) != PROMPT_SHA256:
            raise RuntimeError("validation request instructions differ from the locked prompt")
        posts = oof.posts_from_timeline_input(str(body["input"]))
        if oof.timeline_hash(posts) != by_mapping[request_id]["timeline_sha256"]:
            raise RuntimeError("validation timeline is incomplete or reordered")
        if body["model"] != config["classification"]["model"]:
            raise RuntimeError("validation request uses the wrong model")
        if body["reasoning"] != {"effort": config["classification"]["reasoningEffort"]}:
            raise RuntimeError("validation request uses the wrong reasoning effort")
        if body.get("truncation") != "disabled" or body.get("store") is not False:
            raise RuntimeError("validation request does not preserve the full stateless input")
        if request_id in request_rows and request_rows[request_id] != row:
            raise RuntimeError("a validation retry changed its request body")
        request_rows[request_id] = row
        request_occurrences[request_id] += 1

    first_valid: dict[str, dict[str, Any]] = {}
    errors: dict[str, list[str]] = {}
    for shard, output in completed_output_rows(manifest):
        request_id = str(output.get("custom_id", ""))
        if request_id not in expected_ids or request_id in first_valid:
            continue
        try:
            response = output.get("response")
            if not isinstance(response, dict) or int(response.get("status_code", 0)) != 200:
                raise RuntimeError("validation batch HTTP response is not successful")
            body = response.get("body")
            if not isinstance(body, dict):
                raise RuntimeError("validation response body is missing")
            if body.get("model") != config["classification"]["model"]:
                raise RuntimeError(f"returned model is {body.get('model')!r}")
            parsed = oof.validate_classification(
                json.loads(oof.extract_output_text(body)), selected["evidenceCodes"]
            )
            first_valid[request_id] = {
                "request_id": request_id,
                "prompt_sha256": PROMPT_SHA256,
                "request_sha256": request_manifest[request_id]["request_sha256"],
                "response_id": body.get("id"),
                "model": body.get("model"),
                "prediction": parsed["prediction"],
                "score": float(parsed["diagnosed_probability"]),
                "evidence_codes": parsed["evidence_codes"],
                "counterevidence_codes": parsed["counterevidence_codes"],
                "uncertainty": parsed["uncertainty"],
                "batch_id": shard.get("batchId"),
            }
        except (json.JSONDecodeError, RuntimeError, TypeError, ValueError) as error:
            errors.setdefault(request_id, []).append(str(error))

    missing = sorted(expected_ids - set(first_valid))
    retry = None
    if missing:
        retry = oof.create_retry_shard(manifest, request_rows, request_manifest, missing)
        oof.write_json(manifest_path, manifest)
    normalized_path = paths.responses / "label-free-validation-responses.jsonl"
    normalized_hash = oof.write_jsonl(
        normalized_path, (first_valid[key] for key in sorted(first_valid))
    )
    threshold = float(config["validation"]["predictionThreshold"])
    score_rows = []
    for request_id in sorted(first_valid):
        response = first_valid[request_id]
        score = float(response["score"])
        expected_prediction = oof.CODE_TO_PUBLIC_LABEL[int(score >= threshold)]
        if response["prediction"] != expected_prediction:
            raise RuntimeError("validation response prediction disagrees with the locked threshold")
        score_rows.append(
            {
                "user_id": by_identity[request_id]["user_id"],
                "score": repr(score),
                "model_id": f"gpt-5.6-sol-high-fold-3-{PROMPT_SHA256[:12]}",
            }
        )
    score_hash = oof.write_csv(
        paths.scores / SCORE_FILENAME, ["user_id", "score", "model_id"], score_rows
    )
    lock_time = datetime.fromisoformat(str(lock["lockedAt"]).replace("Z", "+00:00"))
    submission_times = [
        datetime.fromisoformat(str(item["submittedAt"]).replace("Z", "+00:00"))
        for item in manifest["shards"]
        if item.get("submittedAt")
    ]
    prompt_precedes_inference = bool(submission_times) and all(
        lock_time <= value for value in submission_times
    )
    report = {
        "ok": not missing and len(first_valid) == VALIDATION_USERS and prompt_precedes_inference,
        "experimentId": EVALUATION_EXPERIMENT,
        "usersExpected": VALIDATION_USERS,
        "usersWithValidLabelFreeResponse": len(first_valid),
        "missingUsers": len(missing),
        "validationErrorUsers": len(errors),
        "retryShardId": retry["shardId"] if retry else None,
        "duplicateIdenticalRequests": sum(count - 1 for count in request_occurrences.values()),
        "allRetriesByteIdentical": True,
        "selectedOofFold": PROMPT_FOLD,
        "promptSha256": PROMPT_SHA256,
        "promptLockedBeforeInference": prompt_precedes_inference,
        "predictionThreshold": threshold,
        "labelsReadDuringAudit": False,
        "labelsAbsentFromRequests": True,
        "foldAbsentFromRequests": True,
        "originalUserIdsAbsentFromRequestMetadata": True,
        "fullTimelineHashesMatch": True,
        "model": config["classification"]["model"],
        "reasoningEffort": config["classification"]["reasoningEffort"],
        "normalizedResponsesSha256": normalized_hash,
        "labelFreeScoresSha256": score_hash,
    }
    oof.write_json(paths.reports / "label-free-validation-audit.json", report)
    tracked_report = {key: value for key, value in report.items() if key != "retryShardId"}
    oof.write_json(paths.tracked_reports / "fold-3-future-validation-audit.json", tracked_report)
    print(json.dumps(report, indent=2, sort_keys=True))


def evaluate_command(config: dict[str, Any], paths: oof.ArtifactPaths) -> None:
    audit_path = paths.reports / "label-free-validation-audit.json"
    if not audit_path.is_file():
        raise RuntimeError("label-free validation audit is missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("ok"):
        raise RuntimeError("label-free validation audit must pass before labels are opened")
    score_path = paths.scores / SCORE_FILENAME
    if oof.sha256_file(score_path) != audit["labelFreeScoresSha256"]:
        raise RuntimeError("label-free validation scores changed after audit")
    scores = {row["user_id"]: row for row in oof.read_csv(score_path)}
    labels = {
        row["user_id"]: row["label"]
        for row in oof.read_csv(paths.work / "sealed-labels" / "validation-labels.csv")
    }
    if len(scores) != VALIDATION_USERS or set(scores) != set(labels):
        raise RuntimeError("audited validation scores do not align with the sealed labels")
    threshold = float(config["validation"]["predictionThreshold"])
    actual = np.asarray([oof.LABEL_TO_CODE[labels[key]] for key in sorted(labels)], dtype=np.int8)
    predicted = np.asarray(
        [int(float(scores[key]["score"]) >= threshold) for key in sorted(labels)], dtype=np.int8
    )
    metrics = oof.metrics_from_codes(actual, predicted)
    report = {
        "experimentId": EVALUATION_EXPERIMENT,
        "stage": "separately authorized future-validation evaluation",
        "users": VALIDATION_USERS,
        "predictionTarget": "Diagnosed_YN",
        "selectedOofFold": PROMPT_FOLD,
        "selectionRule": "highest observed held-out-fold Macro F1 among the five locked OOF prompts",
        "selectionCaveat": "prompt selection used OOF fold performance; this is a one-shot post-selection validation estimate",
        "promptSha256": PROMPT_SHA256,
        "model": config["classification"]["model"],
        "reasoningEffort": config["classification"]["reasoningEffort"],
        "predictionThreshold": threshold,
        "thresholdSelection": "unchanged 0.5 OOF lock; no validation tuning",
        "labelsJoinedOnlyAfterLabelFreeAudit": True,
        "metrics": metrics,
        "hashChain": {
            "sourceSha256": config["source"]["sha256"],
            "configSha256": config["_configSha256"],
            "promptLockSha256": oof.sha256_file(paths.tracked_manifests / "prompt-lock.json"),
            "labelFreeAuditSha256": oof.sha256_file(audit_path),
            "labelFreeScoresSha256": oof.sha256_file(score_path),
            "sealedLabelsSha256": oof.sha256_file(
                paths.work / "sealed-labels" / "validation-labels.csv"
            ),
        },
    }
    oof.write_json(paths.reports / "fold-3-future-validation-results.json", report)
    oof.write_json(paths.tracked_reports / "fold-3-future-validation-results.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "submit", "status", "fetch", "audit", "evaluate"])
    experiment = Path(__file__).resolve().parents[1]
    project = experiment.parent
    parser.add_argument("--config", type=Path, default=experiment / "config.json")
    parser.add_argument(
        "--work-dir", type=Path, default=project / ".work" / "gpt-validation-fold3"
    )
    parser.add_argument("--source-pkl", type=Path)
    parser.add_argument("--env-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = evaluation_config(config_path)
    paths = oof.ArtifactPaths(args.work_dir.expanduser().resolve(), config_path.parent)
    paths.ensure()
    env_path = args.env_file.expanduser().resolve() if args.env_file else None
    if args.command == "prepare":
        if args.source_pkl is None:
            raise SystemExit("--source-pkl is required for validation prepare")
        prepare_command(args.source_pkl.expanduser().resolve(), config, paths)
    elif args.command == "submit":
        oof.submit_command(config, paths, env_path)
    elif args.command == "status":
        oof.status_command(config, paths, env_path)
    elif args.command == "fetch":
        oof.fetch_command(config, paths, env_path)
    elif args.command == "audit":
        audit_command(config, paths)
    elif args.command == "evaluate":
        evaluate_command(config, paths)


if __name__ == "__main__":
    main()
