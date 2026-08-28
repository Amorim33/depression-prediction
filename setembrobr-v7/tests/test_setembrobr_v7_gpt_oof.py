from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "gpt-experiments/scripts/setembrobr_v7_gpt_oof.py"
SPEC = importlib.util.spec_from_file_location("setembrobr_v7_gpt_oof", SCRIPT)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class FakeEncoding:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


def response_body(model: str, payload: dict, response_id: str) -> dict:
    return {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(payload)}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


class FakeResponses:
    def __init__(self, outputs: list[dict], retrieve_outputs: list[dict] | None = None) -> None:
        self.outputs = list(outputs)
        self.retrieve_outputs = list(retrieve_outputs or [])
        self.requests: list[dict] = []
        self.retrieved: list[str] = []
        self.deleted: list[str] = []

    def create(self, **request: object) -> dict:
        self.requests.append(request)
        return self.outputs.pop(0)

    def retrieve(self, response_id: str) -> dict:
        self.retrieved.append(response_id)
        return self.retrieve_outputs.pop(0)

    def delete(self, response_id: str) -> None:
        self.deleted.append(response_id)


class FakeClient:
    def __init__(self, outputs: list[dict], retrieve_outputs: list[dict] | None = None) -> None:
        self.responses = FakeResponses(outputs, retrieve_outputs)


class SyntheticExperiment:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = root / "SetembroBR-v7-min.pkl"
        self.experiment = root / "gpt-experiments"
        self.work = root / ".work/gpt-experiments"
        self.config_path = self.experiment / "config.json"
        self.paths = pipeline.ArtifactPaths(self.work, self.experiment)
        self.train_ids = [f"train-user-{index:02d}" for index in range(20)]
        self.future_ids = ["future-user-a", "future-user-b"]
        self._write_source()
        self._copy_prompts()
        self._write_config()
        self.config = pipeline.load_config(self.config_path)

    def _write_source(self) -> None:
        rows = []
        for index, user_id in enumerate(self.train_ids):
            label = "yes" if index < 10 else "no"
            rows.append(
                {
                    "User_ID": user_id,
                    "Diagnosed_YN": label,
                    "TextLists": [f"post inicial {index}", f"post final {index}"],
                    "Split": "train",
                }
            )
        for index, user_id in enumerate(self.future_ids):
            rows.append(
                {
                    "User_ID": user_id,
                    "Diagnosed_YN": "yes" if index == 0 else "no",
                    "TextLists": [f"FUTURE_VALIDATION_SECRET_{index}"],
                    "Split": "test",
                }
            )
        pd.DataFrame(rows, columns=pipeline.SOURCE_COLUMNS).to_pickle(self.source)

    def _copy_prompts(self) -> None:
        source = PROJECT / "gpt-experiments/prompts"
        shutil.copytree(source, self.experiment / "prompts")

    def _write_config(self) -> None:
        labels = np.asarray([1] * 10 + [0] * 10, dtype=np.int8)
        folds = pipeline.assign_folds(labels, 5, 42)
        manifest = self.root / "expected.csv"
        manifest_hash = pipeline.write_csv(
            manifest,
            ["user_id", "label", "fold"],
            (
                {
                    "user_id": user_id,
                    "label": "yes" if label == 1 else "no",
                    "fold": int(fold),
                }
                for user_id, label, fold in zip(self.train_ids, labels, folds)
            ),
        )
        payload = {
            "experimentId": "synthetic_gpt_oof",
            "source": {
                "filename": self.source.name,
                "sha256": pipeline.sha256_file(self.source),
                "schema": pipeline.SOURCE_COLUMNS,
                "trainUsers": 20,
                "trainPositive": 10,
                "trainNegative": 10,
                "trainPosts": 40,
            },
            "validation": {
                "seed": 42,
                "foldCount": 5,
                "trainManifestSha256": manifest_hash,
                "predictionThreshold": 0.5,
            },
            "analysis": {
                "model": "gpt-5.6-luna",
                "reasoningEffort": "xhigh",
                "maxOutputTokens": 1000,
                "maxToolCalls": 4,
                "containerMemory": "1g",
                "roles": ["bdi_symptoms"],
            },
            "classification": {
                "model": "gpt-5.6-sol",
                "reasoningEffort": "high",
                "maxOutputTokens": 100,
                "contextWindowTokens": 100000,
                "inputTokenSafetyMargin": 100,
                "truncation": "disabled",
            },
            "batch": {
                "completionWindow": "24h",
                "maxFileBytes": 1000000,
                "maxRequestsPerFile": 100,
                "outputExpiresAfterSeconds": 3600,
            },
            "privacy": {"verbatimShingleWords": 12, "minimumUserIdLengthToScan": 8},
        }
        pipeline.write_json(self.config_path, payload)

    def prepare(self) -> None:
        pipeline.prepare_command(self.source, self.config, self.paths)

    def create_prompt_lock(self) -> None:
        self.paths.ensure()
        folds = []
        for fold in range(5):
            prompt = f"Synthetic classifier prompt for outer partition {fold}."
            prompt_path = self.paths.tracked_prompts / "generated" / f"fold-{fold}.md"
            prompt_hash = pipeline.write_text(prompt_path, prompt)
            folds.append(
                {
                    "heldoutFold": fold,
                    "developmentFolds": [value for value in range(5) if value != fold],
                    "developmentUsers": 16,
                    "heldoutUsers": 4,
                    "outerManifestSha256": pipeline.sha256_file(self.paths.manifests / f"outer-fold-{fold}.json"),
                    "analysisProvenanceSha256": "a" * 64,
                    "finalEvidenceSha256": "b" * 64,
                    "classifierPromptSha256": prompt_hash,
                    "evidenceCodes": ["SELF_PERSISTENCE", "COUNTER_CONTEXT"],
                }
            )
        lock = {
            "experimentId": self.config["experimentId"],
            "configSha256": self.config["_configSha256"],
            "sourceSha256": self.config["source"]["sha256"],
            "trainManifestSha256": self.config["validation"]["trainManifestSha256"],
            "analysisModel": {"model": "gpt-5.6-luna", "reasoningEffort": "xhigh"},
            "classificationModel": {
                "model": "gpt-5.6-sol",
                "reasoningEffort": "high",
                "predictionThreshold": 0.5,
            },
            "allPromptsLockedBeforeInference": True,
            "folds": folds,
            "lockedAt": "2026-01-01T00:00:00Z",
        }
        pipeline.write_json(self.paths.tracked_manifests / "prompt-lock.json", lock)
        pipeline.write_json(
            self.paths.reports / "smoke-report.json",
            {"ok": True, "cases": [], "corpusUsersScored": 0, "promptChangesPermitted": False},
        )


class StrictBlindPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.item = SyntheticExperiment(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prepare_reproduces_folds_and_never_materializes_future_validation(self) -> None:
        self.item.prepare()
        manifest = pipeline.read_csv(self.item.paths.manifests / "train_manifest.csv")
        self.assertEqual(len(manifest), 20)
        self.assertEqual(pipeline.sha256_file(self.item.paths.manifests / "train_manifest.csv"), self.item.config["validation"]["trainManifestSha256"])
        self.assertEqual({int(row["fold"]) for row in manifest}, set(range(5)))
        self.assertFalse(set(self.item.future_ids) & {row["user_id"] for row in manifest})

        mapping = pipeline.read_csv(self.item.paths.mappings / "users.csv")
        self.assertEqual(len(mapping), 20)
        self.assertFalse(set(self.item.future_ids) & {row["user_id"] for row in mapping})
        for fold in range(5):
            outer = json.loads((self.item.paths.manifests / f"outer-fold-{fold}.json").read_text())
            self.assertNotIn(fold, outer["developmentFolds"])
            heldout = list(pipeline.read_gzip_jsonl(self.item.paths.heldout / f"fold-{fold}-unlabeled.jsonl.gz"))
            self.assertTrue(all(set(row) == {"request_id", "posts"} for row in heldout))
            development_ids = set()
            for source_fold in outer["developmentFolds"]:
                rows = pipeline.read_gzip_jsonl(self.item.paths.source_folds / f"fold-{source_fold}-labeled.jsonl.gz")
                development_ids.update(row["opaque_id"] for row in rows)
            self.assertFalse(development_ids & {row["request_id"] for row in heldout})

        all_bytes = b"".join(path.read_bytes() for path in self.item.work.rglob("*" ) if path.is_file())
        self.assertNotIn(b"FUTURE_VALIDATION_SECRET", all_bytes)

    def test_public_artifact_audit_rejects_identity_url_handle_and_quote(self) -> None:
        self.item.prepare()
        mapping = pipeline.read_csv(self.item.paths.mappings / "users.csv")
        corpus = [self.item.paths.source_folds / "fold-0-labeled.jsonl.gz"]
        with self.assertRaises(RuntimeError):
            pipeline.assert_public_text([mapping[0]["user_id"]], mapping, corpus, self.item.config)
        with self.assertRaises(RuntimeError):
            pipeline.assert_public_text(["visit https://example.com"], mapping, corpus, self.item.config)
        with self.assertRaises(RuntimeError):
            pipeline.assert_public_text(["contact @example"], mapping, corpus, self.item.config)


class RequestAndCacheTests(unittest.TestCase):
    def test_openai_schemas_avoid_unsupported_unique_items_keyword(self) -> None:
        schemas = [
            pipeline.analysis_schema("test_role"),
            pipeline.synthesis_schema(),
            pipeline.red_team_schema(),
            pipeline.classification_schema(["SIGNAL_A"]),
        ]
        for schema in schemas:
            self.assertNotIn('"uniqueItems"', json.dumps(schema))

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.item = SyntheticExperiment(Path(self.temporary.name))
        self.item.prepare()
        self.item.create_prompt_lock()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_classifier_request_contains_full_ordered_timeline_and_no_sensitive_fields(self) -> None:
        lock = pipeline.load_prompt_lock(self.item.config, self.item.paths)
        locked = pipeline.fold_lock(lock, 0)
        posts = ["primeiro", "segundo", "terceiro"]
        body = pipeline.classifier_request_body(
            self.item.config,
            "prompt",
            locked["classifierPromptSha256"],
            locked["evidenceCodes"],
            posts,
        )
        self.assertEqual(pipeline.posts_from_timeline_input(body["input"]), posts)
        self.assertEqual(body["model"], "gpt-5.6-sol")
        self.assertEqual(body["reasoning"], {"effort": "high"})
        self.assertEqual(body["truncation"], "disabled")
        self.assertIs(body["store"], False)
        self.assertFalse(
            pipeline.nested_forbidden_keys(
                body, {"label", "fold", "user_id", "userid", "diagnosed_yn", "split"}
            )
        )

    def test_cache_uses_first_semantically_valid_response_and_reuses_it(self) -> None:
        codes = ["SELF_PERSISTENCE"]
        invalid = {
            "prediction": "control",
            "diagnosed_probability": 0.9,
            "evidence_codes": [],
            "counterevidence_codes": [],
            "uncertainty": "high",
        }
        valid = {
            "prediction": "diagnosed",
            "diagnosed_probability": 0.9,
            "evidence_codes": ["SELF_PERSISTENCE"],
            "counterevidence_codes": [],
            "uncertainty": "low",
        }
        client = FakeClient(
            [
                response_body("gpt-5.6-sol", invalid, "response-invalid"),
                response_body("gpt-5.6-sol", valid, "response-valid"),
            ]
        )
        cache = self.item.work / "cache"
        result, provenance = pipeline.call_cached_response(
            client,
            logical_request={"stable": True},
            api_request={"model": "gpt-5.6-sol"},
            cache_dir=cache,
            expected_model="gpt-5.6-sol",
            validator=lambda payload: pipeline.validate_classification(payload, codes),
        )
        self.assertEqual(result, valid)
        self.assertEqual(provenance["attempt"], 2)
        replay, _ = pipeline.call_cached_response(
            FakeClient([]),
            logical_request={"stable": True},
            api_request={"model": "gpt-5.6-sol"},
            cache_dir=cache,
            expected_model="gpt-5.6-sol",
            validator=lambda payload: pipeline.validate_classification(payload, codes),
        )
        self.assertEqual(replay, valid)

    def test_background_response_is_polled_cached_and_deleted(self) -> None:
        valid = {
            "prediction": "control",
            "diagnosed_probability": 0.1,
            "evidence_codes": [],
            "counterevidence_codes": ["SIGNAL_A"],
            "uncertainty": "low",
        }
        queued = {"id": "response-background", "status": "queued", "model": "gpt-5.6-luna"}
        client = FakeClient(
            [queued],
            [response_body("gpt-5.6-luna", valid, "response-background")],
        )
        with mock.patch.object(pipeline.time, "sleep"):
            result, provenance = pipeline.call_cached_response(
                client,
                logical_request={"stable": "background"},
                api_request={"model": "gpt-5.6-luna", "background": True, "store": True},
                cache_dir=self.item.work / "background-cache",
                expected_model="gpt-5.6-luna",
                validator=lambda payload: pipeline.validate_classification(payload, ["SIGNAL_A"]),
            )
        self.assertEqual(result, valid)
        self.assertTrue(provenance["background"])
        self.assertEqual(client.responses.retrieved, ["response-background"])
        self.assertEqual(client.responses.deleted, ["response-background"])


class EndToEndOfflineAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.item = SyntheticExperiment(Path(self.temporary.name))
        self.item.prepare()
        self.item.create_prompt_lock()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_label_free_batch_audit_and_evaluation_cover_every_train_user(self) -> None:
        original_tokenizer = pipeline.tokenizer
        pipeline.tokenizer = lambda _model: FakeEncoding()
        try:
            manifest = pipeline.build_batch_files(self.item.config, self.item.paths)
        finally:
            pipeline.tokenizer = original_tokenizer

        mapping = {row["opaque_id"]: row for row in pipeline.read_csv(self.item.paths.mappings / "users.csv")}
        submitted_at = "2026-01-02T00:00:00Z"
        for shard in manifest["shards"]:
            outputs = []
            for request in pipeline.read_jsonl(Path(shard["path"])):
                item = mapping[request["custom_id"]]
                diagnosed = item["label"] == "yes"
                parsed = {
                    "prediction": "diagnosed" if diagnosed else "control",
                    "diagnosed_probability": 0.9 if diagnosed else 0.1,
                    "evidence_codes": ["SELF_PERSISTENCE"] if diagnosed else [],
                    "counterevidence_codes": [] if diagnosed else ["COUNTER_CONTEXT"],
                    "uncertainty": "low",
                }
                outputs.append(
                    {
                        "id": f"batch-row-{request['custom_id']}",
                        "custom_id": request["custom_id"],
                        "response": {
                            "status_code": 200,
                            "request_id": f"api-{request['custom_id']}",
                            "body": response_body("gpt-5.6-sol", parsed, f"response-{request['custom_id']}"),
                        },
                        "error": None,
                    }
                )
            output_path = self.item.paths.batches / "outputs" / f"{shard['shardId']}.jsonl"
            shard["outputSha256"] = pipeline.write_jsonl(output_path, outputs)
            shard["outputPath"] = str(output_path.resolve())
            shard["batchId"] = f"batch-{shard['shardId']}"
            shard["status"] = "completed"
            shard["submittedAt"] = submitted_at
            shard["fetchedAt"] = "2026-01-03T00:00:00Z"
        pipeline.write_json(self.item.paths.batches / "batch-manifest.json", manifest)

        pipeline.audit_command(self.item.config, self.item.paths)
        audit = json.loads((self.item.paths.reports / "label-free-oof-audit.json").read_text())
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["usersWithValidLabelFreeResponse"], 20)
        pipeline.evaluate_command(self.item.config, self.item.paths)
        report = json.loads((self.item.paths.tracked_reports / "oof-results.json").read_text())
        self.assertEqual(report["users"], 20)
        self.assertEqual(report["overall"]["macroF1"], 1.0)
        self.assertEqual(len(report["folds"]), 5)
        scores = pipeline.read_csv(self.item.paths.scores / "train_oof_gpt-5.6-sol-high.csv")
        self.assertEqual(len(scores), 20)
        self.assertEqual(set(scores[0]), {"user_id", "label", "fold", "score", "model_id"})

    def test_metrics_use_concatenated_predictions(self) -> None:
        actual = np.asarray([1, 1, 0, 0], dtype=np.int8)
        predicted = np.asarray([1, 0, 1, 0], dtype=np.int8)
        metrics = pipeline.metrics_from_codes(actual, predicted)
        self.assertEqual(metrics["confusionMatrix"]["matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["macroF1"], 0.5)


if __name__ == "__main__":
    unittest.main()
