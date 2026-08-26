from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "setembrobr_v7_qwen_logreg", PROJECT / "scripts/setembrobr_v7_qwen_logreg.py"
)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compress_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, destination.open("wb") as output_handle:
        zstd.ZstdCompressor(level=1).copy_stream(input_handle, output_handle)


class SyntheticExperiment:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = root / "SetembroBR-v7-min.pkl"
        self.archive = root / "archive"
        self.output = pipeline.ArtifactPaths(root / "output")
        self.temporary = root / "temporary"
        self.config_path = root / "config.json"
        self.dimension = 3
        self.user_ids = [f"u{index:02d}" for index in range(14)]
        self.target_texts: dict[str, list[str]] = {}
        self.source_texts: dict[str, list[str]] = {}
        self.source_vectors: dict[str, np.ndarray] = {}
        self._write_source()
        archive_values = self._write_archive()
        self._write_config(archive_values)
        self.config = pipeline.load_config(self.config_path)

    def _write_source(self) -> None:
        rows = []
        for index, user_id in enumerate(self.user_ids):
            texts = [f"{user_id}-first", "duplicate", "duplicate", f"{user_id}-last"]
            retained = [texts[0], texts[1], texts[2]] if index % 2 == 0 else [texts[1], texts[3]]
            self.source_texts[user_id] = texts
            self.target_texts[user_id] = retained
            split = "train" if index < 12 else "test"
            label = "yes" if (index < 6 or index == 12) else "no"
            rows.append(
                {
                    "User_ID": user_id,
                    "Diagnosed_YN": label,
                    "TextLists": retained,
                    "Split": split,
                }
            )
        pd.DataFrame(rows, columns=pipeline.SOURCE_COLUMNS).to_pickle(self.source)

    def _write_archive(self) -> dict[str, str]:
        original_parquet = self.root / "source.parquet"
        user_column: list[str] = []
        tweet_indexes: list[int] = []
        text_column: list[str] = []
        vector_rows: list[np.ndarray] = []
        for user_index, user_id in enumerate(self.user_ids):
            vectors = []
            for tweet_index, text in enumerate(self.source_texts[user_id]):
                vector = np.asarray(
                    [user_index + 0.25, tweet_index + 0.5, (user_index + 1) * (tweet_index + 1)],
                    dtype=np.float16,
                )
                vectors.append(vector)
                user_column.append(user_id)
                tweet_indexes.append(tweet_index)
                text_column.append(text)
                vector_rows.append(vector)
            self.source_vectors[user_id] = np.stack(vectors)
        flat = pa.array(np.stack(vector_rows).reshape(-1), type=pa.float16())
        embedding_column = pa.FixedSizeListArray.from_arrays(flat, self.dimension)
        table = pa.table(
            {
                "user_id": pa.array(user_column),
                "tweet_index": pa.array(tweet_indexes, type=pa.int32()),
                "tweet_text": pa.array(text_column),
                "embedding": embedding_column,
            }
        )
        pq.write_table(table, original_parquet)

        shard_relative = "payload/depression/artifacts/tweet_embeddings/train/part-000000-000014.parquet.zst"
        shard_path = self.archive / shard_relative
        compress_file(original_parquet, shard_path)

        pooled_relative = "payload/depression/artifacts/pooled/train_user_mean.npz"
        pooled_path = self.archive / pooled_relative
        pooled_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            pooled_path,
            user_ids=np.asarray(self.user_ids, dtype=object),
            embeddings=np.zeros((len(self.user_ids), self.dimension), dtype=np.float32),
        )

        split_hash = "f" * 64
        generation_payload = {
            "splitManifestHash": split_hash,
            "embedding": {
                "modelId": "synthetic/qwen",
                "modelRevision": "synthetic-revision",
                "embeddingDimension": self.dimension,
                "embeddingStorageDtype": "float16",
            },
        }
        generation_original = self.root / "generation.json"
        generation_original.write_text(json.dumps(generation_payload))
        generation_relative = "payload/depression/artifacts/reports/embedding_generation_manifest.json.zst"
        generation_path = self.archive / generation_relative
        compress_file(generation_original, generation_path)

        records = [
            {
                "sourceId": "depression/artifacts",
                "archivePath": shard_relative,
                "archiveSha256": sha256(shard_path),
                "originalSha256": sha256(original_parquet),
                "originalSize": original_parquet.stat().st_size,
            },
            {
                "sourceId": "depression/artifacts",
                "archivePath": pooled_relative,
                "archiveSha256": sha256(pooled_path),
                "originalSha256": sha256(pooled_path),
                "originalSize": pooled_path.stat().st_size,
            },
        ]
        state_path = self.archive / ".archive-state/manifest.jsonl"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return {
            "shardRelative": shard_relative,
            "generationArchiveSha256": sha256(generation_path),
            "generationSha256": sha256(generation_original),
            "splitHash": split_hash,
        }

    def _write_config(self, archive_values: dict[str, str]) -> None:
        post_count = sum(len(texts) for texts in self.target_texts.values())
        payload = {
            "experimentId": "synthetic_setembrobr_v7",
            "modelId": "synthetic_qwen_logreg",
            "predictionColumn": "qwen_logistic_regression_label",
            "source": {
                "filename": self.source.name,
                "sha256": sha256(self.source),
                "schema": pipeline.SOURCE_COLUMNS,
                "expectedUsers": 14,
                "expectedPosts": post_count,
                "splits": {
                    "train": {"users": 12, "positive": 6, "negative": 6},
                    "test": {"users": 2, "positive": 1, "negative": 1},
                },
            },
            "embeddings": {
                "archiveSourceId": "depression/artifacts",
                "archiveStateManifest": ".archive-state/manifest.jsonl",
                "generationManifest": "payload/depression/artifacts/reports/embedding_generation_manifest.json.zst",
                "generationManifestArchiveSha256": archive_values["generationArchiveSha256"],
                "generationManifestSha256": archive_values["generationSha256"],
                "pooledUserIndex": "payload/depression/artifacts/pooled/train_user_mean.npz",
                "shardGlob": "payload/depression/artifacts/tweet_embeddings/train/*.parquet.zst",
                "expectedTrainShards": 1,
                "sourceUserCount": 14,
                "sourceSplit": "train",
                "sourceSplitManifestSha256": archive_values["splitHash"],
                "modelId": "synthetic/qwen",
                "modelRevision": "synthetic-revision",
                "dimension": self.dimension,
                "storageDtype": "float16",
                "pooling": "float64_sum_divide_then_float32",
                "normalizedPostEmbeddings": True,
            },
            "validation": {
                "seed": 42,
                "foldCount": 5,
                "thresholdMetric": "macroF1",
                "thresholdTieBreakers": ["diagnosedF1", "precision", "recall"],
                "bootstrapSamples": 50,
            },
            "classifier": {
                "standardScaler": True,
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "maxIter": 1000,
                "classWeight": "balanced",
                "randomState": 42,
            },
        }
        self.config_path.write_text(json.dumps(payload, indent=2) + "\n")


class CoreUnitTests(unittest.TestCase):
    def test_locked_config_and_label_mapping(self) -> None:
        config = json.loads((PROJECT / "config.json").read_text())
        self.assertEqual(config["source"]["expectedUsers"], 8002)
        self.assertEqual(config["source"]["expectedPosts"], 4356312)
        self.assertEqual(config["validation"]["seed"], 42)
        self.assertEqual(config["embeddings"]["dimension"], 2560)
        self.assertEqual(pipeline.LABEL_TO_CODE, {"no": 0, "yes": 1})
        self.assertEqual(pipeline.CODE_TO_LABEL, {0: "no", 1: "yes"})

    def test_folds_are_deterministic_and_stratified(self) -> None:
        users = [f"u{index}" for index in range(20)]
        labels = [0] * 10 + [1] * 10
        first = pipeline.assign_folds(users, labels, 5, 42)
        second = pipeline.assign_folds(users, labels, 5, 42)
        np.testing.assert_array_equal(first, second)
        for fold in range(5):
            selected = np.asarray(labels)[first == fold]
            self.assertEqual(selected.tolist().count(0), 2)
            self.assertEqual(selected.tolist().count(1), 2)

    def test_threshold_sweep_uses_declared_tie_breakers(self) -> None:
        actual = np.asarray([0, 0, 1, 1], dtype=np.int8)
        scores = np.asarray([0.1, 0.4, 0.6, 0.9])
        threshold, metrics = pipeline.select_threshold(actual, scores)
        np.testing.assert_array_equal((scores >= threshold).astype(np.int8), actual)
        self.assertEqual(metrics["macroF1"], 1.0)

    def test_duplicate_alignment_preserves_occurrence_and_order(self) -> None:
        indexes = pipeline.match_ordered_posts(["a", "x", "a", "b"], ["a", "a", "b"])
        np.testing.assert_array_equal(indexes, np.asarray([0, 2, 3]))
        for target in (["missing"], ["b", "a"], ["a", "changed"]):
            with self.subTest(target=target), self.assertRaises(RuntimeError):
                pipeline.match_ordered_posts(["a", "b"], target)


class SyntheticArchiveIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.experiment = SyntheticExperiment(Path(self.temporary_directory.name))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_full_strict_blind_pipeline_and_pooling_precision(self) -> None:
        item = self.experiment
        source_hash_before = sha256(item.source)
        pipeline.validate_command(item.source, item.archive, item.config, item.output)
        pipeline.prepare_command(item.source, item.config, item.output)

        sealed_copy = item.root / "sealed-labels.csv"
        shutil.move(item.output.sealed / "test_labels.csv", sealed_copy)
        pipeline.pool_command(item.archive, item.config, item.output, item.temporary)

        train = pd.read_pickle(item.output.sanitized / "train.pkl")
        pool_ids, pool_vectors, counts = pipeline.load_pool(
            item.output.embeddings / "train_user_mean.npz", item.dimension
        )
        first_user = str(train.iloc[0]["User_ID"])
        retained = item.target_texts[first_user]
        source = item.source_texts[first_user]
        positions = pipeline.match_ordered_posts(source, retained)
        expected = (
            item.source_vectors[first_user][positions].astype(np.float64).sum(axis=0) / len(positions)
        ).astype(np.float32)
        observed_position = pool_ids.index(first_user)
        np.testing.assert_array_equal(pool_vectors[observed_position], expected)
        self.assertEqual(int(counts[observed_position]), len(retained))

        pipeline.train_oof_command(item.config, item.output)
        pipeline.audit_oof_command(item.config, item.output)
        pipeline.lock_command(item.config, item.output)
        pipeline.fit_full_command(item.config, item.output)
        pipeline.score_test_command(item.config, item.output)
        pipeline.audit_test_command(item.config, item.output)

        score_rows = pipeline.read_csv(
            item.output.scores / f"test_{item.config['modelId']}.csv"
        )
        self.assertEqual(len(score_rows), 2)
        self.assertEqual(list(score_rows[0]), ["user_id", "score", "model_id"])
        self.assertNotIn("label", score_rows[0])
        self.assertNotIn("fold", score_rows[0])
        self.assertFalse((item.output.sealed / "test_labels.csv").exists())

        shutil.move(sealed_copy, item.output.sealed / "test_labels.csv")
        pipeline.evaluate_command(item.config, item.output)
        pipeline.materialize_command(item.source, item.config, item.output)
        pipeline.audit_final_command(item.source, item.config, item.output)

        derived = pd.read_pickle(item.output.corpus / "SetembroBR-v7-min-qwen-logreg.pkl")
        prediction_column = item.config["predictionColumn"]
        self.assertEqual(int(derived[prediction_column].notna().sum()), 2)
        self.assertTrue(derived.loc[derived["Split"] == "train", prediction_column].isna().all())
        self.assertTrue(derived.loc[derived["Split"] == "test", prediction_column].notna().all())
        self.assertEqual(sha256(item.source), source_hash_before)
        self.assertTrue(json.loads((item.output.reports / "final-audit.json").read_text())["ok"])

    def test_archived_shard_rejects_missing_reordered_and_mismatched_posts(self) -> None:
        item = self.experiment
        records = pipeline.archive_records(item.archive, item.config)
        relative = "payload/depression/artifacts/tweet_embeddings/train/part-000000-000014.parquet.zst"
        source_path = item.archive / relative
        user_id = item.user_ids[0]
        invalid_targets = {
            "missing": [item.target_texts[user_id][0], "not-present"],
            "reordered": list(reversed(item.target_texts[user_id])),
            "mismatched": ["changed-text"],
        }
        for name, target in invalid_targets.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                pipeline.pool_one_shard(
                    source_path,
                    records[relative],
                    [user_id],
                    {user_id: target},
                    item.dimension,
                    item.root / f"bad-{name}.npz",
                    item.config["_configSha256"],
                    item.temporary,
                )

    def test_source_schema_validation_refuses_mutation(self) -> None:
        item = self.experiment
        invalid = pd.read_pickle(item.source).drop(columns=["Split"])
        invalid_path = item.root / "invalid.pkl"
        invalid.to_pickle(invalid_path)
        changed_config = json.loads(item.config_path.read_text())
        changed_config["source"]["sha256"] = sha256(invalid_path)
        changed_path = item.root / "invalid-config.json"
        changed_path.write_text(json.dumps(changed_config))
        with self.assertRaises(RuntimeError):
            pipeline.normalized_source_frame(invalid_path, pipeline.load_config(changed_path))


if __name__ == "__main__":
    unittest.main()
