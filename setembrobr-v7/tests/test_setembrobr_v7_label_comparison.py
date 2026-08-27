from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "setembrobr_v7_label_comparison", SCRIPTS / "setembrobr_v7_label_comparison.py"
)
assert SPEC and SPEC.loader
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def write_npz(path: Path, user_ids: list[str], embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        user_ids=np.asarray(user_ids, dtype=object),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        counts=np.ones(len(user_ids), dtype=np.int32),
    )


class SyntheticComparison:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = root / "SetembroBR-v7-min.pkl"
        self.specialist = root / "corpus.csv"
        self.baseline = root / "baseline"
        self.output = comparison.ComparisonPaths(root / "comparison")
        self.config_path = root / "config.json"
        self.dimension = 4
        self.train_ids = [f"u{index:02d}" for index in range(30)]
        self.test_ids = [f"u{index:02d}" for index in range(30, 40)]
        self.original: dict[str, str] = {}
        self.specialist_labels: dict[str, str] = {}
        self.features: dict[str, np.ndarray] = {}
        self._write_source_and_labels()
        self._write_baseline_artifacts()
        self._write_config()
        self.config = comparison.load_config(self.config_path)

    def _write_source_and_labels(self) -> None:
        rows = []
        specialist_rows = []
        for index, user_id in enumerate([*self.train_ids, *self.test_ids]):
            if index < 30:
                fold = index % 5
                within_fold = index // 5
                original_yes = within_fold in {0, 1}
                specialist_yes = within_fold in {1, 2}
            else:
                test_index = index - 30
                original_yes = test_index < 5
                specialist_yes = test_index in {1, 2, 5, 6}
                fold = -1
            original = "yes" if original_yes else "no"
            specialist = "signal" if specialist_yes else "no_signal"
            self.original[user_id] = original
            self.specialist_labels[user_id] = specialist
            vector = np.asarray(
                [
                    2.0 if original_yes else -2.0,
                    2.0 if specialist_yes else -2.0,
                    float((index % 3) - 1),
                    float(fold) / 5.0,
                ],
                dtype=np.float32,
            )
            self.features[user_id] = vector
            rows.append(
                {
                    "User_ID": user_id,
                    "Diagnosed_YN": original,
                    "TextLists": [f"timeline-{user_id}"],
                    "Split": "train" if index < 30 else "test",
                }
            )
            specialist_rows.append(
                {
                    "user_id": user_id,
                    "label": specialist,
                    "score": "0.9" if specialist_yes else "0.1",
                    "split": "train",
                    "model_id": "synthetic-specialist",
                    "lock_sha256": "f" * 64,
                }
            )
        pd.DataFrame(rows).to_pickle(self.source)
        pd.DataFrame(specialist_rows).to_csv(self.specialist, index=False)

    def _write_baseline_artifacts(self) -> None:
        baseline = self.baseline
        (baseline / "manifests").mkdir(parents=True, exist_ok=True)
        (baseline / "reports").mkdir(parents=True, exist_ok=True)
        (baseline / "ensemble").mkdir(parents=True, exist_ok=True)
        (baseline / "scores").mkdir(parents=True, exist_ok=True)
        (baseline / "manifests/prepared-manifest.json").write_text('{"ok":true}\n')
        train_rows = [
            {
                "user_id": user_id,
                "label": self.original[user_id],
                "fold": index % 5,
            }
            for index, user_id in enumerate(self.train_ids)
        ]
        comparison.write_csv(
            baseline / "manifests/train_manifest.csv",
            ["user_id", "label", "fold"],
            train_rows,
        )
        comparison.write_csv(
            baseline / "manifests/test_inference.csv",
            ["user_id"],
            ({"user_id": user_id} for user_id in self.test_ids),
        )
        train_features = np.stack([self.features[user_id] for user_id in self.train_ids])
        test_features = np.stack([self.features[user_id] for user_id in self.test_ids])
        write_npz(baseline / "embeddings/train_user_mean.npz", self.train_ids, train_features)
        write_npz(baseline / "embeddings/test_user_mean.npz", self.test_ids, test_features)

        classifier_config = {
            "classifier": {
                "standardScaler": True,
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "maxIter": 1000,
                "classWeight": "balanced",
                "randomState": 42,
            }
        }
        labels = np.asarray(
            [comparison.GENERIC_LABEL_TO_CODE[self.original[user_id]] for user_id in self.train_ids]
        )
        folds = np.asarray([index % 5 for index in range(len(self.train_ids))])
        oof = np.full(len(self.train_ids), np.nan, dtype=np.float64)
        for fold in range(5):
            validation = folds == fold
            model = comparison.build_classifier(classifier_config)
            model.fit(train_features[~validation], labels[~validation])
            oof[validation] = comparison.positive_probability(model, train_features[validation])
        threshold, _metrics = comparison.select_threshold(labels, oof)
        full_model = comparison.build_classifier(classifier_config)
        full_model.fit(train_features, labels)
        test_scores = comparison.positive_probability(full_model, test_features)
        comparison.write_csv(
            baseline / "scores/test_qwen3_mean_logreg_s42.csv",
            ["user_id", "score", "model_id"],
            (
                {
                    "user_id": user_id,
                    "score": repr(float(score)),
                    "model_id": "qwen3_mean_logreg_s42",
                }
                for user_id, score in zip(self.test_ids, test_scores)
            ),
        )
        test_actual = np.asarray(
            [comparison.GENERIC_LABEL_TO_CODE[self.original[user_id]] for user_id in self.test_ids]
        )
        test_predicted = (test_scores >= threshold).astype(np.int8)
        test_metrics = comparison.metrics_from_codes(test_actual, test_predicted)
        comparison.write_json(
            baseline / "ensemble/model-lock.json",
            {"threshold": threshold},
        )
        comparison.write_json(
            baseline / "reports/final-test-report.json",
            {"metrics": test_metrics},
        )

    def _write_config(self) -> None:
        payload = {
            "experimentId": "synthetic_label_comparison",
            "source": {
                "filename": self.source.name,
                "sha256": comparison.sha256_file(self.source),
                "expectedUsers": 40,
                "trainUsers": 30,
                "testUsers": 10,
            },
            "specialistLabels": {
                "filename": self.specialist.name,
                "sha256": comparison.sha256_file(self.specialist),
                "userColumn": "user_id",
                "labelColumn": "label",
                "expectedRows": 40,
                "expectedSignal": 14,
                "expectedNoSignal": 26,
            },
            "baselineArtifacts": {
                "preparedManifestSha256": comparison.sha256_file(
                    self.baseline / "manifests/prepared-manifest.json"
                ),
                "trainManifestSha256": comparison.sha256_file(
                    self.baseline / "manifests/train_manifest.csv"
                ),
                "testInferenceSha256": comparison.sha256_file(
                    self.baseline / "manifests/test_inference.csv"
                ),
                "trainPoolSha256": comparison.sha256_file(
                    self.baseline / "embeddings/train_user_mean.npz"
                ),
                "testPoolSha256": comparison.sha256_file(
                    self.baseline / "embeddings/test_user_mean.npz"
                ),
                "originalLockSha256": comparison.sha256_file(
                    self.baseline / "ensemble/model-lock.json"
                ),
                "originalTestScoreSha256": comparison.sha256_file(
                    self.baseline / "scores/test_qwen3_mean_logreg_s42.csv"
                ),
                "originalFinalReportSha256": comparison.sha256_file(
                    self.baseline / "reports/final-test-report.json"
                ),
            },
            "expectedTargets": {
                "original_diagnosis": {
                    "positiveName": "diagnosed",
                    "trainPositive": 10,
                    "trainNegative": 20,
                    "testPositive": 5,
                    "testNegative": 5,
                },
                "specialist_signal": {
                    "positiveName": "signal",
                    "trainPositive": 10,
                    "trainNegative": 20,
                    "testPositive": 4,
                    "testNegative": 6,
                },
            },
            "validation": {
                "seed": 42,
                "foldCount": 5,
                "thresholdMetric": "macroF1",
                "thresholdTieBreakers": ["positiveF1", "precision", "recall"],
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
            "embedding": {
                "modelId": "synthetic/qwen",
                "modelRevision": "synthetic",
                "dimension": self.dimension,
                "pooling": "mean",
            },
        }
        self.config_path.write_text(json.dumps(payload, indent=2) + "\n")


class LabelComparisonUnitTests(unittest.TestCase):
    def test_real_config_pins_both_target_counts(self) -> None:
        config = json.loads((PROJECT / "label-comparison-config.json").read_text())
        self.assertEqual(config["specialistLabels"]["labelColumn"], "label")
        self.assertEqual(config["expectedTargets"]["original_diagnosis"]["trainPositive"], 773)
        self.assertEqual(config["expectedTargets"]["specialist_signal"]["trainPositive"], 377)
        self.assertEqual(config["expectedTargets"]["specialist_signal"]["testPositive"], 86)

    def test_label_mapping_is_explicit(self) -> None:
        self.assertEqual(comparison.normalize_original_label("yes"), "yes")
        self.assertEqual(comparison.normalize_original_label("no"), "no")
        self.assertEqual(comparison.normalize_specialist_label("signal"), "yes")
        self.assertEqual(comparison.normalize_specialist_label("no_signal"), "no")
        with self.assertRaises(RuntimeError):
            comparison.normalize_specialist_label("ambiguous")

    def test_paired_bootstrap_is_deterministic(self) -> None:
        original = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int8)
        specialist = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
        first = comparison.paired_bootstrap_delta(
            original, original, specialist, specialist, 25, 42
        )
        second = comparison.paired_bootstrap_delta(
            original, original, specialist, specialist, 25, 42
        )
        self.assertEqual(first, second)
        self.assertEqual(first, [0.0, 0.0])


class LabelComparisonIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.experiment = SyntheticComparison(Path(self.temporary_directory.name))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_full_paired_pipeline_and_test_label_isolation(self) -> None:
        item = self.experiment
        comparison.prepare_command(
            item.source, item.specialist, item.baseline, item.config, item.output
        )
        sealed_copy = item.root / "sealed-copy"
        shutil.move(item.output.sealed, sealed_copy)

        comparison.train_oof_command(item.baseline, item.config, item.output)
        comparison.audit_oof_command(item.baseline, item.config, item.output)
        comparison.lock_command(item.config, item.output)
        comparison.fit_full_command(item.baseline, item.config, item.output)
        comparison.score_test_command(item.baseline, item.config, item.output)
        comparison.audit_test_command(item.baseline, item.config, item.output)

        for target in comparison.TARGETS:
            score_rows = comparison.read_csv(comparison.target_test_score_path(item.output, target))
            self.assertEqual(set(score_rows[0]), {"user_id", "score", "model_id"})
            self.assertEqual(len(score_rows), 10)
        self.assertFalse(any(item.output.sealed.iterdir()))

        for sealed_file in sealed_copy.iterdir():
            shutil.move(sealed_file, item.output.sealed / sealed_file.name)
        comparison.evaluate_command(item.baseline, item.config, item.output)
        comparison.audit_final_command(
            item.source, item.specialist, item.baseline, item.config, item.output
        )
        report = json.loads((item.output.reports / "final-comparison-report.json").read_text())
        audit = json.loads((item.output.reports / "final-audit.json").read_text())
        self.assertTrue(report["ok"])
        self.assertTrue(report["originalBaselineReproduction"]["ok"])
        self.assertEqual(
            report["originalBaselineReproduction"]["maximumAbsoluteTestScoreDelta"], 0.0
        )
        self.assertEqual(set(report["targets"]), set(comparison.TARGETS))
        self.assertTrue(audit["sameUsersTimelinesSplitFoldsFeaturesClassifier"])
        self.assertTrue(audit["onlyTargetLabelsDiffer"])

    def test_prepare_rejects_missing_specialist_user(self) -> None:
        item = self.experiment
        labels = pd.read_csv(item.specialist).iloc[:-1]
        invalid = item.root / "missing.csv"
        labels.to_csv(invalid, index=False)
        changed = dict(item.config)
        changed["specialistLabels"] = dict(item.config["specialistLabels"])
        changed["specialistLabels"]["sha256"] = comparison.sha256_file(invalid)
        changed["specialistLabels"]["expectedRows"] = len(labels)
        changed["specialistLabels"]["expectedNoSignal"] -= 1
        with self.assertRaisesRegex(RuntimeError, "missing 1 v7 users"):
            comparison.prepare_command(
                item.source, invalid, item.baseline, changed, item.output
            )


if __name__ == "__main__":
    unittest.main()
