from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent


class EnsembleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((PROJECT / "ensemble-config.json").read_text())
        cls.script = (PROJECT / "scripts/setembrobr_v7_ensemble.py").read_text()

    def test_fixed_champion_model_set_is_exact(self) -> None:
        self.assertEqual(
            self.config["ensemble"]["requiredModelIds"],
            [
                "binary_legacy_focal_combined_g1",
                "binary_legacy_logreg_combined_s42",
                "binary_legacy_seq_cnn_top128_s13",
                "binary_stack_logreg_boosted_core",
                "binary_stack_logreg_xgb_tabular",
            ],
        )
        self.assertEqual(set(self.config["sourceChampion"]["weights"].values()), {0.2})

    def test_pinned_feature_helper_hash(self) -> None:
        helper = REPOSITORY / "scripts/raw_ternary_prepare_setembrobr.py"
        if not helper.is_file():
            helper = PROJECT / "shared/raw_ternary_prepare_setembrobr.py"
        observed = hashlib.sha256(helper.read_bytes()).hexdigest()
        self.assertEqual(observed, self.config["featureHelperSha256"])

    def test_test_preparation_requires_oof_lock_first(self) -> None:
        function = self.script[self.script.index("def prepare_split(") : self.script.index("def load_features(")]
        lock_check = function.index('if split == "test":\n        require_lock(output)')
        label_free_read = function.index('sanitized_path = baseline.sanitized / f"{split}.pkl"')
        self.assertLess(lock_check, label_free_read)

    def test_evaluation_requires_label_free_audit_before_sealed_labels(self) -> None:
        function = self.script[self.script.index("def evaluate(") : self.script.index("def run_oof(")]
        audit_check = function.index('label-free score audit is required before opening labels')
        label_read = function.index('sealed_path = baseline.sealed / "test_labels.csv"')
        self.assertLess(audit_check, label_read)

    def test_label_free_score_schema_is_exact(self) -> None:
        self.assertIn(
            '["user_id", "score", "model_id"]',
            self.script,
        )


if __name__ == "__main__":
    unittest.main()
