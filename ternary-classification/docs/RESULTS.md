# SetembroBR Ternary Results

Status: reproduced with the strict-blind ternary workflow. The current train-only lock includes
boosted tabular candidates, bounded selection groups, and pre-registered local weight refinement for
the strongest boosted-core groups. The current train-only lock adds an HGB candidate and uses the
embedding-rich XGBoost variant instead of the shallow XGBoost variant. Stacking candidates were also
tested but were not selected. The current lock was final-test evaluated after the lock existed.

All selection used train OOF probabilities only. Final test labels were read only by
`make ternary-evaluate-test-setembrobr` after the locked ensemble already existed at
`outputs/setembrobr/seed42_ternary_strict_blind/ensemble/ensemble-lock.json`.

## Current Train-Only Lock

- Label policy: `diag_evidence_q20`
- Label policy hash: `06ad79f1e2ac608e7ebf87f2d23a8279b57ede196121302d064c87cbbc3098cd`
- Original split manifest hash: `08ce39f8863fc57165f4b8efe57b4a31b764ab6b974968f0ab0a5e167d44108e`
- Selection group: `tabular_core_xgb_hgb_rich`
- Selection strategy: `exhaustive+local-refine(step=0.01,radius=0.03)`
- Decision rule: `argmax`
- Models: `ternary_hgb_expanded_pca_s42`, `ternary_hier_logreg_gate`, `ternary_mlp_h128_s42`, `ternary_xgb_embedding_rich_s7`, `ternary_xgb_expanded_pca_s13`, `ternary_xgb_tabular_markers_s42`
- Weights: `0.21`, `0.23`, `0.21`, `0.02`, `0.29`, `0.04`

## Train OOF Selection Table

| Rank | Label policy | Group | Decision rule | Models | Macro F1 | Diagnosed F1 | Diagnosed precision | Accuracy |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `diag_evidence_q20` | `tabular_core_xgb_hgb_rich` | `argmax` | 6 | `0.807566` | `0.924277` | `0.913793` | `0.852841` |
| 2 | `diag_evidence_q20` | `tabular_core_xgb_hgb_shallow` | `diagnosed_margin_005` | 6 | `0.807358` | `0.917251` | `0.923729` | `0.847159` |
| 3 | `diag_evidence_q20` | `tabular_core_xgb_s42_shallow` | `diagnosed_margin_005` | 5 | `0.806280` | `0.917910` | `0.922212` | `0.847159` |
| 4 | `diag_evidence_q20` | `tabular_core_xgb_s42_rich` | `diagnosed_margin_005` | 6 | `0.806121` | `0.917680` | `0.924599` | `0.846591` |
| 5 | `diag_evidence_q20` | `tabular_legacy_all` | `argmax` | 4 | `0.796108` | `0.918894` | `0.912168` | `0.843182` |

Row 5 preserves the previous train-only lock under an explicit legacy tabular group.

## Current Final Test Metrics

These metrics belong to the current HGB-rich boosted-core lock. The train-OOF gain did not generalize
to the final test set.

| Metric | Value |
| --- | ---: |
| Macro F1 | `0.242820` |
| Diagnosed F1 | `0.307866` |
| Diagnosed precision | `0.182184` |
| Diagnosed recall | `0.992701` |
| Accuracy | `0.265579` |

## Current Final Test Confusion Matrix

| Actual \ Predicted | diagnosed | control | no-evidence |
| --- | ---: | ---: | ---: |
| diagnosed | 272 | 2 | 0 |
| control | 1221 | 381 | 757 |
| no-evidence | 0 | 0 | 63 |

The earlier pre-boosted-core final test report had Macro F1 `0.315936`, diagnosed F1 `0.312281`,
diagnosed precision `0.185933`, diagnosed recall `0.974453`, and accuracy `0.387611`. That historical
lock remains a prior strict-blind result; it must not be used to reselect or tune the current lock.

## Label-Policy Evidence Summary

| Label policy | Cutoff | Train diagnosed | Train control | Train no-evidence |
| --- | ---: | ---: | ---: | ---: |
| `diag_rel3_zero` | n/a | 1347 | 413 | 0 |
| `diag_rel5_zero` | n/a | 1347 | 413 | 0 |
| `diag_rel6_zero` | n/a | 1347 | 413 | 0 |
| `diag_low_density` | n/a | 1347 | 413 | 0 |
| `diag_top10_avg_lt3` | n/a | 1347 | 413 | 0 |
| `diag_evidence_q10` | `0.446368022` | 1212 | 413 | 135 |
| `diag_evidence_q20` | `0.499652214` | 1077 | 413 | 270 |

## Train-Only Diagnostics

- Fold-combination robustness ranked the current train-only lock first with mean Macro F1 `0.807488` and minimum Macro F1 `0.785349`.
- Group-aware nested OOF split-selection reported held-out train-fold mean Macro F1 `0.789996` and minimum Macro F1 `0.760899`; the OOF lock improved, but this nested mean remains lower than the previous boosted-core check.
- OOF probability diagnostics for the current train-only lock reported Brier score `0.239726`, negative log likelihood `0.396376`, and expected calibration error `0.044242`.
- The train-only single-model leaderboard ranked `diag_evidence_q20/ternary_xgb_expanded_pca_s42` first with OOF Macro F1 `0.788910`.
- Family ablation selected `diag_evidence_q20/tabular_core_xgb_hgb_rich` with OOF Macro F1 `0.807566`.
- The best stacking candidate reached OOF Macro F1 `0.745698` under `diag_evidence_q20`; stacking was not selected by the current train-OOF lock.

## Artifacts

- Lock: `outputs/setembrobr/seed42_ternary_strict_blind/ensemble/ensemble-lock.json`
- Candidate table: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ensemble-candidates.json`
- Train-only robustness: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-robustness.json`
- Nested OOF split-selection: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-nested-oof-selection.json`
- OOF probability diagnostics: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-oof-diagnostics.json`
- Model/policy leaderboard: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-model-policy-leaderboard.json`
- Family ablation: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-family-ablation.json`
- Current final test report: `outputs/setembrobr/seed42_ternary_strict_blind/reports/final-test-report.json`
- GPU run manifest: `outputs/setembrobr/seed42_ternary_strict_blind/gpu-runs/fedora-ternary-seq-oof.json`

Reports are aggregate-only. They contain no raw tweet text.
