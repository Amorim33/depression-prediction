# SetembroBR Ternary Results

Status: reproduced with the strict-blind ternary workflow. The current train-only lock includes
post-report boosted tabular candidates and bounded selection groups; stacking candidates were also
tested but were not selected. The current lock has not been final-test evaluated.

All selection used train OOF probabilities only. For the historical pre-XGBoost run, final test labels were read only by
`make ternary-evaluate-test-setembrobr` after
`outputs/setembrobr/seed42_ternary_strict_blind/ensemble/ensemble-lock.json`
already existed.

## Current Train-Only Lock

- Label policy: `diag_evidence_q20`
- Label policy hash: `06ad79f1e2ac608e7ebf87f2d23a8279b57ede196121302d064c87cbbc3098cd`
- Original split manifest hash: `08ce39f8863fc57165f4b8efe57b4a31b764ab6b974968f0ab0a5e167d44108e`
- Selection group: `tabular_core_xgb_s42_shallow`
- Selection strategy: `exhaustive`
- Decision rule: `diagnosed_margin_005`
- Models: `ternary_hier_logreg_gate`, `ternary_mlp_h128_s42`, `ternary_xgb_expanded_pca_s13`, `ternary_xgb_shallow_pca_s99`, `ternary_xgb_tabular_markers_s42`
- Weights: `0.25`, `0.15`, `0.25`, `0.20`, `0.15`

## Train OOF Selection Table

| Rank | Label policy | Group | Decision rule | Models | Macro F1 | Diagnosed F1 | Diagnosed precision | Accuracy |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `diag_evidence_q20` | `tabular_core_xgb_s42_shallow` | `diagnosed_margin_005` | 5 | `0.804996` | `0.917987` | `0.921422` | `0.846591` |
| 2 | `diag_evidence_q20` | `tabular_core_xgb_s42_rich` | `diagnosed_margin_005` | 6 | `0.803929` | `0.917096` | `0.925331` | `0.844886` |
| 3 | `diag_evidence_q20` | `tabular_legacy_without_relevance_baseline` | `argmax` | 4 | `0.796108` | `0.918894` | `0.912168` | `0.843182` |
| 4 | `diag_evidence_q20` | `tabular_legacy_all` | `argmax` | 4 | `0.796108` | `0.918894` | `0.912168` | `0.843182` |
| 5 | `diag_evidence_q10` | `tabular_core_xgb_s42_rich` | `argmax` | 6 | `0.793001` | `0.900924` | `0.917094` | `0.836364` |

Rows 3-4 preserve the previous train-only lock under explicit legacy tabular groups.

## Historical Final Test Metrics

These metrics belong to the earlier pre-boosted-core lock. They are retained as a historical strict-blind
report. No final test evaluation has been run for the current boosted-core train-only lock.

| Metric | Value |
| --- | ---: |
| Macro F1 | `0.315936` |
| Diagnosed F1 | `0.312281` |
| Diagnosed precision | `0.185933` |
| Diagnosed recall | `0.974453` |
| Accuracy | `0.387611` |

## Historical Final Test Confusion Matrix

| Actual \ Predicted | diagnosed | control | no-evidence |
| --- | ---: | ---: | ---: |
| diagnosed | 267 | 5 | 2 |
| control | 1157 | 730 | 472 |
| no-evidence | 12 | 3 | 48 |

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

- Fold-combination robustness ranked the current train-only lock first with mean Macro F1 `0.804956` and minimum Macro F1 `0.767445`.
- Group-aware nested OOF split-selection reported held-out train-fold mean Macro F1 `0.787314` and minimum Macro F1 `0.760210`.
- OOF probability diagnostics for the current train-only lock reported Brier score `0.243979`, negative log likelihood `0.403552`, and expected calibration error `0.062549`.
- The train-only single-model leaderboard ranked `diag_evidence_q20/ternary_xgb_expanded_pca_s42` first with OOF Macro F1 `0.788910`.
- Family ablation selected `diag_evidence_q20/tabular_core_xgb_s42_shallow` with OOF Macro F1 `0.804996`.
- The best stacking candidate reached OOF Macro F1 `0.745698` under `diag_evidence_q20`; stacking was not selected by the current train-OOF lock.

## Artifacts

- Lock: `outputs/setembrobr/seed42_ternary_strict_blind/ensemble/ensemble-lock.json`
- Candidate table: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ensemble-candidates.json`
- Train-only robustness: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-robustness.json`
- Nested OOF split-selection: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-nested-oof-selection.json`
- OOF probability diagnostics: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-oof-diagnostics.json`
- Model/policy leaderboard: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-model-policy-leaderboard.json`
- Family ablation: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-family-ablation.json`
- Historical pre-XGBoost final report: `outputs/setembrobr/seed42_ternary_strict_blind/reports/final-test-report.json`
- GPU run manifest: `outputs/setembrobr/seed42_ternary_strict_blind/gpu-runs/fedora-ternary-seq-oof.json`

Reports are aggregate-only. They contain no raw tweet text.
