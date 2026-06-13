# SetembroBR Ternary Results

Status: reproduced with the strict-blind ternary workflow.

All selection used train OOF probabilities only. Final test labels were read only by
`make ternary-evaluate-test-setembrobr` after
`outputs/setembrobr/seed42_ternary_strict_blind/ensemble/ensemble-lock.json`
already existed.

## Locked Champion

- Label policy: `diag_evidence_q20`
- Label policy hash: `06ad79f1e2ac608e7ebf87f2d23a8279b57ede196121302d064c87cbbc3098cd`
- Original split manifest hash: `08ce39f8863fc57165f4b8efe57b4a31b764ab6b974968f0ab0a5e167d44108e`
- Selection strategy: `ranked-prefix-pruned(top=10,max=8,step=0.05)`
- Decision rule: `diagnosed_margin_005`
- Models: `ternary_extra_trees_evidence`, `ternary_hier_logreg_gate`, `ternary_logreg_all`, `ternary_mlp_h128_s42`, `ternary_seq_bilstm_top128_s13`, `ternary_seq_transformer_top128_s42`

## Train OOF Selection Table

| Rank | Label policy | Decision rule | Models | Macro F1 | Diagnosed F1 | Diagnosed precision | Accuracy |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `diag_evidence_q20` | `diagnosed_margin_005` | 6 | `0.736756` | `0.886385` | `0.896486` | `0.794318` |
| 2 | `diag_evidence_q10` | `argmax` | 3 | `0.709318` | `0.882039` | `0.878788` | `0.802273` |
| 3 | `diag_low_density` | `diagnosed_margin_010` | 8 | `0.517929` | `0.892630` | `0.899698` | `0.836932` |
| 4 | `diag_rel3_zero` | `diagnosed_margin_010` | 8 | `0.517929` | `0.892630` | `0.899698` | `0.836932` |
| 5 | `diag_rel5_zero` | `diagnosed_margin_010` | 8 | `0.517929` | `0.892630` | `0.899698` | `0.836932` |

## Final Test Metrics

| Metric | Value |
| --- | ---: |
| Macro F1 | `0.315936` |
| Diagnosed F1 | `0.312281` |
| Diagnosed precision | `0.185933` |
| Diagnosed recall | `0.974453` |
| Accuracy | `0.387611` |

## Final Test Confusion Matrix

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

- Fold-combination robustness ranked the locked ensemble first with mean Macro F1 `0.736719` and minimum Macro F1 `0.712095`.
- Nested OOF split-selection selected `diag_evidence_q20` in all 5 outer folds, with held-out train-fold mean Macro F1 `0.725461` and minimum Macro F1 `0.710505`.
- OOF probability diagnostics for the locked ensemble reported Brier score `0.301943`, negative log likelihood `0.524836`, and expected calibration error `0.067121`.
- The train-only single-model leaderboard ranked `diag_evidence_q20/ternary_hier_logreg_gate` first with OOF Macro F1 `0.708104`; the locked ensemble remains higher at `0.736756`.

## Artifacts

- Lock: `outputs/setembrobr/seed42_ternary_strict_blind/ensemble/ensemble-lock.json`
- Candidate table: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ensemble-candidates.json`
- Train-only robustness: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-robustness.json`
- Nested OOF split-selection: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-nested-oof-selection.json`
- OOF probability diagnostics: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-oof-diagnostics.json`
- Model/policy leaderboard: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-model-policy-leaderboard.json`
- Final report: `outputs/setembrobr/seed42_ternary_strict_blind/reports/final-test-report.json`
- GPU run manifest: `outputs/setembrobr/seed42_ternary_strict_blind/gpu-runs/fedora-ternary-seq-oof.json`

Reports are aggregate-only. They contain no raw tweet text.
