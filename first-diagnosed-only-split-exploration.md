# First Diagnosed-Only Split Exploration

Status: current workspace results for the SetembroBR ternary strict-blind workflow.

This note documents how the original binary SetembroBR labels were split into ternary
labels, which heuristics and models were tested, and what the train-OOF and final test
results are so far. Reports remain aggregate-only: no raw tweet text is included.

## Strict-Blind Scope

- Dataset: `setembrobr`
- Seed: `42`
- Output root: `outputs/setembrobr/seed42_ternary_strict_blind/`
- Original split manifest hash: `08ce39f8863fc57165f4b8efe57b4a31b764ab6b974968f0ab0a5e167d44108e`
- Selection objective: train-OOF ternary Macro F1, then diagnosed F1, diagnosed precision, and accuracy.
- Test labels and test prevalence were illegal for label-policy, model, threshold, ensemble, calibration, and early-stopping selection.
- Final test labels were read only after `ensemble-lock.json` existed.

## Original Split

The ternary workflow preserves the existing SetembroBR binary train/test split.

| Split | Binary label | Users |
| --- | --- | ---: |
| train | diagnosed | 1347 |
| train | control | 413 |
| test | diagnosed | 337 |
| test | control | 2359 |

Total users: 4456. Train users receive folds. Test users never receive folds.

For each ternary label policy, train users are assigned to 5 stratified folds using
seed `42`. The split exploration also evaluated all 31 non-empty fold combinations
for robustness and 5 outer train-only nested split-selection checks.

## Ternary Label Heuristic

Source labels are binary: `diagnosed` and `control`. Ternary labels are:

- `diagnosed`
- `control`
- `no-evidence`

The diagnosed-only rule is:

- Controls always remain `control`.
- Only originally diagnosed users can become `no-evidence`.
- `no-evidence` means an originally diagnosed user had weak existing GPT-3.5 relevance evidence according to a pre-registered policy.
- Evidence is derived only from existing database rows and existing `gpt_3_5_relevance`; no embedding APIs were called and no embeddings were regenerated.

Evidence markers used:

- `max_relevance`
- `rel3_count`, `rel5_count`, `rel6_count`, `rel7_count`
- `rel3_ratio`, `rel5_ratio`, `rel6_ratio`, `rel7_ratio`
- `top10_avg_relevance`
- `total_tweets`
- deterministic `evidence_score`

The deterministic evidence score is:

```text
0.30 * clamp01(rel7_ratio * 10)
+ 0.25 * clamp01(rel5_ratio * 6)
+ 0.15 * clamp01(rel3_ratio * 3)
+ 0.15 * clamp01(top10_avg_relevance / 7)
+ 0.15 * clamp01(max_relevance / 7)
```

Quantile cutoffs are computed from train diagnosed users only, locked, and then
applied unchanged.

## Label Policies Tested

Seven ternary heuristics were tested. The first five produced no `no-evidence`
train users on the current SetembroBR split; the two evidence-quantile policies
created the effective ternary split.

| Policy | Rule | Cutoff | Train diagnosed | Train control | Train no-evidence |
| --- | --- | ---: | ---: | ---: | ---: |
| `diag_rel3_zero` | diagnosed with zero tweets at relevance >= 3 | n/a | 1347 | 413 | 0 |
| `diag_rel5_zero` | diagnosed with zero tweets at relevance >= 5 | n/a | 1347 | 413 | 0 |
| `diag_rel6_zero` | diagnosed with zero tweets at relevance >= 6 | n/a | 1347 | 413 | 0 |
| `diag_low_density` | diagnosed with rel3 ratio <= 0.01 | n/a | 1347 | 413 | 0 |
| `diag_top10_avg_lt3` | diagnosed with top-10 average relevance < 3 | n/a | 1347 | 413 | 0 |
| `diag_evidence_q10` | bottom 10% train-diagnosed evidence score | `0.446368022` | 1212 | 413 | 135 |
| `diag_evidence_q20` | bottom 20% train-diagnosed evidence score | `0.499652214` | 1077 | 413 | 270 |

The current locked champion uses `diag_evidence_q20`.

## Models Trained

Twenty model candidates were trained for each of the seven label policies, producing
140 train-OOF score files and 140 label-free test-score files.

Tabular/direct and hierarchical candidates:

- `ternary_logreg_all`
- `ternary_extra_trees_evidence`
- `ternary_xgb_tabular_markers_s42`
- `ternary_xgb_expanded_pca_s13`
- `ternary_xgb_expanded_pca_s42`
- `ternary_xgb_embedding_rich_s7`
- `ternary_xgb_shallow_pca_s99`
- `ternary_hgb_expanded_pca_s42`
- `ternary_hgb_markers_s13`
- `ternary_mlp_h128_s42`
- `ternary_focal_linear_g1`
- `ternary_hier_logreg_gate`
- `ternary_relevance_baseline`

Stacking candidates:

- `ternary_stack_logreg_xgb_tabular`
- `ternary_stack_logreg_boosted_core`
- `ternary_stack_logreg_xgb_variants`

Sequence candidates trained through the Fedora GPU path:

- `ternary_seq_cnn_top128_s42`
- `ternary_seq_cnn_wide_top128_s13`
- `ternary_seq_bilstm_top128_s13`
- `ternary_seq_transformer_top128_s42`

## Selection Grid

The current config contains:

- 7 label policies.
- 20 model candidates.
- 15 ensemble selection groups.
- 5 decision rules: `argmax`, `diagnosed_margin_005`, `diagnosed_margin_010`, `no_evidence_gate_045`, and `no_evidence_gate_055`.
- 105 group-level ensemble candidates from 7 policies x 15 groups.
- 700 single model/policy/rule evaluations from 20 models x 7 policies x 5 decision rules.
- 701 robustness candidates including the locked ensemble.

Selection used train OOF probabilities only.

## Current Locked Champion

The current lock is:

- Label policy: `diag_evidence_q20`
- Label policy hash: `06ad79f1e2ac608e7ebf87f2d23a8279b57ede196121302d064c87cbbc3098cd`
- Selection group: `tabular_core_xgb_hgb_rich`
- Selection strategy: `exhaustive+local-refine(step=0.01,radius=0.03)`
- Decision rule: `argmax`

Selected models and weights:

| Model | Weight |
| --- | ---: |
| `ternary_hgb_expanded_pca_s42` | 0.21 |
| `ternary_hier_logreg_gate` | 0.23 |
| `ternary_mlp_h128_s42` | 0.21 |
| `ternary_xgb_embedding_rich_s7` | 0.02 |
| `ternary_xgb_expanded_pca_s13` | 0.29 |
| `ternary_xgb_tabular_markers_s42` | 0.04 |

Train-OOF metrics for the lock:

| Metric | Value |
| --- | ---: |
| Macro F1 | `0.807566` |
| Accuracy | `0.852841` |
| Diagnosed F1 | `0.924277` |
| Diagnosed precision | `0.913793` |
| Diagnosed recall | `0.935005` |
| Control F1 | `0.670103` |
| No-evidence F1 | `0.828319` |

Train-OOF confusion matrix:

| Actual \ Predicted | diagnosed | control | no-evidence |
| --- | ---: | ---: | ---: |
| diagnosed | 1007 | 68 | 2 |
| control | 94 | 260 | 59 |
| no-evidence | 1 | 35 | 234 |

## Ensemble Progression

| Stage | Policy | Group | Rule | Train-OOF Macro F1 | Diagnosed F1 | Diagnosed precision | Accuracy | Final test Macro F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Legacy tabular lock | `diag_evidence_q20` | `tabular_legacy_all` | `argmax` | `0.796108` | `0.918894` | `0.912168` | `0.843182` | not current |
| Boosted shallow local refine | `diag_evidence_q20` | `tabular_core_xgb_s42_shallow` | `diagnosed_margin_005` | `0.806280` | `0.917910` | `0.922212` | `0.847159` | not evaluated |
| HGB shallow | `diag_evidence_q20` | `tabular_core_xgb_hgb_shallow` | `diagnosed_margin_005` | `0.807358` | `0.917251` | `0.923729` | `0.847159` | not evaluated |
| HGB rich current | `diag_evidence_q20` | `tabular_core_xgb_hgb_rich` | `argmax` | `0.807566` | `0.924277` | `0.913793` | `0.852841` | `0.242820` |

The earlier pre-boosted-core final test report had better final test Macro F1:
`0.315936`, with diagnosed F1 `0.312281`, diagnosed precision `0.185933`,
diagnosed recall `0.974453`, and accuracy `0.387611`. That result is a prior
strict-blind result, but it cannot be used to tune or reselect after seeing test
metrics.

## Single-Model Leaderboard

Top train-OOF single model/policy/rule evaluations:

| Rank | Policy | Model | Family | Rule | Macro F1 | Diagnosed F1 | Diagnosed precision | Accuracy |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `diag_evidence_q20` | `ternary_xgb_expanded_pca_s42` | xgboost | `argmax` | `0.788910` | `0.923923` | `0.912217` | `0.843182` |
| 2 | `diag_evidence_q20` | `ternary_xgb_expanded_pca_s13` | xgboost | `diagnosed_margin_005` | `0.787201` | `0.921803` | `0.913400` | `0.840341` |
| 3 | `diag_evidence_q20` | `ternary_xgb_shallow_pca_s99` | xgboost | `argmax` | `0.781547` | `0.913546` | `0.909761` | `0.832955` |
| 4 | `diag_evidence_q20` | `ternary_xgb_embedding_rich_s7` | xgboost | `diagnosed_margin_005` | `0.778610` | `0.912821` | `0.916667` | `0.829545` |
| 5 | `diag_evidence_q20` | `ternary_hgb_expanded_pca_s42` | hist_gradient_boosting | `argmax` | `0.775173` | `0.921317` | `0.908025` | `0.834091` |

Best stacking candidate:

- `diag_evidence_q20` / `ternary_stack_logreg_xgb_variants` / `argmax`
- OOF Macro F1 `0.745698`
- Diagnosed F1 `0.854082`
- Diagnosed precision `0.947905`
- Accuracy `0.771023`

## Train-Only Robustness

For the current locked ensemble:

| Statistic | Macro F1 | Diagnosed F1 | Diagnosed precision | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Mean over 31 fold combinations | `0.807488` | `0.924275` | `0.913810` | `0.852842` |
| Minimum | `0.785349` | `0.905747` | `0.899543` | `0.843305` |
| Maximum | `0.830380` | `0.931507` | `0.926267` | `0.868946` |
| Std. dev. | `0.010466` | `0.005565` | `0.005577` | `0.006616` |

Nested train-only split selection:

| Metric | Mean | Min | Max | Std. dev. |
| --- | ---: | ---: | ---: | ---: |
| Macro F1 | `0.789996` | `0.760899` | `0.808961` | `0.017948` |
| Diagnosed F1 | `0.916171` | `0.905312` | `0.924138` | `0.006655` |
| Diagnosed precision | `0.914120` | `0.903226` | `0.924528` | `0.008800` |
| Accuracy | `0.838057` | `0.821023` | `0.852691` | `0.010776` |

Nested selection counts:

- Policies: `diag_evidence_q20` selected in all 5 outer folds.
- Groups: `tabular_core_xgb_s42_rich` selected 3 times, `tabular_core_xgb_s42_shallow` selected once, and `tabular_core_xgb_hgb_shallow` selected once.
- Rules: `diagnosed_margin_005` selected twice, `argmax` twice, and `diagnosed_margin_010` once.

OOF probability diagnostics for the current lock:

| Diagnostic | Value |
| --- | ---: |
| Brier score | `0.239726` |
| Negative log likelihood | `0.396376` |
| Expected calibration error | `0.044242` |
| Mean predicted probability, diagnosed | `0.573326` |
| Mean predicted probability, control | `0.268962` |
| Mean predicted probability, no-evidence | `0.157712` |

## Final Test Results

Current final test evaluation for the locked HGB-rich ensemble:

| Metric | Value |
| --- | ---: |
| Macro F1 | `0.242820` |
| Accuracy | `0.265579` |
| Diagnosed F1 | `0.307866` |
| Diagnosed precision | `0.182184` |
| Diagnosed recall | `0.992701` |
| Control F1 | `0.277899` |
| No-evidence F1 | `0.142695` |

Final test confusion matrix:

| Actual \ Predicted | diagnosed | control | no-evidence |
| --- | ---: | ---: | ---: |
| diagnosed | 272 | 2 | 0 |
| control | 1221 | 381 | 757 |
| no-evidence | 0 | 0 | 63 |

Final test supports under the locked `diag_evidence_q20` policy:

- diagnosed: 274
- control: 2359
- no-evidence: 63

The current final test is worse than the earlier pre-boosted-core final test despite
higher train-OOF Macro F1. That means the later train-OOF improvements did not
generalize to the held-out test set.

## Binary Strict-Blind Reference

The binary strict-blind SetembroBR result is not a ternary result, but it is useful
context:

| Metric | Value |
| --- | ---: |
| Binary Macro F1 | `0.507711` |
| Binary accuracy | `0.563798` |
| Binary diagnosed F1 | `0.341545` |
| Binary diagnosed precision | `0.210490` |
| Binary diagnosed recall | `0.905045` |

## Current Interpretation

- The only heuristics that created a real ternary split were `diag_evidence_q10`
  and `diag_evidence_q20`; all zero/density/top10 heuristics left train as binary.
- `diag_evidence_q20` dominated train-only selection, including all five nested
  outer-fold selections.
- Boosted tabular models, especially expanded XGBoost variants, dominated the
  single-model leaderboard.
- The best train-OOF ensemble improved Macro F1 to `0.807566`, but the final test
  Macro F1 dropped to `0.242820`.
- The failure mode on test is heavy overprediction of `diagnosed` and `no-evidence`
  for actual controls.
- No subsequent policy, model, threshold, calibration, or ensemble choice may use
  the observed final test metrics.

## Key Artifacts

- Current lock: `outputs/setembrobr/seed42_ternary_strict_blind/ensemble/ensemble-lock.json`
- Current final test report: `outputs/setembrobr/seed42_ternary_strict_blind/reports/final-test-report.json`
- Candidate table: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ensemble-candidates.json`
- Model/policy leaderboard: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-model-policy-leaderboard.json`
- Label-policy summary: `outputs/setembrobr/seed42_ternary_strict_blind/reports/label-policy-summary.json`
- OOF diagnostics: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-oof-diagnostics.json`
- Robustness report: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-robustness.json`
- Nested OOF split selection: `outputs/setembrobr/seed42_ternary_strict_blind/reports/ternary-nested-oof-selection.json`
- GPU run manifest: `outputs/setembrobr/seed42_ternary_strict_blind/gpu-runs/fedora-ternary-seq-oof.json`
