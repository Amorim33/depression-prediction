# SetembroBR Ternary Strict-Blind Protocol

This experiment is an additive SetembroBR-only workflow for three labels:

- `diagnosed`
- `control`
- `no-evidence`

`no-evidence` is only derived from users whose original binary label is `diagnosed`. Original `control` users always remain `control`.

## Non-Negotiable Rules

1. Use the existing train/test split manifest from the binary strict-blind workflow.
2. Generate aggregate evidence markers from existing Postgres rows only.
3. Do not call embedding APIs and do not regenerate embeddings.
4. Derive label-policy cutoffs from train users only.
5. Assign ternary folds only to train users.
6. Train OOF models using train folds only.
7. Write test score files without labels or folds.
8. Select label policy, model weights, and decision rule from train OOF only.
9. Lock exactly one ensemble before reading final test labels.
10. Read test labels only in `ternary-evaluate-test-setembrobr`.

## Score Schemas

Train OOF files must contain:

```csv
user_id,label,fold,prob_diagnosed,prob_control,prob_no_evidence,model_id,label_policy_id
```

Test score files must contain:

```csv
user_id,prob_diagnosed,prob_control,prob_no_evidence,model_id,label_policy_id
```

Test score files must not contain `label`, `fold`, `actual`, `predicted`, thresholds, or metrics.

## Fedora GPU Sequence Training

Sequence candidates can be trained on the Fedora workstation when CUDA is needed:

```bash
make fedora-ternary-train-seq-oof-setembrobr
```

The target uses the `fedora` SSH alias, starts with `ssh fedora 'nvidia-smi'`, and never targets `127.0.0.1`.
It syncs only the required sequence-training inputs to
`FEDORA_TERNARY_RUN_DIR ?= ~/codex-runs/depression-prediction-setembrobr-ternary`:

- `requirements.txt`
- `scripts/ternary_train_seq_oof_setembrobr.py`
- `scripts/write_gpu_run_manifest.py`
- `ternary-classification/configs/setembrobr.seed42.ternary-strict-blind.json`
- exported source `top128` sequence NPZs
- ternary train manifests
- ternary evidence markers
- ternary label-policy locks

Remote sequence training still uses only train fold labels for OOF generation and writes label-free test probability files.
After the remote run, only ternary `scores/`, `model-manifests/`, and `gpu-runs/` are synced back.
The GPU run manifest records host/device information, command, synced inputs, output hashes, and `usesTestLabelsForTraining: false`.

Full GPU-assisted reproduction is:

```bash
make reproduce-ternary-setembrobr-gpu
```

Local audit, ensemble selection, and final test evaluation run after the Fedora sequence outputs are synced back.

## Tabular Candidate Families

Tabular OOF candidates are CPU-trained locally from existing database rows and evidence-marker
artifacts. The registered families include multinomial logistic regression, balanced ExtraTrees, MLP,
focal linear, hierarchical logistic gates, relevance-only baselines, fixed-hyperparameter XGBoost
classifiers, and fixed-hyperparameter sklearn histogram gradient boosting classifiers adapted from
the older `depression-nlp` embedding/meta-feature experiments.

Boosted tabular candidates use deterministic multiclass probability outputs and balanced train-fold
sample weights. They do not use test labels, test prevalence, or test metrics for fitting, early
stopping, thresholding, model selection, or calibration.

## Stacking Candidate

Stacking targets can be generated after base tabular OOF scores exist:

```bash
make ternary-train-stack-oof-setembrobr
```

Each stacker is a logistic meta-classifier over pre-registered base model probabilities. Its train
OOF rows are produced fold by fold: for each validation fold, the meta-classifier is fit only on the
other train OOF folds and then predicts the held-out fold. Label-free test probabilities are generated
by a meta-classifier fit on all train OOF rows and applied to base label-free test score files.

Stacking must not read test labels, test prevalence, final test reports, or test metrics. It is treated
as another candidate score source and remains subject to the same OOF audit and train-only selector.

## Evidence Markers

Evidence markers are aggregate, label-free features derived from `gpt_3_5_relevance`:

- `max_relevance`
- `rel3_count`, `rel5_count`, `rel6_count`, `rel7_count`
- `rel3_ratio`, `rel5_ratio`, `rel6_ratio`, `rel7_ratio`
- `top10_avg_relevance`
- `total_tweets`
- deterministic `evidence_score`

The deterministic evidence score is versioned as `v1` in label-policy locks.

## Label Policies

The configured policies are:

- `diag_rel3_zero`
- `diag_rel5_zero`
- `diag_rel6_zero`
- `diag_low_density`
- `diag_top10_avg_lt3`
- `diag_evidence_q10`
- `diag_evidence_q20`

Quantile policy cutoffs are computed only from originally diagnosed train users, then locked and applied unchanged.

## Official Objective

The official selection order is:

1. Ternary Macro F1
2. Diagnosed-class F1
3. Diagnosed precision
4. Accuracy

No test metric may be used to alter models, label policies, weights, or decision rules.

Official ensemble selection evaluates the pre-registered `selectionGroups` from the ternary config,
including all-model, tabular-only, boosted-tabular, bounded boosted-core, sequence-only, stacking-only,
and baseline-only groups. Bounded groups with at most the configured exhaustive limit are useful for
testing specific strict-blind model combinations without letting later broad candidate expansion hide
previously strong train-OOF mixtures. The locked champion is the best train-OOF candidate across label
policies, model groups, weights, and decision rules.

The config may also pre-register bounded local weight refinement for specific label-policy and model
group pairs. Local refinement starts from a train-OOF-selected ensemble, keeps only its nonzero model
weights, enumerates nearby probability-simplex weights with the configured finer step and radius, and
keeps a replacement only if it improves the official train-OOF objective. The current refinement is
limited to `diag_evidence_q20` boosted-core groups, including the HGB-shallow and HGB-rich core
groups, uses weight step `0.01`, radius `0.03`, and at most six selected models. It must not read test
scores, test labels, test prevalence, final test reports, or any remote GPU artifacts beyond
label-free candidate probabilities.

## Train-Only Robustness Report

After the OOF audit and ensemble lock, a train-only robustness report can be generated:

```bash
make ternary-robustness-setembrobr
```

This report reads only train OOF probability files and the already-created train-OOF ensemble lock.
It computes metrics over every non-empty combination of train validation folds, ranks candidates by
mean fold-combination Macro F1, and reports minimum fold-combination Macro F1 as a stability check.

The robustness report is diagnostic. It must not read test score files, test labels, final test reports,
or test prevalence, and it must not be used to change the already-locked champion after final test
evaluation has run.

## Nested OOF Split-Selection Report

A stricter train-only split-combination check can be generated with:

```bash
make ternary-nested-oof-selection-setembrobr
```

For each outer train fold, this report selects the label policy, pre-registered model group, model set,
weights, and decision rule using only the other train OOF folds. It then evaluates that inner lock on
the held-out train fold's OOF rows. This estimates whether the selection procedure is stable under
alternate train-only splits.

Nested diagnostics evaluate every configured model group, but use a bounded selector so routine
reproduction stays tractable: groups with more than four models use the same greedy-pruned selector
available to the official ensemble search. If local refinement is configured for an inner
policy/group, the nested selector applies the same train-only bounded refinement on the inner folds
before scoring the held-out train fold. The official champion lock remains selected by the
pre-registered selector before final test evaluation.

The nested report must remain train-only: no test score files, test labels, final test reports, or test
prevalence may be read or used.

## OOF Probability Diagnostics

The locked ensemble's train OOF probabilities can be diagnosed with:

```bash
make ternary-oof-diagnostics-setembrobr
```

This report computes train-only confidence bins, Brier score, negative log likelihood, expected
calibration error, per-fold summaries, and high-confidence OOF errors. It is meant to explain
train-side overconfidence and class skew before any future pre-registered run.

The diagnostics must not read test score files, test labels, final test reports, or test prevalence,
and must not alter the locked champion.

## Model And Policy Leaderboard

Single-model policy/rule rankings can be generated from train OOF probabilities:

```bash
make ternary-model-policy-leaderboard-setembrobr
```

The leaderboard scores every pre-registered ternary model under every configured label policy and
decision rule, then summarizes the strongest model/policy/rule combinations by label policy and model
family. It reads train OOF score files and pre-registered metadata only.

The leaderboard must not read test score files, test labels, final test reports, or test prevalence,
and must not be used to revise a champion after final test evaluation.

## Family Ablation Report

Model-family ensemble ablations can be generated from train OOF probabilities:

```bash
make ternary-family-ablation-setembrobr
```

This report reruns the train-OOF ensemble selector over predefined model groups such as all tabular
models, all sequence models, CNN-only sequence models, BiLSTM/transformer sequence models, and the
relevance-only baseline. It compares each restricted group with the full policy-specific OOF ensemble.

The ablation report must remain train-only and must not read test score files, test labels, final test
reports, or test prevalence.
