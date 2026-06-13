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
