# SetembroBR v7 Qwen3 logistic-regression baseline

This subproject builds a strict-blind user classifier for the final SetembroBR v7 corpus. It reuses the archived, normalized Qwen3 post embeddings, takes the arithmetic mean over exactly the posts retained in each v7 `TextLists`, and fits one scaled L2 logistic regression. No embedding inference, relevance feature, PCA, or legacy full-timeline user vector is used.

The immutable source is `SetembroBR-v7-min.pkl`, SHA-256 `3cde615183d38eaac6788857bc55f14c9b78fa4ad90e2a9007feb51e0e69dc77`. The derived restricted artifact is `SetembroBR-v7-min-qwen-logreg.pkl`; it preserves the original row order and values and adds nullable string column `qwen_logistic_regression_label`. Its 7,602 train values are null and its 400 held-out test values are `yes` or `no`.

## Reproducibility contract

- Corpus: 8,002 unique users and 4,356,312 retained posts.
- Train/test: 7,602/400 users, using the v7 assignments as authoritative.
- Embeddings: `Qwen/Qwen3-Embedding-4B`, revision `5cf2132abc99cad020ac570b19d031efec650f2b`, 2,560 dimensions.
- Pooling: exact ordered-subsequence matching by user, text, occurrence, and tweet order; normalized float16 post vectors accumulated in float64 and emitted as float32 means.
- Validation: seed-42 five-fold stratified OOF over train only.
- Classifier: `StandardScaler`, then balanced L2 logistic regression with `C=1`, `lbfgs`, and `max_iter=1000`.
- Lock: OOF-only Macro F1 threshold selection, with diagnosed F1, precision, and recall as ordered tie-breakers.
- Test boundary: scoring reads a one-column user manifest and label-free pooled vectors, and emits exactly `user_id,score,model_id`. Sealed labels are opened only after test-score audit succeeds.
- Reporting: point metrics plus deterministic class-stratified 2,000-sample bootstrap 95% intervals.

All fixed inputs and hyperparameters are pinned in [`config.json`](config.json). Python dependencies are exact-pinned in [`requirements.lock.txt`](requirements.lock.txt).

## Completed seed-42 baseline

The audited Fedora run completed on 2026-08-26. The train-OOF lock selected threshold `0.7850921027953022` with Macro F1 `0.6473738341525522`. The held-out 400-user result is:

| Metric | Point estimate | Stratified bootstrap 95% interval |
| --- | ---: | ---: |
| Macro F1 | 0.657350040839094 | [0.6096152303537528, 0.7027885568834292] |
| Diagnosed F1 | 0.5742574257425743 | [0.5051177302030544, 0.6381783882629896] |
| Diagnosed precision | 0.8446601941747572 | [0.7767835471567268, 0.9090909090909091] |
| Diagnosed recall | 0.435 | [0.365, 0.5051249999999993] |
| Accuracy | 0.6775 | [0.6375, 0.7175] |

The confusion matrix in `[control, diagnosed]` order is `[[184, 16], [113, 87]]`. The derived corpus SHA-256 is `c8ef0ac905b5ece3904b121438474ffe0732c98475dd2d862bd40144f174dbe2`. See [`artifacts/reports/final-test-report.json`](artifacts/reports/final-test-report.json) and [`artifacts/reports/final-audit.json`](artifacts/reports/final-audit.json) for the locked report and complete artifact chain.

## Staged commands

Create an isolated environment and install the locked dependencies:

```bash
python3 -m venv .venv
make setup PYTHON=.venv/bin/python
```

Every stage has an explicit target:

```bash
make validate
make seal
make pool-embeddings
make train-oof
make audit-oof
make lock-threshold
make fit-full
make score-test
make audit-test
make evaluate-test
make materialize-corpus
make audit-final
```

`make run` follows that chain. Override `SOURCE_PKL`, `ARCHIVE_ROOT`, `OUTPUT_DIR`, or `TEMPORARY_DIR` when needed. Pooling checkpoints one `.npz` per verified archive shard and safely resumes only when the source archive hash, target timeline hash, config hash, dimensions, users, and post counts all still match.

## Fedora workflow

The default Fedora paths point to the workstation source pickle and mounted restricted SSD archive:

```bash
make fedora-sync
make fedora-start
make fedora-status
make fedora-fetch
```

`fedora-start` runs a user transient service named `setembrobr-v7-qwen-logreg`. `fedora-status` shows both service output and the durable job status. `fedora-fetch` retrieves the derived restricted corpus and small JSON locks/reports into the ignored `.work/output/` tree, then promotes only allowlisted aggregate JSON reports and hash-only provenance manifests to `artifacts/`. Bulk pooled vectors, models, scores, and shard checkpoints remain on Fedora.

## Restricted artifacts

The source and derived pickles, sealed labels, per-user manifests and scores, pooled vectors, fitted models, decompressed Parquets, logs, and shard checkpoints are ignored. The repository keeps only code, configuration, tests, the lock, aggregate metrics, and hash-only provenance. Never commit `.work/` or `artifacts/restricted/`.

Run subproject checks with:

```bash
make test PYTHON=.venv/bin/python
```

## Fixed champion architecture transfer

`ensemble-config.json` freezes the repository's five-member depression champion: focal linear,
logistic regression, the top-128 relevance-channel CNN, and both strictly nested logistic
stackers. The fitted models are new v7 models; only the architecture and hyperparameters transfer.
All relevance, temporal, pooled, and sequence inputs are rebuilt from the archived post embeddings
after exact ordered matching to the posts retained by v7.

The ensemble workflow preserves the same strict-blind boundary as the baseline. It prepares and
trains on the 7,602 train users, cross-fits the stackers, selects weights and threshold from OOF
scores, and writes an immutable lock before preparing the 400-user held-out split. Test score files
are exactly `user_id,score,model_id`; sealed labels are opened only after their audit passes.

Run the GPU workflow on Fedora with:

```bash
make fedora-ensemble-start
make fedora-ensemble-status
make fedora-ensemble-fetch
```

The job is resumable at verified embedding shards and fitted fold checkpoints. Bulk features,
sequences, checkpoints, and per-user scores remain ignored on Fedora; the fetch target promotes
only aggregate reports, the OOF lock, and their hash chain into `artifacts/ensemble/champion/`.

### Completed champion-transfer result

The audited seed-42 Fedora run selected OOF threshold `0.68475236363` and OOF Macro F1
`0.7202771988088099`. On the 400 held-out users, the locked ensemble achieved:

| Metric | Point estimate | Stratified bootstrap 95% interval |
| --- | ---: | ---: |
| Macro F1 | 0.6791358057702077 | [0.6343420526953731, 0.7218082463984103] |
| Diagnosed F1 | 0.5973154362416107 | [0.5304659498207885, 0.6601957306447598] |
| Diagnosed precision | 0.9081632653061225 | [0.8484461966604824, 0.9591939806225519] |
| Diagnosed recall | 0.445 | [0.375, 0.515] |
| Accuracy | 0.7 | [0.6625, 0.7375] |

The confusion matrix in `[control, diagnosed]` order is `[[191, 9], [111, 89]]`. This is
`+0.0217857649311137` Macro F1 and nine additional correct users versus the v7 mean-embedding
logistic baseline. It is one held-out point estimate, not a new model-selection signal.

The label-free feature-support audit found that retained-post relevance is bounded at `3` in v7.
Consequently, `rel6` and `rel7` pools collapse to zero, while `rel3` is nonzero for 7,499/7,602
train users and 398/400 test users. The run preserves the frozen champion thresholds rather than
silently rescaling them; this distribution shift is an important limitation of the architecture
transfer. See `artifacts/ensemble/champion/reports/final-test-report.json` and
`artifacts/ensemble/champion/reports/train-feature-support-audit.json`.
