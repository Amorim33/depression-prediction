# AGENTS.md

## Purpose

This repository is the reproducible SetembroBR-only retraining and validation pipeline for depression prediction. It exists to replace legacy test-tuned experiment flows with strict-blind, deterministic workflows.

## Data Source

- Use the existing PostgreSQL database and already-created embeddings.
- Do not call embedding APIs.
- Do not regenerate embeddings.
- Treat the database as read-only.
- Required env var: `DATABASE_URL`.

Default local shape:

```bash
DATABASE_URL=postgresql://embeddings:embeddings@localhost:5437/depression_embeddings
```

Required tables:

- `train_user_emb`, `test_user_emb`
- `train_user_emb_rel3`, `test_user_emb_rel3`
- `train_sub_features`, `test_sub_features`
- `train_embeddings`, `test_embeddings`

Tweet embedding tables must include `tweet_text`, `tweet_index`, `embedding`, and `gpt_3_5_relevance`.

## Scope

Only SetembroBR is in scope for now. Do not add SMHD or cross-corpus workflows unless explicitly requested.

## Metrics

Primary report metrics:

- Macro F1
- Diagnosed-class F1
- Diagnosed precision
- Diagnosed recall

The old `71.51%` Macro F1 result from `depression-nlp` was selected with test-set leakage and must be treated as an oracle upper bound, not a generalization estimate.

## Strict-Blind Rules

Never use test labels or test prevalence for:

- Model selection
- Threshold selection
- Ensemble weight selection
- Early stopping
- OOF generation
- Calibration

Test score files must be label-free: `user_id,score,model_id`.

Train OOF score files must contain: `user_id,label,fold,score,model_id`.

## Required Workflow

Before any final result:

```bash
make lint
make typecheck
make test
make db-check-setembrobr
make audit-oof-setembrobr
```

Full reproduction:

```bash
make reproduce-setembrobr
```

## Reproducibility Rules

- Default seed is `42`.
- Every artifact must reference the split manifest hash.
- OOF rows must equal train users exactly once per model.
- Test rows never receive fold IDs.
- Final test evaluation happens only after an ensemble lock is created from train OOF artifacts.

## Git Policy

- Use conventional commits.
- Run lint and typecheck before committing.
- Open PRs as ready to review.
