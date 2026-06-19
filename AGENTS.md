# AGENTS.md

## Purpose

This repository is the reproducible SetembroBR-only retraining and validation pipeline for depression prediction. It exists to replace legacy test-tuned experiment flows with strict-blind, deterministic workflows.

## Scope

Only SetembroBR is in scope for now. Do not add SMHD or cross-corpus workflows unless explicitly requested.

## Metrics

Primary report metrics:

- Macro F1
- Diagnosed-class F1
- Diagnosed precision
- Diagnosed recall

The old `71.51%` Macro F1 result from `depression-nlp` was selected with test-set leakage and must be treated as a historical leakage reference, not a generalization estimate.

Current seed-42 raw-binary results from the 2026-06-18 experiment thread:

- Raw Qwen3 binary baseline OOF lock: Macro F1 `0.7074286242910172`.
- Relevance-channel OOF lock: Macro F1 `0.7104872543887791`.
- Temporal-relevance OOF lock: Macro F1 `0.7116486257673107`.
- Temporal-relevance sealed test: Macro F1 `0.7134258267691551`, diagnosed F1 `0.49775112443778113`, precision `0.503030303030303`, recall `0.49258160237388726`.
- Temporal-relevance plus full LLM FP disambiguation sealed test: Macro F1 `0.7203273083370678`, diagnosed F1 `0.5033112582781457`, precision `0.5692883895131086`, recall `0.45103857566765576`.

The full LLM disambiguator point estimate fixed `49` false positives and lost `14` true positives versus the temporal base test report. Treat this as a final-test point estimate; paired bootstrap delta vs temporal base crossed zero: `[-0.004882, +0.018171]`.

Human-readable result summary:

- `docs/lock-results-summary.html`
- `outputs/setembrobr/lock-results-summary.html`

Regenerate both copies with:

```bash
make docs-lock-results-summary
```

The renderer is `scripts/render-lock-results-summary.ts`. It must read existing artifacts only, remain deterministic, and keep the docs copy and output mirror byte-identical.

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

LLM disambiguation rules:

- The LLM is a one-way false-positive disambiguator.
- It may change predicted `diagnosed` users to `control` when the LLM says the timeline is not true depression.
- It must never change `control` to `diagnosed`.
- The prompt may be refined on train OOF only.
- Test LLM decision files must be label-free.
- Test labels are read only by the final evaluation script after the LLM decision CSV already exists.
- Reuse existing LLM cache when request hashes match. Call the LLM only for missing or conflicting cached test users when explicitly requested.

## Required Workflow

Before any final result:

```bash
make lint
make typecheck
make test
make audit-oof-setembrobr
```

Full reproduction:

```bash
make reproduce-setembrobr
```

Raw-binary experiment targets added in this thread:

```bash
make raw-binary-relevance-oof-setembrobr
make fedora-raw-binary-relevance-oof-setembrobr
make raw-binary-temporal-relevance-oof-setembrobr
make fedora-raw-binary-temporal-relevance-oof-setembrobr
```

These OOF-only targets must not run final test evaluation, LLM disambiguation, or final test report generation.

## Reproducibility Rules

- Default seed is `42`.
- Every artifact must reference the split manifest hash.
- OOF rows must equal train users exactly once per model.
- Test rows never receive fold IDs.
- Final test evaluation happens only after an ensemble lock is created from train OOF artifacts.
- Existing raw embeddings are reused. Do not call embedding APIs and do not regenerate embeddings.
- Fedora is used for GPU sequence training and full raw-artifact access when local artifacts are incomplete.
- Local `db-check-setembrobr` can fail if `DATABASE_URL` is not reachable; do not replace the database or regenerate data to work around this.

## Project Structure

Important top-level files and directories:

- `configs/`: strict JSON configs. TypeScript config loading does not support inheritance for raw-binary configs, so new raw-binary experiment configs should be full files.
- `scripts/`: TypeScript and Python workflow scripts for prepare, OOF training, stacking, audit, ensemble selection, test evaluation, LLM timeline export, diagnostics, and docs rendering.
- `src/`: shared TypeScript config, metrics, ensemble, audit, raw-binary, raw-ternary, ternary, and LLM-disambiguator logic.
- `tests/`: Bun tests for configs, Makefile targets, metrics, audit behavior, and LLM-disambiguator schema.
- `outputs/setembrobr/`: generated experiment artifacts: manifests, scores, model manifests, ensemble locks, reports, sequences, features, and LLM cache/decision files.
- `docs/`: durable human-readable documentation. `lock-results-summary.html` is the canonical rich result page and should be regenerated after new locks or final reports.

Current raw-binary experiment lanes:

- `seed42_raw_qwen3_binary`: raw binary baseline.
- `seed42_relevance_features_qwen3_binary`: OOF-only relevance-channel experiment; all sequence candidates use `useRelevanceChannel: true`.
- `seed42_temporal_relevance_qwen3_binary`: temporal relevance experiment; sequence export uses `recent_chronological`, all sequence candidates use the relevance channel, and all tabular candidates include `temporal_markers`.

Key temporal-relevance artifacts:

- Config: `configs/setembrobr.seed42.temporal-relevance-qwen3-binary.json`
- Lock: `outputs/setembrobr/seed42_temporal_relevance_qwen3_binary/ensemble/ensemble-lock.json`
- Base final test report: `outputs/setembrobr/seed42_temporal_relevance_qwen3_binary/reports/final-test-report.json`
- Full LLM final test report: `outputs/setembrobr/seed42_temporal_relevance_qwen3_binary/reports/final-test-report-llm-disambiguated.json`
- LLM decisions: `outputs/setembrobr/seed42_temporal_relevance_qwen3_binary/llm-disambiguator/test_decisions_ensemble-lock.csv`
- Temporal OOF diagnostics: `outputs/setembrobr/seed42_temporal_relevance_qwen3_binary/reports/raw_binary_temporal_oof_diagnostics.json`

## Git Policy

- Use conventional commits.
- Run lint and typecheck before committing.
- Open PRs as ready to review.
