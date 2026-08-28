# GPT-5.6 strict-blind OOF experiment

This directory contains the tracked configuration, prompt templates, generated fold prompts,
aggregate evidence, and hash-only reports for a five-fold prompt-classification experiment over
the 7,602-user training partition of SetembroBR v7.

The analysis model is `gpt-5.6-luna` with `xhigh` reasoning. Each outer fold receives six
independent development-set analyses, followed by synthesis, red-team review, and finalization.
The classification model is `gpt-5.6-sol` with `high` reasoning. It receives one complete raw
timeline per request and uses a fixed 0.5 decision boundary. No Sol development-set prompt bake-off
or threshold tuning is performed.

## Strict-blind boundary

Only rows whose immutable source `Split` is `train` are materialized. The 400-user future
validation partition is not written into this experiment, uploaded to OpenAI, scored, summarized,
or evaluated. For outer fold `k`, Luna receives labeled files for the other four folds only. Sol
receives the unlabeled timelines from fold `k` only after all five prompts have been locked.

All corpus-bearing files, API transcripts, opaque-ID mappings, batch inputs and outputs, and
user-level predictions live under `../.work/gpt-experiments/`, which is ignored by Git. Tracked
files must contain no user identifiers, handles, URLs, raw posts, or verbatim corpus excerpts.

## Stages

Install the dedicated API dependencies into the v7 environment and run stages from
`setembrobr-v7/`:

```bash
make gpt-oof-setup
make gpt-oof-prepare
make gpt-oof-analyze
make gpt-oof-lock
make gpt-oof-smoke
make gpt-oof-submit
make gpt-oof-status
make gpt-oof-fetch
make gpt-oof-audit
make gpt-oof-evaluate
```

`gpt-oof-status`, `gpt-oof-fetch`, and `gpt-oof-submit` are resumable. If an account's Batch queue
limit rejects a large shard, the submit stage deterministically bisects that shard without
changing any request. Failed per-user requests are resubmitted only with their original request
body and hash.

The API key is loaded from the repository-root `.env` variable `OPENAI_API_KEY`; it is never
written to logs or artifacts. `make test` is offline and uses a fake client.

## Results contract

After a complete audited run, `reports/oof-results.json` contains five held-out-fold metric blocks
and one overall metric block calculated on concatenated OOF predictions. The restricted score file
under `.work` has exactly:

```text
user_id,label,fold,score,model_id
```

The tracked `prompts/generated/fold-0.md` through `fold-4.md` are the five immutable classifier
prompts. No all-training prompt or future-validation result belongs to this stage.

## Final strict-blind OOF result

The prompt lock SHA-256 is
`625ba6c38cba6669e6a219db7f62340f555397c24963bd1ac2cd6608a53df779`. All 7,602 Sol
responses passed the label-free audit without a retry. At the predeclared 0.5 threshold:

| Fold | Users | Macro F1 |
| ---: | ---: | ---: |
| 0 | 1,521 | 0.4267133071 |
| 1 | 1,521 | 0.6557795541 |
| 2 | 1,520 | 0.6595872280 |
| 3 | 1,520 | 0.7107337432 |
| 4 | 1,520 | 0.6773155282 |

The concatenated OOF Macro F1 is `0.6141735622`. Diagnosed-class F1 is `0.3641121495`,
precision is `0.2560462671`, and recall is `0.6300129366`. The overall confusion matrix is
`[[5414, 1415], [286, 487]]` in `[control, diagnosed]` order. These held-out results were
calculated only after all five prompt hashes were locked and were not used to change a prompt or
threshold. See `reports/oof-results.json` for the complete aggregate report.

The API contract follows the official OpenAI documentation for
[`gpt-5.6-luna`](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[`gpt-5.6-sol`](https://developers.openai.com/api/docs/models/gpt-5.6-sol), and the
[Batch API](https://platform.openai.com/docs/api-reference/batch).
