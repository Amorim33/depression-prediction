# depression-prediction

Reproducible, strict-blind SetembroBR retraining pipeline.

This repository uses the existing PostgreSQL database and already-created embeddings. It does not call embedding APIs and does not regenerate embeddings.

## Setup

```bash
bun install
cp .env.example .env
```

Set `DATABASE_URL` to the existing embeddings database.

Python training scripts require:

```bash
python3 -m pip install -r requirements.txt
```

## Checks

```bash
make lint
make typecheck
make test
make db-check-setembrobr
```

## Full SetembroBR Reproduction

```bash
make reproduce-setembrobr
```

The final test evaluation is only valid after:

1. The split manifest is generated.
2. Train OOF artifacts are generated.
3. `make audit-oof-setembrobr` passes.
4. The ensemble lock is selected from train OOF scores only.

