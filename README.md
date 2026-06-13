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

## Candidate Expansion

Candidate models are pre-registered in `configs/setembrobr.seed42.strict-blind.json`. Run DB-dependent tabular candidates locally and sequence candidates on Fedora's GPU:

```bash
export DATABASE_URL=postgresql://embeddings:embeddings@localhost:5437/depression_embeddings
make lint typecheck test db-check-setembrobr
make manifest-setembrobr export-sequences-setembrobr train-tabular-oof-setembrobr
make train-candidate-tabular-oof-setembrobr
make fedora-candidate-seq-oof-setembrobr
make audit-oof-setembrobr select-ensemble-setembrobr evaluate-test-setembrobr
```

`make train-candidate-oof-setembrobr` runs the local candidate tabular trainer and then the Fedora sequence candidate target. The standalone `make train-candidate-seq-oof-setembrobr` target is intended for the Fedora checkout/run directory, not the Mac workflow.
