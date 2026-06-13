# SetembroBR Strict-Blind Protocol

This project evaluates user-level depression prediction on the SetembroBR splits using the existing Postgres database and precomputed embeddings.

## Non-Negotiable Protocol

1. Build a deterministic split manifest from Postgres.
2. Assign folds only to train users.
3. Train base models in OOF mode using train folds only.
4. Write train OOF scores with labels and fold IDs.
5. Write test scores without labels.
6. Run the OOF leakage audit.
7. Select ensemble weights and threshold only from train OOF scores.
8. Lock the ensemble.
9. Evaluate the locked ensemble once on test labels.

## Why This Exists

The legacy champion result in `depression-nlp` reached Macro F1 71.51%, but model weights, thresholds, and some neural early stopping decisions were selected against the test set. That number is useful as an upper-bound/oracle reference, not as a generalization estimate.

## Database Contract

The existing database is the source of truth. The workflow expects:

- 3072-dimensional user and tweet embeddings.
- Disjoint train/test users.
- Label agreement between user embedding and feature tables.
- Tweet-level text, relevance, embedding, and tweet index columns for stylistic features and sequence exports.

No script in this repository should mutate the database.

## Candidate Model Expansion

The candidate workflow is additive. Baseline artifacts remain valid, and candidate artifacts are emitted under the same strict score contract:

- Train OOF files contain `user_id,label,fold,score,model_id`.
- Test score files contain only `user_id,score,model_id`.
- Candidate IDs, seeds, feature blocks, and hyperparameters are pre-registered in the config before test evaluation.
- PCA, scalers, focal losses, neural early stopping, and tree models are fitted inside train folds only.
- Sequence candidates use exported `.npz` files and should run on Fedora's RTX GPU; local DB reads, OOF audits, ensemble selection, and final evaluation stay on the Mac.

Do not use legacy or current test metrics to prune candidates, tune thresholds, tune ensemble weights, or change calibration.
