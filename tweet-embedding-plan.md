# Embed All SetembroBR Anxiety Tweets on Fedora

## Summary

Extend the existing Qwen3 embedding workflow to support the anxiety CSVs without altering the depression lane, then transfer `dataset/anxiety_tweets.rar` to Fedora, extract it, validate it, and embed every tweet with the RTX 4070 SUPER.

Measured workload:

- 17,752 users across train and test.
- 27,408,040 tweet segments.
- Approximately 153 GiB of tweet-level embeddings.
- Approximately 53 hours based on the previous 37.4-hour depression run.
- Artifacts remain on Fedora because the Mac has only 42 GiB free.

## Implementation Changes

- Make `scripts/raw_qwen3_embeddings_setembrobr.py` support config-driven input profiles:
  - Existing depression profile continues reading relevance pickles unchanged.
  - New anxiety profile reads the four `A` CSVs and splits `Text` on the literal ` # ` delimiter.
  - Preserve blank segments and embed them as `" "`, matching existing encoder behavior.
  - Do not synthesize relevance scores or relevance-threshold pools for anxiety.
- Add an anxiety embedding config using:
  - `Qwen/Qwen3-Embedding-4B` at the revision recorded by the loaded model.
  - 2,560-dimensional normalized embeddings.
  - Float16 Parquet storage, batch size 16, 64 users per resumable shard, seed 42.
  - Expected split totals: 14,200 train users and 3,552 test users.
- Produce:
  - Tweet-level Parquet with `user_id`, `tweet_index`, `tweet_text`, nullable `gpt5_relevance`, and `embedding`.
  - Mean user embeddings and tweet counts.
  - Source hashes, split manifest, model revision, environment versions, GPU details, and generation report.
  - Redacted test labels in generated embedding artifacts; no training, selection, or test evaluation is included.
- Add Make targets for anxiety validation, Fedora synchronization, smoke embedding, full detached embedding, and status inspection. Keep the existing depression targets backward-compatible.

## Fedora Execution

1. Run `make lint`, `make typecheck`, `make test`, and `make audit-oof-setembrobr`.
2. Create `~/codex-runs/depression-prediction-setembrobr-anxiety-embeddings` with separate `repo`, `incoming`, `data`, `artifacts`, and `logs` directories.
3. Transfer the archive with resumable `rsync --partial --append`, then verify its SHA-256 is `a4c6fa4b486b669559bb5b46f56d9914f9271e2b1dbbb0a0b6c8a9626fb6fca7`.
4. Extract on Fedora using a rootless Podman Fedora container with the repository-provided `unar` package, avoiding a host-level sudo installation.
5. Validate the four filenames, extracted sizes, CSV schema, unique users, split counts, labels, and all 27,408,040 tweet segments. Delete only the transferred RAR after successful extraction.
6. Reuse the existing CUDA Python environment and 7.6 GiB Hugging Face model cache from the depression embedding workspace.
7. Require at least 174 GiB free before launch and retain a 20 GiB runtime reserve. Abort cleanly between shards if storage falls below the reserve.
8. Run a two-user-per-split CUDA smoke job and verify dimensionality, normalized vectors, nullable relevance, Parquet readability, and resume behavior.
9. Launch the full job detached with PID, log, and exit-status files. Monitor GPU utilization, disk availability, shard progress, and process status until completion; rerun the same command to resume complete shard pairs after interruption.

## Verification

- Assert exactly 21,769,232 train and 5,638,808 test Parquet rows.
- Verify each user appears once in the user manifest, tweet indexes are contiguous, and no input tweet is omitted or duplicated.
- Check every embedding is float16 on disk, length 2,560, finite, and approximately unit-normalized using full metadata checks plus deterministic samples from every shard.
- Confirm the final manifest references all source hashes, model revision, CUDA device, package versions, counts, and output paths.
- Sync only compact manifests and reports back to the local output mirror; retain the approximately 153 GiB embedding dataset on Fedora.
- Do not run OOF training, model selection, LLM disambiguation, or final-test evaluation as part of this embedding job.

## Assumptions

- "All anxiety tweets" means both diagnosed and control users from all four train/test CSVs.
- The established Qwen3 model, dimensional truncation, normalization, batching, and float16 storage remain the required embedding contract.
- Fedora's existing depression embeddings and model cache must not be deleted or regenerated.
