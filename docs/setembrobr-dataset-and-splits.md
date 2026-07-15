# SetembroBR dataset and experimental splits

This note consolidates the corpus facts and exact split provenance needed to write the dataset and experimental-protocol sections of a paper. Machine-readable values and hashes are in [`setembrobr-dataset-provenance.json`](setembrobr-dataset-provenance.json).

## Paper-ready description

SetembroBR is a large-scale Brazilian Portuguese Twitter/X corpus for user-level prediction of depression and anxiety. The two prediction targets are represented by independent but partially overlapping user sets and must be treated as separate tasks. Positive users were identified from manually reviewed self-reports of a clinically plausible diagnosis or treatment event with a definable temporal endpoint. Their timelines contain public Portuguese posts written before that endpoint; retweets, non-Portuguese posts, and posts shorter than five characters were excluded. The comparison group is pseudo-random rather than clinically confirmed negative: each positive user was matched to seven controls by gender, central timeline date within a two-month window, and number of posts. Control timelines were trimmed to the corresponding positive user's time interval and post count. The corpus was collected with 356 diagnosis/treatment queries from September 2019 through February 2021. The canonical corpus description is Santos, Oliveira, and Paraboni, [“SetembroBR: a social media corpus for depression and anxiety disorder prediction”](https://doi.org/10.1007/s10579-022-09633-0).

For the experiments in this repository, the published v6 train/test files are preserved without repartitioning. Model, ensemble-weight, and threshold selection use only five-fold out-of-fold predictions from the training partition. Test scores contain no labels or folds. Test labels are used only for a single evaluation after the ensemble lock is fixed.

## Corpus construction details

- Platform and language: public Brazilian Portuguese Twitter/X posts.
- Collection period: September 2019 to February 2021.
- Candidate discovery: 356 query strings targeting self-reported diagnosis and treatment, including medication use.
- Positive-user review: reports had to be clinically plausible and identify an exact or approximate first diagnosis/treatment endpoint. Vague, distant, comic, doubtful, multiple, or recurrent events were rejected.
- Timeline boundary: only posts before the earliest valid diagnosis/treatment endpoint were retained.
- Activity filters: at least 80 useful pre-event posts; at most 3,200 publicly available posts collected per user.
- Account filters: accounts with more than 10,000 followers or friends were removed as potentially professional or atypical.
- Comorbidity exclusions stated in the corpus construction: bipolar disorder, borderline disorder, schizophrenia, and autism.
- Control pool: approximately 32,000 pseudo-random users with at least 1,000 Portuguese posts, no query match, and no more than 10,000 followers or friends.
- Matching: exactly seven controls per diagnosed user, matched on gender, central timeline date (maximum two-month difference), and post count. Controls were truncated to match the diagnosed counterpart's interval and length.
- Target relationship: the depression and anxiety datasets are partially overlapping. Their user totals cannot be added to obtain a unique-person total.

These construction facts are reported on printed pages 78-81 of [`VERSAO_FINAL_Defesa___wesley.pdf`](../VERSAO_FINAL_Defesa___wesley.pdf); Table 2 is on printed page 81 (PDF page 82), SHA-256 `c9e075954ee7b22ca81df088843f0fea03f7eba920ae6ac1a0a50c43b60fb2ca`.

## Published corpus-level statistics

Values in this table are the rounded descriptive statistics reported in Table 2 of the thesis/corpus publication.

| Statistic | Depression | Anxiety |
|---|---:|---:|
| Diagnosed users | 1,684 | 2,219 |
| Control users | 11,788 | 15,533 |
| Total users | 13,472 | 17,752 |
| Control:diagnosed ratio | 7:1 | 7:1 |
| Female users | 76.7% | 78.8% |
| Posts | 19.42 million | 27.41 million |
| Words | 231.26 million | 323.75 million |
| Posts per user | 1,441 | 1,543 |
| Words per post | 11.91 | 11.81 |
| Mean timeline span | 546 days | 538 days |
| Maximum timeline span | 4,165 days | 4,211 days |
| Mean friends | 704 | 722 |
| Mean followers | 924 | 954 |
| Mean mentions | 122 | 114 |

The 7:1 matched design makes the positive proportion exactly 12.5% in both targets. This is a study-design prevalence, not an estimate of prevalence in the Brazilian population.

## Exact v6 depression split

| Split | Users | Diagnosed | Control | Positive rate | Tweets | Mean tweets/user | Min-max tweets/user |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 10,776 | 1,347 | 9,429 | 12.50% | 15,580,064 | 1,445.81 | 89-3,200 |
| Test | 2,696 | 337 | 2,359 | 12.50% | 3,839,928 | 1,424.31 | 83-3,200 |
| Total | 13,472 | 1,684 | 11,788 | 12.50% | 19,419,992 | 1,441.51 | 83-3,200 |

The user split is 79.988% train and 20.012% test. Seed 42 assigns the training users to five folds numbered 0-4:

| Fold | Users | Diagnosed | Control |
|---:|---:|---:|---:|
| 0 | 2,156 | 270 | 1,886 |
| 1 | 2,156 | 270 | 1,886 |
| 2 | 2,155 | 269 | 1,886 |
| 3 | 2,155 | 269 | 1,886 |
| 4 | 2,154 | 269 | 1,885 |

Split-manifest SHA-256: `ca8188f4d35d8099c4e76dc3a9fa11053bd861d09c573f1076e368caf9eefe5a`.

Source-file SHA-256 values:

| File | SHA-256 |
|---|---|
| `train/train_D_SetembroBR_v6.csv` | `456a526b09e069e2a9a832f8a5636f880ef78371dd76f85dffe6bb583247f2d2` |
| `train/train_D_c_SetembroBR_v6.csv` | `0489a4eb52f7a9a9ae37b58b83123edae4dcfc6548ec87fddea8dffbd74d6c59` |
| `test/test_D_SetembroBR_v6.csv` | `a8b8ca2486157201beb22753b1f654fa7af49850bb1b1c21fb0fff2f694756c1` |
| `test/test_D_c_SetembroBR_v6.csv` | `92ad0abd94d5b597e92cca48cda4b61050d930916cce2930051c2451d2ca4528` |
| `train/train_D_relevancia_1to10_all.pkl` | `20ae9bb29acf5cc53dddfbcbbba7b87210676e133d5a8f48cbcc5de05d72ea6a` |
| `test/teste_D_relevancia_1to10.pkl` | `a9d0ecffab3ce17c08b198bc04c2f2220712219403d9381b209ef382c84668c3` |

Raw Qwen3 embedding-generation manifest SHA-256: `6b9486df5b4cba222bcccf6f0376773369de60d6da6d760737e233282b12930d`.

### Depression relevance-data quality note

The legacy depression relevance files contain values that are not clean integers in the expected 0-10 range. Validation found 23,382 non-digit train values and 6,377 non-digit test values, affecting 6,150 train and 1,527 test users. It also found 18 out-of-range numeric train values across 14 users and 19 out-of-range numeric test values across 10 users. These values are provenance facts; downstream code must use the documented normalization rather than silently treating the source as a clean ordinal variable.

## Exact v6 anxiety split

| Split | Users | Anxiety | Control | Positive rate | Tweets | Mean tweets/user | Min-max tweets/user |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 14,200 | 1,775 | 12,425 | 12.50% | 21,769,232 | 1,533.04 | 87-3,200 |
| Test | 3,552 | 444 | 3,108 | 12.50% | 5,638,808 | 1,587.50 | 89-3,200 |
| Total | 17,752 | 2,219 | 15,533 | 12.50% | 27,408,040 | 1,543.94 | 87-3,200 |

The user split is 79.991% train and 20.009% test. Seed 42 assigns exactly 2,840 training users to each of five folds numbered 1-5; every fold contains 355 anxiety users and 2,485 controls.

The source parser found eight empty tweet segments in train and two in test after splitting the `" # "` timeline delimiter. These empty segments were recorded during validation and are not counted among the expected tweet totals.

Split-manifest SHA-256: `9457cacad5c1d9e378bfc4aff8c3ac8acdcac5f2eb75ad0926780e7838f6fc61`.

Source-file SHA-256 values:

| File | SHA-256 |
|---|---|
| `train/train_A_SetembroBR_v6.csv` | `eb205b831608a716ddcce256c1431b9df231a0206c438052cd773ba1b920b760` |
| `train/train_A_c_SetembroBR_v6.csv` | `07720e53b1f1c1df8399e3eab6ff65e3fe14034ac3604d59c2748dcca42d7578` |
| `test/test_A_SetembroBR_v6.csv` | `feab08b69220b1e7d07ffa97734c2933aeb18f3a685c0174d31f3c569b5cef30` |
| `test/test_A_c_SetembroBR_v6.csv` | `995a1711987d1c2ba3af34a5ceaf0c8bc1fa48a8e7d15d7a296648ee2cb9328d` |

Raw Qwen3 embedding-generation manifest SHA-256: `c8cd40827f91b298d61a3d2845bb4337c0ae3e8abbc9bf4fdfc552b1aad3c410`.

Anxiety has no original GPT relevance files. The champion experiment derives `anxiety-lexical-v1`, a fixed label-independent lexical proxy, into separate sidecars; it does not modify the raw Parquet embeddings.

## Embedding representation

Both targets use the same frozen representation:

- Model: `Qwen/Qwen3-Embedding-4B`
- Revision: `5cf2132abc99cad020ac570b19d031efec650f2b`
- Vector size: 2,560
- Storage: `float16`
- Batch size: 16
- Shard size: 64 users
- Main tweet-level Parquet codec: ZSTD
- Tweet-level columns: `user_id`, `tweet_index`, `tweet_text`, `gpt5_relevance`, `embedding`

For anxiety, `gpt5_relevance` is structurally present but no GPT relevance source is used. The downstream lexical proxy remains a separate artifact.

## Strict-blind split protocol

The following wording is appropriate for the experimental-methods section:

> We retained the official SetembroBR v6 train/test partition separately for depression and anxiety. All model-family decisions, hyperparameters, ensemble weights, and decision thresholds were selected exclusively from five-fold out-of-fold predictions on the training partition. Stacking features were generated with nested cross-fitting so that each outer validation user was excluded from all corresponding base-model fits. Test inference produced label-free files containing only user ID, score, and model ID. Test labels remained sealed until the ensemble lock and label-free score audit were complete, after which they were opened once for final evaluation.

Important reporting rules:

- Report depression and anxiety separately; do not rank or pool their metrics.
- State that folds are user-level and that every training user appears in validation exactly once per OOF model.
- Report class counts and the fixed 12.5% positive rate; do not call it population prevalence.
- Use Macro F1 as the primary imbalance-aware metric, alongside positive-class F1, precision, recall, accuracy, and the confusion matrix.
- Distinguish OOF train metrics from the single sealed-test estimate.
- Do not interpret “control” as clinically confirmed absence of depression/anxiety.
- Do not add depression and anxiety user totals because the target-specific datasets partially overlap.

## Limitations and ethics relevant to the paper

1. Labels are based on manually reviewed self-reports rather than an independent clinical examination.
2. The control group is pseudo-random and can contain undiagnosed or undisclosed cases. The task is therefore better described as identifying users with explicit pre-diagnosis evidence above the matched background rate, not diagnosing disease.
3. Matching and timeline truncation improve comparability but create an artificial 7:1 class ratio and may limit prevalence calibration.
4. Temporal censoring is target-specific for positive users. Any analysis of recency should acknowledge that the final point is tied to a self-reported event.
5. Social-media language, platform demographics, collection years, and Brazilian Portuguese limit external validity.
6. Mental-health labels and timelines are sensitive personal data even when originally public. Results should be reported in aggregate without exposing user IDs, handles, or verbatim posts.

## Fedora preservation archive

On 2026-07-15, the four Fedora source trees containing the depression/anxiety datasets and embedding artifacts were converted into a restricted, resumable preservation archive at `/home/aluisioamorim/codex-runs/setembrobr-v6-restricted-archive`. Files are independent transfer units: 519 compressible CSV, JSON, Pickle, text, and Parquet objects use Zstandard level 10; 517 already-compressed objects such as NPZ files are preserved byte-for-byte. This avoids a single fragile monolithic archive and permits interrupted `rsync` transfers to resume at file boundaries.

| Archive measurement | Bytes | GiB |
|---|---:|---:|
| Original source payloads | 285,751,959,270 | 266.13 |
| Archived payloads | 266,318,109,486 | 248.03 |
| Space saved | 19,433,849,784 | 18.10 |
| Fedora `/home` free before | 24,374,616,064 | 22.70 |
| Fedora `/home` free after | 43,807,522,816 | 40.80 |

All 1,036 files have a source SHA-256, archive SHA-256, original and archived byte count, codec, mode, modification time, and original path in the durable JSONL manifest. Each source file was removed only after its archive hash and reconstructed-byte hash passed and its manifest record had been flushed. A later independent pass reread every archived object, decompressed every Zstandard object, and reproduced all 1,036 original SHA-256 values. The original source trees are therefore absent until explicitly restored. The machine-readable archive record is [`fedora-setembrobr-archive.json`](fedora-setembrobr-archive.json).

Verification and restoration on Fedora use the bundled tool:

```bash
python3 /home/aluisioamorim/codex-runs/setembrobr-v6-restricted-archive/archive_setembrobr_fedora.py \
  --mode verify \
  --archive-root /home/aluisioamorim/codex-runs/setembrobr-v6-restricted-archive

python3 /home/aluisioamorim/codex-runs/setembrobr-v6-restricted-archive/archive_setembrobr_fedora.py \
  --mode restore \
  --archive-root /home/aluisioamorim/codex-runs/setembrobr-v6-restricted-archive
```

The separate NTFS partition belongs to Windows. It was not used for storage or archive work and remains unmounted. The archive resides entirely on Fedora's Linux `/home` filesystem.

## Archive and redistribution policy

The Fedora archive created for this project is a **restricted preservation/authorized-transfer archive**, not a public dataset release. It contains raw post text, sensitive labels, and embeddings derived from that text. The current [X Developer Policy](https://docs.x.com/developer-terms/policy) generally restricts third-party dataset redistribution to Post/User IDs and imposes additional limits and deletion obligations. Before transferring any archive, confirm:

- the original SetembroBR access agreement and institutional ethics approval;
- the current X Developer Agreement and Policy;
- whether written permission is required from X or the corpus owners;
- a secure recipient, access-control, retention, and deletion plan.

For a public reproducibility package, prefer source-code/configuration files, aggregate statistics, redacted split manifests, artifact hashes, and permitted Post/User IDs. Do not publish the raw text, labels tied to identifiable accounts, or embedding archives in a public repository.
