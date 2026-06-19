#!/usr/bin/env python3
"""Prepare strict-blind raw Qwen3 ternary experiment artifacts.

This script consumes the completed raw embedding workspace. It writes train-only
labels, redacted test manifests, raw-native feature blocks, and derived sequence
NPZs. Test labels are written only to the sealed final-evaluation file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

EXPECTED_USERS = {"train": 10776, "test": 2696}
EMBEDDING_BLOCKS = ["mean", "rel3", "rel6", "rel7"]
EVIDENCE_COLUMNS = [
    "log_total_tweets",
    "max_relevance",
    "rel3_count_log",
    "rel5_count_log",
    "rel6_count_log",
    "rel7_count_log",
    "rel3_ratio",
    "rel5_ratio",
    "rel6_ratio",
    "rel7_ratio",
    "top10_avg_relevance",
    "evidence_score",
]
TEMPORAL_WINDOWS = [
    ("final10", 10),
    ("final50", 50),
    ("final128", 128),
    ("final_third", None),
]
TEMPORAL_BASE_COLUMNS = [
    "tweet_count_log",
    "mean_relevance",
    "max_relevance",
    "rel5_count_log",
    "rel7_count_log",
    "rel5_ratio",
    "rel7_ratio",
    "first_person_negative_ratio",
    "active_ideation_count_log",
    "caregiver_third_party_count_log",
    "indirect_sleep_ratio",
    "indirect_fatigue_ratio",
    "indirect_cry_sad_ratio",
    "indirect_anxiety_ratio",
    "indirect_treatment_ratio",
    "indirect_grief_ratio",
    "indirect_help_seeking_ratio",
    "support_outreach_ratio",
]
TEMPORAL_DELTA_COLUMNS = [
    "recent50_minus_lifetime_mean_relevance",
    "recent50_minus_lifetime_rel7_ratio",
    "recent50_minus_lifetime_first_person_negative_ratio",
    "recent50_minus_lifetime_indirect_tone_ratio",
    "last_rel7_normalized_position",
]
TEMPORAL_COLUMNS = [
    f"{window}_{column}"
    for window, _size in TEMPORAL_WINDOWS
    for column in TEMPORAL_BASE_COLUMNS
] + TEMPORAL_DELTA_COLUMNS

FP_REGEX = re.compile(r"\b(eu|meu|minha|mim|comigo|me)\b", re.IGNORECASE)
NEG_REGEX = re.compile(
    r"\b(triste|vazi[oa]|sozinh[oa]|solit[aá]ri[oa]|cansad[oa]|exaust[oa]|inutil|fracass|derrotad[oa]|desespero|sofrendo|doendo|machuca|arrepend|odei[oa]|horriv[eí]l|pior|terriv[eí]l|dific[ií]l|impossiv[eí]l|raiva|irritad[oa])\b",
    re.IGNORECASE,
)
ACTIVE_IDEATION_REGEX = re.compile(
    r"(quero morrer|vou me matar|vontade de morrer|pensei em suic[ií]d|tentei suic[ií]d|queria me matar|acabar com minha vida|n[aã]o quero mais viver|quero sumir pra sempre)",
    re.IGNORECASE,
)
CAREGIVER_REGEX = re.compile(
    r"(meu (pai|irm|primo|namorad|marido|amigo)|minha (m[aã]e|irm|prima|namorad|amiga|esposa|filh))[^.!?\n]{0,80}(depress|ansiedade|terapia|psic[oó]log|psiquiatr|rem[eé]dio|antidepressiv|suicid)",
    re.IGNORECASE,
)
THIRD_PARTY_REGEX = re.compile(
    r"((ele|ela|eles|elas|voc[eê]|algu[eé]m|pessoa|gente|amig[oa]|fam[ií]lia)[^.!?\n]{0,90}(depress|ansiedade|terapia|psic[oó]log|psiquiatr|rem[eé]dio|antidepressiv|suicid|morrer|morte|triste|ajuda)"
    r"|(depress|ansiedade|terapia|psic[oó]log|psiquiatr|rem[eé]dio|antidepressiv|suicid|morrer|morte|triste|ajuda)[^.!?\n]{0,90}(ele|ela|eles|elas|voc[eê]|algu[eé]m|pessoa|gente|amig[oa]|fam[ií]lia))",
    re.IGNORECASE,
)
SLEEP_REGEX = re.compile(r"\b(ins[oô]nia|dormir|durmo|dormi|dormindo|sono|sem sono|acordad[oa]|virad[oa])\b", re.IGNORECASE)
FATIGUE_REGEX = re.compile(
    r"\b(cansad[oa]|exaust[oa]|esgotad[oa]|des[aâ]nimo|desanimad[oa]|sem energia|sem disposi[cç][aã]o|indispost[oa]|vontade de nada)\b",
    re.IGNORECASE,
)
CRY_SAD_REGEX = re.compile(r"\b(chor(ar|ei|o|ando)|triste|tristeza|bad|arrasad[oa]|p[eé]ssim[oa])\b", re.IGNORECASE)
ANXIETY_REGEX = re.compile(r"\b(ansiedade|ansios[oa]|p[aâ]nico|crise|surt(o|ei|ando)|nervos[oa])\b", re.IGNORECASE)
TREATMENT_REGEX = re.compile(
    r"\b(terapia|psic[oó]log[ao]|psiquiatr[ao]|rem[eé]dio|medica[cç][aã]o|antidepressiv[oa]|fluoxetina|sertralina|rivotril)\b",
    re.IGNORECASE,
)
GRIEF_REGEX = re.compile(r"\b(luto|falec(eu|er)|morreu|perdi|minha perda|meu luto|saudades?|enterro|vel[oó]rio)\b", re.IGNORECASE)
HELP_SEEKING_REGEX = re.compile(
    r"(preciso de ajuda|me ajuda|algu[eé]m me ajuda|socorro|n[aã]o sei o que fazer|n[aã]o aguento mais)",
    re.IGNORECASE,
)
SUPPORT_OUTREACH_REGEX = re.compile(
    r"(se precisar conversar|procure ajuda|busque ajuda|cvv|188|setembro amarelo|estou aqui|conte comigo|projeto|apoio psicol[oó]gico)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/setembrobr.seed42.raw-qwen3-ternary-diagnosed-only.json")
    parser.add_argument("--mode", choices=["prepare", "export-sequences", "all"], default="all")
    parser.add_argument("--raw-artifacts-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--top-n", type=int, default=128)
    parser.add_argument("--sequence-order", choices=["relevance_desc", "recent_chronological"])
    return parser.parse_args()


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_config(path: Path) -> dict[str, Any]:
    parsed = read_json(path)
    parent_ref = parsed.get("extends")
    if not isinstance(parent_ref, str):
        return parsed
    parent_path = Path(parent_ref).expanduser()
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()
    child = {key: value for key, value in parsed.items() if key != "extends"}
    return deep_merge(read_config(parent_path), child)


def deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for key, value in child.items():
        parent_value = merged.get(key)
        if isinstance(parent_value, dict) and isinstance(value, dict):
            merged[key] = deep_merge(parent_value, value)
        else:
            merged[key] = value
    return merged


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return sha256_file(path)


def label_name(value: str | int) -> str:
    text = str(value).strip().lower()
    if text in {"1", "diagnosed"}:
        return "diagnosed"
    if text in {"0", "control"}:
        return "control"
    raise ValueError(f"unsupported raw label: {value!r}")


def label_code(value: str) -> int:
    return 1 if value == "diagnosed" else 0


def read_raw_manifest(raw_artifacts_dir: Path, seed: int) -> list[dict[str, Any]]:
    path = raw_artifacts_dir / "manifests" / f"raw_split_manifest_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    train_folds = [int(row["fold"]) for row in rows if row["split"] == "train" and row.get("fold", "") != ""]
    fold_offset = 1 if train_folds and min(train_folds) == 0 else 0
    for row in rows:
        split = row["split"]
        label = label_name(row["label"])
        fold = ""
        if split == "train":
            fold = int(row["fold"]) + fold_offset
        out.append(
            {
                "dataset": "setembrobr",
                "split": split,
                "user_id": row["user_id"],
                "label": label,
                "label_code": label_code(label),
                "fold": fold,
                "row_hash": row["row_hash"],
            }
        )
    for split, expected in EXPECTED_USERS.items():
        actual = sum(1 for row in out if row["split"] == split)
        if actual != expected:
            raise ValueError(f"{split} raw manifest users: expected {expected}, got {actual}")
    return out


def write_strict_blind_manifests(output_dir: Path, raw_rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    train_rows = [row for row in raw_rows if row["split"] == "train"]
    test_rows = [row for row in raw_rows if row["split"] == "test"]
    strict_rows = []
    for row in raw_rows:
        prelock_row_hash = row["row_hash"] if row["split"] == "train" else redacted_test_row_hash(row["user_id"])
        strict_rows.append(
            {
                "dataset": "setembrobr",
                "split": row["split"],
                "user_id": row["user_id"],
                "label": row["label"] if row["split"] == "train" else -1,
                "fold": row["fold"] if row["split"] == "train" else "",
                "row_hash": prelock_row_hash,
            }
        )
    strict_hash = write_csv(
        output_dir / "manifest" / f"strict_blind_split_manifest_seed{seed}.csv",
        ["dataset", "split", "user_id", "label", "fold", "row_hash"],
        strict_rows,
    )
    train_hash = write_csv(
        output_dir / "manifest" / f"train_binary_manifest_seed{seed}.csv",
        ["dataset", "split", "label", "user_id", "row_hash", "fold"],
        [
            {
                "dataset": "setembrobr",
                "split": "train",
                "label": row["label"],
                "user_id": row["user_id"],
                "row_hash": row["row_hash"],
                "fold": row["fold"],
            }
            for row in train_rows
        ],
    )
    inference_hash = write_csv(
        output_dir / "manifest" / f"test_inference_manifest_seed{seed}.csv",
        ["dataset", "split", "user_id", "label", "fold", "row_hash"],
        [
            {
                "dataset": "setembrobr",
                "split": "test",
                "user_id": row["user_id"],
                "label": -1,
                "fold": "",
                "row_hash": redacted_test_row_hash(row["user_id"]),
            }
            for row in test_rows
        ],
    )
    sealed_hash = write_csv(
        output_dir / "manifest" / f"sealed_test_labels_seed{seed}.csv",
        ["user_id", "binary_label", "label_code", "row_hash"],
        [
            {
                "user_id": row["user_id"],
                "binary_label": row["label"],
                "label_code": row["label_code"],
                "row_hash": row["row_hash"],
            }
            for row in test_rows
        ],
    )
    return {
        "strictBlindManifestHash": strict_hash,
        "trainBinaryManifestHash": train_hash,
        "testInferenceManifestHash": inference_hash,
        "sealedTestLabelsHash": sealed_hash,
        "trainUsers": len(train_rows),
        "testUsers": len(test_rows),
        "trainLabelCounts": count_labels(train_rows),
        "testLabelCountsSealed": count_labels(test_rows),
    }


def redacted_test_row_hash(user_id: str) -> str:
    return sha256_text(f"setembrobr|raw-qwen3|test|-1|{user_id}")


def count_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "control": sum(1 for row in rows if row["label"] == "control"),
        "diagnosed": sum(1 for row in rows if row["label"] == "diagnosed"),
    }


def relevance_value(raw: Any) -> int:
    if raw is None:
        return 0
    try:
        if isinstance(raw, float) and math.isnan(raw):
            return 0
        numeric = int(raw)
    except (TypeError, ValueError):
        return 0
    return min(max(numeric, 0), 10)


class TemporalWindowAgg:
    __slots__ = (
        "total",
        "sum_rel",
        "max_rel",
        "rel5",
        "rel7",
        "selfneg",
        "active",
        "caregiver_third_party",
        "sleep",
        "fatigue",
        "cry_sad",
        "anxiety",
        "treatment",
        "grief",
        "help_seeking",
        "support_outreach",
        "indirect_tone",
    )

    def __init__(self) -> None:
        self.total = 0
        self.sum_rel = 0.0
        self.max_rel = 0
        self.rel5 = 0
        self.rel7 = 0
        self.selfneg = 0
        self.active = 0
        self.caregiver_third_party = 0
        self.sleep = 0
        self.fatigue = 0
        self.cry_sad = 0
        self.anxiety = 0
        self.treatment = 0
        self.grief = 0
        self.help_seeking = 0
        self.support_outreach = 0
        self.indirect_tone = 0

    def add(self, rel: int, flags: dict[str, bool]) -> None:
        self.total += 1
        self.sum_rel += float(rel)
        self.max_rel = max(self.max_rel, rel)
        self.rel5 += int(rel >= 5)
        self.rel7 += int(rel >= 7)
        self.selfneg += int(flags["selfneg"])
        self.active += int(flags["active"])
        self.caregiver_third_party += int(flags["caregiver_third_party"])
        self.sleep += int(flags["sleep"])
        self.fatigue += int(flags["fatigue"])
        self.cry_sad += int(flags["cry_sad"])
        self.anxiety += int(flags["anxiety"])
        self.treatment += int(flags["treatment"])
        self.grief += int(flags["grief"])
        self.help_seeking += int(flags["help_seeking"])
        self.support_outreach += int(flags["support_outreach"])
        self.indirect_tone += int(flags["indirect_tone"])

    def features(self) -> list[float]:
        denom = max(float(self.total), 1.0)
        return [
            math.log1p(self.total),
            self.sum_rel / denom,
            float(self.max_rel),
            math.log1p(self.rel5),
            math.log1p(self.rel7),
            self.rel5 / denom,
            self.rel7 / denom,
            self.selfneg / denom,
            math.log1p(self.active),
            math.log1p(self.caregiver_third_party),
            self.sleep / denom,
            self.fatigue / denom,
            self.cry_sad / denom,
            self.anxiety / denom,
            self.treatment / denom,
            self.grief / denom,
            self.help_seeking / denom,
            self.support_outreach / denom,
        ]


def tweet_flags(text: str) -> dict[str, bool]:
    fp = bool(FP_REGEX.search(text))
    neg = bool(NEG_REGEX.search(text))
    active = bool(ACTIVE_IDEATION_REGEX.search(text))
    caregiver = bool(CAREGIVER_REGEX.search(text))
    third_party = bool(THIRD_PARTY_REGEX.search(text))
    sleep = bool(SLEEP_REGEX.search(text))
    fatigue = bool(FATIGUE_REGEX.search(text))
    cry_sad = bool(CRY_SAD_REGEX.search(text))
    anxiety = bool(ANXIETY_REGEX.search(text))
    treatment = bool(TREATMENT_REGEX.search(text))
    grief = bool(GRIEF_REGEX.search(text))
    help_seeking = bool(HELP_SEEKING_REGEX.search(text))
    support_outreach = bool(SUPPORT_OUTREACH_REGEX.search(text))
    indirect_tone = sleep or fatigue or cry_sad or anxiety or treatment or grief or help_seeking
    return {
        "fp": fp,
        "neg": neg,
        "selfneg": fp and neg,
        "active": active,
        "caregiver": caregiver,
        "caregiver_third_party": caregiver or third_party or support_outreach,
        "sleep": sleep,
        "fatigue": fatigue,
        "cry_sad": cry_sad,
        "anxiety": anxiety,
        "treatment": treatment,
        "grief": grief,
        "help_seeking": help_seeking,
        "support_outreach": support_outreach,
        "indirect_tone": indirect_tone,
    }


def lifetime_tweet_flags(text: str) -> dict[str, bool]:
    fp = bool(FP_REGEX.search(text))
    neg = bool(NEG_REGEX.search(text))
    active = bool(ACTIVE_IDEATION_REGEX.search(text))
    caregiver = bool(CAREGIVER_REGEX.search(text))
    return {
        "fp": fp,
        "neg": neg,
        "selfneg": fp and neg,
        "active": active,
        "caregiver": caregiver,
        "indirect_tone": neg or active or caregiver,
    }


class UserAgg:
    __slots__ = (
        "total",
        "sum_rel",
        "max_rel",
        "rel3",
        "rel5",
        "rel6",
        "rel7",
        "top10",
        "fp",
        "neg",
        "selfneg",
        "active",
        "caregiver",
        "indirect_tone",
        "last_rel7_index",
        "temporal_windows",
    )

    def __init__(self) -> None:
        self.total = 0
        self.sum_rel = 0.0
        self.max_rel = 0
        self.rel3 = 0
        self.rel5 = 0
        self.rel6 = 0
        self.rel7 = 0
        self.top10: list[tuple[int, int]] = []
        self.fp = 0
        self.neg = 0
        self.selfneg = 0
        self.active = 0
        self.caregiver = 0
        self.indirect_tone = 0
        self.last_rel7_index = -1
        self.temporal_windows = {name: TemporalWindowAgg() for name, _size in TEMPORAL_WINDOWS}

    def add(self, text: str, rel: int, tweet_index: int) -> None:
        flags = lifetime_tweet_flags(text)
        self.total += 1
        self.sum_rel += float(rel)
        self.max_rel = max(self.max_rel, rel)
        self.rel3 += int(rel >= 3)
        self.rel5 += int(rel >= 5)
        self.rel6 += int(rel >= 6)
        self.rel7 += int(rel >= 7)
        if rel >= 7:
            self.last_rel7_index = max(self.last_rel7_index, tweet_index)
        self.top10.append((rel, tweet_index))
        self.top10.sort(reverse=True)
        if len(self.top10) > 10:
            self.top10.pop()

        self.fp += int(flags["fp"])
        self.neg += int(flags["neg"])
        self.selfneg += int(flags["selfneg"])
        self.active += int(flags["active"])
        self.caregiver += int(flags["caregiver"])
        self.indirect_tone += int(flags["indirect_tone"])

    def add_temporal(self, text: str, rel: int, tweet_index: int) -> None:
        thresholds = [
            int(math.floor(self.total * 2 / 3)) if size is None else max(self.total - size, 0)
            for _name, size in TEMPORAL_WINDOWS
        ]
        if tweet_index < min(thresholds):
            return
        flags = tweet_flags(text)
        for (name, _size), threshold in zip(TEMPORAL_WINDOWS, thresholds):
            if tweet_index >= threshold:
                self.temporal_windows[name].add(rel, flags)

    def marker(self, user_id: str) -> dict[str, Any]:
        denom = max(float(self.total), 1.0)
        partial = {
            "user_id": user_id,
            "total_tweets": self.total,
            "max_relevance": float(self.max_rel),
            "rel3_count": self.rel3,
            "rel5_count": self.rel5,
            "rel6_count": self.rel6,
            "rel7_count": self.rel7,
            "rel3_ratio": self.rel3 / denom,
            "rel5_ratio": self.rel5 / denom,
            "rel6_ratio": self.rel6 / denom,
            "rel7_ratio": self.rel7 / denom,
            "top10_avg_relevance": sum(rel for rel, _idx in self.top10) / max(len(self.top10), 1),
        }
        return {**partial, "evidence_score": compute_evidence_score(partial)}

    def stylistic(self) -> list[float]:
        denom = max(float(self.total), 1.0)
        return [
            self.fp / denom,
            self.neg / denom,
            self.selfneg / denom,
            self.active / denom,
            self.caregiver / denom,
            math.log1p(self.active),
            math.log1p(self.caregiver),
        ]

    def relevance_counts(self) -> list[float]:
        denom = max(float(self.total), 1.0)
        return [
            self.rel3 / denom,
            self.rel5 / denom,
            self.rel6 / denom,
            self.rel7 / denom,
            math.log1p(self.rel3),
            math.log1p(self.rel5),
            math.log1p(self.rel6),
            math.log1p(self.rel7),
        ]

    def temporal_markers(self) -> list[float]:
        lifetime_denom = max(float(self.total), 1.0)
        lifetime_mean_rel = self.sum_rel / lifetime_denom
        lifetime_rel7_ratio = self.rel7 / lifetime_denom
        lifetime_selfneg_ratio = self.selfneg / lifetime_denom
        lifetime_indirect_ratio = self.indirect_tone / lifetime_denom
        recent50 = self.temporal_windows["final50"]
        recent50_denom = max(float(recent50.total), 1.0)
        recent50_mean_rel = recent50.sum_rel / recent50_denom
        recent50_indirect_ratio = recent50.indirect_tone / recent50_denom
        last_rel7_position = 0.0
        if self.last_rel7_index >= 0 and self.total > 1:
            last_rel7_position = clamp01(self.last_rel7_index / float(self.total - 1))
        elif self.last_rel7_index >= 0:
            last_rel7_position = 1.0
        out: list[float] = []
        for name, _size in TEMPORAL_WINDOWS:
            out.extend(self.temporal_windows[name].features())
        out.extend(
            [
                recent50_mean_rel - lifetime_mean_rel,
                recent50.rel7 / recent50_denom - lifetime_rel7_ratio,
                recent50.selfneg / recent50_denom - lifetime_selfneg_ratio,
                recent50_indirect_ratio - lifetime_indirect_ratio,
                last_rel7_position,
            ]
        )
        return out


def clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def compute_evidence_score(marker: dict[str, Any]) -> float:
    return clamp01(
        0.3 * clamp01(float(marker["rel7_ratio"]) * 10)
        + 0.25 * clamp01(float(marker["rel5_ratio"]) * 6)
        + 0.15 * clamp01(float(marker["rel3_ratio"]) * 3)
        + 0.15 * clamp01(float(marker["top10_avg_relevance"]) / 7)
        + 0.15 * clamp01(float(marker["max_relevance"]) / 7)
    )


def evidence_feature_row(marker: dict[str, Any]) -> list[float]:
    total = max(float(marker["total_tweets"]), 0.0)
    return [
        math.log1p(total),
        float(marker["max_relevance"]),
        math.log1p(float(marker["rel3_count"])),
        math.log1p(float(marker["rel5_count"])),
        math.log1p(float(marker["rel6_count"])),
        math.log1p(float(marker["rel7_count"])),
        float(marker["rel3_ratio"]),
        float(marker["rel5_ratio"]),
        float(marker["rel6_ratio"]),
        float(marker["rel7_ratio"]),
        float(marker["top10_avg_relevance"]),
        float(marker["evidence_score"]),
    ]


def collect_tweet_features(raw_artifacts_dir: Path, split: str) -> dict[str, UserAgg]:
    dataset = ds.dataset(raw_artifacts_dir / "tweet_embeddings" / split, format="parquet")
    aggs: dict[str, UserAgg] = defaultdict(UserAgg)
    for batch in dataset.to_batches(columns=["user_id", "tweet_index", "tweet_text", "gpt5_relevance"], batch_size=65536):
        users = batch.column("user_id").to_pylist()
        indexes = batch.column("tweet_index").to_pylist()
        texts = batch.column("tweet_text").to_pylist()
        relevances = batch.column("gpt5_relevance").to_pylist()
        for uid, tweet_index, text, raw_rel in zip(users, indexes, texts, relevances):
            aggs[str(uid)].add("" if text is None else str(text), relevance_value(raw_rel), int(tweet_index))
    for batch in dataset.to_batches(columns=["user_id", "tweet_index", "tweet_text", "gpt5_relevance"], batch_size=65536):
        users = batch.column("user_id").to_pylist()
        indexes = batch.column("tweet_index").to_pylist()
        texts = batch.column("tweet_text").to_pylist()
        relevances = batch.column("gpt5_relevance").to_pylist()
        for uid, tweet_index, text, raw_rel in zip(users, indexes, texts, relevances):
            aggs[str(uid)].add_temporal("" if text is None else str(text), relevance_value(raw_rel), int(tweet_index))
    return aggs


def write_marker_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    return write_csv(
        path,
        [
            "user_id",
            "total_tweets",
            "max_relevance",
            "rel3_count",
            "rel5_count",
            "rel6_count",
            "rel7_count",
            "rel3_ratio",
            "rel5_ratio",
            "rel6_ratio",
            "rel7_ratio",
            "top10_avg_relevance",
            "evidence_score",
        ],
        rows,
    )


def load_pooled(raw_artifacts_dir: Path, split: str, block: str, ordered_user_ids: list[str]) -> np.ndarray:
    path = raw_artifacts_dir / "pooled" / f"{split}_user_{block}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    try:
        user_ids = data["user_ids"].astype(str)
        embeddings = data["embeddings"].astype(np.float32)
        by_user = {uid: index for index, uid in enumerate(user_ids)}
        missing = [uid for uid in ordered_user_ids if uid not in by_user]
        if missing:
            raise ValueError(f"{split} pooled {block} missing users: {missing[:5]}")
        return embeddings[[by_user[uid] for uid in ordered_user_ids]]
    finally:
        data.close()


def prepare_features(raw_artifacts_dir: Path, output_dir: Path, raw_rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    feature_hashes: dict[str, str] = {}
    feature_dir = output_dir / "features"
    for split in ["train", "test"]:
        split_rows = [row for row in raw_rows if row["split"] == split]
        user_ids = [row["user_id"] for row in split_rows]
        aggs = collect_tweet_features(raw_artifacts_dir, split)
        missing = [uid for uid in user_ids if uid not in aggs]
        if missing:
            raise ValueError(f"{split} tweet feature aggregation missing users: {missing[:5]}")

        marker_rows = [aggs[uid].marker(uid) for uid in user_ids]
        marker_hash = write_marker_csv(output_dir / "evidence-markers" / f"{split}_markers.csv", marker_rows)
        feature_hashes[f"{split}_markers"] = marker_hash
        arrays: dict[str, Any] = {
            "user_ids": np.asarray(user_ids, dtype=object),
            "labels": np.asarray([row["label_code"] if split == "train" else -1 for row in split_rows], dtype=np.int16),
            "true_labels": np.asarray([row["label_code"] if split == "train" else -1 for row in split_rows], dtype=np.int16),
            "folds": np.asarray([int(row["fold"]) if split == "train" else -1 for row in split_rows], dtype=np.int16),
            "evidence_markers": np.asarray([evidence_feature_row(marker) for marker in marker_rows], dtype=np.float32),
            "stylistic": np.asarray([aggs[uid].stylistic() for uid in user_ids], dtype=np.float32),
            "relevance_counts": np.asarray([aggs[uid].relevance_counts() for uid in user_ids], dtype=np.float32),
            "temporal_markers": np.asarray([aggs[uid].temporal_markers() for uid in user_ids], dtype=np.float32),
        }
        for block in EMBEDDING_BLOCKS:
            arrays[block] = load_pooled(raw_artifacts_dir, split, block, user_ids)
        out_path = feature_dir / f"{split}_raw_features.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **arrays)
        feature_hashes[f"{split}_raw_features"] = sha256_file(out_path)

    write_json(
        output_dir / "features" / "raw_feature_manifest.json",
        {
            "dataset": "setembrobr",
            "seed": seed,
            "featureSource": "raw_artifacts",
            "rawArtifactsDir": str(raw_artifacts_dir),
            "evidenceColumns": EVIDENCE_COLUMNS,
            "temporalColumns": TEMPORAL_COLUMNS,
            "featureHashes": feature_hashes,
            "strictBlind": {
                "testLabelsInFeatureNpz": False,
                "testLabelsValue": -1,
                "sealedLabelsOnlyForFinalEvaluation": True,
            },
        },
    )
    return {"featureHashes": feature_hashes}


def select_sequence_indexes(row_indexes: list[int], rels: list[int], indexes: list[Any], top_n: int, sequence_order: str) -> list[int]:
    if sequence_order == "relevance_desc":
        return sorted(row_indexes, key=lambda idx: (rels[idx], int(indexes[idx])), reverse=True)[:top_n]
    if sequence_order == "recent_chronological":
        recent = sorted(row_indexes, key=lambda idx: int(indexes[idx]), reverse=True)[:top_n]
        return sorted(recent, key=lambda idx: int(indexes[idx]))
    raise ValueError(f"unsupported sequence export order: {sequence_order}")


def sequence_sort_order(sequence_order: str) -> list[str]:
    if sequence_order == "relevance_desc":
        return ["gpt5_relevance_desc", "tweet_index_desc"]
    if sequence_order == "recent_chronological":
        return ["tweet_index_desc_take_topN", "tweet_index_asc_within_window"]
    raise ValueError(f"unsupported sequence export order: {sequence_order}")


def export_sequences(raw_artifacts_dir: Path, output_dir: Path, raw_rows: list[dict[str, Any]], seed: int, top_n: int, sequence_order: str = "relevance_desc") -> dict[str, Any]:
    sequence_hashes: dict[str, str] = {}
    embedding_dim = infer_embedding_dim(raw_artifacts_dir)
    for split in ["train", "test"]:
        split_rows = [row for row in raw_rows if row["split"] == split]
        user_ids = [row["user_id"] for row in split_rows]
        user_index = {uid: index for index, uid in enumerate(user_ids)}
        sequences = np.zeros((len(user_ids), top_n, embedding_dim), dtype=np.float16)
        relevances = np.zeros((len(user_ids), top_n), dtype=np.int16)
        lengths = np.zeros(len(user_ids), dtype=np.int32)

        for path in sorted((raw_artifacts_dir / "tweet_embeddings" / split).glob("*.parquet")):
            table = pq.read_table(path, columns=["user_id", "tweet_index", "gpt5_relevance", "embedding"])
            users = table.column("user_id").to_pylist()
            indexes = table.column("tweet_index").to_pylist()
            rels = [relevance_value(value) for value in table.column("gpt5_relevance").to_pylist()]
            by_user: dict[str, list[int]] = defaultdict(list)
            for row_index, uid in enumerate(users):
                if uid in user_index:
                    by_user[str(uid)].append(row_index)
            embedding_col = table.column("embedding").combine_chunks()
            for uid, row_indexes in by_user.items():
                selected = select_sequence_indexes(row_indexes, rels, indexes, top_n, sequence_order)
                if not selected:
                    continue
                taken = embedding_col.take(pa.array(selected, type=pa.int64()))
                values = taken.values.to_numpy(zero_copy_only=False).reshape(len(selected), embedding_dim).astype(np.float16)
                target = user_index[uid]
                length = len(selected)
                sequences[target, :length, :] = values
                relevances[target, :length] = np.asarray([rels[idx] for idx in selected], dtype=np.int16)
                lengths[target] = length

        labels = np.asarray([row["label_code"] if split == "train" else -1 for row in split_rows], dtype=np.int32)
        out_dir = output_dir / "sequences" / f"top{top_n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split}_seq.npz"
        np.savez_compressed(
            out_path,
            user_ids=np.asarray(user_ids, dtype=object),
            labels=labels,
            sequences=sequences,
            lengths=lengths,
            relevances=relevances,
        )
        sequence_hashes[f"{split}_top{top_n}"] = sha256_file(out_path)
    write_json(
        output_dir / "sequences" / f"top{top_n}" / "sequence_manifest.json",
        {
            "dataset": "setembrobr",
            "seed": seed,
            "topN": top_n,
            "embeddingDimension": embedding_dim,
            "sequenceOrder": sequence_order,
            "sortOrder": sequence_sort_order(sequence_order),
            "testLabelsValue": -1,
            "sequenceHashes": sequence_hashes,
        },
    )
    return {"sequenceHashes": sequence_hashes, "embeddingDimension": embedding_dim}


def infer_embedding_dim(raw_artifacts_dir: Path) -> int:
    pooled = np.load(raw_artifacts_dir / "pooled" / "train_user_mean.npz", allow_pickle=True)
    try:
        return int(pooled["embeddings"].shape[1])
    finally:
        pooled.close()


def main() -> None:
    args = parse_args()
    repo_dir = Path.cwd()
    config_path = resolve_path(args.config, repo_dir)
    config = read_config(config_path)
    raw_artifacts_dir = resolve_path(args.raw_artifacts_dir or config["rawArtifactsDir"], repo_dir)
    output_dir = resolve_path(args.output_dir or config["outputDir"], repo_dir)
    seed = int(config["seed"])
    sequence_export = config.get("sequenceExport", {})
    sequence_order = args.sequence_order or sequence_export.get("order", "relevance_desc")
    raw_rows = read_raw_manifest(raw_artifacts_dir, seed)

    manifest_report = write_strict_blind_manifests(output_dir, raw_rows, seed)
    report: dict[str, Any] = {
        "dataset": "setembrobr",
        "seed": seed,
        "config": str(config_path),
        "rawArtifactsDir": str(raw_artifacts_dir),
        "outputDir": str(output_dir),
        "mode": args.mode,
        "manifest": manifest_report,
        "rawEmbeddingManifestHash": sha256_file(raw_artifacts_dir / "reports" / "embedding_generation_manifest.json"),
    }
    if args.mode in {"prepare", "all"}:
        report.update(prepare_features(raw_artifacts_dir, output_dir, raw_rows, seed))
    if args.mode in {"export-sequences", "all"}:
        report.update(export_sequences(raw_artifacts_dir, output_dir, raw_rows, seed, args.top_n, sequence_order))

    write_json(output_dir / "reports" / "raw_ternary_prepare_manifest.json", report)
    print(json.dumps({"status": "ok", "report": str(output_dir / "reports" / "raw_ternary_prepare_manifest.json")}))


if __name__ == "__main__":
    main()
