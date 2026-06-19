#!/usr/bin/env python3
"""Train strict-blind ternary tabular OOF models from existing Postgres data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import psycopg2
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

LABELS = ["diagnosed", "control", "no-evidence"]
LABEL_TO_CODE = {label: index for index, label in enumerate(LABELS)}

V1_COLS = [
    "max_therapy",
    "max_medication",
    "max_selfharm",
    "max_suicidal",
    "max_emptiness",
    "max_depr_self",
    "max_insomnia",
    "max_crying",
]

V2_COLS = [
    "max_sadness",
    "max_pessimism",
    "max_failure",
    "max_anhedonia",
    "max_guilt",
    "max_self_dislike",
    "max_social_withdrawal",
    "max_fatigue",
    "max_anxiety",
    "max_cog_distortion",
    "max_self_harm_passive",
    "max_somatic",
]

EVIDENCE_COLS = [
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

FP_REGEX = r"\y(eu|meu|minha|mim|comigo|me)\y"
NEG_REGEX = r"\y(triste|vazi[oa]|sozinh[oa]|solit[aá]ri[oa]|cansad[oa]|exaust[oa]|inutil|fracass|derrotad[oa]|desespero|sofrendo|doendo|machuca|arrepend|odei[oa]|horriv[eí]l|pior|terriv[eí]l|dific[ií]l|impossiv[eí]l|raiva|irritad[oa])\y"
ACTIVE_IDEATION_REGEX = r"(quero morrer|vou me matar|vontade de morrer|pensei em suic[ií]d|tentei suic[ií]d|queria me matar|acabar com minha vida|n[aã]o quero mais viver|quero sumir pra sempre)"
CAREGIVER_REGEX = r"(meu (pai|irm|primo|namorad|marido|amigo)|minha (m[aã]e|irm|prima|namorad|amiga|esposa|filh))[^.!?\n]{0,80}(depress|ansiedade|terapia|psic[oó]log|psiquiatr|rem[eé]dio|antidepressiv|suicid)"

TABLES_BY_BLOCK = {
    "v1": ("trainSubFeatures", "testSubFeatures"),
    "v2": ("trainV2SubFeatures", "testV2SubFeatures"),
    "rel5_combined": ("trainRel5CombinedFeatures", "testRel5CombinedFeatures"),
    "stylistic": ("trainEmbeddings", "testEmbeddings"),
    "relevance_counts": ("trainEmbeddings", "testEmbeddings"),
    "mean_pca": ("trainUserEmb", "testUserEmb"),
    "rel3_pca": ("trainUserEmbRel3", "testUserEmbRel3"),
    "rel6_pca": ("trainUserEmbRel6", "testUserEmbRel6"),
    "rel7_pca": ("trainUserEmbRel7", "testUserEmbRel7"),
}

EMBEDDING_BLOCKS = {
    "mean_pca": "mean",
    "rel3_pca": "rel3",
    "rel6_pca": "rel6",
    "rel7_pca": "rel7",
}


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    parent_ref = cfg.get("extends")
    if not isinstance(parent_ref, str):
        return cfg
    parent_path = Path(parent_ref).expanduser()
    if not parent_path.is_absolute():
        parent_path = (path.parent / parent_path).resolve()
    child = {key: value for key, value in cfg.items() if key != "extends"}
    return deep_merge(load_config(parent_path), child)


def deep_merge(parent: dict, child: dict) -> dict:
    merged = dict(parent)
    for key, value in child.items():
        parent_value = merged.get(key)
        if isinstance(parent_value, dict) and isinstance(value, dict):
            merged[key] = deep_merge(parent_value, value)
        else:
            merged[key] = value
    return merged


def load_env() -> dict[str, str]:
    env = dict(os.environ)
    path = Path(".env")
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key.strip(), value.strip().strip("\"'"))
    return env


def assert_safe_identifier(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe SQL identifier: {value}")


def parse_vector(raw: str) -> np.ndarray:
    return np.fromstring(raw.strip("[]"), sep=",", dtype=np.float32)


def read_csv_with_hash(path: Path):
    text = path.read_text()
    return list(csv.DictReader(text.splitlines())), hashlib.sha256(text.encode()).hexdigest()


def read_source_test_ids(cfg):
    if cfg.get("featureSource") == "raw_artifacts":
        path = Path(cfg["outputDir"]) / "manifest" / f"test_inference_manifest_seed{cfg['seed']}.csv"
        rows, _hash = read_csv_with_hash(path)
        return [row["user_id"] for row in rows]
    path = Path(cfg["sourceOutputDir"]) / "manifest" / f"split_manifest_seed{cfg['seed']}.csv"
    rows, _hash = read_csv_with_hash(path)
    return [row["user_id"] for row in rows if row["split"] == "test"]


def load_embedding_table(conn, table: str) -> dict[str, np.ndarray]:
    assert_safe_identifier(table)
    cur = conn.cursor()
    cur.execute(f"select user_id, embedding::text from {table}")
    out = {uid: parse_vector(raw) for uid, raw in cur.fetchall()}
    cur.close()
    return out


def load_numeric_table(conn, table: str, columns: list[str], log_columns: set[str] | None = None) -> dict[str, list[float]]:
    assert_safe_identifier(table)
    for column in columns:
        assert_safe_identifier(column)
    log_columns = log_columns or set()
    cur = conn.cursor()
    cur.execute(f"select user_id, {', '.join(columns)} from {table}")
    out: dict[str, list[float]] = {}
    for row in cur.fetchall():
        values = []
        for column, value in zip(columns, row[1:]):
            numeric = 0.0 if value is None else float(value)
            values.append(float(math.log1p(max(numeric, 0.0))) if column in log_columns else numeric)
        out[row[0]] = values
    cur.close()
    return out


def load_tweet_blocks(conn, emb_table: str) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    assert_safe_identifier(emb_table)
    cur = conn.cursor()
    cur.execute(
        f"""
        with agg as (
          select user_id,
                 count(*) as total_tweets,
                 count(*) filter (where tweet_text ~* %s) as fp_count,
                 count(*) filter (where tweet_text ~* %s) as neg_count,
                 count(*) filter (where tweet_text ~* %s and tweet_text ~* %s) as selfneg_count,
                 count(*) filter (where tweet_text ~* %s) as active_ideation_count,
                 count(*) filter (where tweet_text ~* %s) as caregiver_count,
                 count(*) filter (where coalesce(gpt_3_5_relevance, 0) >= 3) as rel3_count,
                 count(*) filter (where coalesce(gpt_3_5_relevance, 0) >= 5) as rel5_count,
                 count(*) filter (where coalesce(gpt_3_5_relevance, 0) >= 6) as rel6_count,
                 count(*) filter (where coalesce(gpt_3_5_relevance, 0) >= 7) as rel7_count
          from {emb_table}
          group by user_id
        )
        select * from agg
        """,
        (FP_REGEX, NEG_REGEX, FP_REGEX, NEG_REGEX, ACTIVE_IDEATION_REGEX, CAREGIVER_REGEX),
    )
    stylistic: dict[str, list[float]] = {}
    relevance_counts: dict[str, list[float]] = {}
    for uid, total, fp, neg, selfneg, active, caregiver, rel3, rel5, rel6, rel7 in cur.fetchall():
        denom = max(float(total), 1.0)
        stylistic[uid] = [
            float(fp) / denom,
            float(neg) / denom,
            float(selfneg) / denom,
            float(active) / denom,
            float(caregiver) / denom,
            float(np.log1p(active)),
            float(np.log1p(caregiver)),
        ]
        relevance_counts[uid] = [
            float(rel3) / denom,
            float(rel5) / denom,
            float(rel6) / denom,
            float(rel7) / denom,
            float(np.log1p(rel3)),
            float(np.log1p(rel5)),
            float(np.log1p(rel6)),
            float(np.log1p(rel7)),
        ]
    cur.close()
    return stylistic, relevance_counts


def load_evidence_markers(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            total = float(row["total_tweets"])
            values = [
                float(np.log1p(max(total, 0.0))),
                float(row["max_relevance"]),
                float(np.log1p(float(row["rel3_count"]))),
                float(np.log1p(float(row["rel5_count"]))),
                float(np.log1p(float(row["rel6_count"]))),
                float(np.log1p(float(row["rel7_count"]))),
                float(row["rel3_ratio"]),
                float(row["rel5_ratio"]),
                float(row["rel6_ratio"]),
                float(row["rel7_ratio"]),
                float(row["top10_avg_relevance"]),
                float(row["evidence_score"]),
            ]
            out[row["user_id"]] = values
    return out


def align_block(
    user_ids: list[str],
    values: dict[str, list[float]] | dict[str, np.ndarray],
    block_name: str,
    fill_missing: bool = False,
) -> np.ndarray:
    missing = [uid for uid in user_ids if uid not in values]
    if missing:
        if fill_missing:
            first = next(iter(values.values()), None)
            if first is None:
                raise RuntimeError(f"{block_name} has no rows and cannot be imputed")
            width = int(first.shape[0]) if isinstance(first, np.ndarray) else len(first)
            zero = np.zeros(width, dtype=np.float32)
            rows = [values.get(uid, zero) for uid in user_ids]
            if isinstance(rows[0], np.ndarray):
                return np.stack(rows).astype(np.float32)
            return np.array(rows, dtype=np.float32)
        sample = ", ".join(missing[:5])
        raise RuntimeError(f"{block_name} missing {len(missing)} users, sample: {sample}")
    rows = [values[uid] for uid in user_ids]
    if isinstance(rows[0], np.ndarray):
        return np.stack(rows).astype(np.float32)
    return np.array(rows, dtype=np.float32)


def load_all_features(conn, cfg, train_manifest_rows, required_blocks: set[str]):
    tables = cfg["database"]["tables"]
    train_ids = [row["user_id"] for row in train_manifest_rows]
    test_ids = read_source_test_ids(cfg)
    train = {
        "user_ids": np.array(train_ids),
        "labels": np.array([LABEL_TO_CODE[row["label"]] for row in train_manifest_rows], dtype=np.int64),
        "binary_labels": np.array([1 if row["binary_label"] == "diagnosed" else 0 for row in train_manifest_rows], dtype=np.int64),
        "folds": np.array([int(row["fold"]) for row in train_manifest_rows], dtype=np.int32),
        "blocks": {},
    }
    split_test = {"user_ids": np.array(test_ids), "blocks": {}}

    if "evidence_markers" in required_blocks:
        train_markers = load_evidence_markers(Path(cfg["outputDir"]) / "evidence-markers" / "train_markers.csv")
        test_markers = load_evidence_markers(Path(cfg["outputDir"]) / "evidence-markers" / "test_markers.csv")
        train["blocks"]["evidence_markers"] = align_block(train_ids, train_markers, "train evidence_markers")
        split_test["blocks"]["evidence_markers"] = align_block(test_ids, test_markers, "test evidence_markers")

    if "stylistic" in required_blocks or "relevance_counts" in required_blocks:
        train_sty, train_rel_counts = load_tweet_blocks(conn, tables["trainEmbeddings"])
        test_sty, test_rel_counts = load_tweet_blocks(conn, tables["testEmbeddings"])
        if "stylistic" in required_blocks:
            train["blocks"]["stylistic"] = align_block(train_ids, train_sty, "train stylistic")
            split_test["blocks"]["stylistic"] = align_block(test_ids, test_sty, "test stylistic")
        if "relevance_counts" in required_blocks:
            train["blocks"]["relevance_counts"] = align_block(train_ids, train_rel_counts, "train relevance_counts")
            split_test["blocks"]["relevance_counts"] = align_block(test_ids, test_rel_counts, "test relevance_counts")

    if "v1" in required_blocks:
        v1_cols = [*V1_COLS, "total_tweets"]
        train["blocks"]["v1"] = align_block(train_ids, load_numeric_table(conn, tables["trainSubFeatures"], v1_cols, {"total_tweets"}), "train v1")
        split_test["blocks"]["v1"] = align_block(test_ids, load_numeric_table(conn, tables["testSubFeatures"], v1_cols, {"total_tweets"}), "test v1")

    if "v2" in required_blocks and "trainV2SubFeatures" in tables and "testV2SubFeatures" in tables:
        v2_cols = [*V2_COLS, "total_tweets"]
        train["blocks"]["v2"] = align_block(
            train_ids,
            load_numeric_table(conn, tables["trainV2SubFeatures"], v2_cols, {"total_tweets"}),
            "train v2",
            fill_missing=True,
        )
        split_test["blocks"]["v2"] = align_block(
            test_ids,
            load_numeric_table(conn, tables["testV2SubFeatures"], v2_cols, {"total_tweets"}),
            "test v2",
            fill_missing=True,
        )

    if "rel5_combined" in required_blocks and "trainRel5CombinedFeatures" in tables and "testRel5CombinedFeatures" in tables:
        rel5_cols = [*V1_COLS, *V2_COLS, "total_tweets", "relevant_tweets"]
        train["blocks"]["rel5_combined"] = align_block(
            train_ids,
            load_numeric_table(conn, tables["trainRel5CombinedFeatures"], rel5_cols, {"total_tweets", "relevant_tweets"}),
            "train rel5_combined",
            fill_missing=True,
        )
        split_test["blocks"]["rel5_combined"] = align_block(
            test_ids,
            load_numeric_table(conn, tables["testRel5CombinedFeatures"], rel5_cols, {"total_tweets", "relevant_tweets"}),
            "test rel5_combined",
            fill_missing=True,
        )

    embedding_tables = {
        "mean": ("trainUserEmb", "testUserEmb"),
        "rel3": ("trainUserEmbRel3", "testUserEmbRel3"),
        "rel6": ("trainUserEmbRel6", "testUserEmbRel6"),
        "rel7": ("trainUserEmbRel7", "testUserEmbRel7"),
    }
    for block_name, (train_key, test_key) in embedding_tables.items():
        if f"{block_name}_pca" not in required_blocks:
            continue
        if train_key not in tables or test_key not in tables:
            continue
        train["blocks"][block_name] = align_block(train_ids, load_embedding_table(conn, tables[train_key]), f"train {block_name}")
        split_test["blocks"][block_name] = align_block(test_ids, load_embedding_table(conn, tables[test_key]), f"test {block_name}")

    return train, split_test


def load_raw_feature_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    out = {
        "user_ids": data["user_ids"].astype(str),
        "labels": data["labels"].astype(np.int64),
        "true_labels": data["true_labels"].astype(np.int64) if "true_labels" in data.files else data["labels"].astype(np.int64),
        "folds": data["folds"].astype(np.int32),
        "blocks": {},
    }
    for block in ["evidence_markers", "stylistic", "relevance_counts", "mean", "rel3", "rel6", "rel7"]:
        if block in data.files:
            out["blocks"][block] = data[block].astype(np.float32)
    data.close()
    return out


def reorder_raw_split(split, user_ids: list[str], labels=None, binary_labels=None, folds=None):
    by_user = {uid: index for index, uid in enumerate(split["user_ids"])}
    missing = [uid for uid in user_ids if uid not in by_user]
    if missing:
        raise RuntimeError(f"raw feature NPZ missing users: {missing[:5]}")
    indexes = np.asarray([by_user[uid] for uid in user_ids], dtype=np.int64)
    out = {
        "user_ids": np.asarray(user_ids),
        "blocks": {name: values[indexes] for name, values in split["blocks"].items()},
    }
    if labels is not None:
        out["labels"] = np.asarray(labels, dtype=np.int64)
    if binary_labels is not None:
        out["binary_labels"] = np.asarray(binary_labels, dtype=np.int64)
    if folds is not None:
        out["folds"] = np.asarray(folds, dtype=np.int32)
    return out


def load_all_raw_features(cfg, train_manifest_rows, required_blocks: set[str]):
    unsupported = sorted(block for block in required_blocks if block not in {"evidence_markers", "stylistic", "relevance_counts", *EMBEDDING_BLOCKS})
    if unsupported:
        raise RuntimeError(f"raw_artifacts featureSource does not support DB-only blocks: {unsupported}")
    required_sources = {EMBEDDING_BLOCKS.get(block, block) for block in required_blocks}
    feature_dir = Path(cfg["outputDir"]) / "features"
    raw_train = load_raw_feature_npz(feature_dir / "train_raw_features.npz")
    raw_test = load_raw_feature_npz(feature_dir / "test_raw_features.npz")
    train_ids = [row["user_id"] for row in train_manifest_rows]
    train_labels = [LABEL_TO_CODE[row["label"]] for row in train_manifest_rows]
    train_binary = [1 if row["binary_label"] == "diagnosed" else 0 for row in train_manifest_rows]
    train_folds = [int(row["fold"]) for row in train_manifest_rows]
    test_ids = read_source_test_ids(cfg)
    train = reorder_raw_split(raw_train, train_ids, train_labels, train_binary, train_folds)
    split_test = reorder_raw_split(raw_test, test_ids)
    if required_sources - set(train["blocks"]):
        raise RuntimeError(f"raw train features missing blocks: {sorted(required_sources - set(train['blocks']))}")
    if required_sources - set(split_test["blocks"]):
        raise RuntimeError(f"raw test features missing blocks: {sorted(required_sources - set(split_test['blocks']))}")
    return train, split_test


def subset(split, mask):
    out = {key: value[mask] for key, value in split.items() if key in {"user_ids", "labels", "binary_labels", "folds"}}
    out["blocks"] = {name: value[mask] for name, value in split["blocks"].items()}
    return out


def build_features(train, val, split_test, candidate):
    train_pieces = []
    val_pieces = []
    test_pieces = []
    pca_components = int(candidate.get("pcaComponents", 80))
    for offset, block_name in enumerate(candidate["featureBlocks"]):
        if block_name in EMBEDDING_BLOCKS:
            source = EMBEDDING_BLOCKS[block_name]
            n_components = min(pca_components, len(train["labels"]) - 1, train["blocks"][source].shape[1])
            if n_components < 1:
                raise RuntimeError(f"{candidate['modelId']} cannot fit PCA for {block_name}")
            pca = PCA(n_components=n_components, random_state=int(candidate["seed"]) + offset * 101)
            train_pieces.append(pca.fit_transform(train["blocks"][source]))
            val_pieces.append(pca.transform(val["blocks"][source]))
            test_pieces.append(pca.transform(split_test["blocks"][source]))
        else:
            train_pieces.append(train["blocks"][block_name])
            val_pieces.append(val["blocks"][block_name])
            test_pieces.append(split_test["blocks"][block_name])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.nan_to_num(np.hstack(train_pieces), copy=False))
    x_val = scaler.transform(np.nan_to_num(np.hstack(val_pieces), copy=False))
    x_test = scaler.transform(np.nan_to_num(np.hstack(test_pieces), copy=False))
    return x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(np.float32)


def normalize_probs(probs):
    probs = np.nan_to_num(probs.astype(np.float64), nan=1.0 / 3.0, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, 1e-9, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def map_sklearn_probs(model, probs):
    out = np.zeros((probs.shape[0], 3), dtype=np.float64)
    for column, code in enumerate(model.classes_):
        out[:, int(code)] = probs[:, column]
    return normalize_probs(out)


def constant_probs(count, code):
    out = np.full((count, 3), 1e-9, dtype=np.float64)
    out[:, int(code)] = 1.0
    return normalize_probs(out)


def balanced_sample_weights(y_train):
    counts = np.bincount(y_train, minlength=3).astype(np.float64)
    class_weights = len(y_train) / (3.0 * np.maximum(counts, 1.0))
    return class_weights[y_train].astype(np.float32)


def fit_predict(candidate, x_train, y_train, binary_train, x_val, x_test):
    family = candidate["family"]
    seed = int(candidate["seed"])
    if family == "multinomial_logreg":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs")
        model.fit(x_train, y_train)
        return map_sklearn_probs(model, model.predict_proba(x_val)), map_sklearn_probs(model, model.predict_proba(x_test))
    if family == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(int(candidate["hiddenSize"]),),
            alpha=float(candidate["alpha"]),
            random_state=seed,
            max_iter=500,
            early_stopping=False,
        )
        model.fit(x_train, y_train)
        return map_sklearn_probs(model, model.predict_proba(x_val)), map_sklearn_probs(model, model.predict_proba(x_test))
    if family == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=int(candidate.get("nEstimators", 400)),
            max_depth=int(candidate.get("maxDepth", 5)),
            min_samples_leaf=int(candidate.get("minSamplesLeaf", 10)),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return map_sklearn_probs(model, model.predict_proba(x_val)), map_sklearn_probs(model, model.predict_proba(x_test))
    if family == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is required for ternary xgboost candidates. Install requirements.txt.")
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            n_estimators=int(candidate.get("nEstimators", 250)),
            max_depth=int(candidate.get("maxDepth", 3)),
            learning_rate=float(candidate.get("learningRate", 0.05)),
            subsample=float(candidate.get("subsample", 0.85)),
            colsample_bytree=float(candidate.get("colsampleBytree", 0.85)),
            reg_lambda=float(candidate.get("regLambda", 3.0)),
            min_child_weight=float(candidate.get("minChildWeight", 2.0)),
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(x_train, y_train, sample_weight=balanced_sample_weights(y_train))
        return map_sklearn_probs(model, model.predict_proba(x_val)), map_sklearn_probs(model, model.predict_proba(x_test))
    if family == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=int(candidate.get("maxIter", 220)),
            learning_rate=float(candidate.get("learningRate", 0.04)),
            max_leaf_nodes=int(candidate.get("maxLeafNodes", 15)),
            max_depth=int(candidate["maxDepth"]) if candidate.get("maxDepth") is not None else None,
            l2_regularization=float(candidate.get("l2Regularization", 0.1)),
            early_stopping=False,
            random_state=seed,
        )
        model.fit(x_train, y_train, sample_weight=balanced_sample_weights(y_train))
        return map_sklearn_probs(model, model.predict_proba(x_val)), map_sklearn_probs(model, model.predict_proba(x_test))
    if family == "focal_linear":
        return fit_focal_predict(x_train, y_train, x_val, x_test, float(candidate.get("gamma", 1.0)), seed)
    if family == "hierarchical_logreg":
        return fit_hierarchical_predict(x_train, y_train, binary_train, x_val, x_test, seed)
    raise RuntimeError(f"Unsupported ternary tabular family: {family}")


def fit_hierarchical_predict(x_train, y_train, binary_train, x_val, x_test, seed):
    if len(set(binary_train.tolist())) < 2:
        positive_val = np.ones(len(x_val), dtype=np.float64) if binary_train[0] == 1 else np.zeros(len(x_val), dtype=np.float64)
        positive_test = np.ones(len(x_test), dtype=np.float64) if binary_train[0] == 1 else np.zeros(len(x_test), dtype=np.float64)
    else:
        binary_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs")
        binary_model.fit(x_train, binary_train)
        positive_val = binary_model.predict_proba(x_val)[:, list(binary_model.classes_).index(1)]
        positive_test = binary_model.predict_proba(x_test)[:, list(binary_model.classes_).index(1)]

    positive_mask = binary_train == 1
    gate_y = (y_train[positive_mask] == LABEL_TO_CODE["diagnosed"]).astype(np.int64)
    if gate_y.size == 0:
        diag_gate_val = np.zeros(len(x_val), dtype=np.float64)
        diag_gate_test = np.zeros(len(x_test), dtype=np.float64)
    elif len(set(gate_y.tolist())) < 2:
        diag_gate_val = np.full(len(x_val), float(gate_y[0]), dtype=np.float64)
        diag_gate_test = np.full(len(x_test), float(gate_y[0]), dtype=np.float64)
    else:
        gate_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed + 101, solver="lbfgs")
        gate_model.fit(x_train[positive_mask], gate_y)
        diag_gate_val = gate_model.predict_proba(x_val)[:, list(gate_model.classes_).index(1)]
        diag_gate_test = gate_model.predict_proba(x_test)[:, list(gate_model.classes_).index(1)]
    return hierarchical_probs(positive_val, diag_gate_val), hierarchical_probs(positive_test, diag_gate_test)


def hierarchical_probs(positive_prob, diag_gate):
    out = np.zeros((len(positive_prob), 3), dtype=np.float64)
    out[:, LABEL_TO_CODE["diagnosed"]] = positive_prob * diag_gate
    out[:, LABEL_TO_CODE["control"]] = 1.0 - positive_prob
    out[:, LABEL_TO_CODE["no-evidence"]] = positive_prob * (1.0 - diag_gate)
    return normalize_probs(out)


def fit_focal_predict(x_train, y_train, x_val, x_test, gamma: float, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    x = torch.from_numpy(x_train).to(device)
    y = torch.from_numpy(y_train.astype(np.int64)).to(device)
    model = torch.nn.Linear(x_train.shape[1], 3).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = torch.from_numpy(len(y_train) / (3.0 * np.maximum(counts, 1.0))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for _epoch in range(300):
        opt.zero_grad()
        logits = model(x).clamp(-30.0, 30.0)
        ce = F.cross_entropy(logits, y, weight=weights, reduction="none")
        pt = torch.exp(-ce)
        loss = (((1.0 - pt) ** gamma) * ce).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("ternary focal linear produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        val_probs = torch.softmax(model(torch.from_numpy(x_val).to(device)).clamp(-30.0, 30.0), dim=1).cpu().numpy()
        test_probs = torch.softmax(model(torch.from_numpy(x_test).to(device)).clamp(-30.0, 30.0), dim=1).cpu().numpy()
    return normalize_probs(val_probs), normalize_probs(test_probs)


def relevance_baseline_probs(split):
    evidence = np.clip(split["blocks"]["evidence_markers"][:, EVIDENCE_COLS.index("evidence_score")], 0.0, 1.0)
    out = np.zeros((len(evidence), 3), dtype=np.float64)
    out[:, LABEL_TO_CODE["diagnosed"]] = evidence + 0.05
    out[:, LABEL_TO_CODE["control"]] = 0.25
    out[:, LABEL_TO_CODE["no-evidence"]] = 1.0 - evidence
    return normalize_probs(out)


def write_scores(path: Path, rows, include_labels: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = (
        ["user_id", "label", "fold", "prob_diagnosed", "prob_control", "prob_no_evidence", "model_id", "label_policy_id"]
        if include_labels
        else ["user_id", "prob_diagnosed", "prob_control", "prob_no_evidence", "model_id", "label_policy_id"]
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in headers})


def db_tables_for_candidate(cfg, feature_blocks: list[str]) -> dict[str, str]:
    if cfg.get("featureSource") == "raw_artifacts":
        return {}
    tables = cfg["database"]["tables"]
    keys: list[str] = []
    for block in feature_blocks:
        if block == "evidence_markers":
            continue
        for key in TABLES_BY_BLOCK[block]:
            if key not in keys:
                keys.append(key)
    return {key: tables[key] for key in keys if key in tables}


def feature_cache_key(candidate, fold: int) -> tuple:
    return (tuple(candidate["featureBlocks"]), int(candidate.get("pcaComponents", 80)), fold)


def train_candidate(cfg, policy_lock, train_manifest_hash, train_all, split_test, candidate, feature_cache) -> None:
    model_id = candidate["modelId"]
    policy_id = policy_lock["policyId"]
    folds = sorted(set(int(fold) for fold in train_all["folds"]))
    oof_rows = []
    test_sum = np.zeros((len(split_test["user_ids"]), 3), dtype=np.float64)
    test_count = 0
    for fold in folds:
        train_mask = train_all["folds"] != fold
        val_mask = train_all["folds"] == fold
        train = subset(train_all, train_mask)
        val = subset(train_all, val_mask)
        fold_candidate = {**candidate, "seed": int(candidate["seed"]) + fold}
        if candidate["family"] == "relevance_baseline":
            val_probs = relevance_baseline_probs(val)
            test_probs = relevance_baseline_probs(split_test)
        else:
            cache_key = feature_cache_key(fold_candidate, fold)
            if cache_key not in feature_cache:
                feature_cache[cache_key] = build_features(train, val, split_test, fold_candidate)
            x_train, x_val, x_test = feature_cache[cache_key]
            val_probs, test_probs = fit_predict(fold_candidate, x_train, train["labels"], train["binary_labels"], x_val, x_test)
        for uid, label_code, prob in zip(val["user_ids"], val["labels"], val_probs):
            oof_rows.append(
                {
                    "user_id": uid,
                    "label": LABELS[int(label_code)],
                    "fold": fold,
                    "prob_diagnosed": f"{float(prob[LABEL_TO_CODE['diagnosed']]):.8f}",
                    "prob_control": f"{float(prob[LABEL_TO_CODE['control']]):.8f}",
                    "prob_no_evidence": f"{float(prob[LABEL_TO_CODE['no-evidence']]):.8f}",
                    "model_id": model_id,
                    "label_policy_id": policy_id,
                }
            )
        test_sum += test_probs
        test_count += 1

    oof_rows.sort(key=lambda row: row["user_id"])
    test_avg = normalize_probs(test_sum / max(test_count, 1))
    test_rows = [
        {
            "user_id": uid,
            "prob_diagnosed": f"{float(prob[LABEL_TO_CODE['diagnosed']]):.8f}",
            "prob_control": f"{float(prob[LABEL_TO_CODE['control']]):.8f}",
            "prob_no_evidence": f"{float(prob[LABEL_TO_CODE['no-evidence']]):.8f}",
            "model_id": model_id,
            "label_policy_id": policy_id,
        }
        for uid, prob in zip(split_test["user_ids"], test_avg)
    ]
    out_dir = Path(cfg["outputDir"])
    artifact_id = f"{policy_id}_{model_id}"
    write_scores(out_dir / "scores" / f"train_oof_{artifact_id}.csv", oof_rows, True)
    write_scores(out_dir / "scores" / f"test_score_{artifact_id}.csv", test_rows, False)
    model_manifest = {
        "modelId": model_id,
        "artifactId": artifact_id,
        "candidate": True,
        "family": candidate["family"],
        "seed": candidate["seed"],
        "originalManifestHash": policy_lock["originalManifestHash"],
        "trainManifestHash": train_manifest_hash,
        "labelPolicyId": policy_id,
        "labelPolicyHash": policy_lock["policyHash"],
        "featureSource": cfg.get("featureSource", "postgres"),
        "rawArtifactsDir": cfg.get("rawArtifactsDir"),
        "dbTables": db_tables_for_candidate(cfg, candidate["featureBlocks"]),
        "featureBlocks": candidate["featureBlocks"],
        "hyperparameters": {key: value for key, value in candidate.items() if key not in {"modelId", "family", "featureBlocks"}},
        "scoreSchema": "ternary-probability-v1",
        "usesTestLabelsForTraining": False,
        "gpuUsed": False,
        "createdAt": "1970-01-01T00:00:00.000Z",
    }
    manifest_path = out_dir / "model-manifests" / f"{artifact_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(model_manifest, indent=2) + "\n")
    print(f"wrote {artifact_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ternary-classification/configs/setembrobr.seed42.ternary-strict-blind.json")
    parser.add_argument("--policy", nargs="*", default=None)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    candidates = list(cfg.get("candidateModels", {}).get("tabular", []))
    if args.only:
        wanted = set(args.only)
        candidates = [candidate for candidate in candidates if candidate["modelId"] in wanted]
    if not candidates:
        raise SystemExit("No ternary tabular candidates selected")

    policies = [policy["policyId"] for policy in cfg["labelPolicies"]]
    if args.policy:
        wanted_policies = set(args.policy)
        policies = [policy for policy in policies if policy in wanted_policies]
    if not policies:
        raise SystemExit("No ternary label policies selected")

    if cfg.get("featureSource") == "raw_artifacts":
        for policy_id in policies:
            train_manifest_path = Path(cfg["outputDir"]) / "manifest" / f"train_manifest_{policy_id}_seed{cfg['seed']}.csv"
            train_manifest_rows, train_manifest_hash = read_csv_with_hash(train_manifest_path)
            policy_lock = json.loads((Path(cfg["outputDir"]) / "label-policies" / f"{policy_id}.json").read_text())
            required_blocks = {block for candidate in candidates for block in candidate["featureBlocks"]}
            train_all, split_test = load_all_raw_features(cfg, train_manifest_rows, required_blocks)
            feature_cache = {}
            for candidate in candidates:
                train_candidate(cfg, policy_lock, train_manifest_hash, train_all, split_test, candidate, feature_cache)
        return

    env = load_env()
    database_url = env.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    conn = psycopg2.connect(database_url)
    try:
        for policy_id in policies:
            train_manifest_path = Path(cfg["outputDir"]) / "manifest" / f"train_manifest_{policy_id}_seed{cfg['seed']}.csv"
            train_manifest_rows, train_manifest_hash = read_csv_with_hash(train_manifest_path)
            policy_lock = json.loads((Path(cfg["outputDir"]) / "label-policies" / f"{policy_id}.json").read_text())
            required_blocks = {block for candidate in candidates for block in candidate["featureBlocks"]}
            train_all, split_test = load_all_features(conn, cfg, train_manifest_rows, required_blocks)
            feature_cache = {}
            for candidate in candidates:
                train_candidate(cfg, policy_lock, train_manifest_hash, train_all, split_test, candidate, feature_cache)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
