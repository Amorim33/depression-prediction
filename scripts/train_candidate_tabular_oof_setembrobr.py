#!/usr/bin/env python3
"""Train strict-blind tabular candidate OOF models from existing Postgres data."""

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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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


def read_manifest(cfg):
    path = Path(cfg["outputDir"]) / "manifest" / f"split_manifest_seed{cfg['seed']}.csv"
    text = path.read_text()
    rows = list(csv.DictReader(text.splitlines()))
    return rows, hashlib.sha256(text.encode()).hexdigest()


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
        raise RuntimeError(f"{block_name} missing {len(missing)} manifest users, sample: {sample}")
    rows = [values[uid] for uid in user_ids]
    if isinstance(rows[0], np.ndarray):
        return np.stack(rows).astype(np.float32)
    return np.array(rows, dtype=np.float32)


def load_all_features(conn, cfg, manifest_rows, required_blocks: set[str]):
    tables = cfg["database"]["tables"]
    train_rows = [row for row in manifest_rows if row["split"] == "train"]
    test_rows = [row for row in manifest_rows if row["split"] == "test"]
    train_ids = [row["user_id"] for row in train_rows]
    test_ids = [row["user_id"] for row in test_rows]
    train = {
        "user_ids": np.array(train_ids),
        "labels": np.array([1 if row["label"] == "diagnosed" else 0 for row in train_rows], dtype=np.int32),
        "folds": np.array([int(row["fold"]) for row in train_rows], dtype=np.int32),
        "blocks": {},
    }
    split_test = {"user_ids": np.array(test_ids), "blocks": {}}

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
        train["blocks"]["v1"] = align_block(
            train_ids,
            load_numeric_table(conn, tables["trainSubFeatures"], v1_cols, {"total_tweets"}),
            "train v1",
        )
        split_test["blocks"]["v1"] = align_block(
            test_ids,
            load_numeric_table(conn, tables["testSubFeatures"], v1_cols, {"total_tweets"}),
            "test v1",
        )

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


def subset(split, mask):
    return {
        key: value[mask] if key in {"user_ids", "labels", "folds"} else value
        for key, value in split.items()
        if key != "blocks"
    } | {"blocks": {name: value[mask] for name, value in split["blocks"].items()}}


def build_features(train, val, split_test, candidate):
    train_pieces = []
    val_pieces = []
    test_pieces = []
    pca_components = int(candidate.get("pcaComponents", 80))
    for offset, block_name in enumerate(candidate["featureBlocks"]):
        if block_name in EMBEDDING_BLOCKS:
            source = EMBEDDING_BLOCKS[block_name]
            for split_name, split in (("train", train), ("val", val), ("test", split_test)):
                if source not in split["blocks"]:
                    raise RuntimeError(f"{candidate['modelId']} requires missing {split_name} block {source}")
            n_components = min(pca_components, len(train["labels"]) - 1, train["blocks"][source].shape[1])
            if n_components < 1:
                raise RuntimeError(f"{candidate['modelId']} cannot fit PCA for {block_name}")
            pca = PCA(n_components=n_components, random_state=int(candidate["seed"]) + offset * 101)
            train_pieces.append(pca.fit_transform(train["blocks"][source]))
            val_pieces.append(pca.transform(val["blocks"][source]))
            test_pieces.append(pca.transform(split_test["blocks"][source]))
        else:
            for split_name, split in (("train", train), ("val", val), ("test", split_test)):
                if block_name not in split["blocks"]:
                    raise RuntimeError(f"{candidate['modelId']} requires missing {split_name} block {block_name}")
            train_pieces.append(train["blocks"][block_name])
            val_pieces.append(val["blocks"][block_name])
            test_pieces.append(split_test["blocks"][block_name])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.nan_to_num(np.hstack(train_pieces), copy=False))
    x_val = scaler.transform(np.nan_to_num(np.hstack(val_pieces), copy=False))
    x_test = scaler.transform(np.nan_to_num(np.hstack(test_pieces), copy=False))
    return x_train.astype(np.float32), x_val.astype(np.float32), x_test.astype(np.float32)


def fit_predict(candidate, x_train, y_train, x_val, x_test):
    family = candidate["family"]
    seed = int(candidate["seed"])
    if family == "logreg":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs")
        model.fit(x_train, y_train)
        return model.predict_proba(x_val)[:, 1], model.predict_proba(x_test)[:, 1]
    if family == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(int(candidate["hiddenSize"]),),
            alpha=float(candidate["alpha"]),
            random_state=seed,
            max_iter=500,
            early_stopping=False,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_val)[:, 1], model.predict_proba(x_test)[:, 1]
    if family == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=int(candidate.get("nEstimators", 400)),
            max_depth=int(candidate.get("maxDepth", 4)),
            min_samples_leaf=int(candidate.get("minSamplesLeaf", 10)),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        return model.predict_proba(x_val)[:, 1], model.predict_proba(x_test)[:, 1]
    if family == "focal_logreg":
        return fit_focal_predict(x_train, y_train, x_val, x_test, float(candidate.get("gamma", 1.0)), seed)
    raise RuntimeError(f"Unsupported tabular candidate family: {family}")


def fit_focal_predict(x_train, y_train, x_val, x_test, gamma: float, seed: int):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    x = torch.from_numpy(x_train).to(device)
    y = torch.from_numpy(y_train.astype(np.float32)).to(device)
    model = torch.nn.Linear(x_train.shape[1], 1).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    positives = max(float((y_train == 1).sum()), 1.0)
    negatives = max(float((y_train == 0).sum()), 1.0)
    pos_weight = len(y_train) / (2.0 * positives)
    neg_weight = len(y_train) / (2.0 * negatives)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for _epoch in range(250):
        opt.zero_grad()
        logits = model(x).squeeze(1).clamp(-30.0, 30.0)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt = torch.exp(-bce)
        weights = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.full_like(y, neg_weight))
        loss = (((1.0 - pt) ** gamma) * bce * weights).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("focal logistic produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(x_val).to(device)).squeeze(1).clamp(-30.0, 30.0)
        holdout_logits = model(torch.from_numpy(x_test).to(device)).squeeze(1).clamp(-30.0, 30.0)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        holdout_probs = torch.sigmoid(holdout_logits).cpu().numpy()
    return np.nan_to_num(val_probs, nan=0.5, posinf=1.0, neginf=0.0), np.nan_to_num(holdout_probs, nan=0.5, posinf=1.0, neginf=0.0)


def write_scores(path: Path, rows, include_labels: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["user_id", "label", "fold", "score", "model_id"] if include_labels else ["user_id", "score", "model_id"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in headers})


def db_tables_for_candidate(cfg, feature_blocks: list[str]) -> dict[str, str]:
    tables = cfg["database"]["tables"]
    keys: list[str] = []
    for block in feature_blocks:
        for key in TABLES_BY_BLOCK[block]:
            if key not in keys:
                keys.append(key)
    return {key: tables[key] for key in keys if key in tables}


def feature_cache_key(candidate, fold: int) -> tuple:
    return (tuple(candidate["featureBlocks"]), int(candidate.get("pcaComponents", 80)), fold)


def train_candidate(cfg, out_dir: Path, manifest_hash: str, train_all, split_test, candidate, feature_cache) -> None:
    model_id = candidate["modelId"]
    folds = sorted(set(int(fold) for fold in train_all["folds"]))
    oof_rows = []
    test_sum = np.zeros(len(split_test["user_ids"]), dtype=np.float64)
    test_count = 0
    for fold in folds:
        train_mask = train_all["folds"] != fold
        val_mask = train_all["folds"] == fold
        train = subset(train_all, train_mask)
        val = subset(train_all, val_mask)
        fold_candidate = {**candidate, "seed": int(candidate["seed"]) + fold}
        cache_key = feature_cache_key(fold_candidate, fold)
        if cache_key not in feature_cache:
            feature_cache[cache_key] = build_features(train, val, split_test, fold_candidate)
        x_train, x_val, x_test = feature_cache[cache_key]
        val_probs, holdout_probs = fit_predict(fold_candidate, x_train, train["labels"], x_val, x_test)
        for uid, label, prob in zip(val["user_ids"], val["labels"], val_probs):
            oof_rows.append(
                {
                    "user_id": uid,
                    "label": "diagnosed" if label == 1 else "control",
                    "fold": fold,
                    "score": f"{float(prob):.8f}",
                    "model_id": model_id,
                }
            )
        test_sum += holdout_probs
        test_count += 1

    oof_rows.sort(key=lambda row: row["user_id"])
    holdout_avg = test_sum / max(test_count, 1)
    test_rows = [
        {"user_id": uid, "score": f"{float(prob):.8f}", "model_id": model_id}
        for uid, prob in zip(split_test["user_ids"], holdout_avg)
    ]
    write_scores(out_dir / "scores" / f"train_oof_{model_id}.csv", oof_rows, True)
    write_scores(out_dir / "scores" / f"test_score_{model_id}.csv", test_rows, False)
    model_manifest = {
        "modelId": model_id,
        "candidate": True,
        "family": candidate["family"],
        "seed": candidate["seed"],
        "manifestHash": manifest_hash,
        "dbTables": db_tables_for_candidate(cfg, candidate["featureBlocks"]),
        "featureBlocks": candidate["featureBlocks"],
        "hyperparameters": {key: value for key, value in candidate.items() if key not in {"modelId", "family", "featureBlocks"}},
        "usesTestLabelsForTraining": False,
        "gpuUsed": False,
        "fedoraGpu": False,
        "createdAt": "1970-01-01T00:00:00.000Z",
    }
    manifest_path = out_dir / "model-manifests" / f"{model_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(model_manifest, indent=2) + "\n")
    print(f"wrote {model_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setembrobr.seed42.strict-blind.json")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    env = load_env()
    database_url = env.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    candidates = list(cfg.get("candidateModels", {}).get("tabular", []))
    if args.only:
        wanted = set(args.only)
        candidates = [candidate for candidate in candidates if candidate["modelId"] in wanted]
    if not candidates:
        raise SystemExit("No tabular candidates selected")

    manifest_rows, manifest_hash = read_manifest(cfg)
    conn = psycopg2.connect(database_url)
    try:
        required_blocks = {block for candidate in candidates for block in candidate["featureBlocks"]}
        train_all, split_test = load_all_features(conn, cfg, manifest_rows, required_blocks)
    finally:
        conn.close()

    out_dir = Path(cfg["outputDir"])
    feature_cache = {}
    for candidate in candidates:
        train_candidate(cfg, out_dir, manifest_hash, train_all, split_test, candidate, feature_cache)


if __name__ == "__main__":
    main()
