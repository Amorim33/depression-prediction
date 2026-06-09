#!/usr/bin/env python3
"""Train strict-blind tabular OOF models from existing Postgres embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import psycopg2
from sklearn.decomposition import PCA
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

FP_REGEX = r"\y(eu|meu|minha|mim|comigo|me)\y"
NEG_REGEX = r"\y(triste|vazi[oa]|sozinh[oa]|solit[aá]ri[oa]|cansad[oa]|exaust[oa]|inutil|fracass|derrotad[oa]|desespero|sofrendo|doendo|machuca|arrepend|odei[oa]|horriv[eí]l|pior|terriv[eí]l|dific[ií]l|impossiv[eí]l|raiva|irritad[oa])\y"
ACTIVE_IDEATION_REGEX = r"(quero morrer|vou me matar|vontade de morrer|pensei em suic[ií]d|tentei suic[ií]d|queria me matar|acabar com minha vida|n[aã]o quero mais viver|quero sumir pra sempre)"
CAREGIVER_REGEX = r"(meu (pai|irm|primo|namorad|marido|amigo)|minha (m[aã]e|irm|prima|namorad|amiga|esposa|filh))[^.!?\n]{0,80}(depress|ansiedade|terapia|psic[oó]log|psiquiatr|rem[eé]dio|antidepressiv|suicid)"


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


def parse_vector(raw: str) -> np.ndarray:
    return np.fromstring(raw.strip("[]"), sep=",", dtype=np.float32)


def read_manifest(cfg):
    path = Path(cfg["outputDir"]) / "manifest" / "split_manifest_seed42.csv"
    text = path.read_text()
    manifest_hash = hashlib.sha256(text.encode()).hexdigest()
    rows = list(csv.DictReader(text.splitlines()))
    return rows, manifest_hash


def load_stylistic(conn, emb_table: str) -> dict[str, list[float]]:
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
                 count(*) filter (where tweet_text ~* %s) as caregiver_count
          from {emb_table}
          group by user_id
        )
        select * from agg
        """,
        (FP_REGEX, NEG_REGEX, FP_REGEX, NEG_REGEX, ACTIVE_IDEATION_REGEX, CAREGIVER_REGEX),
    )
    out = {}
    for uid, total, fp, neg, selfneg, active, caregiver in cur.fetchall():
        denom = max(float(total), 1.0)
        out[uid] = [
            float(fp) / denom,
            float(neg) / denom,
            float(selfneg) / denom,
            float(active) / denom,
            float(caregiver) / denom,
            float(np.log1p(active)),
        ]
    cur.close()
    return out


def load_split(conn, cfg, manifest_rows, split: str):
    tables = cfg["database"]["tables"]
    user_emb = tables[f"{split}UserEmb"]
    rel3_emb = tables[f"{split}UserEmbRel3"]
    sub_features = tables[f"{split}SubFeatures"]
    emb_table = tables[f"{split}Embeddings"]
    labels_by_user = {r["user_id"]: (1 if r["label"] == "diagnosed" else 0) for r in manifest_rows if r["split"] == split}
    folds_by_user = {r["user_id"]: int(r["fold"]) for r in manifest_rows if r["split"] == "train"}
    stylistic = load_stylistic(conn, emb_table)

    cur = conn.cursor()
    cur.execute(f"select user_id, embedding::text from {user_emb}")
    mean = {uid: parse_vector(raw) for uid, raw in cur.fetchall()}
    cur.execute(f"select user_id, embedding::text from {rel3_emb}")
    rel3 = {uid: parse_vector(raw) for uid, raw in cur.fetchall()}
    cur.execute(f"select user_id, {', '.join(V1_COLS)}, total_tweets from {sub_features}")
    sub = {row[0]: list(map(float, row[1:9])) + [float(np.log1p(row[9]))] for row in cur.fetchall()}
    cur.close()

    user_ids = sorted(uid for uid in labels_by_user if uid in mean and uid in rel3 and uid in sub and uid in stylistic)
    return {
        "user_ids": np.array(user_ids),
        "labels": np.array([labels_by_user[uid] for uid in user_ids], dtype=np.int32),
        "folds": np.array([folds_by_user.get(uid, -1) for uid in user_ids], dtype=np.int32),
        "v1": np.array([sub[uid] for uid in user_ids], dtype=np.float32),
        "sty": np.array([stylistic[uid] for uid in user_ids], dtype=np.float32),
        "mean": np.stack([mean[uid] for uid in user_ids]).astype(np.float32),
        "rel3": np.stack([rel3[uid] for uid in user_ids]).astype(np.float32),
    }


def build_features(train, val, test, model_id: str, seed: int):
    use_rel3 = model_id != "baseline_logreg"
    mean_pca = PCA(n_components=min(100, len(train["labels"]) - 1, train["mean"].shape[1]), random_state=seed)
    train_mean = mean_pca.fit_transform(train["mean"])
    val_mean = mean_pca.transform(val["mean"])
    test_mean = mean_pca.transform(test["mean"])

    pieces_train = [train["v1"], train_mean]
    pieces_val = [val["v1"], val_mean]
    pieces_test = [test["v1"], test_mean]
    if use_rel3:
        rel3_pca = PCA(n_components=min(100, len(train["labels"]) - 1, train["rel3"].shape[1]), random_state=seed + 101)
        pieces_train = [train["v1"], train["sty"], train_mean, rel3_pca.fit_transform(train["rel3"])]
        pieces_val = [val["v1"], val["sty"], val_mean, rel3_pca.transform(val["rel3"])]
        pieces_test = [test["v1"], test["sty"], test_mean, rel3_pca.transform(test["rel3"])]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.hstack(pieces_train))
    return x_train, scaler.transform(np.hstack(pieces_val)), scaler.transform(np.hstack(pieces_test))


def fit_predict(model_id: str, x_train, y_train, x_val, x_test, seed: int):
    if model_id == "mlp_128_alpha_001":
        model = MLPClassifier(hidden_layer_sizes=(128,), alpha=0.01, random_state=seed, max_iter=400, early_stopping=False)
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs")
    model.fit(x_train, y_train)
    return model.predict_proba(x_val)[:, 1], model.predict_proba(x_test)[:, 1]


def subset(data, mask):
    return {key: value[mask] for key, value in data.items()}


def write_scores(path: Path, rows, include_labels: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["user_id", "label", "fold", "score", "model_id"] if include_labels else ["user_id", "score", "model_id"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in headers})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setembrobr.seed42.strict-blind.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    env = load_env()
    database_url = env.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    manifest_rows, manifest_hash = read_manifest(cfg)
    conn = psycopg2.connect(database_url)
    try:
        train_all = load_split(conn, cfg, manifest_rows, "train")
        test = load_split(conn, cfg, manifest_rows, "test")
    finally:
        conn.close()

    out_dir = Path(cfg["outputDir"])
    tabular_models = ["baseline_logreg", "combined_all_logreg", "combined_all_focal_g1", "mlp_128_alpha_001"]
    for model_id in tabular_models:
        oof_rows = []
        test_sum = np.zeros(len(test["user_ids"]), dtype=np.float64)
        test_count = 0
        for fold in sorted(set(train_all["folds"])):
            train_mask = train_all["folds"] != fold
            val_mask = train_all["folds"] == fold
            train = subset(train_all, train_mask)
            val = subset(train_all, val_mask)
            x_train, x_val, x_test = build_features(train, val, test, model_id, cfg["seed"] + int(fold))
            val_probs, test_probs = fit_predict(model_id, x_train, train["labels"], x_val, x_test, cfg["seed"] + int(fold))
            for uid, label, prob in zip(val["user_ids"], val["labels"], val_probs):
                oof_rows.append({
                    "user_id": uid,
                    "label": "diagnosed" if label == 1 else "control",
                    "fold": int(fold),
                    "score": f"{float(prob):.8f}",
                    "model_id": model_id,
                })
            test_sum += test_probs
            test_count += 1

        oof_rows.sort(key=lambda row: row["user_id"])
        test_probs = test_sum / max(test_count, 1)
        test_rows = [
            {"user_id": uid, "score": f"{float(prob):.8f}", "model_id": model_id}
            for uid, prob in zip(test["user_ids"], test_probs)
        ]
        write_scores(out_dir / "scores" / f"train_oof_{model_id}.csv", oof_rows, True)
        write_scores(out_dir / "scores" / f"test_score_{model_id}.csv", test_rows, False)
        manifest = {
            "modelId": model_id,
            "seed": cfg["seed"],
            "manifestHash": manifest_hash,
            "dbTables": cfg["database"]["tables"],
            "usesTestLabelsForTraining": False,
            "createdAt": "1970-01-01T00:00:00.000Z",
        }
        manifest_path = out_dir / "model-manifests" / f"{model_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {model_id}")


if __name__ == "__main__":
    main()

