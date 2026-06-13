#!/usr/bin/env python3
"""Export top-N tweet embeddings from the existing Postgres database.

The test split export is label-free: test labels are not stored in the NPZ.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import psycopg2


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


def load_manifest(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def export_split(conn, cfg, manifest_rows, split: str, n_tweets: int, output_dir: Path) -> None:
    out_path = output_dir / f"{split}_seq.npz"
    if out_path.exists():
        print(f"exists {out_path}")
        return

    table = cfg["database"]["tables"][f"{split}Embeddings"]
    split_rows = [r for r in manifest_rows if r["split"] == split]
    user_ids = [r["user_id"] for r in split_rows]
    labels = np.array([1 if r["label"] == "diagnosed" else 0 for r in split_rows], dtype=np.int32)
    if split == "test":
        labels[:] = -1
    user_index = {uid: index for index, uid in enumerate(user_ids)}

    sequences = np.zeros((len(user_ids), n_tweets, cfg["database"]["embeddingDimension"]), dtype=np.float16)
    lengths = np.zeros(len(user_ids), dtype=np.int32)
    relevances = np.zeros((len(user_ids), n_tweets), dtype=np.int16)

    cur = conn.cursor(name=f"stream_{split}_{n_tweets}")
    cur.itersize = 2000
    cur.execute(
        f"""
        select user_id, emb_text, rel
        from (
          select user_id,
                 embedding::text as emb_text,
                 coalesce(gpt_3_5_relevance, 0) as rel,
                 row_number() over (
                   partition by user_id
                   order by coalesce(gpt_3_5_relevance, 0) desc, tweet_index desc
                 ) as rank
          from {table}
        ) ranked
        where rank <= %s
        order by user_id, rank
        """,
        (n_tweets,),
    )

    slots: dict[str, int] = {}
    for uid, raw_embedding, relevance in cur:
        if uid not in user_index:
            continue
        slot = slots.get(uid, 0)
        if slot >= n_tweets:
            continue
        row_index = user_index[uid]
        sequences[row_index, slot] = parse_vector(raw_embedding).astype(np.float16)
        relevances[row_index, slot] = int(relevance or 0)
        lengths[row_index] = slot + 1
        slots[uid] = slot + 1

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        user_ids=np.array(user_ids),
        labels=labels,
        sequences=sequences,
        lengths=lengths,
        relevances=relevances,
    )
    print(f"wrote {out_path}")


def derive_split_from_larger(seq_root: Path, split: str, n_tweets: int, output_dir: Path) -> bool:
    out_path = output_dir / f"{split}_seq.npz"
    if out_path.exists():
        print(f"exists {out_path}")
        return True

    sources = []
    for candidate_dir in seq_root.glob("top*"):
        if not candidate_dir.is_dir():
            continue
        try:
            source_n = int(candidate_dir.name.removeprefix("top"))
        except ValueError:
            continue
        source_path = candidate_dir / f"{split}_seq.npz"
        if source_n > n_tweets and source_path.exists():
            sources.append((source_n, source_path))
    if not sources:
        return False

    _source_n, source_path = sorted(sources)[0]
    data = np.load(source_path, allow_pickle=True)
    user_ids = data["user_ids"].astype(str)
    labels = data["labels"].astype(np.int32) if "labels" in data.files else np.full(len(user_ids), -1, dtype=np.int32)
    if split == "test":
        labels[:] = -1
    sequences = data["sequences"][:, :n_tweets, :].astype(data["sequences"].dtype, copy=True)
    lengths = np.minimum(data["lengths"].astype(np.int32), n_tweets)
    relevances = data["relevances"][:, :n_tweets].astype(data["relevances"].dtype, copy=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        user_ids=user_ids,
        labels=labels,
        sequences=sequences,
        lengths=lengths,
        relevances=relevances,
    )
    print(f"derived {out_path} from {source_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setembrobr.seed42.strict-blind.json")
    parser.add_argument("--n", type=int, nargs="*", default=[64, 128, 256])
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    env = load_env()
    database_url = env.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    manifest_path = Path(cfg["outputDir"]) / "manifest" / "split_manifest_seed42.csv"
    manifest_rows = load_manifest(manifest_path)
    seq_root = Path(cfg["outputDir"]) / "sequences"
    pending: list[tuple[int, str, Path]] = []
    for n_tweets in args.n:
        output_dir = seq_root / f"top{n_tweets}"
        for split in ("train", "test"):
            if not derive_split_from_larger(seq_root, split, n_tweets, output_dir):
                pending.append((n_tweets, split, output_dir))
    if not pending:
        return

    conn = psycopg2.connect(database_url)
    try:
        for n_tweets, split, output_dir in pending:
            export_split(conn, cfg, manifest_rows, split, n_tweets, output_dir)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
