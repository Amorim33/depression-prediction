#!/usr/bin/env python3
"""Build the fixed, label-free anxiety relevance proxy without changing raw embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

PROXY_KIND = "anxiety-lexical-v1"
PROXY_FORMULA = {
    "direct_anxiety": 3,
    "self_experience": 2,
    "acute_symptom": 2,
    "treatment_or_help": 1,
    "active_or_recent": 1,
    "direct_and_acute": 1,
    "third_party_or_outreach": -4,
    "minimum": 0,
    "maximum": 10,
}
PATTERN_TEXT = {
    "direct_anxiety": r"\b(ansiedade|ansios[oa]s?|panico|sindrome do panico|ataque de panico|ataque de ansiedade|crise de ansiedade)\b",
    "self_token": r"\b(eu|me|mim|comigo|estou|to|tenho|tive|sinto|senti|minha ansiedade)\b",
    "acute_symptom": r"\b(falta de ar|coracao acelerado|taquicardia|palpitac(?:ao|oes)|tremor(?:es)?|hiperventil(?:acao|ando)|aperto no peito|suor frio|nausea|enjoo|desmai(?:o|ei)|formigamento)\b",
    "treatment_or_help": r"\b(terapia|psicolog[oa]|psiquiatr[oa]|remedio|medicacao|ansiolitic[oa]|rivotril|clonazepam|sertralina|fluoxetina|preciso de ajuda|me ajuda|socorro|nao sei o que fazer)\b",
    "active_or_recent": r"\b(agora|hoje|neste momento|nesse momento|acabei de|comecou|bateu|estou|to|tive)\b",
    "third_party_or_outreach": r"\b(meu pai|minha mae|meu irmao|minha irma|meu amigo|minha amiga|ele|ela|eles|elas|voce|alguem|procure ajuda|busque ajuda|conte comigo|apoio psicologico|projeto)\b",
}
PATTERNS = {name: re.compile(value) for name, value in PATTERN_TEXT.items()}
SELF_EVIDENCE_PATTERN = re.compile(
    f"(?:{PATTERN_TEXT['direct_anxiety']})|(?:{PATTERN_TEXT['acute_symptom']})"
)


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def within_window(text: str, left: re.Pattern[str], right: re.Pattern[str], window: int = 80) -> bool:
    left_matches = list(left.finditer(text))
    right_matches = list(right.finditer(text))
    return any(abs(left_match.start() - right_match.start()) <= window for left_match in left_matches for right_match in right_matches)


def anxiety_relevance_score(value: str) -> int:
    text = normalize_text(value)
    direct = bool(PATTERNS["direct_anxiety"].search(text))
    acute = bool(PATTERNS["acute_symptom"].search(text))
    if not direct and not acute:
        return 0
    self_experience = within_window(
        text,
        PATTERNS["self_token"],
        SELF_EVIDENCE_PATTERN,
    )
    treatment_or_help = bool(PATTERNS["treatment_or_help"].search(text))
    active_or_recent = bool(PATTERNS["active_or_recent"].search(text))
    third_party_or_outreach = bool(PATTERNS["third_party_or_outreach"].search(text))
    score = (
        3 * int(direct)
        + 2 * int(self_experience)
        + 2 * int(acute)
        + int(treatment_or_help)
        + int(active_or_recent)
        + int(direct and acute)
        - 4 * int(third_party_or_outreach)
    )
    return min(max(score, 0), 10)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve(value: str, repo: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo / path).resolve()


def verify_raw_artifacts(config: dict[str, Any], raw_dir: Path) -> None:
    manifest_path = raw_dir / "reports" / "embedding_generation_manifest.json"
    split_path = raw_dir / "manifests" / f"raw_split_manifest_seed{config['seed']}.csv"
    if sha256_file(manifest_path) != config["rawEmbeddingManifestSha256"]:
        raise RuntimeError("raw anxiety embedding manifest hash mismatch")
    if sha256_file(split_path) != config["rawSplitManifestSha256"]:
        raise RuntimeError("raw anxiety split manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("embedding", {}).get("modelRevision") != config["rawEmbeddingModelRevision"]:
        raise RuntimeError("raw anxiety embedding model revision mismatch")


def require_post_lock(output_dir: Path) -> None:
    lock = output_dir / "ensemble" / "ensemble-lock.json"
    audit = output_dir / "reports" / "oof-audit.json"
    if not lock.exists() or not audit.exists():
        raise RuntimeError("test relevance requires an existing ensemble lock and OOF audit")
    if not json.loads(audit.read_text()).get("ok"):
        raise RuntimeError("test relevance refused because the OOF audit did not pass")


def alignment_hash(users: list[str], indexes: list[int]) -> str:
    digest = hashlib.sha256()
    for user_id, tweet_index in zip(users, indexes):
        digest.update(user_id.encode())
        digest.update(b"\0")
        digest.update(str(tweet_index).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def pool_shard(users: list[str], scores: np.ndarray, embeddings: np.ndarray, thresholds: list[int]) -> dict[str, np.ndarray]:
    ordered_users = list(dict.fromkeys(users))
    indexes_by_user: dict[str, list[int]] = {user_id: [] for user_id in ordered_users}
    for index, user_id in enumerate(users):
        indexes_by_user[user_id].append(index)
    payload: dict[str, np.ndarray] = {"user_ids": np.asarray(ordered_users, dtype=object)}
    dimension = embeddings.shape[1]
    for threshold in thresholds:
        vectors = []
        counts = []
        for user_id in ordered_users:
            row_indexes = np.asarray(indexes_by_user[user_id], dtype=np.int64)
            selected = row_indexes[scores[row_indexes] >= threshold]
            counts.append(int(selected.size))
            vectors.append(
                np.zeros(dimension, dtype=np.float32)
                if selected.size == 0
                else embeddings[selected].astype(np.float32).mean(axis=0)
            )
        payload[f"rel{threshold}_embeddings"] = np.stack(vectors).astype(np.float32)
        payload[f"rel{threshold}_counts"] = np.asarray(counts, dtype=np.int32)
    return payload


def build_split(config: dict[str, Any], repo: Path, split: str, force: bool) -> None:
    import pyarrow.parquet as pq

    raw_dir = resolve(config["rawArtifactsDir"], repo)
    output_dir = resolve(config["outputDir"], repo)
    work_dir = resolve(config.get("workDir", config["outputDir"]), repo)
    verify_raw_artifacts(config, raw_dir)
    if split == "test":
        require_post_lock(output_dir)

    thresholds = [int(value) for value in config["relevanceProxy"]["poolThresholds"]]
    source_paths = sorted((raw_dir / "tweet_embeddings" / split).glob("*.parquet"))
    if not source_paths:
        raise RuntimeError(f"no raw anxiety embedding shards for {split}")
    proxy_dir = resolve(config["relevanceProxy"]["artifactDir"], repo)
    sidecar_dir = proxy_dir / "sidecars" / split
    pool_shard_dir = work_dir / "relevance-proxy" / "pool-shards" / split
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    pool_shard_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    pool_paths = []

    for source_path in source_paths:
        stem = source_path.stem
        sidecar_path = sidecar_dir / f"{stem}.npz"
        shard_pool_path = pool_shard_dir / f"{stem}.npz"
        if force or not sidecar_path.exists() or not shard_pool_path.exists():
            table = pq.read_table(source_path, columns=["user_id", "tweet_index", "tweet_text", "embedding"])
            users = [str(value) for value in table.column("user_id").to_pylist()]
            indexes = [int(value) for value in table.column("tweet_index").to_pylist()]
            texts = ["" if value is None else str(value) for value in table.column("tweet_text").to_pylist()]
            scores = np.fromiter((anxiety_relevance_score(text) for text in texts), dtype=np.uint8, count=len(texts))
            embedding_column = table.column("embedding").combine_chunks()
            embeddings = embedding_column.values.to_numpy(zero_copy_only=False).reshape(len(users), -1).astype(np.float16)
            align_hash = alignment_hash(users, indexes)
            np.savez_compressed(sidecar_path, scores=scores, alignment_sha256=np.asarray(align_hash))
            np.savez_compressed(shard_pool_path, **pool_shard(users, scores, embeddings, thresholds))
        else:
            sidecar = np.load(sidecar_path, allow_pickle=False)
            align_hash = str(sidecar["alignment_sha256"])
            sidecar.close()
        with np.load(sidecar_path, allow_pickle=False) as sidecar:
            row_count = int(sidecar["scores"].shape[0])
        entries.append(
            {
                "source": source_path.name,
                "sourceSha256": sha256_file(source_path),
                "sidecar": str(sidecar_path.relative_to(proxy_dir)),
                "sidecarSha256": sha256_file(sidecar_path),
                "poolShardSha256": sha256_file(shard_pool_path),
                "alignmentSha256": align_hash,
                "rowCount": row_count,
            }
        )
        pool_paths.append(shard_pool_path)
        print(f"proxy {split} {source_path.name}")

    pool_files = [np.load(path, allow_pickle=True) for path in pool_paths]
    try:
        user_ids = np.concatenate([item["user_ids"] for item in pool_files])
        pooled_dir = work_dir / "relevance-proxy" / "pooled"
        pooled_dir.mkdir(parents=True, exist_ok=True)
        pooled_hashes = {}
        for threshold in thresholds:
            path = pooled_dir / f"{split}_user_rel{threshold}.npz"
            np.savez_compressed(
                path,
                user_ids=user_ids,
                embeddings=np.concatenate([item[f"rel{threshold}_embeddings"] for item in pool_files]).astype(np.float32),
                counts=np.concatenate([item[f"rel{threshold}_counts"] for item in pool_files]).astype(np.int32),
            )
            pooled_hashes[f"rel{threshold}"] = sha256_file(path)
    finally:
        for item in pool_files:
            item.close()

    definition = {
        "kind": PROXY_KIND,
        "predictionTarget": "anxiety",
        "formula": PROXY_FORMULA,
        "patterns": PATTERN_TEXT,
        "normalization": "NFKD casefold with combining marks removed",
        "usesTrainLabels": False,
        "usesTestLabels": False,
        "usesTestDistribution": False,
        "rawEmbeddingManifestSha256": config["rawEmbeddingManifestSha256"],
        "rawSplitManifestSha256": config["rawSplitManifestSha256"],
        "poolThresholds": thresholds,
    }
    definition_path = proxy_dir / "proxy-definition.json"
    if definition_path.exists() and json.loads(definition_path.read_text()) != definition:
        raise RuntimeError("the frozen anxiety relevance proxy definition changed")
    write_json(definition_path, definition)
    write_json(
        proxy_dir / f"{split}-proxy-manifest.json",
        {
            "kind": PROXY_KIND,
            "split": split,
            "definitionSha256": sha256_file(definition_path),
            "rawSplitManifestSha256": config["rawSplitManifestSha256"],
            "expectedUsers": config["expectedUsers"][split],
            "shards": entries,
            "pooledHashes": pooled_hashes,
        },
    )
    for threshold in thresholds:
        write_json(
            proxy_dir / f"{split}-rel{threshold}-pool-manifest.json",
            {
                "kind": PROXY_KIND,
                "split": split,
                "threshold": threshold,
                "pooling": "mean",
                "zeroVectorFallback": True,
                "pooledSha256": pooled_hashes[f"rel{threshold}"],
                "sourceProxyManifestSha256": sha256_file(proxy_dir / f"{split}-proxy-manifest.json"),
                "regenerableWorkArtifact": True,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setembrobr.seed42.anxiety-temporal-champion-qwen3-binary.json")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repo = Path.cwd()
    build_split(load_config(resolve(args.config, repo)), repo, args.split, args.force)


if __name__ == "__main__":
    main()
