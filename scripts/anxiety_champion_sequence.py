#!/usr/bin/env python3
"""Train anxiety CNN OOF checkpoints or score test sequences after the OOF lock."""

from __future__ import annotations

import argparse
import csv
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from anxiety_champion_relevance import load_config, require_post_lock, resolve, sha256_file, write_json


class Cnn1D(nn.Module):
    def __init__(self, input_dim: int, num_filters: int, dropout: float):
        super().__init__()
        kernels = (3, 5, 7)
        self.convs = nn.ModuleList(
            [nn.Conv1d(input_dim, num_filters, kernel, padding=kernel // 2) for kernel in kernels]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernels), 2)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            hidden = F.relu(conv(values))
            hidden = hidden.masked_fill(mask.unsqueeze(1).expand_as(hidden) < 0.5, float("-inf"))
            pooled.append(hidden.max(dim=2).values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text().splitlines()))


def load_sequences(path: Path, use_relevance_channel: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        sequences = data["sequences"].astype(np.float16, copy=False)
        result = {
            "user_ids": data["user_ids"].astype(str),
            "sequences": sequences,
            "lengths": data["lengths"].astype(np.int32),
        }
        if use_relevance_channel:
            result["relevances"] = np.clip(
                data["relevances"].astype(np.float16) / np.float16(10.0), 0.0, 1.0
            )
        return result


def reorder(split: dict[str, np.ndarray], user_ids: list[str]) -> dict[str, np.ndarray]:
    if split["user_ids"].tolist() != user_ids:
        raise RuntimeError("sequence artifact order does not match the immutable split manifest")
    return split


def masks(lengths: np.ndarray, sequence_length: int) -> np.ndarray:
    positions = np.arange(sequence_length)[None, :]
    return (positions < lengths[:, None]).astype(np.float32)


def build_model(candidate: dict[str, Any], input_dim: int) -> Cnn1D:
    if candidate["family"] != "cnn":
        raise RuntimeError(f"unsupported anxiety sequence family {candidate['family']}")
    return Cnn1D(input_dim, int(candidate.get("numFilters", 64)), float(candidate.get("dropout", 0.4)))


def indexed_values(split: dict[str, np.ndarray], indexes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = torch.from_numpy(split["sequences"])[indexes]
    if "relevances" in split:
        relevance = torch.from_numpy(split["relevances"])[indexes].unsqueeze(2)
        sequences = torch.cat([sequences, relevance], dim=2)
    lengths = split["lengths"][indexes.numpy()]
    mask = torch.from_numpy(masks(lengths, sequences.shape[1]))
    return sequences, mask


def infer(
    model: nn.Module,
    device: torch.device,
    split: dict[str, np.ndarray],
    batch_size: int,
    indexes: np.ndarray | None = None,
) -> np.ndarray:
    selected = np.arange(len(split["user_ids"]), dtype=np.int64) if indexes is None else indexes
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch_indexes = torch.from_numpy(selected[start : start + batch_size])
            values, mask = indexed_values(split, batch_indexes)
            logits = model(
                values.to(device).float(),
                mask.to(device),
            )
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.clip(np.nan_to_num(np.concatenate(probabilities), nan=0.5, posinf=1.0, neginf=0.0), 0, 1)


def train_fold(
    training: dict[str, np.ndarray], train_indexes: np.ndarray, candidate: dict[str, Any], seed: int
) -> tuple[nn.Module, torch.device]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(candidate.get("batchSize", 64))
    input_dim = training["sequences"].shape[2] + (1 if "relevances" in training else 0)
    model = build_model(candidate, input_dim).to(device)
    labels = torch.from_numpy(training["labels"])
    index_tensor = torch.from_numpy(train_indexes)
    loader = DataLoader(
        TensorDataset(index_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    selected_labels = labels[index_tensor]
    positives = max(int((selected_labels == 1).sum()), 1)
    negatives = max(int((selected_labels == 0).sum()), 1)
    weights = torch.tensor(
        [len(selected_labels) / (2 * negatives), len(selected_labels) / (2 * positives)], dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for _epoch in range(int(candidate.get("epochs", 25))):
        model.train()
        for (batch_indexes,) in loader:
            values, value_mask = indexed_values(training, batch_indexes)
            target = labels[batch_indexes]
            optimizer.zero_grad()
            loss = loss_fn(model(values.to(device).float(), value_mask.to(device)), target.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("anxiety CNN produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model, device


def checkpoint_payload(model: nn.Module, candidate: dict[str, Any], input_dim: int) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "inputDim": input_dim,
        "stateDict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def write_scores(path: Path, rows: list[dict[str, Any]], include_labels: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["user_id", "label", "fold", "score", "model_id"] if include_labels else ["user_id", "score", "model_id"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows({header: row[header] for header in headers} for row in rows)


def train_oof(config: dict[str, Any], repo: Path) -> None:
    output_dir = resolve(config["outputDir"], repo)
    work_dir = resolve(config.get("workDir", config["outputDir"]), repo)
    manifest_path = output_dir / "manifest" / f"train_binary_manifest_seed{config['seed']}.csv"
    manifest = read_csv(manifest_path)
    user_ids = [row["user_id"] for row in manifest]
    labels = np.asarray([1 if row["label"] == "diagnosed" else 0 for row in manifest], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in manifest], dtype=np.int32)
    for candidate in config["candidateModels"]["sequence"]:
        model_id = candidate["modelId"]
        train = reorder(
            load_sequences(
                work_dir / "sequences" / f"top{candidate['topN']}" / "train_seq.npz",
                bool(candidate.get("useRelevanceChannel", False)),
            ),
            user_ids,
        )
        train["labels"] = labels
        rows = []
        checkpoint_hashes = {}
        for fold in sorted(set(folds.tolist())):
            train_indexes = np.flatnonzero(folds != fold).astype(np.int64)
            validation_indexes = np.flatnonzero(folds == fold).astype(np.int64)
            fold_seed = int(candidate["seed"]) + fold
            model, device = train_fold(train, train_indexes, candidate, fold_seed)
            probabilities = infer(model, device, train, int(candidate.get("batchSize", 64)), validation_indexes)
            checkpoint = output_dir / "checkpoints" / "sequence" / model_id / f"fold-{fold}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            input_dim = train["sequences"].shape[2] + (1 if "relevances" in train else 0)
            torch.save(checkpoint_payload(model, candidate, input_dim), checkpoint)
            checkpoint_hashes[str(fold)] = sha256_file(checkpoint)
            for user_id, label, probability in zip(
                train["user_ids"][validation_indexes], train["labels"][validation_indexes], probabilities
            ):
                rows.append(
                    {
                        "user_id": user_id,
                        "label": "diagnosed" if int(label) == 1 else "control",
                        "fold": fold,
                        "score": f"{float(probability):.8f}",
                        "model_id": model_id,
                    }
                )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        write_scores(output_dir / "scores" / f"train_oof_{model_id}.csv", sorted(rows, key=lambda row: row["user_id"]), True)
        write_json(
            output_dir / "model-manifests" / f"{model_id}.json",
            {
                "modelId": model_id,
                "candidate": True,
                "supportOnly": False,
                "family": candidate["family"],
                "seed": candidate["seed"],
                "predictionTarget": "anxiety",
                "featureSource": "raw_artifacts",
                "manifestHash": sha256_file(output_dir / "manifest" / f"strict_blind_split_manifest_seed{config['seed']}.csv"),
                "trainManifestHash": sha256_file(manifest_path),
                "relevanceProxyKind": config["relevanceProxy"]["kind"],
                "relevanceProxyDefinitionHash": sha256_file(output_dir / "relevance-proxy" / "proxy-definition.json"),
                "checkpointHashes": checkpoint_hashes,
                "artifactHashes": {
                    "strictBlindManifestSha256": sha256_file(
                        output_dir / "manifest" / f"strict_blind_split_manifest_seed{config['seed']}.csv"
                    ),
                    "trainManifestSha256": sha256_file(manifest_path),
                    "trainSequenceManifestSha256": sha256_file(
                        work_dir / "sequences" / f"top{candidate['topN']}" / "train-sequence-manifest.json"
                    ),
                    "relevanceProxyDefinitionSha256": sha256_file(
                        output_dir / "relevance-proxy" / "proxy-definition.json"
                    ),
                },
                "featureBlocks": [f"sequence_top{candidate['topN']}", "relevance_channel"],
                "hyperparameters": {key: value for key, value in candidate.items() if key not in {"modelId", "family"}},
                "scoreSchema": "binary-score-v1",
                "usesTestLabelsForTraining": False,
                "usesTestScoresForTraining": False,
                "testArtifactsReadDuringOof": False,
                "fixedEpochs": True,
                "gpuUsed": torch.cuda.is_available(),
                "fedoraGpu": torch.cuda.is_available() and "fedora" in platform.node().lower(),
                "deviceName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "createdAt": "1970-01-01T00:00:00.000Z",
            },
        )
        print(f"wrote train OOF {model_id}")


def score_test(config: dict[str, Any], repo: Path) -> None:
    output_dir = resolve(config["outputDir"], repo)
    work_dir = resolve(config.get("workDir", config["outputDir"]), repo)
    require_post_lock(output_dir)
    test_manifest = read_csv(output_dir / "manifest" / f"test_inference_manifest_seed{config['seed']}.csv")
    user_ids = [row["user_id"] for row in test_manifest]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for candidate in config["candidateModels"]["sequence"]:
        model_id = candidate["modelId"]
        test = reorder(
            load_sequences(
                work_dir / "sequences" / f"top{candidate['topN']}" / "test_seq.npz",
                bool(candidate.get("useRelevanceChannel", False)),
            ),
            user_ids,
        )
        total = np.zeros(len(user_ids), dtype=np.float64)
        checkpoints = sorted((output_dir / "checkpoints" / "sequence" / model_id).glob("fold-*.pt"))
        if len(checkpoints) != int(config["foldCount"]):
            raise RuntimeError(f"{model_id}: expected {config['foldCount']} checkpoints, got {len(checkpoints)}")
        for checkpoint in checkpoints:
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
            model = build_model(payload["candidate"], int(payload["inputDim"])).to(device)
            model.load_state_dict(payload["stateDict"])
            total += infer(model, device, test, int(candidate.get("batchSize", 64)))
            del model
        rows = [
            {"user_id": user_id, "score": f"{float(probability):.8f}", "model_id": model_id}
            for user_id, probability in zip(user_ids, total / len(checkpoints))
        ]
        write_scores(output_dir / "scores" / f"test_score_{model_id}.csv", rows, False)
        print(f"wrote label-free test score {model_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setembrobr.seed42.anxiety-temporal-champion-qwen3-binary.json")
    parser.add_argument("--stage", choices=["oof", "score-test"], required=True)
    args = parser.parse_args()
    repo = Path.cwd()
    config = load_config(resolve(args.config, repo))
    train_oof(config, repo) if args.stage == "oof" else score_test(config, repo)


if __name__ == "__main__":
    main()
