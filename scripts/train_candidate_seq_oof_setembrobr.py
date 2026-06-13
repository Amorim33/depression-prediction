#!/usr/bin/env python3
"""Train strict-blind sequence candidate OOF models from exported NPZ files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Cnn1D(nn.Module):
    def __init__(self, input_dim: int, num_filters: int = 64, dropout: float = 0.4, wide: bool = False):
        super().__init__()
        kernels = (3, 5, 7, 9) if wide else (3, 5, 7)
        self.convs = nn.ModuleList([nn.Conv1d(input_dim, num_filters, kernel, padding=kernel // 2) for kernel in kernels])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernels), 2)

    def forward(self, x, mask):
        x = x.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = h.masked_fill(mask.unsqueeze(1).expand_as(h) < 0.5, float("-inf"))
            pooled.append(h.max(dim=2).values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class BiLstmClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 96, dropout: float = 0.35):
        super().__init__()
        projected = hidden_size * 2
        self.proj = nn.Linear(input_dim, projected)
        self.lstm = nn.LSTM(projected, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 4, 2)

    def forward(self, x, mask):
        x = F.relu(self.proj(x))
        h, _state = self.lstm(x)
        expanded = mask.unsqueeze(2)
        h = h * expanded
        denom = expanded.sum(dim=1).clamp_min(1.0)
        mean_pool = h.sum(dim=1) / denom
        max_pool = h.masked_fill(expanded < 0.5, float("-inf")).max(dim=1).values
        return self.fc(self.dropout(torch.cat([mean_pool, max_pool], dim=1)))


class TinyTransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_size: int = 192, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, mask):
        h = self.proj(x) + self.pos[:, : x.shape[1], :]
        h = self.encoder(h, src_key_padding_mask=mask < 0.5)
        expanded = mask.unsqueeze(2)
        pooled = (h * expanded).sum(dim=1) / expanded.sum(dim=1).clamp_min(1.0)
        return self.fc(self.dropout(pooled))


def read_manifest(cfg):
    path = Path(cfg["outputDir"]) / "manifest" / f"split_manifest_seed{cfg['seed']}.csv"
    text = path.read_text()
    rows = list(csv.DictReader(text.splitlines()))
    return rows, hashlib.sha256(text.encode()).hexdigest()


def load_npz(path: Path, include_labels: bool):
    data = np.load(path, allow_pickle=True)
    out = {
        "user_ids": data["user_ids"].astype(str),
        "sequences": data["sequences"].astype(np.float32),
        "lengths": data["lengths"].astype(np.int32),
    }
    if include_labels:
        out["labels"] = data["labels"].astype(np.int64)
    return out


def masks(lengths, seq_len):
    out = np.zeros((len(lengths), seq_len), dtype=np.float32)
    for index, length in enumerate(lengths):
        out[index, : min(int(length), seq_len)] = 1.0
    return out


def infer(model, device, split, batch_size: int):
    seq = torch.from_numpy(split["sequences"])
    mask = torch.from_numpy(masks(split["lengths"], seq.shape[1]))
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(seq), batch_size):
            logits = model(seq[start : start + batch_size].to(device), mask[start : start + batch_size].to(device))
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def macro_f1(probs, labels):
    candidates = np.unique(np.concatenate([np.linspace(0, 1, 101), np.quantile(probs, np.linspace(0, 1, 41))]))
    best = 0.0
    for threshold in candidates:
        pred = (probs > threshold).astype(np.int64)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        tn = np.sum((pred == 0) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        diag = 2 * precision * recall / max(precision + recall, 1e-12)
        cprecision = tn / max(tn + fn, 1)
        crecall = tn / max(tn + fp, 1)
        ctrl = 2 * cprecision * crecall / max(cprecision + crecall, 1e-12)
        best = max(best, float((diag + ctrl) / 2))
    return best


def build_model(candidate, input_dim: int, seq_len: int) -> nn.Module:
    family = candidate["family"]
    if family == "cnn":
        return Cnn1D(input_dim, int(candidate.get("numFilters", 64)), float(candidate.get("dropout", 0.4)), wide=False)
    if family == "cnn_wide":
        return Cnn1D(input_dim, int(candidate.get("numFilters", 96)), float(candidate.get("dropout", 0.45)), wide=True)
    if family == "bilstm":
        return BiLstmClassifier(input_dim, int(candidate.get("hiddenSize", 96)), float(candidate.get("dropout", 0.35)))
    if family == "tiny_transformer":
        return TinyTransformerClassifier(input_dim, seq_len, int(candidate.get("hiddenSize", 192)), float(candidate.get("dropout", 0.25)))
    raise RuntimeError(f"Unsupported sequence candidate family: {family}")


def train_fold(train_split, val_split, candidate, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(candidate.get("batchSize", 48))
    model = build_model(candidate, train_split["sequences"].shape[2], train_split["sequences"].shape[1]).to(device)
    seq = torch.from_numpy(train_split["sequences"])
    mask = torch.from_numpy(masks(train_split["lengths"], seq.shape[1]))
    labels = torch.from_numpy(train_split["labels"])
    loader = DataLoader(
        TensorDataset(seq, mask, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    pos = max(int((labels == 1).sum()), 1)
    neg = max(int((labels == 0).sum()), 1)
    weights = torch.tensor([len(labels) / (2 * neg), len(labels) / (2 * pos)], dtype=torch.float32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    best_score = -1.0
    best_state = None
    stale = 0
    for _epoch in range(int(candidate.get("epochs", 45))):
        model.train()
        for xb, mb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device), mb.to(device)), yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_probs = infer(model, device, val_split, batch_size)
        score = macro_f1(val_probs, val_split["labels"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 6:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return model, device


def subset(split, mask):
    return {key: value[mask] for key, value in split.items()}


def write_scores(path: Path, rows, include_labels: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["user_id", "label", "fold", "score", "model_id"] if include_labels else ["user_id", "score", "model_id"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in headers})


def validate_train_ids(train_ids, fold_by_user, label_by_user):
    missing_folds = [uid for uid in train_ids if uid not in fold_by_user]
    missing_labels = [uid for uid in train_ids if uid not in label_by_user]
    if missing_folds or missing_labels:
        raise RuntimeError(f"Sequence export train users do not match manifest: folds={len(missing_folds)} labels={len(missing_labels)}")


def train_candidate(cfg, out_dir: Path, manifest_hash: str, manifest_rows, candidate):
    model_id = candidate["modelId"]
    top_n = int(candidate["topN"])
    seq_dir = out_dir / "sequences" / f"top{top_n}"
    train_all = load_npz(seq_dir / "train_seq.npz", include_labels=True)
    split_test = load_npz(seq_dir / "test_seq.npz", include_labels=False)
    fold_by_user = {row["user_id"]: int(row["fold"]) for row in manifest_rows if row["split"] == "train"}
    label_by_user = {row["user_id"]: row["label"] for row in manifest_rows if row["split"] == "train"}
    validate_train_ids(train_all["user_ids"], fold_by_user, label_by_user)
    folds = np.array([fold_by_user[uid] for uid in train_all["user_ids"]], dtype=np.int32)
    train_all["labels"] = np.array([1 if label_by_user[uid] == "diagnosed" else 0 for uid in train_all["user_ids"]], dtype=np.int64)

    oof_rows = []
    holdout_sum = np.zeros(len(split_test["user_ids"]), dtype=np.float64)
    count = 0
    for fold in sorted(set(folds)):
        train_mask = folds != fold
        val_mask = folds == fold
        fold_seed = int(candidate["seed"]) + int(fold)
        model, device = train_fold(subset(train_all, train_mask), subset(train_all, val_mask), candidate, fold_seed)
        val_split = subset(train_all, val_mask)
        val_probs = infer(model, device, val_split, int(candidate.get("batchSize", 48)))
        for uid, label, prob in zip(val_split["user_ids"], val_split["labels"], val_probs):
            oof_rows.append(
                {
                    "user_id": uid,
                    "label": "diagnosed" if label == 1 else "control",
                    "fold": int(fold),
                    "score": f"{float(prob):.8f}",
                    "model_id": model_id,
                }
            )
        holdout_sum += infer(model, device, split_test, int(candidate.get("batchSize", 48)))
        count += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_rows.sort(key=lambda row: row["user_id"])
    holdout_avg = holdout_sum / max(count, 1)
    holdout_rows = [
        {"user_id": uid, "score": f"{float(prob):.8f}", "model_id": model_id}
        for uid, prob in zip(split_test["user_ids"], holdout_avg)
    ]
    write_scores(out_dir / "scores" / f"train_oof_{model_id}.csv", oof_rows, True)
    write_scores(out_dir / "scores" / f"test_score_{model_id}.csv", holdout_rows, False)
    tables = cfg["database"]["tables"]
    cuda_available = torch.cuda.is_available()
    model_manifest = {
        "modelId": model_id,
        "candidate": True,
        "family": candidate["family"],
        "seed": candidate["seed"],
        "manifestHash": manifest_hash,
        "dbTables": {
            "trainEmbeddings": tables["trainEmbeddings"],
            "testEmbeddings": tables["testEmbeddings"],
        },
        "featureBlocks": [f"sequence_top{top_n}"],
        "sequenceTopN": top_n,
        "hyperparameters": {key: value for key, value in candidate.items() if key not in {"modelId", "family"}},
        "usesTestLabelsForTraining": False,
        "gpuUsed": cuda_available,
        "fedoraGpu": cuda_available and "fedora" in platform.node().lower(),
        "deviceName": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        "createdAt": "1970-01-01T00:00:00.000Z",
    }
    path = out_dir / "model-manifests" / f"{model_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_manifest, indent=2) + "\n")
    print(f"wrote {model_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setembrobr.seed42.strict-blind.json")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    candidates = list(cfg.get("candidateModels", {}).get("sequence", []))
    if args.only:
        wanted = set(args.only)
        candidates = [candidate for candidate in candidates if candidate["modelId"] in wanted]
    if not candidates:
        raise SystemExit("No sequence candidates selected")

    manifest_rows, manifest_hash = read_manifest(cfg)
    out_dir = Path(cfg["outputDir"])
    for candidate in candidates:
        train_candidate(cfg, out_dir, manifest_hash, manifest_rows, candidate)


if __name__ == "__main__":
    main()
