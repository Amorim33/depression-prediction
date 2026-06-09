#!/usr/bin/env python3
"""Train strict-blind sequence CNN OOF models.

Early stopping uses only train-fold validation users. Test labels are neither
loaded nor used by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Cnn1D(nn.Module):
    def __init__(self, input_dim: int, num_filters: int = 64, dropout: float = 0.4):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(input_dim, num_filters, k, padding=k // 2) for k in (3, 5, 7)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * 3, 2)

    def forward(self, x, mask):
        x = x.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = h.masked_fill(mask.unsqueeze(1).expand_as(h) < 0.5, float("-inf"))
            pooled.append(h.max(dim=2).values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


def read_manifest(cfg):
    path = Path(cfg["outputDir"]) / "manifest" / "split_manifest_seed42.csv"
    text = path.read_text()
    rows = list(csv.DictReader(text.splitlines()))
    return rows, hashlib.sha256(text.encode()).hexdigest()


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    return {
        "user_ids": data["user_ids"].astype(str),
        "labels": data["labels"].astype(np.int64),
        "sequences": data["sequences"].astype(np.float32),
        "lengths": data["lengths"].astype(np.int32),
    }


def masks(lengths, seq_len):
    out = np.zeros((len(lengths), seq_len), dtype=np.float32)
    for i, length in enumerate(lengths):
        out[i, : min(int(length), seq_len)] = 1.0
    return out


def infer(model, device, split, batch_size):
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
    best = 0.0
    for threshold in np.unique(np.concatenate([np.linspace(0, 1, 101), np.quantile(probs, np.linspace(0, 1, 41))])):
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


def train_fold(train_split, val_split, seed, epochs=40, batch_size=64):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Cnn1D(train_split["sequences"].shape[2]).to(device)
    seq = torch.from_numpy(train_split["sequences"])
    mask = torch.from_numpy(masks(train_split["lengths"], seq.shape[1]))
    labels = torch.from_numpy(train_split["labels"])
    loader = DataLoader(TensorDataset(seq, mask, labels), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    pos = max(int((labels == 1).sum()), 1)
    neg = max(int((labels == 0).sum()), 1)
    weights = torch.tensor([len(labels) / (2 * neg), len(labels) / (2 * pos)], dtype=torch.float32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    best_score = -1.0
    best_state = None
    stale = 0
    for _epoch in range(epochs):
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/setembrobr.seed42.strict-blind.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    manifest_rows, manifest_hash = read_manifest(cfg)
    fold_by_user = {r["user_id"]: int(r["fold"]) for r in manifest_rows if r["split"] == "train"}
    label_by_user = {r["user_id"]: r["label"] for r in manifest_rows if r["split"] == "train"}
    out_dir = Path(cfg["outputDir"])
    jobs = [
        ("seq_cnn_top128_s42", 128, 42),
        ("seq_cnn_top128_s13", 128, 13),
        ("seq_cnn_top256_s13", 256, 13),
    ]
    for model_id, top_n, seed in jobs:
        seq_dir = out_dir / "sequences" / f"top{top_n}"
        train_all = load_npz(seq_dir / "train_seq.npz")
        test_split = load_npz(seq_dir / "test_seq.npz")
        folds = np.array([fold_by_user[uid] for uid in train_all["user_ids"]], dtype=np.int32)
        train_all["labels"] = np.array([1 if label_by_user[uid] == "diagnosed" else 0 for uid in train_all["user_ids"]], dtype=np.int64)
        oof_rows = []
        test_sum = np.zeros(len(test_split["user_ids"]), dtype=np.float64)
        count = 0
        for fold in sorted(set(folds)):
            train_mask = folds != fold
            val_mask = folds == fold
            model, device = train_fold(subset(train_all, train_mask), subset(train_all, val_mask), seed + int(fold))
            val_split = subset(train_all, val_mask)
            val_probs = infer(model, device, val_split, 64)
            for uid, label, prob in zip(val_split["user_ids"], val_split["labels"], val_probs):
                oof_rows.append({
                    "user_id": uid,
                    "label": "diagnosed" if label == 1 else "control",
                    "fold": int(fold),
                    "score": f"{float(prob):.8f}",
                    "model_id": model_id,
                })
            test_sum += infer(model, device, test_split, 64)
            count += 1
        oof_rows.sort(key=lambda row: row["user_id"])
        test_rows = [
            {"user_id": uid, "score": f"{float(prob):.8f}", "model_id": model_id}
            for uid, prob in zip(test_split["user_ids"], test_sum / max(count, 1))
        ]
        write_scores(out_dir / "scores" / f"train_oof_{model_id}.csv", oof_rows, True)
        write_scores(out_dir / "scores" / f"test_score_{model_id}.csv", test_rows, False)
        model_manifest = {
            "modelId": model_id,
            "seed": seed,
            "manifestHash": manifest_hash,
            "dbTables": cfg["database"]["tables"],
            "usesTestLabelsForTraining": False,
            "sequenceTopN": top_n,
            "createdAt": "1970-01-01T00:00:00.000Z",
        }
        path = out_dir / "model-manifests" / f"{model_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model_manifest, indent=2) + "\n")
        print(f"wrote {model_id}")


if __name__ == "__main__":
    main()
