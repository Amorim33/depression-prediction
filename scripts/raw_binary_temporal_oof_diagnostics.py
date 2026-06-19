#!/usr/bin/env python3
"""Write descriptive OOF diagnostics for temporal raw-binary experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_features(output_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    data = np.load(output_dir / "features" / "train_raw_features.npz", allow_pickle=True)
    try:
        user_ids = data["user_ids"].astype(str)
        temporal = data["temporal_markers"].astype(np.float32)
        evidence = data["evidence_markers"].astype(np.float32)
    finally:
        data.close()
    manifest = json.loads((output_dir / "features" / "raw_feature_manifest.json").read_text())
    return user_ids, temporal, list(manifest["temporalColumns"]), evidence, list(manifest["evidenceColumns"])


def selected_scores(output_dir: Path, lock: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    total_weight = 0.0
    raw_weights = lock["weights"]
    weight_entries = raw_weights.items() if isinstance(raw_weights, dict) else ((entry["key"], entry["value"]) for entry in raw_weights)
    for model_id, raw_weight in weight_entries:
        weight = float(raw_weight)
        total_weight += weight
        for row in read_csv(output_dir / "scores" / f"train_oof_{model_id}.csv"):
            scores[row["user_id"]] = scores.get(row["user_id"], 0.0) + float(row["score"]) * weight
    if total_weight <= 0:
        raise RuntimeError("ensemble lock has no positive selected weight")
    return {user_id: score / total_weight for user_id, score in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_config(Path(args.config))
    output_dir = Path(cfg["outputDir"])
    lock = json.loads((output_dir / "ensemble" / "ensemble-lock.json").read_text())
    threshold = float(lock["threshold"])
    scores = selected_scores(output_dir, lock)
    user_ids, temporal, columns, evidence, evidence_columns = load_features(output_dir)
    col = {name: index for index, name in enumerate(columns)}
    evidence_col = {name: index for index, name in enumerate(evidence_columns)}

    labels = {row["user_id"]: row["label"] for row in read_csv(output_dir / "manifest" / f"train_binary_manifest_seed{cfg['seed']}.csv")}
    fp_historical_recent_quiet = 0
    fn_recent_indirect_low_score = 0
    fp_total = 0
    fn_total = 0
    examples: dict[str, list[dict[str, Any]]] = {"fp_historical_recent_quiet": [], "fn_recent_indirect_low_score": []}
    for row_index, user_id in enumerate(user_ids):
        score = scores[user_id]
        label = labels[user_id]
        pred = "diagnosed" if score > threshold else "control"
        values = temporal[row_index]
        evidence_values = evidence[row_index]
        lifetime_last_rel7 = float(values[col["last_rel7_normalized_position"]])
        lifetime_rel7_count = float(np.expm1(evidence_values[evidence_col["rel7_count_log"]]))
        recent50_rel7 = float(values[col["final50_rel7_ratio"]])
        recent50_indirect = float(values[col["final50_indirect_anxiety_ratio"]] + values[col["final50_indirect_sleep_ratio"]] + values[col["final50_indirect_fatigue_ratio"]] + values[col["final50_indirect_cry_sad_ratio"]])
        if label == "control" and pred == "diagnosed":
            fp_total += 1
            if lifetime_rel7_count >= 1 and lifetime_last_rel7 < 0.75 and recent50_rel7 < 0.02:
                fp_historical_recent_quiet += 1
                if len(examples["fp_historical_recent_quiet"]) < 10:
                    examples["fp_historical_recent_quiet"].append(
                        {
                            "user_id": user_id,
                            "score": score,
                            "lifetime_rel7_count": lifetime_rel7_count,
                            "last_rel7_normalized_position": lifetime_last_rel7,
                            "final50_rel7_ratio": recent50_rel7,
                        }
                    )
        elif label == "diagnosed" and pred == "control":
            fn_total += 1
            if recent50_indirect >= 0.08 and score < threshold:
                fn_recent_indirect_low_score += 1
                if len(examples["fn_recent_indirect_low_score"]) < 10:
                    examples["fn_recent_indirect_low_score"].append(
                        {"user_id": user_id, "score": score, "final50_indirect_signal_sum": recent50_indirect}
                    )

    report = {
        "dataset": "setembrobr",
        "seed": cfg["seed"],
        "config": args.config,
        "lockPath": str(output_dir / "ensemble" / "ensemble-lock.json"),
        "threshold": threshold,
        "descriptiveOnly": True,
        "selectionInput": False,
        "fpTotal": fp_total,
        "fnTotal": fn_total,
        "fpHistoricalHighRelevanceRecentQuiet": fp_historical_recent_quiet,
        "fnRecentIndirectToneLowScore": fn_recent_indirect_low_score,
        "examples": examples,
    }
    out_path = output_dir / "reports" / "raw_binary_temporal_oof_diagnostics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "report": str(out_path), "summary": report}))


if __name__ == "__main__":
    main()
