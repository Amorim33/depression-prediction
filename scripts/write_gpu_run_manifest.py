#!/usr/bin/env python3
"""Write a small manifest for Fedora GPU sequence-training runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def summarize_input(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": relative(path), "exists": False, "fileCount": 0, "totalBytes": 0}
    if path.is_file():
        return {
            "path": relative(path),
            "exists": True,
            "fileCount": 1,
            "totalBytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return {
        "path": relative(path),
        "exists": True,
        "fileCount": len(files),
        "totalBytes": sum(item.stat().st_size for item in files),
    }


def output_hashes(paths: list[Path]) -> list[dict[str, str]]:
    hashes: list[dict[str, str]] = []
    for path in paths:
        if path.is_file():
            hashes.append({"path": relative(path), "sha256": sha256_file(path)})
            continue
        if not path.exists():
            continue
        for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
            hashes.append({"path": relative(file_path), "sha256": sha256_file(file_path)})
    return hashes


def cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False, "deviceName": "cpu"}
    try:
        import torch

        info["torchVersion"] = torch.__version__
        info["cudaVersion"] = torch.version.cuda
        info["available"] = bool(torch.cuda.is_available())
        if info["available"]:
            info["deviceName"] = torch.cuda.get_device_name(0)
            info["deviceCount"] = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - depends on remote runtime
        info["error"] = str(exc)
    return info


def nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return {"available": False, "command": " ".join(command)}
    return {
        "available": result.returncode == 0,
        "command": " ".join(command),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ternary-classification/configs/setembrobr.seed42.ternary-strict-blind.json")
    parser.add_argument("--host-label", default="fedora")
    parser.add_argument("--command", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    output_dir = Path(config["outputDir"])
    source_output_dir = Path(config["sourceOutputDir"])
    output_path = Path(args.output) if args.output else output_dir / "gpu-runs" / "fedora-ternary-seq-oof.json"
    command = args.command or f"python scripts/ternary_train_seq_oof_setembrobr.py --config {args.config}"
    synced_inputs = [
        config_path,
        source_output_dir / "sequences",
        output_dir / "manifest",
        output_dir / "evidence-markers",
        output_dir / "label-policies",
    ]
    hashed_outputs = [
        output_dir / "scores",
        output_dir / "model-manifests",
    ]

    manifest = {
        "dataset": config["dataset"],
        "seed": config["seed"],
        "runKind": "fedora-ternary-sequence-oof",
        "hostLabel": args.host_label,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "command": command,
        "configPath": relative(config_path),
        "outputDir": config["outputDir"],
        "scoreSchema": "ternary-probability-v1",
        "usesTestLabelsForTraining": False,
        "cuda": cuda_info(),
        "nvidiaSmi": nvidia_smi(),
        "syncedInputs": [summarize_input(path) for path in synced_inputs],
        "outputHashes": output_hashes(hashed_outputs),
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
