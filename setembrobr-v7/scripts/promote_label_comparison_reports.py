#!/usr/bin/env python3
"""Promote only aggregate label-comparison reports and locks."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPORTS = [
    "prepare-report.json",
    "oof-audit.json",
    "oof-report.json",
    "test-score-audit.json",
    "final-comparison-report.json",
    "final-audit.json",
]
LOCKS = ["original_diagnosis.json", "specialist_signal.json"]


def copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise RuntimeError(f"required comparison artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in REPORTS:
        copy_required(args.output_dir / "reports" / name, args.artifacts_dir / "reports" / name)
    for name in LOCKS:
        copy_required(args.output_dir / "locks" / name, args.artifacts_dir / "locks" / name)


if __name__ == "__main__":
    main()
