#!/usr/bin/env python3
"""Promote only aggregate v7 champion reports and hash-only provenance."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ALLOWLIST = {
    "ensemble/ensemble-lock.json",
    "reports/train-feature-preparation.json",
    "reports/test-feature-preparation.json",
    "reports/train-feature-support-audit.json",
    "reports/test-feature-support-audit.json",
    "reports/oof-audit.json",
    "reports/lock-provenance.json",
    "reports/label-free-test-score-manifest.json",
    "reports/final-test-report.json",
    "reports/final-artifact-chain.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()
    chain = json.loads((args.output_dir / "reports/final-artifact-chain.json").read_text())
    if not chain.get("ok"):
        raise RuntimeError("passing final ensemble artifact chain is required")
    for relative in sorted(ALLOWLIST):
        source = args.output_dir / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = args.artifacts_dir / "champion" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


if __name__ == "__main__":
    main()
