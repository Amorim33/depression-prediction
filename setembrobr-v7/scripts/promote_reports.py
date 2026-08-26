#!/usr/bin/env python3
"""Copy only reviewable, label-free aggregate artifacts into the repository."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ALLOWLIST = {
    "reports/source-audit.json",
    "reports/embedding-pool-audit.json",
    "reports/oof-audit.json",
    "reports/label-free-test-score-manifest.json",
    "reports/final-test-report.json",
    "reports/derived-corpus-manifest.json",
    "reports/final-audit.json",
    "manifests/prepared-manifest.json",
    "manifests/oof-model-manifest.json",
    "manifests/full-fit-model-manifest.json",
    "ensemble/model-lock.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args()

    final_audit_path = args.output_dir / "reports/final-audit.json"
    final_audit = json.loads(final_audit_path.read_text())
    if not final_audit.get("ok"):
        raise RuntimeError("final audit must pass before reports can be promoted")

    for relative in sorted(ALLOWLIST):
        source = args.output_dir / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = args.artifacts_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


if __name__ == "__main__":
    main()
