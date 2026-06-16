"""Command-line entry points for PhysicsCourt."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(prog="physicscourt")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("phase0", help="Run model smoke tests and write the environment report.")
    sub.add_parser("benchmark", help="Run the full benchmark once later phases are implemented.")
    sub.add_parser("report", help="Regenerate result tables and figures once evaluation exists.")

    analyze = sub.add_parser("analyze", help="Analyze one user video once detectors are implemented.")
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--prompt-box", default=None, help="x,y,w,h box for the target object.")

    args = parser.parse_args()
    root = _repo_root()

    if args.command == "phase0":
        subprocess.run(
            [
                "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
                str(root / "scripts" / "smoke_models.py"),
                "--config",
                str(root / "config" / "models.yaml"),
                "--report",
                str(root / "results" / "environment_report.json"),
                "--device",
                "auto",
                "--fp16",
            ],
            check=True,
        )
        return

    raise SystemExit(f"`physicscourt {args.command}` is scheduled for a later phase.")

