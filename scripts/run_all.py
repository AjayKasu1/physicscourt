#!/usr/bin/env python3
"""Phase dispatcher.

Only Phase 0 is implemented at this point; later phases will extend this script
instead of adding one-off orchestration commands.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"


def main() -> None:
    subprocess.run(
        [
            PYTHON,
            str(ROOT / "scripts" / "smoke_models.py"),
            "--config",
            str(ROOT / "config" / "models.yaml"),
            "--report",
            str(ROOT / "results" / "environment_report.json"),
            "--device",
            "auto",
            "--fp16",
        ],
        check=True,
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc

