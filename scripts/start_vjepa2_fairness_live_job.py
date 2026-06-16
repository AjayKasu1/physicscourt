#!/usr/bin/env python3
"""Start the resumable V-JEPA 2 live fairness sweep in the background."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
LOG_PATH = ROOT / "results" / "logs" / "vjepa2_fairness_live.log"
STATUS_PATH = ROOT / "results" / "vjepa2_fairness_live_job.json"


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = LOG_PATH.open("ab")
    command = [
        "/usr/bin/caffeinate",
        "-is",
        PYTHON,
        str(ROOT / "scripts" / "run_vjepa2_fairness_job.py"),
        "--device",
        "auto",
        "--fp16",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    payload = {
        "status": "started",
        "pid": process.pid,
        "command": " ".join(command),
        "heartbeat": str(ROOT / "results" / "vjepa2_fairness_live_report.json"),
        "log": str(LOG_PATH),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with STATUS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
