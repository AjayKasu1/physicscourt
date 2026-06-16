#!/usr/bin/env python3
"""Start the overnight Detector A cache job in the background."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
LOG_PATH = ROOT / "results" / "detector_a_cache_job.log"
STATUS_PATH = ROOT / "results" / "detector_a_cache_job.json"


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = LOG_PATH.open("ab")
    process = subprocess.Popen(
        [PYTHON, str(ROOT / "scripts" / "cache_detector_a.py")],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    payload = {
        "status": "started",
        "pid": process.pid,
        "command": f"{PYTHON} scripts/cache_detector_a.py",
        "log": str(LOG_PATH),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with STATUS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

