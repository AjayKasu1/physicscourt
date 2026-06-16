#!/usr/bin/env python3
"""Run Detector A cache passes sequentially without evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
STATUS_PATH = ROOT / "results" / "detector_a_cache_job.json"


def write_status(payload: dict[str, object]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    with STATUS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def run_step(detector: str, timing_report: str) -> int:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "run_detector_a.py"),
        "--detector",
        detector,
        "--timing-report",
        str(ROOT / timing_report),
        "--device",
        "auto",
        "--fp16",
        "--continue-on-error",
    ]
    write_status({"status": "running", "phase": detector, "command": " ".join(cmd)})
    completed = subprocess.run(cmd, cwd=ROOT, env=env)
    return int(completed.returncode)


def main() -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    write_status({"status": "running", "phase": "vjepa2", "started_at": started_at})
    vjepa_code = run_step("vjepa2", "results/vjepa_timing.json")
    if vjepa_code != 0:
        write_status({"status": "failed", "phase": "vjepa2", "started_at": started_at, "returncode": vjepa_code})
        raise SystemExit(vjepa_code)

    dino_code = run_step("dino_latent", "results/dino_timing.json")
    status = "complete" if dino_code == 0 else "failed"
    write_status(
        {
            "status": status,
            "phase": "done" if dino_code == 0 else "dino_latent",
            "started_at": started_at,
            "returncodes": {"vjepa2": vjepa_code, "dino_latent": dino_code},
        }
    )
    raise SystemExit(dino_code)


if __name__ == "__main__":
    main()

