#!/usr/bin/env python3
"""Remove repo-owned generated artifacts.

The global Hugging Face cache is intentionally left alone; it can be shared by
other projects and should not be deleted by a repo-local make target.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
console = Console()


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-gb", type=float, default=25.0)
    args = parser.parse_args()

    targets = [
        ROOT / ".cache",
        ROOT / "results" / "features",
        ROOT / "results" / "overlays",
        ROOT / "results" / "tmp",
        ROOT / "data" / "synthetic_generator" / "generated",
    ]

    before = sum(dir_size(path) for path in targets)
    for path in targets:
        if path.exists():
            shutil.rmtree(path)
            console.print(f"removed {path}")
    after = sum(dir_size(path) for path in targets)
    console.print(
        {
            "repo_cache_limit_gb": args.limit_gb,
            "removed_bytes": before - after,
            "remaining_bytes": after,
        }
    )


if __name__ == "__main__":
    main()

