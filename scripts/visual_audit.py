#!/usr/bin/env python3
"""Render visual audit sheets from the actual MP4 clips and cached scores."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MPL_CONFIG_DIR = ROOT / "results" / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "results" / ".cache"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physicscourt.pipeline.clip_dataset import ClipRecord
from physicscourt.pipeline.score_utils import load_score_cache
from physicscourt.pipeline.video_io import read_frame_rgb


@dataclass(frozen=True)
class PairSelection:
    category: str
    label: str
    pair_id: str
    possible_clip_id: str
    impossible_clip_id: str
    detector_b_margin: float
    vjepa2_margin: float
    violation_frame: int


def load_manifest(path: Path) -> list[ClipRecord]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    return [ClipRecord(**record) for record in payload["records"]]


def load_report_rows(path: Path, detector: str) -> dict[str, dict[str, Any]]:
    report = json.loads(path.read_text())
    return {row["clip_id"]: row for row in report["detectors"][detector]["rows"]}


def pair_margins(
    records: list[ClipRecord],
    detector_b_rows: dict[str, dict[str, Any]],
    vjepa_rows: dict[str, dict[str, Any]],
) -> dict[str, list[PairSelection]]:
    by_pair: dict[str, dict[bool, ClipRecord]] = {}
    for record in records:
        if record.split != "test":
            continue
        by_pair.setdefault(record.pair_id, {})[record.possible] = record

    out: dict[str, list[PairSelection]] = {}
    for pair_id, pair in by_pair.items():
        if True not in pair or False not in pair:
            continue
        possible = pair[True]
        impossible = pair[False]
        if possible.clip_id not in detector_b_rows or impossible.clip_id not in detector_b_rows:
            continue
        b_margin = float(detector_b_rows[impossible.clip_id]["clip_level_score"]) - float(
            detector_b_rows[possible.clip_id]["clip_level_score"]
        )
        v_margin = float(vjepa_rows[impossible.clip_id]["clip_level_score"]) - float(
            vjepa_rows[possible.clip_id]["clip_level_score"]
        )
        selection = PairSelection(
            category=impossible.category,
            label="",
            pair_id=pair_id,
            possible_clip_id=possible.clip_id,
            impossible_clip_id=impossible.clip_id,
            detector_b_margin=b_margin,
            vjepa2_margin=v_margin,
            violation_frame=int(impossible.violation_frame or 0),
        )
        out.setdefault(impossible.category, []).append(selection)
    return out


def choose_pairs(by_category: dict[str, list[PairSelection]]) -> dict[str, list[PairSelection]]:
    selected: dict[str, list[PairSelection]] = {}
    for category, pairs in by_category.items():
        ordered = sorted(pairs, key=lambda item: item.detector_b_margin)
        miss = ordered[0]
        hit = ordered[-1]
        selected[category] = [
            PairSelection(**{**hit.__dict__, "label": "detector_b_hit"}),
            PairSelection(**{**miss.__dict__, "label": "detector_b_miss"}),
        ]
    return selected


def frame_indices(t_star: int, num_frames: int) -> list[int]:
    return sorted({0, max(0, t_star - 8), t_star, min(num_frames - 1, t_star + 8), num_frames - 1})


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int] = (20, 20, 20)) -> None:
    draw.text(xy, text, fill=fill, font=ImageFont.load_default())


def resize_frame(frame: np.ndarray, width: int) -> Image.Image:
    image = Image.fromarray(frame)
    height = int(round(image.height * (width / image.width)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def render_contact_sheet(
    *,
    category: str,
    selections: list[PairSelection],
    records_by_id: dict[str, ClipRecord],
    output_dir: Path,
    thumb_width: int,
) -> Path:
    t = selections[0].violation_frame
    indices = frame_indices(t, records_by_id[selections[0].impossible_clip_id].num_frames)
    label_w = 300
    top_h = 48
    row_gap = 10
    col_gap = 8
    thumb_h = int(round(thumb_width * records_by_id[selections[0].impossible_clip_id].height / records_by_id[selections[0].impossible_clip_id].width))
    rows = len(selections) * 2
    sheet_w = label_w + len(indices) * thumb_width + (len(indices) - 1) * col_gap
    sheet_h = top_h + rows * thumb_h + (rows - 1) * row_gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw_label(draw, (8, 8), f"{category}: actual MP4 frames; dotted column is t*={t}")
    for col, frame_idx in enumerate(indices):
        x = label_w + col * (thumb_width + col_gap)
        draw_label(draw, (x + 4, 28), f"f{frame_idx}")

    row_idx = 0
    for selection in selections:
        for possible, clip_id in [(True, selection.possible_clip_id), (False, selection.impossible_clip_id)]:
            record = records_by_id[clip_id]
            y = top_h + row_idx * (thumb_h + row_gap)
            label = "possible" if possible else "impossible"
            draw_label(
                draw,
                (8, y + 6),
                f"{selection.label}\n{label}\n{clip_id}\nB margin={selection.detector_b_margin:.2f}\nV-JEPA 2 margin={selection.vjepa2_margin:.2f}",
            )
            for col, frame_idx in enumerate(indices):
                frame = read_frame_rgb(Path(record.video_path), frame_idx)
                thumb = resize_frame(frame, thumb_width)
                x = label_w + col * (thumb_width + col_gap)
                sheet.paste(thumb, (x, y))
                if frame_idx == t:
                    frame_draw = ImageDraw.Draw(sheet)
                    frame_draw.rectangle([x, y, x + thumb_width - 1, y + thumb_h - 1], outline=(220, 38, 38), width=4)
            row_idx += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{category}_contact_sheet.png"
    sheet.save(path)
    return path


def score_array(features_dir: Path, detector: str, clip_id: str) -> np.ndarray:
    cached = load_score_cache(features_dir / detector / f"{clip_id}.npz")
    return np.asarray(cached["smoothed_score"], dtype=np.float32)


def render_score_overlay(
    *,
    category: str,
    selections: list[PairSelection],
    output_dir: Path,
    features_dir: Path,
) -> Path:
    detectors = [("vjepa2", "V-JEPA 2"), ("dino_latent", "DINO"), ("li_state_rules", "SSRD")]
    fig, axes = plt.subplots(len(detectors), len(selections), figsize=(11, 7), sharex=True)
    if len(selections) == 1:
        axes = np.asarray(axes).reshape(len(detectors), 1)
    for col, selection in enumerate(selections):
        for row, (detector, label) in enumerate(detectors):
            ax = axes[row, col]
            possible = score_array(features_dir, detector, selection.possible_clip_id)
            impossible = score_array(features_dir, detector, selection.impossible_clip_id)
            ax.plot(possible, color="#2563eb", linewidth=1.8, label="possible")
            ax.plot(impossible, color="#dc2626", linewidth=1.8, label="impossible")
            ax.axvline(selection.violation_frame, color="black", linestyle=":", linewidth=1.8)
            ax.grid(True, alpha=0.22)
            ax.set_title(f"{label} / {selection.label}" if row == 0 else label)
            if col == 0:
                ax.set_ylabel("z score")
            if row == len(detectors) - 1:
                ax.set_xlabel("frame")
            if row == 0 and col == len(selections) - 1:
                ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"{category}: possible vs impossible smoothed scores")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{category}_score_overlay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "synthetic_manifest.yaml")
    parser.add_argument("--detector-a-report", type=Path, default=ROOT / "results" / "detector_a_report.json")
    parser.add_argument("--detector-b-report", type=Path, default=ROOT / "results" / "detector_b_report.json")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "results" / "features")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures" / "visual_audit")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "visual_audit_report.json")
    parser.add_argument("--thumb-width", type=int, default=160)
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    records_by_id = {record.clip_id: record for record in records}
    detector_b_rows = load_report_rows(args.detector_b_report, "li_state_rules")
    vjepa_rows = load_report_rows(args.detector_a_report, "vjepa2")
    selected = choose_pairs(pair_margins(records, detector_b_rows, vjepa_rows))
    audit_rows: list[dict[str, Any]] = []

    for category, selections in sorted(selected.items()):
        contact_path = render_contact_sheet(
            category=category,
            selections=selections,
            records_by_id=records_by_id,
            output_dir=args.output_dir,
            thumb_width=args.thumb_width,
        )
        score_path = render_score_overlay(
            category=category,
            selections=selections,
            output_dir=args.output_dir,
            features_dir=args.features_dir,
        )
        audit_rows.append(
            {
                "category": category,
                "contact_sheet": str(contact_path),
                "score_overlay": str(score_path),
                "selections": [selection.__dict__ for selection in selections],
            }
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        json.dump({"rows": audit_rows}, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.report}")
    for row in audit_rows:
        print(row["category"], row["contact_sheet"], row["score_overlay"])


if __name__ == "__main__":
    main()
