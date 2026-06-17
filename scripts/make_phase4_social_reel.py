#!/usr/bin/env python3
"""Render a vertical social reel for Phase 4 edited-real results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
FALLBACK_FONT = Path("/System/Library/Fonts/SFNS.ttf")


@dataclass(frozen=True)
class DetectorSpec:
    key: str
    label: str
    status: str


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    title: str
    subtitle: str
    possible_path: Path
    impossible_path: Path
    possible_clip_id: str
    impossible_clip_id: str
    t_star: int
    bottom_line: str
    detectors: tuple[DetectorSpec, ...]


PAIRS = (
    PairSpec(
        pair_id="edited_real_object_permanence_000",
        title="Test 1: missing ball",
        subtitle="The ball should come back out. The edit removes it.",
        possible_path=ROOT / "data/edited_real/processed/object_permanence/pair000/possible.mp4",
        impossible_path=ROOT / "data/edited_real/processed/object_permanence/pair000/impossible.mp4",
        possible_clip_id="edited_real_object_permanence_000_possible",
        impossible_clip_id="edited_real_object_permanence_000_impossible",
        t_star=32,
        bottom_line="Result: SSRD catches the missing ball. V-JEPA 2 and DINOv2 miss it.",
        detectors=(
            DetectorSpec("vjepa2", "V-JEPA 2", "wrong"),
            DetectorSpec("dino_latent", "DINOv2", "wrong"),
            DetectorSpec("li_state_rules", "SSRD", "correct"),
        ),
    ),
    PairSpec(
        pair_id="edited_real_continuity_teleportation_000",
        title="Test 2: teleport jump",
        subtitle="The ball skips the middle path and reappears left.",
        possible_path=ROOT / "data/edited_real/processed/continuity_teleportation/pair000/possible.mp4",
        impossible_path=ROOT / "data/edited_real/processed/continuity_teleportation/pair000/impossible.mp4",
        possible_clip_id="edited_real_continuity_teleportation_000_possible",
        impossible_clip_id="edited_real_continuity_teleportation_000_impossible",
        t_star=16,
        bottom_line="Result: nobody catches this one cleanly. Teleport is still a miss.",
        detectors=(
            DetectorSpec("vjepa2", "V-JEPA 2", "wrong"),
            DetectorSpec("dino_latent", "DINOv2", "wrong"),
            DetectorSpec("li_state_rules", "SSRD", "tied"),
        ),
    ),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if not path.exists():
        path = FALLBACK_FONT
    return ImageFont.truetype(str(path), size)


def read_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {path}")
    return frames


def load_score(features_dir: Path, detector: str, clip_id: str) -> np.ndarray:
    path = features_dir / detector / f"{clip_id}.npz"
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["smoothed_score"], dtype=np.float32)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=22, fill=fill, outline=outline, width=2 if outline else 1)


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: str,
    text_font: ImageFont.FreeTypeFont,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=text_font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((xy[0] - w // 2, xy[1] - h // 2), text, fill=fill, font=text_font)


def paste_panel(
    canvas: Image.Image,
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    accent: str,
) -> None:
    x0, y0, x1, y1 = box
    panel = Image.fromarray(frame).resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    canvas.paste(panel, (x0, y0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box, radius=18, outline=accent, width=4)
    pill = (x0 + 16, y0 + 14, x0 + 190, y0 + 54)
    draw.rounded_rectangle(pill, radius=18, fill="#111827")
    draw.text((pill[0] + 18, pill[1] + 8), label, fill="white", font=font(22, True))


def line_points(
    values: np.ndarray,
    chart: tuple[int, int, int, int],
    ymin: float,
    ymax: float,
    *,
    start_index: int = 0,
    total_frames: int = 72,
) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = chart
    if values.size == 1:
        return [(x0, (y0 + y1) // 2)]
    denom = max(1e-6, ymax - ymin)
    pts = []
    for i, value in enumerate(values):
        frame_index = start_index + i
        x = int(x0 + (x1 - x0) * frame_index / max(1, total_frames - 1))
        y = int(y1 - (float(value) - ymin) / denom * (y1 - y0))
        pts.append((x, y))
    return pts


def draw_polyline(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], color: str, width: int = 5) -> None:
    if len(pts) > 1:
        draw.line(pts, fill=color, width=width, joint="curve")


def wrap_text(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if draw.textbbox((0, 0), candidate, font=text_font)[2] <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: str,
    text_font: ImageFont.FreeTypeFont,
    max_width: int,
    line_gap: int = 8,
) -> int:
    y = xy[1]
    for line in wrap_text(draw, text, text_font, max_width):
        draw.text((xy[0], y), line, fill=fill, font=text_font)
        bbox = draw.textbbox((xy[0], y), line, font=text_font)
        y += bbox[3] - bbox[1] + line_gap
    return y


def draw_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    detector: DetectorSpec,
    possible: np.ndarray,
    impossible: np.ndarray,
    current: int,
    t_star: int,
) -> None:
    x0, y0, x1, y1 = box
    rounded(draw, box, "#111827", "#273449")
    header_y = y0 + 18
    draw.text((x0 + 22, header_y), detector.label, fill="white", font=font(26, True))

    status_fill = {"correct": "#166534", "wrong": "#991b1b", "tied": "#854d0e"}[detector.status]
    status_label = {"correct": "correct", "wrong": "wrong", "tied": "tie"}[detector.status]
    pill = (x1 - 160, y0 + 16, x1 - 24, y0 + 54)
    draw.rounded_rectangle(pill, radius=18, fill=status_fill)
    draw_centered(draw, ((pill[0] + pill[2]) // 2, (pill[1] + pill[3]) // 2), status_label, "white", font(20, True))

    chart = (x0 + 40, y0 + 72, x1 - 36, y1 - 28)
    draw.rectangle(chart, fill="#0b1220")
    for k in range(4):
        yy = chart[1] + (chart[3] - chart[1]) * k // 3
        draw.line((chart[0], yy, chart[2], yy), fill="#1f2a3d", width=1)

    values = np.concatenate([possible, impossible])
    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))
    if ymax - ymin < 1e-3:
        ymin -= 1.0
        ymax += 1.0
    pad = (ymax - ymin) * 0.12
    ymin -= pad
    ymax += pad

    draw_polyline(draw, line_points(possible, chart, ymin, ymax), "#1e3a8a", width=3)
    draw_polyline(draw, line_points(impossible, chart, ymin, ymax), "#7f1d1d", width=3)
    draw_polyline(draw, line_points(possible[: current + 1], chart, ymin, ymax), "#60a5fa")
    draw_polyline(draw, line_points(impossible[: current + 1], chart, ymin, ymax), "#ef4444")

    for marker, color, width in ((t_star, "#facc15", 3), (current, "#ffffff", 2)):
        xx = int(chart[0] + (chart[2] - chart[0]) * marker / 71)
        draw.line((xx, chart[1], xx, chart[3]), fill=color, width=width)

    draw.text((chart[0], chart[3] + 4), "blue normal", fill="#93c5fd", font=font(18))
    draw.text((chart[0] + 148, chart[3] + 4), "red edited", fill="#fca5a5", font=font(18))
    draw.text((chart[2] - 210, chart[3] + 4), "higher = more suspicious", fill="#cbd5e1", font=font(16))


def make_frame(
    pair: PairSpec,
    possible_frames: list[np.ndarray],
    impossible_frames: list[np.ndarray],
    scores: dict[str, tuple[np.ndarray, np.ndarray]],
    clip_frame: int,
    frame_index: int,
    total_frames: int,
) -> Image.Image:
    canvas = Image.new("RGB", (1080, 1920), "#05070d")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1080, 1920), fill="#05070d")

    draw.text((54, 42), "PhysicsCourt", fill="#f8fafc", font=font(62, True))
    draw.text((56, 112), pair.title, fill="#facc15", font=font(42, True))
    draw.text((56, 166), pair.subtitle, fill="#cbd5e1", font=font(28))

    pf = possible_frames[min(clip_frame, len(possible_frames) - 1)]
    imf = impossible_frames[min(clip_frame, len(impossible_frames) - 1)]
    paste_panel(canvas, pf, (50, 250, 530, 520), "normal", "#60a5fa")
    paste_panel(canvas, imf, (550, 250, 1030, 520), "edited", "#ef4444")

    if abs(clip_frame - pair.t_star) <= 2:
        draw.rounded_rectangle((330, 536, 750, 592), radius=22, fill="#facc15")
        draw_centered(draw, (540, 564), "violation starts here", "#111827", font(30, True))

    chart_y = 635
    for idx, detector in enumerate(pair.detectors):
        poss, imp = scores[detector.key]
        draw_chart(draw, (56, chart_y + idx * 278, 1024, chart_y + idx * 278 + 238), detector, poss, imp, clip_frame, pair.t_star)

    progress_w = int(960 * frame_index / max(1, total_frames - 1))
    draw.rounded_rectangle((60, 1516, 1020, 1530), radius=7, fill="#1f2937")
    draw.rounded_rectangle((60, 1516, 60 + progress_w, 1530), radius=7, fill="#facc15")

    rounded(draw, (56, 1570, 1024, 1810), "#111827", "#273449")
    y = draw_wrapped(draw, (92, 1608), pair.bottom_line, "#f8fafc", font(32, True), 860, line_gap=8)
    y += 14
    y = draw_wrapped(
        draw,
        (92, y),
        "Same video pair. Same frozen calibration. No hand-picked model prompt.",
        "#cbd5e1",
        font(24),
        860,
        line_gap=6,
    )
    draw.text((92, min(y + 14, 1750)), "This is why edited-real validation matters.", fill="#facc15", font=font(31, True))

    draw.text((56, 1858), "github.com/AjayKasu1/physicscourt", fill="#94a3b8", font=font(26))
    return canvas


def make_intro(frame_number: int, total: int) -> Image.Image:
    canvas = Image.new("RGB", (1080, 1920), "#05070d")
    draw = ImageDraw.Draw(canvas)
    draw.text((74, 330), "Can AI spot", fill="#f8fafc", font=font(86, True))
    draw.text((74, 430), "impossible physics?", fill="#facc15", font=font(86, True))
    draw.text((78, 570), "V-JEPA 2 vs DINOv2 vs SSRD", fill="#cbd5e1", font=font(42, True))
    draw.rounded_rectangle((76, 720, 1004, 1010), radius=36, fill="#111827", outline="#273449", width=3)
    draw.text((116, 782), "Two edited-real ball tests", fill="#f8fafc", font=font(46, True))
    draw.text((116, 862), "1. Missing ball", fill="#93c5fd", font=font(38, True))
    draw.text((116, 928), "2. Teleport jump", fill="#fca5a5", font=font(38, True))
    draw.text((78, 1180), "Watch the red curve.", fill="#f8fafc", font=font(54, True))
    draw.text((78, 1250), "Higher means the model thinks the clip is more suspicious.", fill="#cbd5e1", font=font(32))
    progress_w = int(928 * frame_number / max(1, total - 1))
    draw.rounded_rectangle((76, 1510, 1004, 1526), radius=8, fill="#1f2937")
    draw.rounded_rectangle((76, 1510, 76 + progress_w, 1526), radius=8, fill="#facc15")
    draw.text((76, 1816), "PhysicsCourt", fill="#94a3b8", font=font(34, True))
    return canvas


def make_outro(frame_number: int, total: int) -> Image.Image:
    canvas = Image.new("RGB", (1080, 1920), "#05070d")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 260), "The honest result", fill="#facc15", font=font(72, True))
    items = [
        ("Missing ball", "SSRD catches it"),
        ("Teleport jump", "all three miss"),
        ("Next test", "more edited-real pairs"),
    ]
    y = 470
    for title, result in items:
        rounded(draw, (70, y, 1010, y + 180), "#111827", "#273449")
        draw.text((112, y + 42), title, fill="#f8fafc", font=font(44, True))
        draw.text((112, y + 100), result, fill="#cbd5e1", font=font(34, True))
        y += 230
    draw.text((70, 1350), "Not a demo trick. A benchmark artifact.", fill="#f8fafc", font=font(44, True))
    draw.text((70, 1420), "Repo has clips, curves, reports, and GCP timings.", fill="#cbd5e1", font=font(31))
    draw.text((70, 1660), "github.com/AjayKasu1/physicscourt", fill="#93c5fd", font=font(34, True))
    progress_w = int(940 * frame_number / max(1, total - 1))
    draw.rounded_rectangle((70, 1770, 1010, 1786), radius=8, fill="#1f2937")
    draw.rounded_rectangle((70, 1770, 70 + progress_w, 1786), radius=8, fill="#facc15")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=ROOT / "results/features_phase4_processed")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/media/physicscourt_phase4_reel.mp4")
    parser.add_argument("--poster", type=Path, default=ROOT / "docs/media/physicscourt_phase4_reel_poster.png")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1080, 1920))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {args.output}")

    intro_frames = args.fps * 2
    pair_frames = args.fps * 6
    outro_frames = args.fps * 3

    poster = make_intro(0, intro_frames)
    poster.save(args.poster)

    for i in range(intro_frames):
        frame = make_intro(i, intro_frames)
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))

    for pair in PAIRS:
        possible_frames = read_video(pair.possible_path)
        impossible_frames = read_video(pair.impossible_path)
        scores = {
            detector.key: (
                load_score(args.features_dir, detector.key, pair.possible_clip_id),
                load_score(args.features_dir, detector.key, pair.impossible_clip_id),
            )
            for detector in pair.detectors
        }
        for i in range(pair_frames):
            clip_frame = min(71, int(round(i / max(1, pair_frames - 1) * 71)))
            frame = make_frame(pair, possible_frames, impossible_frames, scores, clip_frame, i, pair_frames)
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))

    for i in range(outro_frames):
        frame = make_outro(i, outro_frames)
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"wrote {args.output}")
    print(f"wrote {args.poster}")


if __name__ == "__main__":
    main()
