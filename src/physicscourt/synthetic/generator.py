"""Controlled synthetic physical-impossibility videos.

This module serves both hypotheses by making matched possible/impossible pairs
whose pixels are identical until the recorded violation frame. Detector A gets a
clean surprise benchmark; Detector B gets object prompts and category labels for
explicit state checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from physicscourt.pipeline.clip_dataset import ClipRecord, write_manifest
from physicscourt.pipeline.video_io import read_frame_rgb, write_video_mp4

CATEGORIES: dict[str, str] = {
    "object_permanence": "object permanence",
    "solidity": "solidity",
    "continuity_teleportation": "continuity",
    "gravity_support": "gravity and support",
    "spontaneous_vanishing": "spontaneous appearance or vanishing",
    "momentum_causality": "momentum and causality",
}

BACKGROUND_VARIANTS = ("plain", "textured", "plain_shake", "textured_shake")


@dataclass(frozen=True)
class SyntheticConfig:
    width: int = 512
    height: int = 512
    fps: int = 24
    duration_seconds: float = 3.0
    seed: int = 1729

    @property
    def num_frames(self) -> int:
        return int(round(self.duration_seconds * self.fps))


@dataclass(frozen=True)
class ScenarioSpec:
    category: str
    pair_id: str
    split: str
    seed: int
    style_seed: int
    background_variant: str
    background_source: str
    photorealism_gap_control: bool
    violation_frame: int
    ball_radius: int
    ball_color: tuple[int, int, int]
    object_prompt_box_xywh: list[int]
    prompt_point_xy: list[int]


class SyntheticGenerator:
    """Render matched possible/impossible clips from small analytic scenes."""

    def __init__(self, config: SyntheticConfig) -> None:
        self.config = config

    def make_spec(
        self,
        category: str,
        pair_id: str,
        split: str,
        seed: int,
        background_variant: str,
        photorealism_gap_control: bool = False,
    ) -> ScenarioSpec:
        rng = np.random.default_rng(seed)
        radius = int(rng.integers(18, 31))
        color = tuple(int(v) for v in rng.integers(70, 245, size=3))
        t_star = self._violation_frame(category)
        prompt_center = self._initial_center(category, radius)
        prompt_box = [
            int(prompt_center[0] - radius),
            int(prompt_center[1] - radius),
            int(radius * 2),
            int(radius * 2),
        ]
        source = "procedural_photo_like" if photorealism_gap_control else "procedural"
        return ScenarioSpec(
            category=category,
            pair_id=pair_id,
            split=split,
            seed=seed,
            style_seed=int(rng.integers(0, 2**31 - 1)),
            background_variant=background_variant,
            background_source=source,
            photorealism_gap_control=photorealism_gap_control,
            violation_frame=t_star,
            ball_radius=radius,
            ball_color=color,
            object_prompt_box_xywh=prompt_box,
            prompt_point_xy=[int(prompt_center[0]), int(prompt_center[1])],
        )

    def render_pair(self, spec: ScenarioSpec) -> tuple[list[np.ndarray], list[np.ndarray]]:
        possible = [self.render_frame(spec, frame_idx, possible=True) for frame_idx in range(self.config.num_frames)]
        impossible = [
            self.render_frame(spec, frame_idx, possible=False) for frame_idx in range(self.config.num_frames)
        ]
        return possible, impossible

    def render_frame(self, spec: ScenarioSpec, frame_idx: int, possible: bool) -> np.ndarray:
        frame = self._background(spec, frame_idx)
        painter = self._scenario_painter(spec.category)
        painter(frame, spec, frame_idx, possible)
        return self._apply_camera_shake(frame, spec, frame_idx)

    def make_record(self, spec: ScenarioSpec, possible: bool, video_path: Path) -> ClipRecord:
        suffix = "possible" if possible else "impossible"
        return ClipRecord(
            clip_id=f"{spec.pair_id}_{suffix}",
            pair_id=spec.pair_id,
            source="synthetic",
            split=spec.split,
            category=spec.category,
            infant_construct=CATEGORIES[spec.category],
            possible=possible,
            violation_kind=spec.category,
            violation_frame=None if possible else spec.violation_frame,
            video_path=str(video_path),
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            num_frames=self.config.num_frames,
            prompt_box_xywh=list(spec.object_prompt_box_xywh),
            prompt_point_xy=list(spec.prompt_point_xy),
            object_color_rgb=list(spec.ball_color),
            background_variant=spec.background_variant,
            background_source=spec.background_source,
            photorealism_gap_control=spec.photorealism_gap_control,
            seed=spec.seed,
        )

    def assert_pair_identity_until_violation(self, spec: ScenarioSpec) -> None:
        for frame_idx in range(spec.violation_frame):
            possible = self.render_frame(spec, frame_idx, possible=True)
            impossible = self.render_frame(spec, frame_idx, possible=False)
            if not np.array_equal(possible, impossible):
                raise AssertionError(f"{spec.category} pair diverged before t_star at frame {frame_idx}")

    def _violation_frame(self, category: str) -> int:
        n = self.config.num_frames
        positions = {
            "object_permanence": 42,
            "solidity": 34,
            "continuity_teleportation": 36,
            "gravity_support": 38,
            "spontaneous_vanishing": 36,
            "momentum_causality": 32,
        }
        return min(max(positions[category], 12), n - 18)

    def _initial_center(self, category: str, radius: int) -> tuple[float, float]:
        h = self.config.height
        starts = {
            "object_permanence": (70 + radius, h * 0.55),
            "solidity": (70 + radius, h * 0.56),
            "continuity_teleportation": (70 + radius, h * 0.50),
            "gravity_support": (110 + radius, h * 0.42),
            "spontaneous_vanishing": (80 + radius, h * 0.52),
            "momentum_causality": (80 + radius, h * 0.58),
        }
        return starts[category]

    def _background(self, spec: ScenarioSpec, frame_idx: int) -> np.ndarray:
        rng = np.random.default_rng(spec.style_seed)
        h, w = self.config.height, self.config.width
        if spec.photorealism_gap_control:
            return self._photo_like_background(rng, frame_idx)
        if "textured" in spec.background_variant:
            base = rng.normal(116, 18, size=(h, w, 3)).astype(np.float32)
            base = cv2.GaussianBlur(base, (0, 0), sigmaX=5.0)
            tint = np.array(rng.integers(20, 70, size=3), dtype=np.float32)
            return np.clip(base + tint, 0, 255).astype(np.uint8)
        color = np.array(rng.integers(38, 92, size=3), dtype=np.uint8)
        grad = np.linspace(0, 42, h, dtype=np.uint8)[:, None]
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for channel in range(3):
            frame[:, :, channel] = np.clip(int(color[channel]) + grad, 0, 255)
        return frame

    def _photo_like_background(self, rng: np.random.Generator, frame_idx: int) -> np.ndarray:
        del frame_idx
        h, w = self.config.height, self.config.width
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        wall = np.array(rng.integers([80, 85, 90], [150, 155, 165]), dtype=np.uint8)
        floor = np.array(rng.integers([80, 65, 52], [135, 115, 95]), dtype=np.uint8)
        horizon = int(h * 0.58)
        frame[:horizon, :, :] = wall
        frame[horizon:, :, :] = floor

        for _ in range(10):
            x0 = int(rng.integers(-40, w - 30))
            y0 = int(rng.integers(20, h - 80))
            x1 = int(x0 + rng.integers(45, 150))
            y1 = int(y0 + rng.integers(30, 120))
            color = tuple(int(v) for v in rng.integers(55, 210, size=3))
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness=-1)
        frame = cv2.GaussianBlur(frame, (0, 0), sigmaX=6.0)
        noise = rng.normal(0, 4, size=frame.shape)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _apply_camera_shake(self, frame: np.ndarray, spec: ScenarioSpec, frame_idx: int) -> np.ndarray:
        if "shake" not in spec.background_variant:
            return frame
        rng = np.random.default_rng(spec.style_seed + 99)
        dxs = rng.normal(0, 1.4, size=self.config.num_frames).cumsum()
        dys = rng.normal(0, 1.0, size=self.config.num_frames).cumsum()
        dx = float(np.clip(dxs[frame_idx], -5, 5))
        dy = float(np.clip(dys[frame_idx], -4, 4))
        transform = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        return cv2.warpAffine(frame, transform, (self.config.width, self.config.height), borderMode=cv2.BORDER_REFLECT)

    def _scenario_painter(self, category: str) -> Callable[[np.ndarray, ScenarioSpec, int, bool], None]:
        painters = {
            "object_permanence": self._paint_object_permanence,
            "solidity": self._paint_solidity,
            "continuity_teleportation": self._paint_continuity,
            "gravity_support": self._paint_gravity,
            "spontaneous_vanishing": self._paint_vanishing,
            "momentum_causality": self._paint_momentum,
        }
        return painters[category]

    def _draw_ball(self, frame: np.ndarray, center: tuple[float, float], spec: ScenarioSpec) -> None:
        cv2.circle(
            frame,
            (int(round(center[0])), int(round(center[1]))),
            spec.ball_radius,
            spec.ball_color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            frame,
            (int(round(center[0] - spec.ball_radius * 0.32)), int(round(center[1] - spec.ball_radius * 0.32))),
            max(3, spec.ball_radius // 5),
            (255, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    def _paint_object_permanence(self, frame: np.ndarray, spec: ScenarioSpec, frame_idx: int, possible: bool) -> None:
        w, h = self.config.width, self.config.height
        y = h * 0.55
        occluder = (int(w * 0.42), int(h * 0.26), int(w * 0.60), int(h * 0.71))
        start_x = w * 0.14
        exit_x = occluder[2] + spec.ball_radius + w * 0.06
        speed = (exit_x - start_x) / max(1, spec.violation_frame)
        x = start_x + frame_idx * speed
        if possible or frame_idx < spec.violation_frame:
            self._draw_ball(frame, (x, y), spec)
        cv2.rectangle(frame, (occluder[0], occluder[1]), (occluder[2], occluder[3]), (162, 169, 178), -1)
        cv2.rectangle(frame, (occluder[0], occluder[1]), (occluder[2], occluder[3]), (84, 92, 102), 3)

    def _paint_solidity(self, frame: np.ndarray, spec: ScenarioSpec, frame_idx: int, possible: bool) -> None:
        w, h = self.config.width, self.config.height
        wall_x = int(w * 0.59)
        y = h * 0.56
        contact = spec.violation_frame
        x_contact = wall_x - spec.ball_radius - 2
        if frame_idx < contact:
            x = w * 0.15 + (x_contact - w * 0.15) * frame_idx / contact
        elif possible:
            x = x_contact - (frame_idx - contact) * w * 0.013
        else:
            x = x_contact + (frame_idx - contact) * w * 0.013
        self._draw_ball(frame, (x, y), spec)
        cv2.rectangle(frame, (wall_x, int(h * 0.18)), (wall_x + int(w * 0.07), int(h * 0.83)), (190, 194, 199), -1)
        cv2.rectangle(frame, (wall_x, int(h * 0.18)), (wall_x + int(w * 0.07), int(h * 0.83)), (88, 92, 98), 3)

    def _paint_continuity(self, frame: np.ndarray, spec: ScenarioSpec, frame_idx: int, possible: bool) -> None:
        w, h = self.config.width, self.config.height
        y = h * 0.50
        if possible or frame_idx < spec.violation_frame:
            x = w * 0.15 + frame_idx * w * 0.011
        else:
            x = w * 0.15 + frame_idx * w * 0.011 + w * 0.29
            y -= h * 0.16
        self._draw_ball(frame, (x, y), spec)
        cv2.line(frame, (int(w * 0.10), int(y + h * 0.14)), (int(w * 0.90), int(y + h * 0.14)), (90, 110, 118), 2)

    def _paint_gravity(self, frame: np.ndarray, spec: ScenarioSpec, frame_idx: int, possible: bool) -> None:
        w, h = self.config.width, self.config.height
        ledge_y = int(h * 0.53)
        ledge_end = int(w * 0.59)
        start_x = w * 0.23
        x_edge = ledge_end + spec.ball_radius
        if frame_idx < spec.violation_frame:
            x = start_x + (x_edge - start_x) * frame_idx / spec.violation_frame
            y = ledge_y - spec.ball_radius
        else:
            dt = frame_idx - spec.violation_frame
            x = x_edge + dt * w * 0.009
            if possible:
                y = ledge_y - spec.ball_radius + h * 0.00082 * dt * dt
            else:
                y = ledge_y - spec.ball_radius - 2
        cv2.rectangle(frame, (int(w * 0.13), ledge_y), (ledge_end, ledge_y + int(h * 0.055)), (122, 101, 82), -1)
        cv2.rectangle(frame, (int(w * 0.13), ledge_y), (ledge_end, ledge_y + int(h * 0.055)), (76, 61, 49), 2)
        self._draw_ball(frame, (x, y), spec)

    def _paint_vanishing(self, frame: np.ndarray, spec: ScenarioSpec, frame_idx: int, possible: bool) -> None:
        w, h = self.config.width, self.config.height
        y = h * 0.52
        x = w * 0.18 + frame_idx * w * 0.011
        visible = possible or frame_idx < spec.violation_frame
        if visible:
            self._draw_ball(frame, (x, y), spec)
        cv2.line(frame, (int(w * 0.08), int(y + h * 0.13)), (int(w * 0.92), int(y + h * 0.13)), (88, 103, 108), 2)

    def _paint_momentum(self, frame: np.ndarray, spec: ScenarioSpec, frame_idx: int, possible: bool) -> None:
        w, h = self.config.width, self.config.height
        y = h * 0.58
        reverse = spec.violation_frame
        bumper_x = int(w * 0.68)
        speed = w * 0.0098
        if frame_idx < reverse:
            x = w * 0.17 + frame_idx * speed
        elif possible:
            pre = w * 0.17 + frame_idx * speed
            if pre < bumper_x - spec.ball_radius:
                x = pre
            else:
                bounce_frame = reverse + int(0.18 * self.config.num_frames)
                x = bumper_x - spec.ball_radius - (frame_idx - bounce_frame) * speed
        else:
            x_at_reverse = w * 0.17 + reverse * speed
            x = x_at_reverse - (frame_idx - reverse) * speed
        cv2.rectangle(frame, (bumper_x, int(y - h * 0.18)), (bumper_x + int(w * 0.055), int(y + h * 0.18)), (104, 150, 166), -1)
        cv2.rectangle(frame, (bumper_x, int(y - h * 0.18)), (bumper_x + int(w * 0.055), int(y + h * 0.18)), (52, 88, 102), 3)
        self._draw_ball(frame, (x, y), spec)


def _pair_specs(
    generator: SyntheticGenerator,
    split: str,
    pairs_per_category: int,
    seed: int,
    photo_controls: bool,
) -> Iterable[ScenarioSpec]:
    variants = list(BACKGROUND_VARIANTS)
    for category_index, category in enumerate(CATEGORIES):
        for pair_index in range(pairs_per_category):
            variant = variants[(category_index + pair_index) % len(variants)]
            pair_seed = seed + category_index * 10_000 + pair_index * 101
            pair_id = f"{split}_{category}_{pair_index:03d}"
            yield generator.make_spec(category, pair_id, split, pair_seed, variant)
            if photo_controls and split == "test":
                yield generator.make_spec(
                    category,
                    f"{pair_id}_photo",
                    split,
                    pair_seed,
                    "photo_like",
                    photorealism_gap_control=True,
                )


def generate_dataset(
    output_root: Path,
    manifest_path: Path,
    config: SyntheticConfig,
    pairs_per_category: int = 10,
    calibration_per_category: int = 2,
    photo_controls: bool = True,
    overwrite: bool = False,
    montage_path: Path | None = None,
) -> list[ClipRecord]:
    generator = SyntheticGenerator(config)
    records: list[ClipRecord] = []

    specs = list(_pair_specs(generator, "test", pairs_per_category, config.seed, photo_controls))
    specs.extend(_pair_specs(generator, "calibration", calibration_per_category, config.seed + 500_000, False))

    for spec in specs:
        generator.assert_pair_identity_until_violation(spec)
        possible_frames, impossible_frames = generator.render_pair(spec)
        category_dir = output_root / spec.split / spec.category
        possible_path = category_dir / f"{spec.pair_id}_possible.mp4"
        impossible_path = category_dir / f"{spec.pair_id}_impossible.mp4"

        if overwrite or not possible_path.exists():
            write_video_mp4(possible_frames, possible_path, config.fps)
        records.append(generator.make_record(spec, True, possible_path))

        if spec.split == "test":
            if overwrite or not impossible_path.exists():
                write_video_mp4(impossible_frames, impossible_path, config.fps)
            records.append(generator.make_record(spec, False, impossible_path))

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "physicscourt.synthetic.generator",
        "seed": config.seed,
        "width": config.width,
        "height": config.height,
        "fps": config.fps,
        "num_frames": config.num_frames,
        "pairs_per_category": pairs_per_category,
        "calibration_normals_per_category": calibration_per_category,
        "photo_controls": photo_controls,
        "photo_control_note": "Procedural photo-like backgrounds are used for license-clean automation.",
    }
    write_manifest(manifest_path, records, metadata)
    if montage_path is not None:
        write_spotcheck_montage(records, montage_path)
    return records


def write_spotcheck_montage(records: list[ClipRecord], output_path: Path, max_clips: int = 12) -> None:
    candidates = [record for record in records if not record.possible and not record.photorealism_gap_control]
    by_category: dict[str, list[ClipRecord]] = {category: [] for category in CATEGORIES}
    for record in candidates:
        by_category[record.category].append(record)
    selected: list[ClipRecord] = []
    offset = 0
    while len(selected) < max_clips:
        added = False
        for category in CATEGORIES:
            bucket = by_category[category]
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
                if len(selected) == max_clips:
                    break
        if not added:
            break
        offset += 1
    if not selected:
        return

    tile_w, tile_h = 320, 220
    cols = 3
    rows = int(np.ceil(len(selected) / cols))
    montage = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    montage[:, :, :] = (24, 27, 31)

    for idx, record in enumerate(selected):
        row, col = divmod(idx, cols)
        frame_idx = int(record.violation_frame or 0)
        frame = read_frame_rgb(Path(record.video_path), frame_idx)
        frame = cv2.resize(frame, (tile_w, tile_h - 36), interpolation=cv2.INTER_AREA)
        y0 = row * tile_h
        x0 = col * tile_w
        montage[y0 : y0 + tile_h - 36, x0 : x0 + tile_w] = frame
        label = f"{record.category}  t={frame_idx}"
        cv2.putText(
            montage,
            label,
            (x0 + 10, y0 + tile_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 235, 240),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
