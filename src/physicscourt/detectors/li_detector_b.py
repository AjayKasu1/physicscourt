"""Detector B: explicit object-state reconstruction and rule scores."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from physicscourt.pipeline.clip_dataset import ClipRecord
from physicscourt.pipeline.score_utils import median_filter_1d
from physicscourt.pipeline.video_io import iter_video_frames_rgb
from physicscourt.utils.torch_runtime import clear_torch, select_device, tensor_batch_to_device

COLOR_DISTANCE_THRESHOLD = 30.0


@dataclass(frozen=True)
class StageTiming:
    clip_id: str
    seconds: float
    num_frames: int
    stage: str
    device: str = ""
    dtype: str = ""


def save_npz_cache(path: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {"metadata_json": np.array([metadata], dtype=object)}
    payload.update(arrays)
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **payload)
    tmp.replace(path)


def load_npz_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        loaded: dict[str, Any] = {key: data[key] for key in data.files}
    if "metadata_json" in loaded:
        loaded["metadata"] = loaded["metadata_json"][0]
    return loaded


def clip_metadata(clip: ClipRecord, stage: str) -> dict[str, Any]:
    return {
        "clip_id": clip.clip_id,
        "pair_id": clip.pair_id,
        "stage": stage,
        "split": clip.split,
        "category": clip.category,
        "possible": clip.possible,
        "violation_frame": clip.violation_frame,
        "video_path": clip.video_path,
        "width": clip.width,
        "height": clip.height,
        "fps": clip.fps,
        "num_frames": clip.num_frames,
    }


def _clip_box_xyxy(box: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = box.astype(np.float32, copy=True)
    clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, width - 1)
    clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, height - 1)
    if clipped[2] <= clipped[0]:
        clipped[2] = min(width - 1, clipped[0] + 1)
    if clipped[3] <= clipped[1]:
        clipped[3] = min(height - 1, clipped[1] + 1)
    return clipped


def prompt_box_xyxy(clip: ClipRecord) -> np.ndarray:
    x, y, w, h = clip.prompt_box_xywh
    return _clip_box_xyxy(np.array([x, y, x + w, y + h], dtype=np.float32), clip.width, clip.height)


def prompt_query_points(clip: ClipRecord) -> np.ndarray:
    box = prompt_box_xyxy(clip)
    x0, y0, x1, y1 = box.tolist()
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    return np.array(
        [
            [cx, cy],
            [x0 + (x1 - x0) * 0.30, cy],
            [x0 + (x1 - x0) * 0.70, cy],
            [cx, y0 + (y1 - y0) * 0.30],
            [cx, y0 + (y1 - y0) * 0.70],
        ],
        dtype=np.float32,
    )


def sampled_frame_indices(num_frames: int, stride: int) -> np.ndarray:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if num_frames < 1:
        return np.zeros((0,), dtype=np.int32)
    return np.arange(0, num_frames, stride, dtype=np.int32)


def interpolate_sampled_values(sample_indices: np.ndarray, sample_values: np.ndarray, num_frames: int) -> np.ndarray:
    values = np.asarray(sample_values, dtype=np.float32)
    if values.shape[0] != sample_indices.shape[0]:
        raise ValueError("sample_values must have one row per sampled frame")
    if num_frames == 0:
        return np.zeros((0, *values.shape[1:]), dtype=np.float32)
    if sample_indices.size == 0:
        return np.full((num_frames, *values.shape[1:]), np.nan, dtype=np.float32)

    flat = values.reshape(values.shape[0], -1)
    out = np.empty((num_frames, flat.shape[1]), dtype=np.float32)
    frame_axis = np.arange(num_frames, dtype=np.float32)
    sample_axis = sample_indices.astype(np.float32)
    for dim in range(flat.shape[1]):
        column = flat[:, dim]
        finite = np.isfinite(column)
        if finite.sum() == 0:
            out[:, dim] = np.nan
        elif finite.sum() == 1:
            out[:, dim] = column[finite][0]
        else:
            out[:, dim] = np.interp(frame_axis, sample_axis[finite], column[finite]).astype(np.float32)
    return out.reshape((num_frames, *values.shape[1:]))


def nearest_sampled_masks(sample_indices: np.ndarray, sample_masks: np.ndarray, num_frames: int) -> np.ndarray:
    if num_frames == 0:
        return np.zeros((0, *sample_masks.shape[1:]), dtype=sample_masks.dtype)
    if sample_indices.size == 0:
        return np.zeros((num_frames, *sample_masks.shape[1:]), dtype=sample_masks.dtype)
    frame_axis = np.arange(num_frames)
    nearest = np.abs(frame_axis[:, None] - sample_indices[None, :]).argmin(axis=1)
    return sample_masks[nearest]


class CoTrackerStage:
    """Track prompt points through the clip with the official CoTracker model."""

    name = "li_cotracker"

    def __init__(self, model_id: str, checkpoint_file: str, device: str = "auto", track_size: int = 192) -> None:
        from cotracker.predictor import CoTrackerPredictor
        from huggingface_hub import hf_hub_download

        self.device = select_device(device)
        self.dtype = torch.float32
        self.track_size = track_size
        checkpoint = hf_hub_download(repo_id=model_id, filename=checkpoint_file)
        self.model = CoTrackerPredictor(checkpoint=checkpoint, offline=True, v2=False, window_len=60)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    def close(self) -> None:
        del self.model
        clear_torch()

    def process_clip(self, clip: ClipRecord) -> tuple[dict[str, np.ndarray], StageTiming]:
        started = time.perf_counter()
        frames = list(iter_video_frames_rgb(Path(clip.video_path)))
        resized = []
        for frame in frames:
            small = cv2.resize(frame, (self.track_size, self.track_size), interpolation=cv2.INTER_AREA)
            resized.append(small.astype(np.float32).transpose(2, 0, 1))
        video = torch.from_numpy(np.stack(resized, axis=0)).unsqueeze(0)
        video = video.to(device=self.device, dtype=self.dtype)

        scale_x = self.track_size / float(clip.width)
        scale_y = self.track_size / float(clip.height)
        points = prompt_query_points(clip)
        scaled_points = points.copy()
        scaled_points[:, 0] *= scale_x
        scaled_points[:, 1] *= scale_y
        queries_np = np.column_stack([np.zeros((scaled_points.shape[0],), dtype=np.float32), scaled_points])
        queries = torch.from_numpy(queries_np[None, :, :]).to(device=self.device, dtype=self.dtype)

        with torch.inference_mode():
            tracks, visibility = self.model(video, queries=queries)
            tracks_np = tracks.detach().float().cpu().numpy()[0]
            visibility_np = visibility.detach().float().cpu().numpy()[0]

        tracks_np[:, :, 0] /= scale_x
        tracks_np[:, :, 1] /= scale_y
        tracks_np[:, :, 0] = np.clip(tracks_np[:, :, 0], 0, clip.width - 1)
        tracks_np[:, :, 1] = np.clip(tracks_np[:, :, 1], 0, clip.height - 1)
        arrays = {
            "tracks_xy": tracks_np.astype(np.float32),
            "visibility": visibility_np.astype(np.float32),
            "query_points_xy": points.astype(np.float32),
            "track_size": np.array([self.track_size], dtype=np.int32),
        }
        del video, queries, tracks, visibility
        timing = StageTiming(
            clip.clip_id,
            float(time.perf_counter() - started),
            len(frames),
            self.name,
            self.device.type,
            str(self.dtype).replace("torch.", ""),
        )
        return arrays, timing


class SAM2Stage:
    """Segment sampled frames with SAM 2 image prompts and interpolate state."""

    name = "li_sam2"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        fp16: bool = True,
        mask_size: int = 128,
        frame_stride: int = 4,
    ) -> None:
        from transformers import Sam2Model, Sam2Processor

        self.device = select_device(device)
        self.dtype = torch.float16 if fp16 and self.device.type in {"cuda", "mps"} else torch.float32
        self.mask_size = mask_size
        self.frame_stride = frame_stride
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id, dtype=self.dtype, low_cpu_mem_usage=True)
        self.model.to(self.device)
        self.model.eval()

    def close(self) -> None:
        del self.model
        del self.processor
        clear_torch()

    def process_clip(self, clip: ClipRecord, cotracker_cache: Path) -> tuple[dict[str, np.ndarray], StageTiming]:
        started = time.perf_counter()
        tracks_data = load_npz_cache(cotracker_cache)
        tracks = np.asarray(tracks_data["tracks_xy"], dtype=np.float32)
        visibility = np.asarray(tracks_data["visibility"], dtype=np.float32)
        frames = list(iter_video_frames_rgb(Path(clip.video_path)))
        sample_indices = sampled_frame_indices(len(frames), self.frame_stride)

        sampled_masks = np.zeros((len(sample_indices), self.mask_size, self.mask_size), dtype=np.uint8)
        sampled_areas = np.zeros((len(sample_indices),), dtype=np.float32)
        sampled_centers = np.full((len(sample_indices), 2), np.nan, dtype=np.float32)
        sampled_bboxes = np.full((len(sample_indices), 4), np.nan, dtype=np.float32)
        sampled_iou_scores = np.full((len(sample_indices),), np.nan, dtype=np.float32)
        prompt_boxes = np.zeros((len(frames), 4), dtype=np.float32)

        base_box = prompt_box_xyxy(clip)
        base_w = max(4.0, float(base_box[2] - base_box[0]))
        base_h = max(4.0, float(base_box[3] - base_box[1]))
        last_box = base_box
        for idx in range(len(frames)):
            visible = visibility[idx] > 0.5 if idx < visibility.shape[0] else np.zeros((tracks.shape[1],), dtype=bool)
            points = tracks[idx, visible] if idx < tracks.shape[0] else np.zeros((0, 2), dtype=np.float32)
            if points.shape[0] >= 2:
                x0 = float(np.nanmin(points[:, 0])) - base_w * 0.35
                y0 = float(np.nanmin(points[:, 1])) - base_h * 0.35
                x1 = float(np.nanmax(points[:, 0])) + base_w * 0.35
                y1 = float(np.nanmax(points[:, 1])) + base_h * 0.35
                box = _clip_box_xyxy(np.array([x0, y0, x1, y1], dtype=np.float32), clip.width, clip.height)
                last_box = box
            elif np.isfinite(last_box).all():
                box = last_box
            else:
                box = base_box
            prompt_boxes[idx] = box

        for sample_pos, idx in enumerate(sample_indices):
            frame = frames[int(idx)]
            box = prompt_boxes[int(idx)]
            inputs = self.processor(images=Image.fromarray(frame), input_boxes=[[box.tolist()]], return_tensors="pt")
            inputs = tensor_batch_to_device(inputs, self.device, self.dtype if self.device.type != "cpu" else None)
            with torch.inference_mode():
                outputs = self.model(**inputs, multimask_output=False)
                processed = self.processor.post_process_masks(
                    outputs.pred_masks.detach().cpu(),
                    inputs["original_sizes"].detach().cpu(),
                )[0]

            mask_logits = processed.squeeze().float().numpy()
            while mask_logits.ndim > 2:
                mask_logits = mask_logits[0]
            mask = mask_logits > 0.0
            small_mask = cv2.resize(
                mask.astype(np.uint8),
                (self.mask_size, self.mask_size),
                interpolation=cv2.INTER_NEAREST,
            )
            sampled_masks[sample_pos] = small_mask
            ys, xs = np.nonzero(mask)
            sampled_areas[sample_pos] = float(mask.sum())
            if xs.size:
                sampled_centers[sample_pos] = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)
                sampled_bboxes[sample_pos] = np.array(
                    [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                    dtype=np.float32,
                )
            if hasattr(outputs, "iou_scores"):
                iou = outputs.iou_scores.detach().float().cpu().numpy().reshape(-1)
                if iou.size:
                    sampled_iou_scores[sample_pos] = float(iou[0])
            del inputs, outputs, processed

        masks = nearest_sampled_masks(sample_indices, sampled_masks, len(frames))
        areas = interpolate_sampled_values(sample_indices, sampled_areas[:, None], len(frames))[:, 0]
        centers = interpolate_sampled_values(sample_indices, sampled_centers, len(frames))
        bboxes = interpolate_sampled_values(sample_indices, sampled_bboxes, len(frames))
        iou_scores = interpolate_sampled_values(sample_indices, sampled_iou_scores[:, None], len(frames))[:, 0]
        timing = StageTiming(
            clip.clip_id,
            float(time.perf_counter() - started),
            len(frames),
            self.name,
            self.device.type,
            str(self.dtype).replace("torch.", ""),
        )
        return {
            "mask_128": masks,
            "mask_area_px": areas,
            "mask_center_xy": centers,
            "mask_bbox_xyxy": bboxes,
            "prompt_box_xyxy": prompt_boxes,
            "iou_score": iou_scores,
            "sampled_frame_indices": sample_indices.astype(np.int32),
            "frame_stride": np.array([self.frame_stride], dtype=np.int32),
        }, timing


class DepthStage:
    """Estimate sampled monocular depth and interpolate compact low-res maps."""

    name = "li_depth"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        fp16: bool = True,
        depth_size: int = 128,
        frame_stride: int = 4,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = select_device(device)
        self.dtype = torch.float16 if fp16 and self.device.type in {"cuda", "mps"} else torch.float32
        self.depth_size = depth_size
        self.frame_stride = frame_stride
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id, dtype=self.dtype, low_cpu_mem_usage=True)
        self.model.to(self.device)
        self.model.eval()

    def close(self) -> None:
        del self.model
        del self.processor
        clear_torch()

    def process_clip(self, clip: ClipRecord) -> tuple[dict[str, np.ndarray], StageTiming]:
        started = time.perf_counter()
        frames = list(iter_video_frames_rgb(Path(clip.video_path)))
        sample_indices = sampled_frame_indices(len(frames), self.frame_stride)
        sampled_depth_maps = []
        for idx in sample_indices:
            frame = frames[int(idx)]
            inputs = self.processor(images=Image.fromarray(frame), return_tensors="pt")
            inputs = tensor_batch_to_device(inputs, self.device, self.dtype if self.device.type != "cpu" else None)
            with torch.inference_mode():
                outputs = self.model(**inputs)
                depth = outputs.predicted_depth.detach().float().cpu().numpy().squeeze()
            depth = cv2.resize(depth, (self.depth_size, self.depth_size), interpolation=cv2.INTER_AREA)
            sampled_depth_maps.append(depth.astype(np.float32))
            del inputs, outputs
        sampled = np.stack(sampled_depth_maps, axis=0)
        stacked = interpolate_sampled_values(sample_indices, sampled, len(frames)).astype(np.float16)
        timing = StageTiming(
            clip.clip_id,
            float(time.perf_counter() - started),
            int(stacked.shape[0]),
            self.name,
            self.device.type,
            str(self.dtype).replace("torch.", ""),
        )
        return {
            "depth_128": stacked,
            "depth_min": np.array([float(np.nanmin(stacked))], dtype=np.float32),
            "depth_max": np.array([float(np.nanmax(stacked))], dtype=np.float32),
            "sampled_frame_indices": sample_indices.astype(np.int32),
            "frame_stride": np.array([self.frame_stride], dtype=np.int32),
        }, timing


def reconstruct_state_and_score(clip: ClipRecord, features_dir: Path) -> tuple[dict[str, np.ndarray], StageTiming]:
    started = time.perf_counter()
    cot = load_npz_cache(features_dir / CoTrackerStage.name / f"{clip.clip_id}.npz")
    sam = load_npz_cache(features_dir / SAM2Stage.name / f"{clip.clip_id}.npz")
    dep = load_npz_cache(features_dir / DepthStage.name / f"{clip.clip_id}.npz")
    frames = list(iter_video_frames_rgb(Path(clip.video_path)))

    tracks = np.asarray(cot["tracks_xy"], dtype=np.float32)
    visibility = np.asarray(cot["visibility"], dtype=np.float32)
    masks = np.asarray(sam["mask_128"], dtype=np.uint8)
    mask_area = np.asarray(sam["mask_area_px"], dtype=np.float32)
    mask_centers = np.asarray(sam["mask_center_xy"], dtype=np.float32)
    depth = np.asarray(dep["depth_128"], dtype=np.float32)

    num_frames = min(len(frames), tracks.shape[0], masks.shape[0], depth.shape[0])
    full_area = float(clip.width * clip.height)
    prompt_area = float(max(1, clip.prompt_box_xywh[2] * clip.prompt_box_xywh[3]))
    expected_area = max(prompt_area * 0.45, 50.0)
    color = np.asarray(clip.object_color_rgb, dtype=np.float32)

    color_conf = np.zeros((num_frames,), dtype=np.float32)
    color_area_px = np.zeros((num_frames,), dtype=np.float32)
    color_centers = np.full((num_frames, 2), np.nan, dtype=np.float32)
    depth_mean = np.full((num_frames,), np.nan, dtype=np.float32)
    area_conf = np.clip(mask_area[:num_frames] / expected_area, 0.0, 1.5).astype(np.float32) / 1.5
    track_visibility = np.clip(np.nanmean(visibility[:num_frames], axis=1), 0.0, 1.0).astype(np.float32)
    centers = np.full((num_frames, 2), np.nan, dtype=np.float32)

    for idx in range(num_frames):
        frame_float = frames[idx].astype(np.float32)
        color_distance = np.linalg.norm(frame_float - color, axis=2)
        color_mask = color_distance < COLOR_DISTANCE_THRESHOLD
        color_area_px[idx] = float(color_mask.sum())
        color_y, color_x = np.nonzero(color_mask)
        if color_x.size:
            color_centers[idx] = np.array([float(color_x.mean()), float(color_y.mean())], dtype=np.float32)

        mask_small = masks[idx].astype(bool)
        if mask_small.any():
            frame_small = cv2.resize(frames[idx], (128, 128), interpolation=cv2.INTER_AREA)
            pixels = frame_small[mask_small].astype(np.float32)
            if pixels.size:
                distance = float(np.linalg.norm(pixels.mean(axis=0) - color))
                color_conf[idx] = float(np.exp(-distance / 80.0))
                depth_mean[idx] = float(depth[idx][mask_small].mean())
        if np.isfinite(color_centers[idx]).all():
            centers[idx] = color_centers[idx]
        elif np.isfinite(mask_centers[idx]).all():
            centers[idx] = mask_centers[idx]
        else:
            visible = visibility[idx] > 0.5
            points = tracks[idx, visible]
            if points.size:
                centers[idx] = np.nanmean(points, axis=0)

    mask_presence = np.clip(area_conf * color_conf * np.maximum(track_visibility, 0.25), 0.0, 1.0).astype(np.float32)
    color_presence = np.clip(color_area_px / expected_area, 0.0, 1.0).astype(np.float32)
    presence = np.maximum(mask_presence, color_presence).astype(np.float32)
    centers = _fill_missing_centers(centers, tracks[:num_frames], visibility[:num_frames])
    velocity = np.zeros_like(centers, dtype=np.float32)
    if num_frames > 1:
        velocity[1:] = centers[1:] - centers[:-1]
        velocity[0] = velocity[1]
    speed = np.linalg.norm(velocity, axis=1).astype(np.float32)

    rule_scores, rule_names = _rule_scores(clip, presence, centers, velocity, speed, depth_mean)
    raw_score = rule_scores.max(axis=1).astype(np.float32)
    smoothed_score = median_filter_1d(raw_score, 5)
    arrays = {
        "raw_score": raw_score,
        "smoothed_score": smoothed_score,
        "rule_scores": rule_scores.astype(np.float32),
        "rule_names_json": np.array([json.dumps(rule_names)], dtype=object),
        "presence": presence,
        "center_xy": centers.astype(np.float32),
        "velocity_xy": velocity.astype(np.float32),
        "speed_px": speed,
        "mask_area_fraction": (mask_area[:num_frames] / full_area).astype(np.float32),
        "mask_presence": mask_presence,
        "color_area_fraction": (color_area_px / full_area).astype(np.float32),
        "color_area_px": color_area_px,
        "color_center_xy": color_centers.astype(np.float32),
        "color_confidence": color_conf,
        "track_visibility": track_visibility,
        "depth_mean": depth_mean,
    }
    timing = StageTiming(clip.clip_id, float(time.perf_counter() - started), num_frames, "li_state_rules", "none", "none")
    return arrays, timing


def _fill_missing_centers(centers: np.ndarray, tracks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    filled = centers.astype(np.float32, copy=True)
    for idx in range(filled.shape[0]):
        if np.isfinite(filled[idx]).all():
            continue
        visible = visibility[idx] > 0.5
        points = tracks[idx, visible]
        if points.size:
            filled[idx] = np.nanmean(points, axis=0)
    finite = np.isfinite(filled).all(axis=1)
    if not finite.any():
        return np.nan_to_num(filled, nan=0.0)
    valid_idx = np.flatnonzero(finite)
    for dim in range(2):
        filled[:, dim] = np.interp(np.arange(filled.shape[0]), valid_idx, filled[valid_idx, dim])
    return filled


def _absence_run(absence: np.ndarray) -> np.ndarray:
    out = np.zeros_like(absence, dtype=np.float32)
    run = 0.0
    for idx, value in enumerate(absence):
        if value > 0.45:
            run += float(value)
        else:
            run = 0.0
        out[idx] = run
    return out


def _rule_scores(
    clip: ClipRecord,
    presence: np.ndarray,
    centers: np.ndarray,
    velocity: np.ndarray,
    speed: np.ndarray,
    depth_mean: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    del depth_mean
    num_frames = presence.shape[0]
    rule_columns: list[np.ndarray] = []
    names: list[str] = []
    absence = 1.0 - presence
    median_speed = float(np.nanmedian(speed[5:])) if num_frames > 6 else float(np.nanmedian(speed))
    median_speed = max(median_speed, 1.0)
    frame_axis = np.arange(num_frames, dtype=np.float32)
    late_ramp = np.clip((frame_axis - num_frames * 0.45) / max(1.0, num_frames * 0.18), 0.0, 1.0)

    def add(name: str, values: np.ndarray) -> None:
        names.append(name)
        rule_columns.append(np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=5.0, neginf=0.0))

    add("permanence_late_absence", late_ramp * absence)
    add("permanence_absence_run", np.clip(_absence_run(absence) / 12.0, 0.0, 3.0) * late_ramp)
    add(
        "permanence_absence_after_rightward_motion",
        np.clip((centers[:, 0] - clip.width * 0.58) / max(1.0, clip.width * 0.20), 0.0, 2.0) * absence,
    )

    prev_presence = np.r_[presence[0], presence[:-1]]
    add("vanishing_presence_drop", np.clip(prev_presence - presence, 0.0, 1.0) * 3.0)
    add("vanishing_sustained_absence", np.clip(_absence_run(absence) / 8.0, 0.0, 2.0) * late_ramp)
    add("vanishing_absence_before_exit", absence * (centers[:, 0] < clip.width * 0.82))

    jump = speed / median_speed
    add("continuity_speed_jump", np.clip(jump - 2.5, 0.0, 5.0) * presence)
    add("continuity_vertical_jump", np.clip(np.abs(velocity[:, 1]) / median_speed - 1.5, 0.0, 5.0) * presence)
    add("continuity_absolute_speed", np.clip(speed / max(clip.width * 0.08, 1.0), 0.0, 5.0) * presence)

    wall_x = clip.width * 0.59
    penetration = np.clip((centers[:, 0] - wall_x) / max(1.0, clip.width * 0.10), 0.0, 4.0)
    add("solidity_wall_penetration", penetration * presence)
    add("solidity_rightward_penetration", np.clip(velocity[:, 0] / median_speed, 0.0, 4.0) * penetration)
    add("solidity_late_penetration", (frame_axis > num_frames * 0.45).astype(np.float32) * penetration)

    edge_x = clip.width * 0.59
    beyond_edge = np.clip((centers[:, 0] - edge_x) / max(1.0, clip.width * 0.15), 0.0, 3.0)
    downward = np.clip(velocity[:, 1], 0.0, None)
    floating = np.clip(1.0 - downward / max(median_speed, 2.0), 0.0, 1.0)
    add("gravity_beyond_edge_floating", beyond_edge * floating * presence)
    add(
        "gravity_high_beyond_edge",
        beyond_edge * np.clip((clip.height * 0.55 - centers[:, 1]) / max(1.0, clip.height * 0.20), 0.0, 2.0),
    )
    add("gravity_late_beyond_edge", late_ramp * beyond_edge)

    accel = np.zeros_like(speed, dtype=np.float32)
    if num_frames > 2:
        accel[1:] = np.linalg.norm(velocity[1:] - velocity[:-1], axis=1)
    median_accel = max(float(np.nanmedian(accel[5:])), 1.0)
    add("momentum_acceleration_spike", np.clip(accel / median_accel - 2.0, 0.0, 5.0) * presence)
    add("momentum_leftward_reversal", np.clip(-velocity[:, 0] / median_speed, 0.0, 4.0) * (centers[:, 0] < clip.width * 0.62) * presence)
    add("momentum_speed_burst", np.clip(speed / median_speed - 2.5, 0.0, 5.0) * presence)

    scores = np.stack(rule_columns, axis=1).astype(np.float32)
    return scores, names
