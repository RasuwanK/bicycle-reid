"""Video processing engine for detection, within-camera tracking, and visualization.

Coordinates video decoding, YOLO detection, within-camera tracking,
visual embedding extraction, and frame annotation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Callable
import cv2
import numpy as np

from bicycle_reid.contracts.models import Detection, Track
from bicycle_reid.detection.yolo_detector import YOLODetector
from bicycle_reid.tracking.tracker import WithinCameraTracker
from bicycle_reid.representation.vision_encoder import (
    PretrainedVisionEncoder,
    extract_crop_from_frame,
)
from bicycle_reid.retrieval.similarity import (
    compute_similarity_matrix,
    retrieve_candidates,
    RetrievalResult,
)
from bicycle_reid.config.settings import AppConfig

logger = logging.getLogger(__name__)

# Color palette for tracks (BGR format for OpenCV)
TRACK_COLORS = [
    (56, 56, 255),    # Red
    (255, 112, 31),   # Blue
    (31, 222, 112),   # Green
    (23, 197, 255),   # Orange / Yellow
    (214, 39, 178),   # Magenta
    (0, 215, 255),    # Gold
    (180, 105, 255),  # Pink
    (255, 191, 0),    # Cyan
]


def get_color_for_id(track_id: str) -> tuple[int, int, int]:
    """Deterministically map a track ID string to a distinct BGR color."""
    idx = abs(hash(track_id)) % len(TRACK_COLORS)
    return TRACK_COLORS[idx]


def annotate_frame(
    frame: np.ndarray,
    detections: Sequence[Detection],
    tracks_by_id: dict[str, Track] | None = None,
    draw_trajectory: bool = True,
) -> np.ndarray:
    """Draw bounding boxes, labels, and trajectory trails onto a video frame."""
    annotated = frame.copy()
    tracks_by_id = tracks_by_id or {}

    # 1. Optionally draw trajectories for active tracks
    if draw_trajectory and tracks_by_id:
        for trk_id, trk in tracks_by_id.items():
            if trk.trajectory and len(trk.trajectory) > 1:
                color = get_color_for_id(trk_id)
                pts = np.array([[int(x), int(y)] for x, y in trk.trajectory], dtype=np.int32)
                cv2.polylines(annotated, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

    # 2. Draw detection boxes and labels
    for det in detections:
        x1, y1, x2, y2 = [int(round(coord)) for coord in det.bbox_xyxy]
        color = get_color_for_id(det.track_id or det.detection_id)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

        # Label text
        label_id = det.track_id if det.track_id else det.class_name
        label = f"{label_id} ({det.confidence:.2f})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Draw filled background for label
        bg_y1 = max(0, y1 - txt_h - baseline - 4)
        bg_y2 = y1
        bg_x2 = min(annotated.shape[1], x1 + txt_w + 6)
        cv2.rectangle(annotated, (x1, bg_y1), (bg_x2, bg_y2), color, -1)

        # Draw text in white or dark depending on background
        cv2.putText(
            annotated,
            label,
            (x1 + 3, bg_y2 - baseline - 1),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return annotated


@dataclass
class VideoProcessingResult:
    """Summary of processing for a single camera video or image source."""
    camera_id: str
    source_path: str
    total_frames: int
    fps: float
    duration_seconds: float
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    annotated_video_path: str | None = None
    preview_frames: list[np.ndarray] = field(default_factory=list)  # Sampled annotated frames
    track_crops: dict[str, np.ndarray] = field(default_factory=dict)  # Best crop per track_id
    track_embeddings: dict[str, np.ndarray] = field(default_factory=dict)  # Feature vector per track_id


class VideoProcessor:
    """Processes video/image sources: detection, tracking, feature extraction, annotation."""

    def __init__(
        self,
        config: AppConfig | None = None,
        detector: YOLODetector | None = None,
        encoder: PretrainedVisionEncoder | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.detector = detector or YOLODetector(self.config.detection)
        self.encoder = encoder or PretrainedVisionEncoder(self.config.representation)

    def process_source(
        self,
        source_path: str | Path,
        camera_id: str,
        max_frames: int = 150,
        frame_step: int = 1,
        conf_threshold: float | None = None,
        output_video_path: str | Path | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> VideoProcessingResult:
        """Run detection, within-camera tracking, and visual embedding on a media source."""
        source_str = str(source_path)
        cap = cv2.VideoCapture(source_str)

        if not cap.isOpened():
            # Check if it's a still image
            img = cv2.imread(source_str)
            if img is not None:
                return self._process_still_image(img, source_str, camera_id, conf_threshold)
            raise ValueError(f"Cannot open video or image source: {source_path}")

        total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames_in_file / fps if fps > 0 else 0.0

        if conf_threshold is not None:
            self.detector.config.confidence_threshold = conf_threshold

        tracker = WithinCameraTracker(
            camera_id=camera_id,
            iou_threshold=0.3,
            max_lost_frames=5,
        )

        all_detections: list[Detection] = []
        annotated_frames: list[np.ndarray] = []
        raw_crops_by_track: dict[str, list[tuple[float, np.ndarray]]] = {}  # track_id -> [(conf, crop)]

        # Video writer for annotated footage if requested
        writer = None
        if output_video_path:
            out_p = Path(output_video_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

        frame_idx = 0
        processed_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue

                if processed_count >= max_frames:
                    break

                timestamp = frame_idx / fps if fps > 0 else 0.0

                # 1. Detect
                frame_dets = self.detector.detect_frame(
                    frame=frame,
                    camera_id=camera_id,
                    frame_index=frame_idx,
                    timestamp_seconds=timestamp,
                )

                # 2. Track
                tracked_dets = tracker.update(frame_dets)
                all_detections.extend(tracked_dets)

                # 3. Extract crops for active tracks
                for det in tracked_dets:
                    crop = extract_crop_from_frame(
                        frame,
                        det.bbox_xyxy,
                        padding=self.config.representation.crop_padding,
                    )
                    t_id = det.track_id or det.detection_id
                    if t_id not in raw_crops_by_track:
                        raw_crops_by_track[t_id] = []
                    raw_crops_by_track[t_id].append((det.confidence, crop))

                # 4. Annotate
                annotated = annotate_frame(frame, tracked_dets)
                if writer:
                    writer.write(annotated)

                # Keep sample preview frames (at most 30 preview frames)
                if processed_count % max(1, (max_frames // 30)) == 0:
                    annotated_frames.append(annotated)

                processed_count += 1
                frame_idx += 1

                if progress_callback and max_frames > 0:
                    progress_callback(
                        min(0.9, processed_count / max_frames),
                        f"Processing {camera_id}: frame {processed_count}/{min(total_frames_in_file, max_frames)}"
                    )
        finally:
            cap.release()
            if writer:
                writer.release()

        # Finalize tracks
        tracks = tracker.finish()

        # Build best crop and visual embeddings per track
        track_crops: dict[str, np.ndarray] = {}
        track_embeddings: dict[str, np.ndarray] = {}

        for trk in tracks:
            t_id = trk.track_id
            crop_list = raw_crops_by_track.get(t_id, [])
            if crop_list:
                # Pick crop with highest confidence
                crop_list.sort(key=lambda item: item[0], reverse=True)
                best_crop = crop_list[0][1]
                track_crops[t_id] = best_crop

                # Extract visual embedding
                emb = self.encoder.extract_crop(best_crop)
                track_embeddings[t_id] = emb

        if progress_callback:
            progress_callback(1.0, f"Completed tracking for {camera_id} ({len(tracks)} tracks)")

        return VideoProcessingResult(
            camera_id=camera_id,
            source_path=source_str,
            total_frames=processed_count,
            fps=fps,
            duration_seconds=duration,
            detections=all_detections,
            tracks=tracks,
            annotated_video_path=str(output_video_path) if output_video_path else None,
            preview_frames=annotated_frames,
            track_crops=track_crops,
            track_embeddings=track_embeddings,
        )

    def _process_still_image(
        self,
        img: np.ndarray,
        source_str: str,
        camera_id: str,
        conf_threshold: float | None = None,
    ) -> VideoProcessingResult:
        """Handle single image input as a single-frame source."""
        if conf_threshold is not None:
            self.detector.config.confidence_threshold = conf_threshold

        tracker = WithinCameraTracker(camera_id=camera_id)
        dets = self.detector.detect_frame(img, camera_id=camera_id, frame_index=0, timestamp_seconds=0.0)
        tracked_dets = tracker.update(dets)
        tracks = tracker.finish()

        track_crops: dict[str, np.ndarray] = {}
        track_embeddings: dict[str, np.ndarray] = {}

        for det in tracked_dets:
            crop = extract_crop_from_frame(img, det.bbox_xyxy, padding=self.config.representation.crop_padding)
            t_id = det.track_id or det.detection_id
            track_crops[t_id] = crop
            track_embeddings[t_id] = self.encoder.extract_crop(crop)

        annotated = annotate_frame(img, tracked_dets)

        return VideoProcessingResult(
            camera_id=camera_id,
            source_path=source_str,
            total_frames=1,
            fps=1.0,
            duration_seconds=1.0,
            detections=tracked_dets,
            tracks=tracks,
            preview_frames=[annotated],
            track_crops=track_crops,
            track_embeddings=track_embeddings,
        )

    def match_cross_camera(
        self,
        result_a: VideoProcessingResult,
        result_b: VideoProcessingResult,
        top_k: int = 5,
        min_threshold: float = 0.35,
    ) -> tuple[dict[str, RetrievalResult], np.ndarray]:
        """Compute cross-camera similarity and candidate retrieval between tracks in Camera A and Camera B."""
        tracks_a = result_a.tracks
        tracks_b = result_b.tracks

        if not tracks_a or not tracks_b:
            return {}, np.empty((0, 0), dtype=np.float32)

        ids_a = [t.track_id for t in tracks_a if t.track_id in result_a.track_embeddings]
        embs_a = np.array([result_a.track_embeddings[tid] for tid in ids_a])

        ids_b = [t.track_id for t in tracks_b if t.track_id in result_b.track_embeddings]
        cams_b = [t.camera_id for t in tracks_b if t.track_id in result_b.track_embeddings]
        embs_b = np.array([result_b.track_embeddings[tid] for tid in ids_b])

        if len(embs_a) == 0 or len(embs_b) == 0:
            return {}, np.empty((0, 0), dtype=np.float32)

        similarity_matrix = compute_similarity_matrix(embs_a, embs_b)
        retrieval_map: dict[str, RetrievalResult] = {}

        for i, q_id in enumerate(ids_a):
            q_emb = embs_a[i]
            res = retrieve_candidates(
                query_id=q_id,
                query_camera_id=result_a.camera_id,
                query_embedding=q_emb,
                candidate_ids=ids_b,
                candidate_camera_ids=cams_b,
                candidate_embeddings=embs_b,
                top_k=top_k,
                min_threshold=min_threshold,
            )
            retrieval_map[q_id] = res

        return retrieval_map, similarity_matrix
