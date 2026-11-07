"""Within-camera tracking implementation with camera-scoped IDs and quality scoring.

Conforms to AGENTS.md Section 4.3 & Section 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

from bicycle_reid.contracts.models import Detection, Track

logger = logging.getLogger(__name__)


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    box_a_area = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    box_b_area = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


@dataclass
class TrackState:
    """Internal tracking state for within-camera tracking."""
    track_id: str
    camera_id: str
    detections: list[Detection] = field(default_factory=list)
    lost_frames: int = 0
    is_active: bool = True

    @property
    def latest_bbox(self) -> tuple[float, float, float, float]:
        if not self.detections:
            return (0.0, 0.0, 0.0, 0.0)
        return self.detections[-1].bbox_xyxy

    @property
    def start_time(self) -> float | None:
        return self.detections[0].timestamp_seconds if self.detections else None

    @property
    def end_time(self) -> float | None:
        return self.detections[-1].timestamp_seconds if self.detections else None

    def add_detection(self, detection: Detection) -> None:
        detection.track_id = self.track_id
        self.detections.append(detection)
        self.lost_frames = 0


class WithinCameraTracker:
    """Within-camera multi-object tracker for bicycles and motorcycles.
    
    Generates deterministic, camera-scoped track IDs (e.g. CAM_A:track_0001)
    and produces Track contract objects with trajectory and quality metrics.
    """

    def __init__(
        self,
        camera_id: str,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 5,
        min_hits: int = 1,
    ) -> None:
        self.camera_id = camera_id
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_hits = min_hits
        
        self._next_track_num = 1
        self._active_tracks: list[TrackState] = []
        self._finished_tracks: list[TrackState] = []

    def update(self, detections: Sequence[Detection]) -> list[Detection]:
        """Update tracks with a new frame's detections and assign track IDs.
        
        Returns the list of detections with their track_id assigned.
        """
        # If no active tracks, all detections start new tracks
        if not self._active_tracks:
            for det in detections:
                new_track = self._create_new_track(det)
                self._active_tracks.append(new_track)
            return list(detections)

        if not detections:
            # All active tracks lose a frame
            for trk in self._active_tracks:
                trk.lost_frames += 1
            self._prune_lost_tracks()
            return []

        # Build IoU cost matrix
        num_tracks = len(self._active_tracks)
        num_dets = len(detections)
        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for t_idx, trk in enumerate(self._active_tracks):
            for d_idx, det in enumerate(detections):
                iou_matrix[t_idx, d_idx] = compute_iou(trk.latest_bbox, det.bbox_xyxy)

        # Greedy bipartite matching
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if iou_matrix.size > 0:
            flat_indices = np.argsort(-iou_matrix, axis=None)
            for idx in flat_indices:
                t_idx, d_idx = np.unravel_index(idx, iou_matrix.shape)
                if t_idx in matched_tracks or d_idx in matched_dets:
                    continue
                if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                    self._active_tracks[t_idx].add_detection(detections[d_idx])
                    matched_tracks.add(t_idx)
                    matched_dets.add(d_idx)

        # Handle unmatched active tracks
        for t_idx, trk in enumerate(self._active_tracks):
            if t_idx not in matched_tracks:
                trk.lost_frames += 1

        # Handle unmatched detections (start new tracks)
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_track = self._create_new_track(det)
                self._active_tracks.append(new_track)

        self._prune_lost_tracks()
        return list(detections)

    def _create_new_track(self, detection: Detection) -> TrackState:
        track_id = f"{self.camera_id}:track_{self._next_track_num:04d}"
        self._next_track_num += 1
        track = TrackState(track_id=track_id, camera_id=self.camera_id)
        track.add_detection(detection)
        return track

    def _prune_lost_tracks(self) -> None:
        retained: list[TrackState] = []
        for trk in self._active_tracks:
            if trk.lost_frames > self.max_lost_frames:
                trk.is_active = False
                self._finished_tracks.append(trk)
            else:
                retained.append(trk)
        self._active_tracks = retained

    def finish(self) -> list[Track]:
        """Finalize tracking and return all tracked objects as Track contracts."""
        all_states = self._finished_tracks + self._active_tracks
        self._finished_tracks = []
        self._active_tracks = []

        tracks: list[Track] = []
        for state in all_states:
            if len(state.detections) < self.min_hits:
                continue

            # Compute trajectory centroids
            trajectory: list[tuple[float, float]] = []
            confidences: list[float] = []
            areas: list[float] = []

            best_det_idx = 0
            best_det_score = -1.0

            for i, det in enumerate(state.detections):
                x1, y1, x2, y2 = det.bbox_xyxy
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                trajectory.append((cx, cy))
                confidences.append(det.confidence)
                areas.append(det.area)

                score = det.confidence * np.sqrt(max(1.0, det.area))
                if score > best_det_score:
                    best_det_score = score
                    best_det_idx = i

            # Estimate movement direction
            direction = self._estimate_direction(trajectory)

            # Quality metrics
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            avg_area = float(np.mean(areas)) if areas else 0.0
            duration = (
                (state.end_time - state.start_time)
                if (state.start_time is not None and state.end_time is not None)
                else 0.0
            )

            quality = {
                "detection_count": len(state.detections),
                "mean_confidence": round(avg_conf, 3),
                "max_confidence": round(float(np.max(confidences)), 3) if confidences else 0.0,
                "mean_area": round(avg_area, 1),
                "duration_seconds": round(duration, 2),
                "is_fragmented": len(state.detections) < 3,
                "best_frame_index": state.detections[best_det_idx].frame_index,
            }

            rep_ref = f"frame_{state.detections[best_det_idx].frame_index}"

            track = Track(
                track_id=state.track_id,
                camera_id=self.camera_id,
                start_time_seconds=state.start_time,
                end_time_seconds=state.end_time,
                detections=state.detections,
                representative_frame_refs=[rep_ref],
                trajectory=trajectory,
                direction=direction,
                quality=quality,
            )
            tracks.append(track)

        return tracks

    @staticmethod
    def _estimate_direction(trajectory: list[tuple[float, float]]) -> str:
        if len(trajectory) < 2:
            return "STATIONARY"

        dx = trajectory[-1][0] - trajectory[0][0]
        dy = trajectory[-1][1] - trajectory[0][1]
        dist = np.hypot(dx, dy)

        if dist < 20.0:
            return "STATIONARY"

        if abs(dx) > abs(dy):
            return "LEFT_TO_RIGHT" if dx > 0 else "RIGHT_TO_LEFT"
        else:
            return "TOP_TO_BOTTOM" if dy > 0 else "BOTTOM_TO_TOP"
