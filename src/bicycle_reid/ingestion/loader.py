"""Media ingestion module for reading images and video frames."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator
import cv2
import numpy as np


@dataclass
class FrameInfo:
    camera_id: str
    frame_index: int
    timestamp_seconds: float | None
    image: np.ndarray  # BGR


@dataclass
class SourceMetadata:
    source_path: str
    source_type: str  # "image" or "video"
    width: int
    height: int
    total_frames: int
    fps: float
    duration_seconds: float | None
    is_valid: bool
    error_message: str | None = None


class MediaSource:
    """Handles loading either a still image or a video stream with frame metadata."""

    def __init__(self, source_path: str | Path, camera_id: str | None = None):
        self.source_path = Path(source_path)
        self.camera_id = camera_id or self.source_path.stem
        self.metadata = self._probe()

    def _probe(self) -> SourceMetadata:
        if not self.source_path.exists():
            return SourceMetadata(
                source_path=str(self.source_path),
                source_type="unknown",
                width=0,
                height=0,
                total_frames=0,
                fps=0.0,
                duration_seconds=None,
                is_valid=False,
                error_message=f"File not found: {self.source_path}",
            )

        suffix = self.source_path.suffix.lower()
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if suffix in image_extensions:
            img = cv2.imread(str(self.source_path))
            if img is None:
                return SourceMetadata(
                    source_path=str(self.source_path),
                    source_type="image",
                    width=0,
                    height=0,
                    total_frames=0,
                    fps=0.0,
                    duration_seconds=None,
                    is_valid=False,
                    error_message=f"Could not decode image at {self.source_path}",
                )
            h, w = img.shape[:2]
            return SourceMetadata(
                source_path=str(self.source_path),
                source_type="image",
                width=w,
                height=h,
                total_frames=1,
                fps=0.0,
                duration_seconds=0.0,
                is_valid=True,
            )

        # Try video capture
        cap = cv2.VideoCapture(str(self.source_path))
        if not cap.isOpened():
            return SourceMetadata(
                source_path=str(self.source_path),
                source_type="video",
                width=0,
                height=0,
                total_frames=0,
                fps=0.0,
                duration_seconds=None,
                is_valid=False,
                error_message=f"Could not open video file at {self.source_path}",
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if (fps > 0 and total_frames > 0) else None
        cap.release()

        return SourceMetadata(
            source_path=str(self.source_path),
            source_type="video",
            width=width,
            height=height,
            total_frames=total_frames,
            fps=fps,
            duration_seconds=duration,
            is_valid=True,
        )

    def frames(self, frame_step: int = 1) -> Generator[FrameInfo, None, None]:
        if not self.metadata.is_valid:
            raise RuntimeError(self.metadata.error_message or "Invalid media source")

        if self.metadata.source_type == "image":
            img = cv2.imread(str(self.source_path))
            if img is not None:
                yield FrameInfo(
                    camera_id=self.camera_id,
                    frame_index=0,
                    timestamp_seconds=0.0,
                    image=img,
                )
            return

        cap = cv2.VideoCapture(str(self.source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.source_path}")

        frame_idx = 0
        fps = self.metadata.fps if self.metadata.fps > 0 else 30.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    timestamp = frame_idx / fps
                    yield FrameInfo(
                        camera_id=self.camera_id,
                        frame_index=frame_idx,
                        timestamp_seconds=timestamp,
                        image=frame,
                    )
                frame_idx += 1
        finally:
            cap.release()
