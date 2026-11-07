"""YOLO detector implementation for bicycles and motorcycles."""

from __future__ import annotations

import logging
from typing import Sequence
import numpy as np
import torch
from ultralytics import YOLO

from bicycle_reid.contracts.models import Detection
from bicycle_reid.detection.base import BaseDetector
from bicycle_reid.config.settings import DetectionConfig

logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    """YOLO detector leveraging transfer learning without training."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        
        # Determine device
        if self.config.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        logger.info(f"Loading YOLO model {self.config.model_name} on device {self.device}")
        self.model = YOLO(self.config.model_name)
        
        # Build map from class name to class id
        self.target_class_ids: set[int] = set()
        for idx, name in self.model.names.items():
            if name.lower() in [cls.lower() for cls in self.config.target_classes]:
                self.target_class_ids.add(idx)

    def detect_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        frame_index: int = 0,
        timestamp_seconds: float | None = None,
    ) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            classes=list(self.target_class_ids) if self.target_class_ids else None,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confidences = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes_xyxy, confidences, cls_ids):
            class_name = self.model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            
            # Check bounding box area
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area = width * height
            if area < self.config.min_box_area:
                continue

            detection = Detection(
                camera_id=camera_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                bbox_xyxy=(x1, y1, x2, y2),
                class_name=class_name,
                confidence=float(conf),
            )
            detections.append(detection)

        return detections
