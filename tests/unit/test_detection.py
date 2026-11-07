"""Unit tests for detector filtering and interface."""

from unittest.mock import MagicMock
import numpy as np
from bicycle_reid.contracts.models import Detection
from bicycle_reid.detection.base import BaseDetector
from bicycle_reid.detection.yolo_detector import YOLODetector
from bicycle_reid.config.settings import DetectionConfig


class DummyDetector(BaseDetector):
    def detect_frame(self, frame, camera_id, frame_index=0, timestamp_seconds=None):
        return [
            Detection(
                camera_id=camera_id,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                bbox_xyxy=(10.0, 10.0, 60.0, 60.0),
                class_name="bicycle",
                confidence=0.91,
            )
        ]


def test_base_detector_interface():
    det = DummyDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    results = det.detect_frame(frame, camera_id="CAM_A", frame_index=2, timestamp_seconds=0.1)
    assert len(results) == 1
    assert results[0].class_name == "bicycle"
    assert results[0].camera_id == "CAM_A"
    assert results[0].frame_index == 2


def test_yolo_detector_empty_frame():
    cfg = DetectionConfig(model_name="yolov8n.pt", device="cpu")
    detector = YOLODetector(cfg)
    empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
    dets = detector.detect_frame(empty_frame, camera_id="CAM_A")
    assert dets == []
