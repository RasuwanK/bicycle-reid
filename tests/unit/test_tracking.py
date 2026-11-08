"""Unit tests for within-camera tracking."""

import pytest
from bicycle_reid.contracts.models import Detection
from bicycle_reid.tracking.tracker import WithinCameraTracker, compute_iou


def test_compute_iou():
    box_a = (0.0, 0.0, 10.0, 10.0)
    box_b = (0.0, 0.0, 10.0, 10.0)
    assert compute_iou(box_a, box_b) == pytest.approx(1.0)

    # Disjoint
    box_c = (20.0, 20.0, 30.0, 30.0)
    assert compute_iou(box_a, box_c) == 0.0

    # 50% overlap
    box_d = (5.0, 0.0, 15.0, 10.0)
    # intersection: 5 * 10 = 50. Union: 100 + 100 - 50 = 150. IoU = 50 / 150 = 1/3
    assert compute_iou(box_a, box_d) == pytest.approx(1.0 / 3.0)


def test_tracker_camera_scoped_ids():
    tracker = WithinCameraTracker(camera_id="CAM_X", iou_threshold=0.3, max_lost_frames=2)

    det_f0 = Detection(
        camera_id="CAM_X",
        frame_index=0,
        timestamp_seconds=0.0,
        bbox_xyxy=(10.0, 10.0, 50.0, 50.0),
        class_name="motorcycle",
        confidence=0.9,
    )

    tracked = tracker.update([det_f0])
    assert len(tracked) == 1
    assert tracked[0].track_id == "CAM_X:track_0001"

    # Frame 1: moved slightly, should match same track
    det_f1 = Detection(
        camera_id="CAM_X",
        frame_index=1,
        timestamp_seconds=0.1,
        bbox_xyxy=(12.0, 11.0, 52.0, 51.0),
        class_name="motorcycle",
        confidence=0.92,
    )
    tracked_1 = tracker.update([det_f1])
    assert len(tracked_1) == 1
    assert tracked_1[0].track_id == "CAM_X:track_0001"

    tracks = tracker.finish()
    assert len(tracks) == 1
    t = tracks[0]
    assert t.track_id == "CAM_X:track_0001"
    assert t.camera_id == "CAM_X"
    assert t.start_time_seconds == 0.0
    assert t.end_time_seconds == 0.1
    assert len(t.detections) == 2
    assert len(t.trajectory) == 2
    assert t.quality["detection_count"] == 2
