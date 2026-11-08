"""Unit tests for AGENTS.md Section 5 data contracts."""

from bicycle_reid.contracts.models import (
    CameraMetadata,
    Detection,
    Track,
    TrackRepresentation,
    EvidenceItem,
    CandidateReview,
)


def test_detection_contract_and_geometry():
    det = Detection(
        camera_id="CAM_01",
        frame_index=12,
        timestamp_seconds=0.48,
        bbox_xyxy=(10.0, 20.0, 110.0, 120.0),
        class_name="motorcycle",
        confidence=0.88,
    )
    assert det.width == 100.0
    assert det.height == 100.0
    assert det.area == 10000.0
    assert "CAM_01:f12:motorcycle" in det.detection_id
    d_dict = det.to_dict()
    assert d_dict["confidence"] == 0.88
    assert d_dict["class_name"] == "motorcycle"


def test_camera_metadata_unknown_offset():
    meta = CameraMetadata(
        camera_id="CAM_A",
        video_id="video_01.mp4",
        clock_offset_seconds=None,
        metadata_quality="UNSYNCHRONIZED",
    )
    assert meta.clock_offset_seconds is None
    assert meta.metadata_quality == "UNSYNCHRONIZED"


def test_candidate_review_honesty_principle():
    review = CandidateReview(
        source_track_id="CAM_A:track_01",
        candidate_track_id="CAM_B:track_05",
        visual_similarity=0.92,
        candidate_rank=1,
        candidate_margin=0.04,
        outcome="UNCERTAIN",
        confidence_label="LOW",
        calibrated_probability=None,  # Not calibrated yet!
        explanation="Candidate competition margin is small (0.04); requires verification.",
    )
    assert review.calibrated_probability is None
    assert review.visual_similarity == 0.92
    assert review.outcome == "UNCERTAIN"
