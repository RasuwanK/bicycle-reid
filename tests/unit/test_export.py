"""Unit tests for CSV export functions."""

import io
import pandas as pd
from bicycle_reid.contracts.models import Detection, Track
from bicycle_reid.retrieval.similarity import RetrievalResult, MatchCandidate
from bicycle_reid.app.export import (
    detections_to_dataframe,
    tracks_to_dataframe,
    export_detections_csv,
    export_tracks_csv,
)


def test_detections_to_dataframe():
    det = Detection(
        camera_id="CAM_A",
        frame_index=5,
        timestamp_seconds=0.25,
        bbox_xyxy=(10.0, 20.0, 110.0, 220.0),
        class_name="bicycle",
        confidence=0.88,
        track_id="CAM_A:track_0001",
    )

    retrieval_map = {
        "CAM_A:track_0001": RetrievalResult(
            query_id="CAM_A:track_0001",
            query_camera_id="CAM_A",
            top_candidates=[
                MatchCandidate(candidate_id="CAM_B:track_0002", camera_id="CAM_B", similarity_score=0.85, rank=1)
            ],
            best_candidate_id="CAM_B:track_0002",
            best_score=0.85,
            second_best_score=None,
            margin=None,
        )
    }

    df = detections_to_dataframe([det], retrieval_map)
    assert len(df) == 1
    assert df.loc[0, "camera_id"] == "CAM_A"
    assert df.loc[0, "track_id"] == "CAM_A:track_0001"
    assert df.loc[0, "frame_index"] == 5
    assert df.loc[0, "matched_candidate_id"] == "CAM_B:track_0002"
    assert df.loc[0, "similarity_score"] == 0.85
    assert df.loc[0, "match_outcome"] == "ACCEPTED"


def test_export_csv_output():
    det = Detection(
        camera_id="CAM_B",
        frame_index=0,
        timestamp_seconds=0.0,
        bbox_xyxy=(5.0, 5.0, 50.0, 60.0),
        class_name="motorcycle",
        confidence=0.75,
        track_id="CAM_B:track_0001",
    )
    csv_str = export_detections_csv([det])
    lines = csv_str.strip().split("\n")
    assert len(lines) == 2  # Header + 1 row
    assert "detection_id,track_id,camera_id" in lines[0]
    assert "CAM_B:track_0001,CAM_B" in lines[1]
