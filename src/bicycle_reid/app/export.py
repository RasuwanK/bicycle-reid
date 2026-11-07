"""CSV Export utilities for detection and tracking status.

Conforms to user requirements for exporting current detection status to CSV formats.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Sequence
import pandas as pd

from bicycle_reid.contracts.models import Detection, Track
from bicycle_reid.retrieval.similarity import RetrievalResult


def detections_to_dataframe(
    detections: Sequence[Detection],
    retrieval_map: dict[str, RetrievalResult] | None = None,
) -> pd.DataFrame:
    """Convert a sequence of Detection objects into a pandas DataFrame."""
    retrieval_map = retrieval_map or {}
    rows = []

    for det in detections:
        # Check if this detection or its track has a retrieval result
        match_info = retrieval_map.get(det.detection_id)
        if not match_info and det.track_id:
            match_info = retrieval_map.get(det.track_id)

        top_cand_id = ""
        sim_score = None
        margin = None
        outcome = "NOT_EVALUATED"

        if match_info and match_info.top_candidates:
            top_cand = match_info.top_candidates[0]
            top_cand_id = top_cand.candidate_id
            sim_score = round(top_cand.similarity_score, 4)
            margin = round(match_info.margin, 4) if match_info.margin is not None else None
            
            # Simple provisional outcome based on AGENTS.md rules
            if sim_score >= 0.75 and (margin is None or margin >= 0.05):
                outcome = "ACCEPTED"
            elif sim_score >= 0.40:
                outcome = "UNCERTAIN"
            else:
                outcome = "NO_MATCH"

        x1, y1, x2, y2 = det.bbox_xyxy
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)

        row = {
            "detection_id": det.detection_id,
            "track_id": det.track_id or "UNTRACKED",
            "camera_id": det.camera_id,
            "frame_index": det.frame_index,
            "timestamp_seconds": round(det.timestamp_seconds, 3) if det.timestamp_seconds is not None else None,
            "class_name": det.class_name,
            "confidence": round(det.confidence, 4),
            "bbox_x1": round(x1, 1),
            "bbox_y1": round(y1, 1),
            "bbox_x2": round(x2, 1),
            "bbox_y2": round(y2, 1),
            "bbox_width": round(w, 1),
            "bbox_height": round(h, 1),
            "bbox_area": round(w * h, 1),
            "matched_candidate_id": top_cand_id,
            "similarity_score": sim_score,
            "candidate_margin": margin,
            "match_outcome": outcome,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "detection_id", "track_id", "camera_id", "frame_index",
            "timestamp_seconds", "class_name", "confidence",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "bbox_width", "bbox_height", "bbox_area",
            "matched_candidate_id", "similarity_score", "candidate_margin", "match_outcome"
        ])

    return pd.DataFrame(rows)


def tracks_to_dataframe(
    tracks: Sequence[Track],
    retrieval_map: dict[str, RetrievalResult] | None = None,
) -> pd.DataFrame:
    """Convert a sequence of Track objects into a summary pandas DataFrame."""
    retrieval_map = retrieval_map or {}
    rows = []

    for trk in tracks:
        match_info = retrieval_map.get(trk.track_id)
        top_cand_id = ""
        sim_score = None
        margin = None
        outcome = "NOT_EVALUATED"

        if match_info and match_info.top_candidates:
            top_cand = match_info.top_candidates[0]
            top_cand_id = top_cand.candidate_id
            sim_score = round(top_cand.similarity_score, 4)
            margin = round(match_info.margin, 4) if match_info.margin is not None else None
            if sim_score >= 0.75 and (margin is None or margin >= 0.05):
                outcome = "ACCEPTED"
            elif sim_score >= 0.40:
                outcome = "UNCERTAIN"
            else:
                outcome = "NO_MATCH"

        quality = trk.quality or {}
        row = {
            "track_id": trk.track_id,
            "camera_id": trk.camera_id,
            "detection_count": len(trk.detections),
            "start_time_seconds": round(trk.start_time_seconds, 3) if trk.start_time_seconds is not None else None,
            "end_time_seconds": round(trk.end_time_seconds, 3) if trk.end_time_seconds is not None else None,
            "duration_seconds": quality.get("duration_seconds", 0.0),
            "mean_confidence": quality.get("mean_confidence", 0.0),
            "direction": trk.direction or "UNKNOWN",
            "is_fragmented": quality.get("is_fragmented", False),
            "matched_candidate_id": top_cand_id,
            "similarity_score": sim_score,
            "candidate_margin": margin,
            "match_outcome": outcome,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "track_id", "camera_id", "detection_count", "start_time_seconds",
            "end_time_seconds", "duration_seconds", "mean_confidence",
            "direction", "is_fragmented", "matched_candidate_id",
            "similarity_score", "candidate_margin", "match_outcome"
        ])

    return pd.DataFrame(rows)


def export_detections_csv(
    detections: Sequence[Detection],
    retrieval_map: dict[str, RetrievalResult] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Generate CSV string of detection status, optionally saving to disk."""
    df = detections_to_dataframe(detections, retrieval_map)
    csv_string = df.to_csv(index=False)
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(csv_string, encoding="utf-8")
    return csv_string


def export_tracks_csv(
    tracks: Sequence[Track],
    retrieval_map: dict[str, RetrievalResult] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Generate CSV string of track summary status, optionally saving to disk."""
    df = tracks_to_dataframe(tracks, retrieval_map)
    csv_string = df.to_csv(index=False)
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(csv_string, encoding="utf-8")
    return csv_string
