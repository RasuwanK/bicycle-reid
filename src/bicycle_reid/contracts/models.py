"""Data contracts conforming to AGENTS.md Section 5."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class CameraMetadata:
    camera_id: str
    video_id: str
    timezone: str | None = None
    clock_offset_seconds: float | None = None
    clock_offset_uncertainty_seconds: float | None = None
    location_description: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    viewing_direction: str | None = None
    metadata_quality: str = "PROVISIONAL"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Detection:
    camera_id: str
    frame_index: int
    timestamp_seconds: float | None
    bbox_xyxy: tuple[float, float, float, float]
    class_name: str
    confidence: float
    detection_id: str = ""

    def __post_init__(self) -> None:
        if not self.detection_id:
            x1, y1, x2, y2 = [int(round(coord)) for coord in self.bbox_xyxy]
            self.detection_id = f"{self.camera_id}:f{self.frame_index}:{self.class_name}_{x1}_{y1}_{x2}_{y2}"

    @property
    def width(self) -> float:
        return max(0.0, self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Track:
    track_id: str
    camera_id: str
    start_time_seconds: float | None
    end_time_seconds: float | None
    detections: list[Detection] = field(default_factory=list)
    representative_frame_refs: list[str] = field(default_factory=list)
    trajectory: list[tuple[float, float]] | None = None
    direction: str | None = None
    quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Alias for backward/domain compatibility with AGENTS.md
MotorcycleTrack = Track
BicycleTrack = Track


@dataclass
class TrackRepresentation:
    track_id: str
    embedding: list[float]
    frame_embeddings: list[list[float]] = field(default_factory=list)
    frame_quality: list[dict[str, Any]] = field(default_factory=list)
    aggregation_method: str = "mean"
    model_name: str = "resnet50"
    model_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    evidence_type: str
    state: str  # SUPPORTED | CONTRADICTED | UNKNOWN | WEAK
    observed_value: Any | None
    expected_value: Any | None
    reliability: str  # HIGH | MEDIUM | LOW | UNKNOWN
    explanation: str
    source_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateReview:
    source_track_id: str
    candidate_track_id: str | None
    visual_similarity: float | None
    candidate_rank: int | None
    candidate_margin: float | None
    evidence: list[EvidenceItem] = field(default_factory=list)
    outcome: str = "UNCERTAIN"  # ACCEPTED | UNCERTAIN | REJECTED | NO_MATCH
    confidence_label: str = "LOW"
    calibrated_probability: float | None = None
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
