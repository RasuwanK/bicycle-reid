"""Configuration models and loader for bicycle-reid."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.30
    target_classes: list[str] = field(default_factory=lambda: ["bicycle", "motorcycle"])
    primary_class: str = "bicycle"
    device: str = "auto"
    min_box_area: float = 400.0  # filter out tiny noise artifacts


@dataclass
class RepresentationConfig:
    model_name: str = "resnet50"
    weights: str = "DEFAULT"
    embedding_dim: int = 2048
    device: str = "auto"
    crop_padding: float = 0.05


@dataclass
class RetrievalConfig:
    top_k: int = 5
    min_similarity_threshold: float = 0.35
    competition_margin_threshold: float = 0.05


@dataclass
class AppConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        det_data = data.get("detection", {})
        rep_data = data.get("representation", {})
        ret_data = data.get("retrieval", {})

        return cls(
            detection=DetectionConfig(**det_data),
            representation=RepresentationConfig(**rep_data),
            retrieval=RetrievalConfig(**ret_data),
        )


def load_default_config(config_path: str | Path | None = None) -> AppConfig:
    if config_path:
        return AppConfig.from_yaml(config_path)

    # Check default locations
    default_candidates = [
        Path("configs/default.yaml"),
        Path(__file__).resolve().parent.parent.parent.parent / "configs" / "default.yaml",
    ]
    for candidate in default_candidates:
        if candidate.exists():
            return AppConfig.from_yaml(candidate)

    return AppConfig()
