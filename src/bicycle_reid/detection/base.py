"""Abstract detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from bicycle_reid.contracts.models import Detection


class BaseDetector(ABC):
    """Abstract interface for object detectors."""

    @abstractmethod
    def detect_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        frame_index: int = 0,
        timestamp_seconds: float | None = None,
    ) -> list[Detection]:
        """Detect objects of target classes in a single image frame (BGR format)."""
        pass
