"""Abstract feature extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseFeatureExtractor(ABC):
    """Abstract interface for extracting embeddings from object crops."""

    @abstractmethod
    def extract_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Extract a 1D L2-normalized embedding vector for a single BGR image crop."""
        pass

    @abstractmethod
    def extract_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Extract a 2D array of L2-normalized embedding vectors (N, D) for a batch of crops."""
        pass
