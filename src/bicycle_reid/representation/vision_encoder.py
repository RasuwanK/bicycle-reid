"""Deep visual feature extractor utilizing pretrained vision encoders."""

from __future__ import annotations

import logging
from typing import Sequence
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from bicycle_reid.representation.base import BaseFeatureExtractor
from bicycle_reid.config.settings import RepresentationConfig

logger = logging.getLogger(__name__)


def extract_crop_from_frame(
    frame: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    padding: float = 0.05,
) -> np.ndarray:
    """Extract a cropped region from a frame with optional contextual padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = box_w * padding
    pad_y = box_h * padding

    x1_pad = int(max(0, round(x1 - pad_x)))
    y1_pad = int(max(0, round(y1 - pad_y)))
    x2_pad = int(min(w, round(x2 + pad_x)))
    y2_pad = int(min(h, round(y2 + pad_y)))

    crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
    if crop.size == 0:
        # Fallback to zero image
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
    return crop


class PretrainedVisionEncoder(BaseFeatureExtractor):
    """Transfer learning feature extractor using torchvision pretrained backbones."""

    def __init__(self, config: RepresentationConfig | None = None) -> None:
        self.config = config or RepresentationConfig()

        if self.config.device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        logger.info(
            f"Initializing {self.config.model_name} feature extractor on device {self.device}"
        )
        self.model, self.embedding_dim = self._build_model()
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _build_model(self) -> tuple[nn.Module, int]:
        model_name = self.config.model_name.lower()
        if "resnet50" in model_name:
            weights = models.ResNet50_Weights.DEFAULT if self.config.weights == "DEFAULT" else None
            m = models.resnet50(weights=weights)
            # Remove classification head, keep avgpool
            modules = list(m.children())[:-1]
            feature_extractor = nn.Sequential(*modules, nn.Flatten())
            return feature_extractor, 2048

        elif "mobilenet" in model_name:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if self.config.weights == "DEFAULT" else None
            m = models.mobilenet_v3_small(weights=weights)
            feature_extractor = nn.Sequential(
                m.features,
                m.avgpool,
                nn.Flatten(),
            )
            return feature_extractor, 576

        else:
            raise ValueError(f"Unsupported model architecture: {self.config.model_name}")

    def _preprocess_crop(self, crop_bgr: np.ndarray) -> torch.Tensor:
        if crop_bgr is None or crop_bgr.size == 0:
            crop_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        return self.transform(pil_img)

    @torch.no_grad()
    def extract_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        tensor = self._preprocess_crop(crop_bgr).unsqueeze(0).to(self.device)
        feats = self.model(tensor)
        # L2-normalize
        feats = nn.functional.normalize(feats, p=2, dim=1)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        if not crops_bgr:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        tensors = [self._preprocess_crop(c) for c in crops_bgr]
        batch = torch.stack(tensors, dim=0).to(self.device)
        feats = self.model(batch)
        feats = nn.functional.normalize(feats, p=2, dim=1)
        return feats.cpu().numpy().astype(np.float32)
