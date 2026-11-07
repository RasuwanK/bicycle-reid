"""Unit tests for visual feature extraction and crop handling."""

import numpy as np
import pytest
from bicycle_reid.representation.vision_encoder import (
    extract_crop_from_frame,
    PretrainedVisionEncoder,
)
from bicycle_reid.config.settings import RepresentationConfig


def test_extract_crop_padding_and_bounds():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Box from (10, 10) to (50, 50), width 40, height 40
    # 10% padding = 4 pixels on each side -> (6, 6) to (54, 54) -> shape 48x48
    crop = extract_crop_from_frame(frame, (10.0, 10.0, 50.0, 50.0), padding=0.10)
    assert crop.shape[0] == 48
    assert crop.shape[1] == 48


def test_vision_encoder_embedding_shape_and_norm():
    # Use mobilenet_v3_small for fast unit test
    cfg = RepresentationConfig(
        model_name="mobilenet_v3_small",
        weights="DEFAULT",
        device="cpu",
    )
    encoder = PretrainedVisionEncoder(cfg)
    crop = np.full((64, 64, 3), 150, dtype=np.uint8)

    emb = encoder.extract_crop(crop)
    assert emb.shape == (576,)
    norm = np.linalg.norm(emb)
    assert norm == pytest.approx(1.0, abs=1e-4)

    # Batch extraction
    batch_embs = encoder.extract_batch([crop, crop])
    assert batch_embs.shape == (2, 576)
    assert np.linalg.norm(batch_embs[0]) == pytest.approx(1.0, abs=1e-4)
