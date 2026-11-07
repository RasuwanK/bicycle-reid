"""Unit tests for media ingestion and metadata validation."""

from pathlib import Path
import cv2
import numpy as np
from bicycle_reid.ingestion.loader import MediaSource


def test_missing_file_metadata(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist.mp4"
    source = MediaSource(non_existent)
    assert not source.metadata.is_valid
    assert "File not found" in (source.metadata.error_message or "")


def test_valid_image_metadata(tmp_path: Path):
    test_img = np.full((120, 160, 3), 128, dtype=np.uint8)
    img_path = tmp_path / "test_frame.jpg"
    cv2.imwrite(str(img_path), test_img)

    source = MediaSource(img_path, camera_id="CAM_TEST")
    assert source.metadata.is_valid
    assert source.metadata.source_type == "image"
    assert source.metadata.width == 160
    assert source.metadata.height == 120
    assert source.metadata.total_frames == 1

    frames = list(source.frames())
    assert len(frames) == 1
    assert frames[0].camera_id == "CAM_TEST"
    assert frames[0].image.shape == (120, 160, 3)
