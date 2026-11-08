"""Command-line interface for Motorcycle Re-Identification MVP."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import cv2
import numpy as np

from bicycle_reid.config.settings import load_default_config, AppConfig
from bicycle_reid.pipeline import MotorcycleReIDPipeline


def create_demo_assets(output_dir: Path) -> tuple[Path, Path]:
    """Ensure two demo test frames exist containing motorcycles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    img_a_path = output_dir / "camera_a_sample.jpg"
    img_b_path = output_dir / "camera_b_sample.jpg"

    if img_a_path.exists() and img_b_path.exists():
        return img_a_path, img_b_path

    # Try downloading darknet sample or creating simulated camera views
    import urllib.request
    try:
        url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/dog.jpg"
        urllib.request.urlretrieve(url, str(img_a_path))
        img = cv2.imread(str(img_a_path))
        if img is not None:
            h, w = img.shape[:2]
            pts1 = np.float32([[50, 50], [w - 50, 50], [50, h - 50], [w - 50, h - 50]])
            pts2 = np.float32([[30, 70], [w - 70, 40], [60, h - 30], [w - 40, h - 60]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            cam_b = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            cam_b = cv2.convertScaleAbs(cam_b, alpha=0.92, beta=12)
            cv2.imwrite(str(img_b_path), cam_b)
            return img_a_path, img_b_path
    except Exception as exc:
        logging.warning(f"Could not download sample imagery: {exc}. Generating synthetic frames.")

    # Synthetic fallback
    frame_a = np.full((480, 640, 3), 200, dtype=np.uint8)
    frame_b = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(img_a_path), frame_a)
    cv2.imwrite(str(img_b_path), frame_b)
    return img_a_path, img_b_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-Camera Motorcycle Re-Identification MVP (Transfer Learning Baseline)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--source-a",
        type=str,
        default=None,
        help="Path to Camera A image or video file.",
    )
    parser.add_argument(
        "--source-b",
        type=str,
        default=None,
        help="Path to Camera B image or video file.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[],
        help="List of media sources to analyze.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run MVP on sample demo imagery.",
    )
    parser.add_argument(
        "--ui",
        "--web",
        action="store_true",
        dest="web_ui",
        help="Launch the interactive Streamlit Web Interface.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold (default from config).",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["resnet50", "mobilenet_v3_small"],
        default=None,
        help="Visual feature encoder backbone.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.web_ui:
        import subprocess
        app_file = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
        print(f"[*] Starting Streamlit Web UI at {app_file}...")
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)]
        return subprocess.call(cmd)

    config = load_default_config(args.config)
    if args.conf is not None:
        config.detection.confidence_threshold = args.conf
    if args.encoder is not None:
        config.representation.model_name = args.encoder

    sources: list[str | Path] = []
    camera_ids: list[str] = []

    if args.source_a:
        sources.append(args.source_a)
        camera_ids.append("CAM_A")
    if args.source_b:
        sources.append(args.source_b)
        camera_ids.append("CAM_B")
    if args.sources:
        for s in args.sources:
            sources.append(s)
            camera_ids.append(Path(s).stem)

    if not sources or args.demo:
        cctv_a = Path("data/demo/camera_a_cctv.mp4")
        cctv_b = Path("data/demo/camera_b_cctv.mp4")
        if cctv_a.exists() and cctv_b.exists():
            sources = [cctv_a, cctv_b]
            camera_ids = ["CAM_A", "CAM_B"]
        else:
            demo_dir = Path("data/demo")
            print(f"[*] No inputs provided or --demo requested. Generating sample test assets in {demo_dir}...")
            cam_a_file, cam_b_file = create_demo_assets(demo_dir)
            sources = [cam_a_file, cam_b_file]
            camera_ids = ["CAM_A", "CAM_B"]

    print(f"[*] Initializing Motorcycle Re-ID MVP Pipeline...")
    pipeline = MotorcycleReIDPipeline(config)

    print(f"[*] Processing {len(sources)} source(s)...")
    pipeline.process_sources(sources, camera_ids=camera_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
