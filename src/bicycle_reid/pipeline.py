"""End-to-end MVP pipeline connecting detection, feature extraction, and candidate retrieval."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence
import numpy as np

from bicycle_reid.config.settings import AppConfig, load_default_config
from bicycle_reid.contracts.models import Detection
from bicycle_reid.ingestion.loader import MediaSource
from bicycle_reid.detection.yolo_detector import YOLODetector
from bicycle_reid.representation.vision_encoder import (
    PretrainedVisionEncoder,
    extract_crop_from_frame,
)
from bicycle_reid.retrieval.similarity import (
    compute_similarity_matrix,
    retrieve_candidates,
    RetrievalResult,
)
from bicycle_reid.reporting.console import (
    print_detection_summary,
    print_similarity_matrix_report,
    print_retrieval_results,
)

logger = logging.getLogger(__name__)


class BicycleReIDPipeline:
    """MVP Pipeline for bicycle detection, transfer learning embedding, and similarity scoring."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_default_config()
        self.detector = YOLODetector(self.config.detection)
        self.encoder = PretrainedVisionEncoder(self.config.representation)

    def process_sources(
        self,
        source_paths: Sequence[str | Path],
        camera_ids: Sequence[str] | None = None,
        max_frames_per_source: int = 10,
    ) -> dict[str, object]:
        """Process one or more image/video sources and compute bicycle similarity."""
        detections_by_source: dict[str, list[Detection]] = {}
        crops_by_source: dict[str, list[np.ndarray]] = {}
        embeddings_by_source: dict[str, np.ndarray] = {}

        # 1. Video Ingestion & Detection
        for i, src_path in enumerate(source_paths):
            cam_id = camera_ids[i] if (camera_ids and i < len(camera_ids)) else Path(src_path).stem
            media = MediaSource(src_path, camera_id=cam_id)
            if not media.metadata.is_valid:
                logger.error(f"Failed to load source {src_path}: {media.metadata.error_message}")
                detections_by_source[cam_id] = []
                crops_by_source[cam_id] = []
                embeddings_by_source[cam_id] = np.empty((0, self.encoder.embedding_dim), dtype=np.float32)
                continue

            all_dets: list[Detection] = []
            all_crops: list[np.ndarray] = []

            for frame_count, frame_info in enumerate(media.frames(frame_step=1)):
                if frame_count >= max_frames_per_source:
                    break
                frame_dets = self.detector.detect_frame(
                    frame=frame_info.image,
                    camera_id=cam_id,
                    frame_index=frame_info.frame_index,
                    timestamp_seconds=frame_info.timestamp_seconds,
                )
                for det in frame_dets:
                    crop = extract_crop_from_frame(
                        frame_info.image,
                        det.bbox_xyxy,
                        padding=self.config.representation.crop_padding,
                    )
                    all_dets.append(det)
                    all_crops.append(crop)

            detections_by_source[cam_id] = all_dets
            crops_by_source[cam_id] = all_crops

            # 2. Visual Feature Extraction (Transfer Learning)
            if all_crops:
                embeddings = self.encoder.extract_batch(all_crops)
            else:
                embeddings = np.empty((0, self.encoder.embedding_dim), dtype=np.float32)
            embeddings_by_source[cam_id] = embeddings

        # 3. Console Output: Detection Summary
        print_detection_summary(detections_by_source)

        # 4. Pairwise Comparison & Retrieval
        source_keys = list(detections_by_source.keys())
        all_detections_flat: list[Detection] = []
        all_embeddings_list: list[np.ndarray] = []

        for cam_id in source_keys:
            all_detections_flat.extend(detections_by_source[cam_id])
            if len(embeddings_by_source[cam_id]) > 0:
                all_embeddings_list.append(embeddings_by_source[cam_id])

        if all_embeddings_list:
            all_embeddings_matrix = np.vstack(all_embeddings_list)
        else:
            all_embeddings_matrix = np.empty((0, self.encoder.embedding_dim), dtype=np.float32)

        retrieval_results: list[RetrievalResult] = []
        similarity_matrix = np.empty((0, 0), dtype=np.float32)

        if len(all_detections_flat) >= 2:
            all_ids = [d.detection_id for d in all_detections_flat]
            similarity_matrix = compute_similarity_matrix(all_embeddings_matrix, all_embeddings_matrix)
            print_similarity_matrix_report(all_ids, all_ids, similarity_matrix)

            # If multiple sources, perform cross-source retrieval
            if len(source_keys) >= 2:
                src_cam_a = source_keys[0]
                src_cam_b = source_keys[1]
                dets_a = detections_by_source[src_cam_a]
                embs_a = embeddings_by_source[src_cam_a]
                dets_b = detections_by_source[src_cam_b]
                embs_b = embeddings_by_source[src_cam_b]

                if len(dets_a) > 0 and len(dets_b) > 0:
                    cand_ids_b = [d.detection_id for d in dets_b]
                    cand_cams_b = [d.camera_id for d in dets_b]

                    for q_idx, q_det in enumerate(dets_a):
                        q_emb = embs_a[q_idx]
                        res = retrieve_candidates(
                            query_id=q_det.detection_id,
                            query_camera_id=q_det.camera_id,
                            query_embedding=q_emb,
                            candidate_ids=cand_ids_b,
                            candidate_camera_ids=cand_cams_b,
                            candidate_embeddings=embs_b,
                            top_k=self.config.retrieval.top_k,
                            min_threshold=self.config.retrieval.min_similarity_threshold,
                        )
                        retrieval_results.append(res)

                    print_retrieval_results(
                        retrieval_results,
                        margin_threshold=self.config.retrieval.competition_margin_threshold,
                    )

        return {
            "detections_by_source": detections_by_source,
            "embeddings_by_source": embeddings_by_source,
            "similarity_matrix": similarity_matrix,
            "retrieval_results": retrieval_results,
        }
