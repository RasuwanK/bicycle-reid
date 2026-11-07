"""Unit tests for similarity computation, margin calculations, and candidate retrieval."""

import numpy as np
import pytest
from bicycle_reid.retrieval.similarity import (
    compute_cosine_similarity,
    compute_similarity_matrix,
    retrieve_candidates,
)


def test_cosine_similarity_identical_and_orthogonal():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    assert compute_cosine_similarity(v1, v2) == pytest.approx(1.0)
    assert compute_cosine_similarity(v1, v3) == pytest.approx(0.0)
    assert compute_cosine_similarity(v1, zero) == 0.0


def test_similarity_matrix_pairwise():
    mat_a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mat_b = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)

    sim = compute_similarity_matrix(mat_a, mat_b)
    assert sim.shape == (2, 3)
    assert sim[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert sim[0, 1] == pytest.approx(0.0, abs=1e-3)
    assert sim[0, 2] == pytest.approx(0.7071, abs=1e-3)


def test_candidate_retrieval_and_margin():
    q_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cand_embs = np.array(
        [
            [0.9, 0.1, 0.0],   # score ~0.99
            [0.7, 0.7, 0.0],   # score ~0.70
            [0.1, 0.9, 0.0],   # score ~0.11
        ],
        dtype=np.float32,
    )
    cand_ids = ["cand_1", "cand_2", "cand_3"]
    cand_cams = ["CAM_B", "CAM_B", "CAM_B"]

    res = retrieve_candidates(
        query_id="query_1",
        query_camera_id="CAM_A",
        query_embedding=q_emb,
        candidate_ids=cand_ids,
        candidate_camera_ids=cand_cams,
        candidate_embeddings=cand_embs,
        top_k=2,
    )

    assert len(res.top_candidates) == 2
    assert res.best_candidate_id == "cand_1"
    assert res.best_score is not None
    assert res.second_best_score is not None
    assert res.margin is not None
    assert res.margin == pytest.approx(res.best_score - res.second_best_score)
    assert res.margin > 0.10


def test_candidate_retrieval_empty_no_match():
    q_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    res = retrieve_candidates(
        query_id="query_1",
        query_camera_id="CAM_A",
        query_embedding=q_emb,
        candidate_ids=[],
        candidate_camera_ids=[],
        candidate_embeddings=np.empty((0, 3), dtype=np.float32),
    )
    assert res.best_candidate_id is None
    assert res.best_score is None
    assert res.margin is None
    assert len(res.top_candidates) == 0
