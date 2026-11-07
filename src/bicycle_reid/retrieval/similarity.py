"""Cross-source visual candidate retrieval and similarity computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class MatchCandidate:
    candidate_id: str
    camera_id: str
    similarity_score: float
    rank: int


@dataclass
class RetrievalResult:
    query_id: str
    query_camera_id: str
    top_candidates: list[MatchCandidate]
    best_candidate_id: str | None
    best_score: float | None
    second_best_score: float | None
    margin: float | None


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_similarity_matrix(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two batches of embeddings.
    
    If embeddings are already L2-normalized, matrix is simply A @ B.T.
    Otherwise, safe normalization is applied.
    """
    if len(embeddings_a) == 0 or len(embeddings_b) == 0:
        return np.empty((len(embeddings_a), len(embeddings_b)), dtype=np.float32)

    norm_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    norm_a[norm_a == 0] = 1e-8
    norm_b[norm_b == 0] = 1e-8

    a_normed = embeddings_a / norm_a
    b_normed = embeddings_b / norm_b

    sim_matrix = np.dot(a_normed, b_normed.T)
    # Clip numerical precision drift
    return np.clip(sim_matrix, -1.0, 1.0)


def retrieve_candidates(
    query_id: str,
    query_camera_id: str,
    query_embedding: np.ndarray,
    candidate_ids: Sequence[str],
    candidate_camera_ids: Sequence[str],
    candidate_embeddings: np.ndarray,
    top_k: int = 5,
    min_threshold: float = 0.0,
) -> RetrievalResult:
    """Rank candidate items by cosine similarity against the query embedding."""
    if len(candidate_embeddings) == 0 or len(candidate_ids) == 0:
        return RetrievalResult(
            query_id=query_id,
            query_camera_id=query_camera_id,
            top_candidates=[],
            best_candidate_id=None,
            best_score=None,
            second_best_score=None,
            margin=None,
        )

    # Compute similarities
    scores = compute_similarity_matrix(
        np.expand_dims(query_embedding, axis=0), candidate_embeddings
    )[0]

    # Sort in descending order
    sorted_indices = np.argsort(-scores)

    ranked_candidates: list[MatchCandidate] = []
    for rank_idx, idx in enumerate(sorted_indices, start=1):
        score = float(scores[idx])
        if score < min_threshold and rank_idx > 1:
            continue
        ranked_candidates.append(
            MatchCandidate(
                candidate_id=candidate_ids[idx],
                camera_id=candidate_camera_ids[idx],
                similarity_score=score,
                rank=rank_idx,
            )
        )
        if len(ranked_candidates) >= top_k:
            break

    best_candidate_id = ranked_candidates[0].candidate_id if ranked_candidates else None
    best_score = ranked_candidates[0].similarity_score if ranked_candidates else None
    second_best_score = (
        ranked_candidates[1].similarity_score if len(ranked_candidates) > 1 else None
    )
    margin = (
        (best_score - second_best_score)
        if (best_score is not None and second_best_score is not None)
        else None
    )

    return RetrievalResult(
        query_id=query_id,
        query_camera_id=query_camera_id,
        top_candidates=ranked_candidates,
        best_candidate_id=best_candidate_id,
        best_score=best_score,
        second_best_score=second_best_score,
        margin=margin,
    )
