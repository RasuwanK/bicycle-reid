"""Console reporting and evidence presentation conforming to AGENTS.md principles."""

from __future__ import annotations

from typing import Sequence
from bicycle_reid.contracts.models import Detection
from bicycle_reid.retrieval.similarity import RetrievalResult


def print_banner(title: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {title.upper()}")
    print(f"{sep}")


def print_detection_summary(detections_by_source: dict[str, list[Detection]]) -> None:
    print_banner("Motorcycle Detection Summary")
    total_dets = sum(len(d) for d in detections_by_source.values())
    print(f"Total sources analyzed: {len(detections_by_source)}")
    print(f"Total motorcycle detections: {total_dets}\n")

    for src_id, dets in detections_by_source.items():
        print(f"▸ Source: [{src_id}] — ({len(dets)} detection{'s' if len(dets) != 1 else ''})")
        if not dets:
            print("    (No targets detected)")
            continue
        for i, d in enumerate(dets, start=1):
            x1, y1, x2, y2 = [int(round(c)) for c in d.bbox_xyxy]
            print(
                f"    [{i}] ID: {d.detection_id:<28} | Class: {d.class_name:<10} "
                f"| Conf: {d.confidence:.1%} | BBox: [{x1}, {y1}, {x2}, {y2}] ({d.width:.0f}x{d.height:.0f})"
            )
    print()


def print_similarity_matrix_report(
    item_ids_a: Sequence[str],
    item_ids_b: Sequence[str],
    sim_matrix: Sequence[Sequence[float]],
) -> None:
    print_banner("Cross-Detection Visual Similarity Matrix")
    print("NOTE: Values represent raw Cosine Similarity [-1.0, 1.0] in visual embedding space.")
    print("      Raw similarity is a visual feature metric, NOT a calibrated identity probability.\n")

    # Header
    col_width = 16
    header = f"{'Query / Candidate':<30} | " + " | ".join(f"{id_b[:col_width]:>{col_width}}" for id_b in item_ids_b)
    print(header)
    print("-" * len(header))

    for id_a, row in zip(item_ids_a, sim_matrix):
        row_str = " | ".join(f"{score:>16.4f}" for score in row)
        print(f"{id_a[:30]:<30} | {row_str}")
    print()


def print_retrieval_results(
    results: list[RetrievalResult],
    margin_threshold: float = 0.05,
) -> None:
    print_banner("Cross-Source Candidate Retrieval & Review")
    
    for res in results:
        print(f"▸ Query Detection: {res.query_id} (Camera: {res.query_camera_id})")
        if not res.top_candidates:
            print("    Outcome: NO_MATCH (No candidates available in target source)\n")
            continue

        print("    Top Candidate Matches (Ranked by Pretrained Vision Encoder Similarity):")
        for cand in res.top_candidates:
            print(
                f"      Rank {cand.rank}: {cand.candidate_id:<28} "
                f"(Source: {cand.camera_id}) -> Visual Similarity: {cand.similarity_score:.4f}"
            )

        # Competition analysis
        if res.margin is not None:
            print(f"    Candidate Margin (Rank 1 vs Rank 2): {res.margin:.4f}")
            if res.margin < margin_threshold:
                print(
                    f"    ⚠️  AMBIGUITY WARNING: Margin ({res.margin:.4f}) is below threshold ({margin_threshold:.4f}). "
                    "Close candidate competition suggests potential identity confusion."
                )
            else:
                print(f"    ✓ Clear visual candidate separation (Margin: {res.margin:.4f})")
        else:
            print("    (Only one candidate available; no competitor margin)")

        # Honest confidence disclosure
        best = res.top_candidates[0]
        outcome = "UNCERTAIN"
        rationale = []

        if best.similarity_score < 0.40:
            outcome = "REJECTED"
            rationale.append(f"Visual similarity ({best.similarity_score:.4f}) is too low.")
        elif res.margin is not None and res.margin < margin_threshold:
            outcome = "UNCERTAIN"
            rationale.append("Candidate competition is ambiguous (small margin between top candidates).")
        elif best.similarity_score >= 0.70:
            outcome = "PROVISIONAL_SUPPORT"
            rationale.append("Strong visual embedding alignment using transfer learning baseline.")
        else:
            outcome = "UNCERTAIN"
            rationale.append("Moderate visual similarity; requires independent temporal/movement verification.")

        print(f"    Preliminary Outcome: {outcome}")
        print(f"    Evidence Rationale : {'; '.join(rationale)}")
        print()
