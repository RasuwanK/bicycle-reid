"""Streamlit Web Interface for CCTV Video Footage View, Tracking, Similarity Detection, and CSV Export.

Run with:
    streamlit run src/bicycle_reid/app/streamlit_app.py
or:
    python src/main.py --ui
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from bicycle_reid.app.video_processor import VideoProcessor, VideoProcessingResult
from bicycle_reid.app.export import (
    detections_to_dataframe,
    tracks_to_dataframe,
    export_detections_csv,
    export_tracks_csv,
)
from bicycle_reid.config.settings import AppConfig, DetectionConfig, RepresentationConfig, RetrievalConfig


# Initialize Streamlit Page
st.set_page_config(
    page_title="CCTV Bicycle Re-ID & Tracking",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished interface
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 18px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 12px;
    }
    .badge-accepted {
        background-color: #2e7d32;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .badge-uncertain {
        background-color: #ef6c00;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .badge-nomatch {
        background-color: #c62828;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_video_processor(conf: float, min_sim: float, top_k: int) -> VideoProcessor:
    """Cache the VideoProcessor and vision models so weights are loaded once."""
    config = AppConfig(
        detection=DetectionConfig(confidence_threshold=conf),
        retrieval=RetrievalConfig(top_k=top_k, min_similarity_threshold=min_sim),
    )
    return VideoProcessor(config=config)


def save_uploaded_file(uploaded_file) -> Path:
    """Save Streamlit UploadedFile to a temporary file on disk."""
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_dir = Path(tempfile.gettempdir()) / "bicycle_reid_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target_path = temp_dir / f"{uploaded_file.file_id}_{uploaded_file.name}"
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path


def main():
    st.title("🚲 Cross-Camera Bicycle Re-Identification & Tracking System")
    st.markdown(
        "Standalone CCTV video footage analyzer with **within-camera tracking**, "
        "**transfer-learning visual similarity**, and **one-click CSV detection status export**."
    )

    # Sidebar: Input Sources & Configuration
    st.sidebar.header("⚙️ Configuration & Sources")

    default_demo_a = Path("data/demo/camera_a.mp4")
    default_demo_b = Path("data/demo/camera_b.mp4")

    # Camera A source selection
    st.sidebar.subheader("📹 Camera A Source")
    source_type_a = st.sidebar.radio(
        "Camera A input type:",
        ["Built-in Sample Footage", "Upload Video / Image"],
        key="src_type_a",
    )
    if source_type_a == "Upload Video / Image":
        file_a = st.sidebar.file_uploader(
            "Upload Camera A video (MP4, AVI, MOV) or image",
            type=["mp4", "avi", "mov", "jpg", "jpeg", "png"],
            key="file_a",
        )
        path_a = save_uploaded_file(file_a) if file_a else None
    else:
        path_a = default_demo_a if default_demo_a.exists() else Path("src/data/demo/camera_a_sample.jpg")

    # Camera B source selection
    st.sidebar.subheader("📹 Camera B Source")
    source_type_b = st.sidebar.radio(
        "Camera B input type:",
        ["Built-in Sample Footage", "Upload Video / Image"],
        key="src_type_b",
    )
    if source_type_b == "Upload Video / Image":
        file_b = st.sidebar.file_uploader(
            "Upload Camera B video (MP4, AVI, MOV) or image",
            type=["mp4", "avi", "mov", "jpg", "jpeg", "png"],
            key="file_b",
        )
        path_b = save_uploaded_file(file_b) if file_b else None
    else:
        path_b = default_demo_b if default_demo_b.exists() else Path("src/data/demo/camera_b_sample.jpg")

    # Processing Parameters
    st.sidebar.subheader("🔧 Detection & Tracking Parameters")
    conf_thresh = st.sidebar.slider("Detection Confidence", min_value=0.10, max_value=0.90, value=0.30, step=0.05)
    max_frames = st.sidebar.slider("Max Frames per Source", min_value=5, max_value=150, value=30, step=5)
    frame_step = st.sidebar.selectbox("Frame Sampling Step", [1, 2, 3, 5], index=0)
    top_k = st.sidebar.slider("Top-K Candidates", min_value=1, max_value=5, value=3)
    min_sim_thresh = st.sidebar.slider("Min Similarity Threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05)

    run_pipeline = st.sidebar.button("🚀 Run Tracking & Re-ID Pipeline", type="primary", use_container_width=True)

    # Initialize session state for results if not present
    if "results" not in st.session_state:
        st.session_state.results = None

    if run_pipeline:
        if not path_a or not Path(path_a).exists():
            st.error("Camera A source is invalid or missing.")
            return
        if not path_b or not Path(path_b).exists():
            st.error("Camera B source is invalid or missing.")
            return

        processor = get_video_processor(conf_thresh, min_sim_thresh, top_k)

        progress_bar = st.progress(0, text="Initializing detection and tracking pipeline...")
        status_text = st.empty()

        def update_progress(ratio: float, msg: str):
            progress_bar.progress(int(ratio * 100), text=msg)

        status_text.info("Processing Camera A footage...")
        res_a = processor.process_source(
            source_path=path_a,
            camera_id="CAM_A",
            max_frames=max_frames,
            frame_step=frame_step,
            conf_threshold=conf_thresh,
            progress_callback=update_progress,
        )

        status_text.info("Processing Camera B footage...")
        res_b = processor.process_source(
            source_path=path_b,
            camera_id="CAM_B",
            max_frames=max_frames,
            frame_step=frame_step,
            conf_threshold=conf_thresh,
            progress_callback=update_progress,
        )

        status_text.info("Extracting visual representations and performing cross-camera similarity...")
        retrieval_map, sim_matrix = processor.match_cross_camera(
            res_a, res_b, top_k=top_k, min_threshold=min_sim_thresh
        )

        progress_bar.progress(100, text="Pipeline processing complete!")
        status_text.success("Analysis complete!")

        st.session_state.results = {
            "res_a": res_a,
            "res_b": res_b,
            "retrieval_map": retrieval_map,
            "sim_matrix": sim_matrix,
        }

    # Render results if available
    if st.session_state.results is not None:
        data = st.session_state.results
        res_a: VideoProcessingResult = data["res_a"]
        res_b: VideoProcessingResult = data["res_b"]
        retrieval_map = data["retrieval_map"]
        sim_matrix = data["sim_matrix"]

        all_detections = res_a.detections + res_b.detections
        all_tracks = res_a.tracks + res_b.tracks

        # Main Navigation Tabs
        tab_footage, tab_similarity, tab_export = st.tabs([
            "📹 Video Footage & Tracking View",
            "🔍 Similarity Detection & Re-ID",
            "📊 Detection Status & CSV Export",
        ])

        # ----------------------------------------------------
        # TAB 1: Video Footage View & Within-Camera Tracking
        # ----------------------------------------------------
        with tab_footage:
            st.subheader("📹 CCTV Footage View & Tracking Overview")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### 🎥 Camera A")
                st.caption(f"Source: `{Path(res_a.source_path).name}` | Frames: {res_a.total_frames} | FPS: {res_a.fps:.1f}")
                m1, m2 = st.columns(2)
                m1.metric("Tracked Bicycles", len(res_a.tracks))
                m2.metric("Total Detections", len(res_a.detections))

                if res_a.preview_frames:
                    frame_num_a = st.slider(
                        "Frame Scrubber (Camera A)",
                        0,
                        len(res_a.preview_frames) - 1,
                        0,
                        key="scrubber_a",
                    )
                    # Convert BGR to RGB for Streamlit display
                    rgb_frame_a = cv2.cvtColor(res_a.preview_frames[frame_num_a], cv2.COLOR_BGR2RGB)
                    st.image(rgb_frame_a, caption=f"Camera A - Sample Frame {frame_num_a + 1}", use_container_width=True)
                else:
                    st.warning("No video frames detected for Camera A.")

            with col_b:
                st.markdown("### 🎥 Camera B")
                st.caption(f"Source: `{Path(res_b.source_path).name}` | Frames: {res_b.total_frames} | FPS: {res_b.fps:.1f}")
                m3, m4 = st.columns(2)
                m3.metric("Tracked Bicycles", len(res_b.tracks))
                m4.metric("Total Detections", len(res_b.detections))

                if res_b.preview_frames:
                    frame_num_b = st.slider(
                        "Frame Scrubber (Camera B)",
                        0,
                        len(res_b.preview_frames) - 1,
                        0,
                        key="scrubber_b",
                    )
                    rgb_frame_b = cv2.cvtColor(res_b.preview_frames[frame_num_b], cv2.COLOR_BGR2RGB)
                    st.image(rgb_frame_b, caption=f"Camera B - Sample Frame {frame_num_b + 1}", use_container_width=True)
                else:
                    st.warning("No video frames detected for Camera B.")

            # Tracklet Visual Gallery
            st.markdown("---")
            st.subheader("🖼️ Detected Tracklet Visual Gallery")
            st.markdown("Representative crops extracted for each camera-scoped tracklet:")

            gallery_col_a, gallery_col_b = st.columns(2)
            with gallery_col_a:
                st.markdown("**Camera A Tracks:**")
                if not res_a.tracks:
                    st.write("No tracks generated in Camera A.")
                for trk in res_a.tracks:
                    crop = res_a.track_crops.get(trk.track_id)
                    with st.expander(f"📍 Track `{trk.track_id}` ({len(trk.detections)} detections)", expanded=True):
                        c1, c2 = st.columns([1, 2])
                        if crop is not None:
                            c1.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                        q = trk.quality or {}
                        c2.write(f"- **Duration:** {q.get('duration_seconds', 0.0)}s")
                        c2.write(f"- **Mean Confidence:** {q.get('mean_confidence', 0.0):.2f}")
                        c2.write(f"- **Direction:** `{trk.direction}`")

            with gallery_col_b:
                st.markdown("**Camera B Tracks:**")
                if not res_b.tracks:
                    st.write("No tracks generated in Camera B.")
                for trk in res_b.tracks:
                    crop = res_b.track_crops.get(trk.track_id)
                    with st.expander(f"📍 Track `{trk.track_id}` ({len(trk.detections)} detections)", expanded=True):
                        c1, c2 = st.columns([1, 2])
                        if crop is not None:
                            c1.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                        q = trk.quality or {}
                        c2.write(f"- **Duration:** {q.get('duration_seconds', 0.0)}s")
                        c2.write(f"- **Mean Confidence:** {q.get('mean_confidence', 0.0):.2f}")
                        c2.write(f"- **Direction:** `{trk.direction}`")

        # ----------------------------------------------------
        # TAB 2: Similarity Detection & Cross-Camera Re-ID
        # ----------------------------------------------------
        with tab_similarity:
            st.subheader("🔍 Cross-Camera Track Similarity & Re-ID Matches")

            if not res_a.tracks or not res_b.tracks:
                st.warning("Both cameras must have at least one detected track to compute cross-camera similarity.")
            else:
                # Similarity Matrix Heatmap/Table
                st.markdown("#### 📐 Pairwise Visual Similarity Matrix")
                track_ids_a = [t.track_id for t in res_a.tracks]
                track_ids_b = [t.track_id for t in res_b.tracks]

                if sim_matrix.size > 0:
                    sim_df = pd.DataFrame(sim_matrix, index=track_ids_a, columns=track_ids_b)
                    st.dataframe(sim_df.style.background_gradient(cmap="Blues", vmin=0.0, vmax=1.0).format("{:.3f}"))
                else:
                    st.write("No embeddings available for similarity matrix computation.")

                st.markdown("---")
                st.markdown("#### 🎯 Independent Match Review & Candidates")

                for trk_a in res_a.tracks:
                    q_id = trk_a.track_id
                    ret_res = retrieval_map.get(q_id)

                    with st.container():
                        st.markdown(f"### Query: `{q_id}` (Camera A)")
                        crop_a = res_a.track_crops.get(q_id)

                        card_col1, card_col2, card_col3 = st.columns([1, 1, 2])
                        with card_col1:
                            if crop_a is not None:
                                st.image(cv2.cvtColor(crop_a, cv2.COLOR_BGR2RGB), caption=f"Query {q_id}", use_container_width=True)

                        if ret_res and ret_res.top_candidates:
                            top_cand = ret_res.top_candidates[0]
                            cand_id = top_cand.candidate_id
                            sim_score = top_cand.similarity_score
                            crop_b = res_b.track_crops.get(cand_id)

                            with card_col2:
                                if crop_b is not None:
                                    st.image(cv2.cvtColor(crop_b, cv2.COLOR_BGR2RGB), caption=f"Top Match {cand_id}", use_container_width=True)

                            with card_col3:
                                # Determine outcome
                                margin = ret_res.margin
                                if sim_score >= 0.75 and (margin is None or margin >= 0.05):
                                    outcome_badge = "<span class='badge-accepted'>ACCEPTED</span>"
                                    explanation = "High visual similarity supported with adequate candidate separation."
                                elif sim_score >= 0.40:
                                    outcome_badge = "<span class='badge-uncertain'>UNCERTAIN</span>"
                                    explanation = "Plausible similarity, but insufficient margin or moderate confidence."
                                else:
                                    outcome_badge = "<span class='badge-nomatch'>NO_MATCH</span>"
                                    explanation = "Visual similarity below minimum confidence threshold."

                                st.markdown(f"**Outcome:** {outcome_badge}", unsafe_allow_html=True)
                                st.write(f"- **Top Candidate:** `{cand_id}`")
                                st.write(f"- **Visual Similarity:** `{sim_score:.3f}`")
                                st.write(f"- **Candidate Margin:** `{margin:.3f}`" if margin is not None else "- **Candidate Margin:** `N/A (Single Candidate)`")
                                st.info(f"**Explanation:** {explanation}")

                                # Show competing candidates if any
                                if len(ret_res.top_candidates) > 1:
                                    st.markdown("**Competing Candidates:**")
                                    comp_df = pd.DataFrame([
                                        {"Rank": c.rank, "Candidate Track": c.candidate_id, "Similarity Score": f"{c.similarity_score:.3f}"}
                                        for c in ret_res.top_candidates
                                    ])
                                    st.dataframe(comp_df, hide_index=True)
                        else:
                            with card_col3:
                                st.markdown("<span class='badge-nomatch'>NO_MATCH</span>", unsafe_allow_html=True)
                                st.info("No candidates in Camera B passed the minimum similarity threshold.")

                        st.markdown("---")

        # ----------------------------------------------------
        # TAB 3: Detection Status & CSV Export
        # ----------------------------------------------------
        with tab_export:
            st.subheader("📊 Detection & Tracking Status (CSV Export)")
            st.markdown(
                "Export and review the frame-by-frame detection records and within-camera tracking status in standard CSV formats."
            )

            # Export buttons row
            dets_df = detections_to_dataframe(all_detections, retrieval_map)
            tracks_df = tracks_to_dataframe(all_tracks, retrieval_map)

            csv_detections = dets_df.to_csv(index=False)
            csv_tracks = tracks_df.to_csv(index=False)

            exp_col1, exp_col2, _ = st.columns([1, 1, 2])
            with exp_col1:
                st.download_button(
                    label="📥 Download Detection Status (CSV)",
                    data=csv_detections,
                    file_name="bicycle_detection_status.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                )

            with exp_col2:
                st.download_button(
                    label="📥 Download Track Summary (CSV)",
                    data=csv_tracks,
                    file_name="bicycle_tracks_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # Metrics
            st.markdown("### 📈 Detection Summary Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Detections", len(all_detections))
            m2.metric("Tracked Objects", len(all_tracks))
            m3.metric("Camera A Detections", len(res_a.detections))
            m4.metric("Camera B Detections", len(res_b.detections))

            # Interactive DataFrame
            st.markdown("### 🔍 Frame-by-Frame Detection Records")
            st.caption("Filters and sorting are active. Click table headers to sort.")

            # Filter options
            filter_cam = st.multiselect(
                "Filter by Camera:",
                options=["CAM_A", "CAM_B"],
                default=["CAM_A", "CAM_B"],
            )
            filtered_df = dets_df[dets_df["camera_id"].isin(filter_cam)]
            st.dataframe(filtered_df, use_container_width=True)

            # Track Summary Table
            st.markdown("### 📋 Tracklet Summary Table")
            st.dataframe(tracks_df, use_container_width=True)

    else:
        # Initial Landing view before clicking Run
        st.info("👈 Select your camera video footage or use the built-in sample footage in the sidebar, then click **'🚀 Run Tracking & Re-ID Pipeline'** to begin.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 System Features")
            st.markdown(
                """
                - **CCTV Video Footage View:** View footage, scrub through annotated frames, and inspect bounding boxes.
                - **Within-Camera Tracking:** Generates camera-scoped track IDs (`CAM_A:track_0001`, `CAM_B:track_0001`) with trajectory trails.
                - **Similarity Detection & Transfer Learning:** Computes deep visual representations using pretrained encoders and cosine similarity matching.
                - **Evidence Review:** Honest evaluation with `ACCEPTED`, `UNCERTAIN`, and `NO_MATCH` states and margin checks.
                - **CSV Export:** One-click instant export of all detection coordinates, timestamps, confidence scores, and match results.
                """
            )
        with col2:
            st.markdown("### 🛠️ Quick Start Guide")
            st.markdown(
                """
                1. Keep the default **Built-in Sample Footage** selected in the sidebar.
                2. Adjust **Confidence** or **Max Frames** if desired.
                3. Click **'🚀 Run Tracking & Re-ID Pipeline'**.
                4. Inspect video frames, similarity matrix, and download your **CSV report**!
                """
            )


if __name__ == "__main__":
    main()
