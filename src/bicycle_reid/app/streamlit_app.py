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
    page_title="CCTV Motorcycle Re-ID & Tracking",
    page_icon="🏍️",
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
    .track-card-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 6px 10px;
        border-radius: 6px 6px 0 0;
        font-weight: bold;
        font-size: 0.85em;
        display: flex;
        justify-content: space-between;
    }
    .track-meta-box {
        background: #fdfdfd;
        border: 1px solid #e0e0e0;
        border-top: none;
        border-radius: 0 0 6px 6px;
        padding: 8px 10px;
        font-size: 0.8em;
        line-height: 1.4;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_video_processor(conf: float, min_sim: float, top_k: int) -> VideoProcessor:
    """Cache the VideoProcessor and vision models so weights are loaded once."""
    config = AppConfig(
        detection=DetectionConfig(
            confidence_threshold=conf,
            target_classes=["motorcycle"],
            primary_class="motorcycle",
        ),
        retrieval=RetrievalConfig(top_k=top_k, min_similarity_threshold=min_sim),
    )
    return VideoProcessor(config=config)


def save_uploaded_file(uploaded_file) -> Path:
    """Save Streamlit UploadedFile to a temporary file on disk."""
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temp_dir = Path(tempfile.gettempdir()) / "motorcycle_reid_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target_path = temp_dir / f"{uploaded_file.file_id}_{uploaded_file.name}"
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path


def get_available_footage_sources() -> dict[str, Path]:
    """Discover available CCTV video footage in data directories."""
    sources: dict[str, Path] = {}
    
    # Priority CCTV footage (camera_a_cctv and camera_b_cctv)
    cctv_a = Path("data/demo/camera_a_cctv.mp4")
    cctv_b = Path("data/demo/camera_b_cctv.mp4")
    if cctv_a.exists():
        sources["Real CCTV Footage - Camera A (camera_a_cctv.mp4)"] = cctv_a
    if cctv_b.exists():
        sources["Real CCTV Footage - Camera B (camera_b_cctv.mp4)"] = cctv_b

    # Any other demo videos
    demo_dir = Path("data/demo")
    if demo_dir.exists():
        for p in sorted(demo_dir.glob("*.mp4")):
            key = f"Demo Video: {p.name}"
            if key not in sources and p not in sources.values():
                sources[key] = p
                
    # Fallback samples if needed
    sample_a = Path("src/data/demo/camera_a_sample.jpg")
    sample_b = Path("src/data/demo/camera_b_sample.jpg")
    if sample_a.exists():
        sources["Sample Frame - Camera A"] = sample_a
    if sample_b.exists():
        sources["Sample Frame - Camera B"] = sample_b

    return sources


def main():
    st.title("🏍️ Cross-Camera Motorcycle Re-Identification & Tracking System")
    st.markdown(
        "Standalone CCTV video footage analyzer with **motorcycle detection**, **within-camera tracking**, "
        "**transfer-learning visual similarity**, and **one-click CSV detection status export**."
    )

    # Sidebar: Input Sources & Configuration
    st.sidebar.header("⚙️ Configuration & Sources")

    available_sources = get_available_footage_sources()
    source_keys = list(available_sources.keys())

    # Find sensible defaults for Camera A and Camera B
    default_idx_a = 0
    default_idx_b = min(1, len(source_keys) - 1) if len(source_keys) > 1 else 0

    for idx, k in enumerate(source_keys):
        if "camera_a_cctv" in k.lower():
            default_idx_a = idx
        elif "camera_b_cctv" in k.lower():
            default_idx_b = idx

    # Camera A source selection
    st.sidebar.subheader("📹 Camera A Source")
    source_type_a = st.sidebar.radio(
        "Camera A input type:",
        ["Real CCTV Footage (Built-in)", "Upload Video / Image"],
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
        chosen_a = st.sidebar.selectbox(
            "Select Camera A CCTV Footage:",
            options=source_keys,
            index=default_idx_a,
            key="sel_a",
        )
        path_a = available_sources[chosen_a] if chosen_a else None

    # Camera B source selection
    st.sidebar.subheader("📹 Camera B Source")
    source_type_b = st.sidebar.radio(
        "Camera B input type:",
        ["Real CCTV Footage (Built-in)", "Upload Video / Image"],
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
        chosen_b = st.sidebar.selectbox(
            "Select Camera B CCTV Footage:",
            options=source_keys,
            index=default_idx_b,
            key="sel_b",
        )
        path_b = available_sources[chosen_b] if chosen_b else None

    # Processing Parameters
    st.sidebar.subheader("🔧 Detection & Tracking Parameters")
    conf_thresh = st.sidebar.slider("Motorcycle Detection Confidence", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
    max_frames = st.sidebar.slider("Max Frames per Source", min_value=5, max_value=150, value=40, step=5)
    frame_step = st.sidebar.selectbox("Frame Sampling Step", [1, 2, 3, 5], index=0)
    top_k = st.sidebar.slider("Top-K Candidates", min_value=1, max_value=5, value=3)
    min_sim_thresh = st.sidebar.slider("Min Similarity Threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05)

    run_pipeline = st.sidebar.button("🚀 Run Tracking & Re-ID Pipeline", type="primary", width="stretch")

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

        progress_bar = st.progress(0, text="Initializing motorcycle detection and tracking pipeline...")
        status_text = st.empty()

        def update_progress(ratio: float, msg: str):
            progress_bar.progress(int(ratio * 100), text=msg)

        status_text.info("Processing Camera A CCTV footage for motorcycles...")
        res_a = processor.process_source(
            source_path=path_a,
            camera_id="CAM_A",
            max_frames=max_frames,
            frame_step=frame_step,
            conf_threshold=conf_thresh,
            progress_callback=update_progress,
        )

        status_text.info("Processing Camera B CCTV footage for motorcycles...")
        res_b = processor.process_source(
            source_path=path_b,
            camera_id="CAM_B",
            max_frames=max_frames,
            frame_step=frame_step,
            conf_threshold=conf_thresh,
            progress_callback=update_progress,
        )

        status_text.info("Extracting visual representations and performing cross-camera motorcycle similarity...")
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
            st.subheader("📹 CCTV Footage View & Within-Camera Motorcycle Tracking")
            
            # Playback Controls Bar
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 4])
            with ctrl_col1:
                play_realtime = st.button("▶️ Play Realtime Video", type="secondary", width="stretch")
            with ctrl_col2:
                playback_speed = st.selectbox("Playback Speed", options=[0.5, 1.0, 1.5, 2.0], index=1)
            with ctrl_col3:
                st.caption("Press **Play Realtime Video** to stream annotated detection frames live at native CCTV frame rate.")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### 🎥 Camera A")
                st.caption(f"Source: `{Path(res_a.source_path).name}` | Frames: {res_a.total_frames} | FPS: {res_a.fps:.1f}")
                m1, m2 = st.columns(2)
                m1.metric("Tracked Motorcycles", len(res_a.tracks))
                m2.metric("Total Detections", len(res_a.detections))
                holder_a = st.empty()

            with col_b:
                st.markdown("### 🎥 Camera B")
                st.caption(f"Source: `{Path(res_b.source_path).name}` | Frames: {res_b.total_frames} | FPS: {res_b.fps:.1f}")
                m3, m4 = st.columns(2)
                m3.metric("Tracked Motorcycles", len(res_b.tracks))
                m4.metric("Total Detections", len(res_b.detections))
                holder_b = st.empty()

            # Realtime Playback or Interactive Scrubber
            if play_realtime:
                import time
                max_play_frames = max(len(res_a.preview_frames), len(res_b.preview_frames))
                nominal_fps = max(res_a.fps, res_b.fps, 15.0)
                frame_delay = (1.0 / nominal_fps) / playback_speed

                for idx in range(max_play_frames):
                    if idx < len(res_a.preview_frames):
                        frame_rgb_a = cv2.cvtColor(res_a.preview_frames[idx], cv2.COLOR_BGR2RGB)
                        holder_a.image(frame_rgb_a, caption=f"Camera A - Live Frame {idx + 1}/{len(res_a.preview_frames)}", width="stretch")
                    if idx < len(res_b.preview_frames):
                        frame_rgb_b = cv2.cvtColor(res_b.preview_frames[idx], cv2.COLOR_BGR2RGB)
                        holder_b.image(frame_rgb_b, caption=f"Camera B - Live Frame {idx + 1}/{len(res_b.preview_frames)}", width="stretch")
                    time.sleep(frame_delay)
            else:
                # Default scrubbers
                with col_a:
                    if res_a.preview_frames:
                        frame_num_a = st.slider(
                            "Frame Scrubber (Camera A)",
                            0,
                            len(res_a.preview_frames) - 1,
                            0,
                            key="scrubber_a",
                        )
                        rgb_frame_a = cv2.cvtColor(res_a.preview_frames[frame_num_a], cv2.COLOR_BGR2RGB)
                        holder_a.image(rgb_frame_a, caption=f"Camera A - Frame {frame_num_a + 1}/{len(res_a.preview_frames)}", width="stretch")
                    else:
                        holder_a.warning("No video frames processed for Camera A.")

                with col_b:
                    if res_b.preview_frames:
                        frame_num_b = st.slider(
                            "Frame Scrubber (Camera B)",
                            0,
                            len(res_b.preview_frames) - 1,
                            0,
                            key="scrubber_b",
                        )
                        rgb_frame_b = cv2.cvtColor(res_b.preview_frames[frame_num_b], cv2.COLOR_BGR2RGB)
                        holder_b.image(rgb_frame_b, caption=f"Camera B - Frame {frame_num_b + 1}/{len(res_b.preview_frames)}", width="stretch")
                    else:
                        holder_b.warning("No video frames processed for Camera B.")

            # Tracklet Visual Gallery
            st.markdown("---")
            st.subheader("🖼️ Detected Motorcycle Tracklet Visual Gallery")
            st.caption("Representative motorcycle crops and tracking metadata arranged in an aligned responsive card grid.")

            cam_tab_a, cam_tab_b = st.tabs([
                f"🎥 Camera A Tracks ({len(res_a.tracks)})",
                f"🎥 Camera B Tracks ({len(res_b.tracks)})",
            ])

            def render_track_grid(tracks, track_crops, cam_label):
                if not tracks:
                    st.info(f"No motorcycle tracks generated for {cam_label}.")
                    return

                # Render in an aligned 3-column grid
                n_cols = 3
                for row_start in range(0, len(tracks), n_cols):
                    cols = st.columns(n_cols)
                    for col_idx in range(n_cols):
                        item_idx = row_start + col_idx
                        if item_idx < len(tracks):
                            trk = tracks[item_idx]
                            crop = track_crops.get(trk.track_id)
                            q = trk.quality or {}
                            with cols[col_idx]:
                                # Card container with border
                                with st.container(border=True):
                                    st.markdown(f"**📍 `{trk.track_id}`**")
                                    if crop is not None:
                                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                        st.image(crop_rgb, width="stretch")
                                    else:
                                        st.caption("No crop available")
                                    
                                    # Aligned metrics row
                                    meta_c1, meta_c2 = st.columns(2)
                                    meta_c1.metric("Detections", len(trk.detections))
                                    meta_c2.metric("Mean Conf", f"{q.get('mean_confidence', 0.0):.2f}")
                                    st.caption(f"⏱️ **Duration:** {q.get('duration_seconds', 0.0):.2f}s | 🧭 **Dir:** `{trk.direction or 'N/A'}`")

            with cam_tab_a:
                render_track_grid(res_a.tracks, res_a.track_crops, "Camera A")

            with cam_tab_b:
                render_track_grid(res_b.tracks, res_b.track_crops, "Camera B")

        # ----------------------------------------------------
        # TAB 2: Similarity Detection & Cross-Camera Re-ID
        # ----------------------------------------------------
        with tab_similarity:
            st.subheader("🔍 Cross-Camera Track Similarity & Re-ID Matches")

            if not res_a.tracks or not res_b.tracks:
                st.warning("Both cameras must have at least one detected motorcycle track to compute cross-camera similarity.")
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
                                st.image(cv2.cvtColor(crop_a, cv2.COLOR_BGR2RGB), caption=f"Query {q_id}", width="stretch")

                        if ret_res and ret_res.top_candidates:
                            top_cand = ret_res.top_candidates[0]
                            cand_id = top_cand.candidate_id
                            sim_score = top_cand.similarity_score
                            crop_b = res_b.track_crops.get(cand_id)

                            with card_col2:
                                if crop_b is not None:
                                    st.image(cv2.cvtColor(crop_b, cv2.COLOR_BGR2RGB), caption=f"Top Match {cand_id}", width="stretch")

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
            st.subheader("📊 Motorcycle Detection & Tracking Status (CSV Export)")
            st.markdown(
                "Export and review the frame-by-frame motorcycle detection records and within-camera tracking status in standard CSV formats."
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
                    file_name="motorcycle_detection_status.csv",
                    mime="text/csv",
                    type="primary",
                    width="stretch",
                )

            with exp_col2:
                st.download_button(
                    label="📥 Download Track Summary (CSV)",
                    data=csv_tracks,
                    file_name="motorcycle_tracks_summary.csv",
                    mime="text/csv",
                    width="stretch",
                )

            # Metrics
            st.markdown("### 📈 Detection Summary Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Detections", len(all_detections))
            m2.metric("Tracked Motorcycles", len(all_tracks))
            m3.metric("Camera A Detections", len(res_a.detections))
            m4.metric("Camera B Detections", len(res_b.detections))

            # Interactive DataFrame
            st.markdown("### 🔍 Frame-by-Frame Motorcycle Detection Records")
            st.caption("Filters and sorting are active. Click table headers to sort.")

            # Filter options
            filter_cam = st.multiselect(
                "Filter by Camera:",
                options=["CAM_A", "CAM_B"],
                default=["CAM_A", "CAM_B"],
            )
            filtered_df = dets_df[dets_df["camera_id"].isin(filter_cam)]
            st.dataframe(filtered_df, width="stretch")

            # Track Summary Table
            st.markdown("### 📋 Motorcycle Tracklet Summary Table")
            st.dataframe(tracks_df, width="stretch")

    else:
        # Initial Landing view before clicking Run
        st.info("👈 Select your real CCTV footage or upload files in the sidebar, then click **'🚀 Run Tracking & Re-ID Pipeline'** to begin.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 System Features")
            st.markdown(
                """
                - **CCTV Video Footage View:** View footage, scrub through annotated frames, and inspect motorcycle bounding boxes.
                - **Within-Camera Tracking:** Generates camera-scoped track IDs (`CAM_A:track_0001`, `CAM_B:track_0001`) with trajectory trails.
                - **Motorcycle Similarity Detection:** Computes deep visual representations using pretrained encoders and cosine similarity matching.
                - **Independent Evidence Review:** Honest evaluation with `ACCEPTED`, `UNCERTAIN`, and `NO_MATCH` states and margin checks.
                - **CSV Export:** One-click instant export of all detection coordinates, timestamps, confidence scores, and match results.
                """
            )
        with col2:
            st.markdown("### 🛠️ Quick Start Guide")
            st.markdown(
                """
                1. Select the real CCTV video footage (`camera_a_cctv.mp4` and `camera_b_cctv.mp4`) in the sidebar.
                2. Adjust **Motorcycle Detection Confidence** or **Max Frames** if desired.
                3. Click **'🚀 Run Tracking & Re-ID Pipeline'**.
                4. Inspect video frames, similarity matrix, and download your **CSV report**!
                """
            )


if __name__ == "__main__":
    main()
