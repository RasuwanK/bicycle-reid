import json
import uuid

def make_element(
    el_type,
    x,
    y,
    width,
    height,
    text="",
    stroke_color="#1e1e1e",
    bg_color="transparent",
    fill_style="solid",
    font_size=16,
    text_align="center",
    vertical_align="middle",
    stroke_width=2,
    roundness={"type": 3},
    stroke_style="solid",
    roughness=1,
    opacity=100,
    points=None,
    start_binding=None,
    end_binding=None,
    start_arrowhead=None,
    end_arrowhead=None,
):
    el_id = str(uuid.uuid4())[:12]
    element = {
        "id": el_id,
        "type": el_type,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": stroke_color,
        "backgroundColor": bg_color,
        "fillStyle": fill_style,
        "strokeWidth": stroke_width,
        "strokeStyle": stroke_style,
        "roughness": roughness,
        "opacity": opacity,
        "groupIds": [],
        "frameId": None,
        "roundness": roundness,
        "seed": abs(hash(el_id)) % 1000000,
        "version": 1,
        "versionNonce": abs(hash(el_id + "nonce")) % 1000000,
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
    }

    if el_type == "text":
        element["text"] = text
        element["fontSize"] = font_size
        element["fontFamily"] = 1  # Virgil / clean sans
        element["textAlign"] = text_align
        element["verticalAlign"] = vertical_align
        element["baseline"] = font_size
        element["containerId"] = None
        element["originalText"] = text
        element["lineHeight"] = 1.25

    if el_type in ("arrow", "line"):
        element["points"] = points or [[0, 0], [width, height]]
        element["lastCommittedPoint"] = None
        element["startBinding"] = start_binding
        element["endBinding"] = end_binding
        element["startArrowhead"] = start_arrowhead
        element["endArrowhead"] = end_arrowhead or "arrow"

    return element


def build_system_diagram():
    elements = []

    # Title Card
    title_box = make_element(
        "rectangle",
        x=200,
        y=40,
        width=1100,
        height=80,
        bg_color="#e3f2fd",
        stroke_color="#1565c0",
        stroke_width=2,
        roundness={"type": 3},
    )
    title_text = make_element(
        "text",
        x=220,
        y=55,
        width=1060,
        height=50,
        text="🏍️ Cross-Camera Motorcycle Re-Identification System Architecture\nEvidence-Fusion & Independent Decision Verification Pipeline",
        font_size=20,
        stroke_color="#0d47a1",
        text_align="center",
    )
    elements.extend([title_box, title_text])

    # ----------------------------------------------------
    # SECTION 1: DUAL CCTV INGESTION
    # ----------------------------------------------------
    # Group box
    ingest_grp = make_element(
        "rectangle",
        x=100,
        y=160,
        width=300,
        height=540,
        bg_color="#fafafa",
        stroke_color="#9e9e9e",
        stroke_style="dashed",
        roundness={"type": 3},
    )
    ingest_grp_lbl = make_element(
        "text",
        x=115,
        y=175,
        width=270,
        height=24,
        text="1. Video Ingestion & Decoding",
        font_size=16,
        stroke_color="#424242",
        text_align="left",
    )
    elements.extend([ingest_grp, ingest_grp_lbl])

    # Camera A Box
    cam_a_box = make_element(
        "rectangle",
        x=130,
        y=220,
        width=240,
        height=110,
        bg_color="#e8f5e9",
        stroke_color="#2e7d32",
        roundness={"type": 3},
    )
    cam_a_txt = make_element(
        "text",
        x=140,
        y=235,
        width=220,
        height=80,
        text="📹 Camera A Video (CCTV)\n• Format: MP4 / AVI / MOV\n• Native FPS & Timestamps\n• Source Metadata validation",
        font_size=13,
        stroke_color="#1b5e20",
        text_align="left",
    )
    elements.extend([cam_a_box, cam_a_txt])

    # Camera B Box
    cam_b_box = make_element(
        "rectangle",
        x=130,
        y=370,
        width=240,
        height=110,
        bg_color="#e8f5e9",
        stroke_color="#2e7d32",
        roundness={"type": 3},
    )
    cam_b_txt = make_element(
        "text",
        x=140,
        y=385,
        width=220,
        height=80,
        text="📹 Camera B Video (CCTV)\n• Format: MP4 / AVI / MOV\n• Independent Clock / Offset\n• Viewpoint & Calibration",
        font_size=13,
        stroke_color="#1b5e20",
        text_align="left",
    )
    elements.extend([cam_b_box, cam_b_txt])

    # Streamlit Upload / Live Ingest
    ui_ingest_box = make_element(
        "rectangle",
        x=130,
        y=520,
        width=240,
        height=150,
        bg_color="#fff3e0",
        stroke_color="#e65100",
        roundness={"type": 3},
    )
    ui_ingest_txt = make_element(
        "text",
        x=140,
        y=535,
        width=220,
        height=120,
        text="🖥️ Streamlit Web Interface\n• Built-in Footage Selector\n• Dynamic File Uploader\n• FPS & Sampling Step (1..5)\n• Confidence threshold slider",
        font_size=13,
        stroke_color="#bf360c",
        text_align="left",
    )
    elements.extend([ui_ingest_box, ui_ingest_txt])

    # ----------------------------------------------------
    # SECTION 2: DETECTION & WITHIN-CAMERA TRACKING
    # ----------------------------------------------------
    det_grp = make_element(
        "rectangle",
        x=450,
        y=160,
        width=300,
        height=540,
        bg_color="#fafafa",
        stroke_color="#9e9e9e",
        stroke_style="dashed",
        roundness={"type": 3},
    )
    det_grp_lbl = make_element(
        "text",
        x=465,
        y=175,
        width=270,
        height=24,
        text="2. Detection & Tracking",
        font_size=16,
        stroke_color="#424242",
        text_align="left",
    )
    elements.extend([det_grp, det_grp_lbl])

    # YOLO Detector
    yolo_box = make_element(
        "rectangle",
        x=480,
        y=220,
        width=240,
        height=130,
        bg_color="#f3e5f5",
        stroke_color="#7b1fa2",
        roundness={"type": 3},
    )
    yolo_txt = make_element(
        "text",
        x=490,
        y=235,
        width=220,
        height=100,
        text="🎯 YOLO Motorcycle Detector\n• YOLOv8 Transfer Learning\n• Target class: motorcycle\n• Bounding box [x1,y1,x2,y2]\n• Area & Confidence filtering",
        font_size=13,
        stroke_color="#4a148c",
        text_align="left",
    )
    elements.extend([yolo_box, yolo_txt])

    # Within-camera Tracker
    tracker_box = make_element(
        "rectangle",
        x=480,
        y=380,
        width=240,
        height=140,
        bg_color="#e0f2f1",
        stroke_color="#00796b",
        roundness={"type": 3},
    )
    tracker_txt = make_element(
        "text",
        x=490,
        y=395,
        width=220,
        height=110,
        text="🔄 Within-Camera Tracker\n• Camera-scoped Track IDs\n  (e.g. CAM_A:track_0001)\n• IoU Association & Replay\n• Trajectory & Motion vector\n• Frame quality & Sharpness",
        font_size=13,
        stroke_color="#004d40",
        text_align="left",
    )
    elements.extend([tracker_box, tracker_txt])

    # Live Annotation & Stream
    annot_box = make_element(
        "rectangle",
        x=480,
        y=550,
        width=240,
        height=120,
        bg_color="#fce4ec",
        stroke_color="#c2185b",
        roundness={"type": 3},
    )
    annot_txt = make_element(
        "text",
        x=490,
        y=565,
        width=220,
        height=90,
        text="🎞️ Live Annotation Engine\n• Realtime Video Playback\n• Bounding box overlay\n• Track IDs & Trajectory trails\n• Scrubbable frame buffer",
        font_size=13,
        stroke_color="#880e4f",
        text_align="left",
    )
    elements.extend([annot_box, annot_txt])

    # ----------------------------------------------------
    # SECTION 3: REPRESENTATION & EMBEDDING
    # ----------------------------------------------------
    rep_grp = make_element(
        "rectangle",
        x=800,
        y=160,
        width=300,
        height=540,
        bg_color="#fafafa",
        stroke_color="#9e9e9e",
        stroke_style="dashed",
        roundness={"type": 3},
    )
    rep_grp_lbl = make_element(
        "text",
        x=815,
        y=175,
        width=270,
        height=24,
        text="3. Representation & Quality",
        font_size=16,
        stroke_color="#424242",
        text_align="left",
    )
    elements.extend([rep_grp, rep_grp_lbl])

    # Representative Frame Selection
    frame_sel_box = make_element(
        "rectangle",
        x=830,
        y=220,
        width=240,
        height=130,
        bg_color="#ede7f6",
        stroke_color="#512da8",
        roundness={"type": 3},
    )
    frame_sel_txt = make_element(
        "text",
        x=840,
        y=235,
        width=220,
        height=100,
        text="🖼️ Quality-Aware Selection\n• Representative Crop Extraction\n• Padding & Context preservation\n• Blur / Sharpness scoring\n• Viewpoint diversity filtering",
        font_size=13,
        stroke_color="#311b92",
        text_align="left",
    )
    elements.extend([frame_sel_box, frame_sel_txt])

    # Deep Visual Feature Extractor
    encoder_box = make_element(
        "rectangle",
        x=830,
        y=380,
        width=240,
        height=130,
        bg_color="#e1f5fe",
        stroke_color="#0288d1",
        roundness={"type": 3},
    )
    encoder_txt = make_element(
        "text",
        x=840,
        y=395,
        width=220,
        height=100,
        text="🧠 Pretrained Vision Encoder\n• ResNet-50 / MobileNetV3\n• Global Avg Pool Embeddings\n• L2 Feature Normalization\n• Zero-shot Transfer Learning",
        font_size=13,
        stroke_color="#01579b",
        text_align="left",
    )
    elements.extend([encoder_box, encoder_txt])

    # Track Representation Aggregator
    agg_box = make_element(
        "rectangle",
        x=830,
        y=540,
        width=240,
        height=130,
        bg_color="#fff8e1",
        stroke_color="#ffa000",
        roundness={"type": 3},
    )
    agg_txt = make_element(
        "text",
        x=840,
        y=555,
        width=220,
        height=100,
        text="📊 Tracklet Aggregator\n• Multi-frame pooling\n• Camera-scoped embeddings\n• Track duration & confidence\n• Output: TrackRepresentation",
        font_size=13,
        stroke_color="#ff6f00",
        text_align="left",
    )
    elements.extend([agg_box, agg_txt])

    # ----------------------------------------------------
    # SECTION 4: RETRIEVAL & EVIDENCE VERIFICATION
    # ----------------------------------------------------
    match_grp = make_element(
        "rectangle",
        x=1150,
        y=160,
        width=320,
        height=540,
        bg_color="#fafafa",
        stroke_color="#9e9e9e",
        stroke_style="dashed",
        roundness={"type": 3},
    )
    match_grp_lbl = make_element(
        "text",
        x=1165,
        y=175,
        width=290,
        height=24,
        text="4. Retrieval & Evidence Fusion",
        font_size=16,
        stroke_color="#424242",
        text_align="left",
    )
    elements.extend([match_grp, match_grp_lbl])

    # Similarity Matrix & Candidate Retrieval
    retrieval_box = make_element(
        "rectangle",
        x=1180,
        y=220,
        width=260,
        height=130,
        bg_color="#e8eaf6",
        stroke_color="#303f9f",
        roundness={"type": 3},
    )
    retrieval_txt = make_element(
        "text",
        x=1190,
        y=235,
        width=240,
        height=100,
        text="🔍 Candidate Retrieval\n• Pairwise Cosine Similarity\n• Full Similarity Matrix [N x M]\n• Top-K Candidates (Rank 1..K)\n• Margin Calculation (1st vs 2nd)",
        font_size=13,
        stroke_color="#1a237e",
        text_align="left",
    )
    elements.extend([retrieval_box, retrieval_txt])

    # Independent Evidence Verifier
    verifier_box = make_element(
        "rectangle",
        x=1180,
        y=380,
        width=260,
        height=160,
        bg_color="#ffebee",
        stroke_color="#d32f2f",
        roundness={"type": 3},
    )
    verifier_txt = make_element(
        "text",
        x=1190,
        y=395,
        width=240,
        height=130,
        text="⚖️ Independent Verifier (AGENTS.md)\n• Strict Hard-Constraint Checks\n• Travel Time / Timestamp Plausibility\n• Spatiotemporal & Movement Rules\n• Ambiguity & Competitor Margin\n• Never forces top-1 match",
        font_size=13,
        stroke_color="#b71c1c",
        text_align="left",
    )
    elements.extend([verifier_box, verifier_txt])

    # Final Outcome & Decision States
    outcome_box = make_element(
        "rectangle",
        x=1180,
        y=560,
        width=260,
        height=120,
        bg_color="#f1f8e9",
        stroke_color="#689f38",
        roundness={"type": 3},
    )
    outcome_txt = make_element(
        "text",
        x=1190,
        y=570,
        width=240,
        height=100,
        text="🏁 Honest Decision States:\n• ✅ ACCEPTED (Strong & Uncontradicted)\n• ⚠️ UNCERTAIN (Low margin / Ambiguous)\n• 🚫 REJECTED (Temporal contradiction)\n• ❓ NO_MATCH (Under min threshold)",
        font_size=12,
        stroke_color="#33691e",
        text_align="left",
    )
    elements.extend([outcome_box, outcome_txt])

    # ----------------------------------------------------
    # SECTION 5: PRESENTATION, PLAYBACK & CSV EXPORT
    # ----------------------------------------------------
    export_box = make_element(
        "rectangle",
        x=100,
        y=740,
        width=1370,
        height=120,
        bg_color="#f3e5f5",
        stroke_color="#6a1b9a",
        stroke_width=2,
        roundness={"type": 3},
    )
    export_txt = make_element(
        "text",
        x=120,
        y=755,
        width=1330,
        height=90,
        text="📊 Output, Presentation & Data Contracts\n• Realtime Video Playback: Synchronized side-by-side player with detection bounding boxes & track trajectories\n• Aligned Tracklet Gallery: High-resolution visual crops, track durations, mean detector confidence & direction\n• Instant CSV Export: Full audit logs (motorcycle_detection_status.csv & motorcycle_tracks_summary.csv)",
        font_size=14,
        stroke_color="#4a148c",
        text_align="left",
    )
    elements.extend([export_box, export_txt])

    # ----------------------------------------------------
    # ARROWS & WORKFLOW CONNECTORS
    # ----------------------------------------------------
    # Cam A & B -> YOLO
    arrow1 = make_element("arrow", x=370, y=275, width=110, height=0, points=[[0, 0], [110, 0]], stroke_color="#1976d2")
    arrow2 = make_element("arrow", x=370, y=425, width=110, height=-100, points=[[0, 0], [110, -100]], stroke_color="#1976d2")
    # YOLO -> Tracker
    arrow3 = make_element("arrow", x=600, y=350, width=0, height=30, points=[[0, 0], [0, 30]], stroke_color="#7b1fa2")
    # Tracker -> Annotation Engine
    arrow4 = make_element("arrow", x=600, y=520, width=0, height=30, points=[[0, 0], [0, 30]], stroke_color="#00796b")
    # Tracker -> Representative Frame Selection
    arrow5 = make_element("arrow", x=720, y=440, width=110, height=-160, points=[[0, 0], [110, -160]], stroke_color="#00796b")
    # Frame Sel -> Encoder
    arrow6 = make_element("arrow", x=950, y=350, width=0, height=30, points=[[0, 0], [0, 30]], stroke_color="#512da8")
    # Encoder -> Aggregator
    arrow7 = make_element("arrow", x=950, y=510, width=0, height=30, points=[[0, 0], [0, 30]], stroke_color="#0288d1")
    # Aggregator -> Retrieval
    arrow8 = make_element("arrow", x=1070, y=600, width=110, height=-320, points=[[0, 0], [110, -320]], stroke_color="#ff6f00")
    # Retrieval -> Verifier
    arrow9 = make_element("arrow", x=1310, y=350, width=0, height=30, points=[[0, 0], [0, 30]], stroke_color="#303f9f")
    # Verifier -> Outcome
    arrow10 = make_element("arrow", x=1310, y=540, width=0, height=20, points=[[0, 0], [0, 20]], stroke_color="#d32f2f")
    # Outcome & Annotator -> Export / UI bottom bar
    arrow11 = make_element("arrow", x=1310, y=680, width=0, height=60, points=[[0, 0], [0, 60]], stroke_color="#2e7d32")
    arrow12 = make_element("arrow", x=600, y=670, width=0, height=70, points=[[0, 0], [0, 70]], stroke_color="#c2185b")

    elements.extend([arrow1, arrow2, arrow3, arrow4, arrow5, arrow6, arrow7, arrow8, arrow9, arrow10, arrow11, arrow12])

    data = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "viewBackgroundColor": "#ffffff",
            "gridSize": 20,
        },
        "files": {},
    }
    return data

if __name__ == "__main__":
    diagram = build_system_diagram()
    output_path = "assets/architecture_diagram.excalidraw"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diagram, f, indent=2)
    print(f"Successfully generated {output_path} with {len(diagram['elements'])} elements.")
