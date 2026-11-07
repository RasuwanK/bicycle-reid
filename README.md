# Cross-Camera Bicycle Re-Identification System

![Project Cover Image](assets/cover.png)
*(Place your project cover or architecture banner image at `assets/cover.png`)*

---

## 🎯 Goal

The goal of this project is to build a standalone computer-vision application that determines whether a bicycle observed in CCTV Video A is the same physical bicycle observed in CCTV Video B.

Unlike traditional demos that simply return the nearest neighbor, this system operates as an **evidence-fusion and decision-support system**:
- **Track-Level Comparison**: Compares tracklets and multi-frame representations rather than a single arbitrary still image.
- **Transfer Learning (No Custom Retraining Required)**: Uses pretrained YOLO detection and deep visual encoders (ResNet-50 / MobileNetV3) for feature extraction.
- **Honest Confidence**: Treats cosine similarity as a feature space distance, never mistaking it for a calibrated probability.
- **Independent Evidence Review**: Supports `ACCEPTED`, `UNCERTAIN`, `REJECTED`, and `NO_MATCH` outcomes, accounting for candidate competition, margins, and contradictions.

---

## 🎬 Demo Video & Media

| Media Type | Expected File Path | Description |
| :--- | :--- | :--- |
| **Cover Image** | [`assets/cover.png`](assets/cover.png) | High-level banner, system diagram, or cover visual |
| **Demo Video** | [`assets/demo.mp4`](assets/demo.mp4) | Walkthrough screen recording of the console/UI matching run |

> [!TIP]
> To preview a video directly on GitHub, place an MP4 video or an animated GIF in `assets/` and link it below:
> 
> [![Watch Demo Video](https://img.shields.io/badge/Demo%20Video-assets%2Fdemo.mp4-blue?style=for-the-badge&logo=youtube)](assets/demo.mp4)

---

## 🛠️ How to Build & Install

### Prerequisites
- Python >= 3.10
- Linux / macOS / Windows
- NVIDIA GPU with CUDA support recommended (CPU fallback supported automatically)

### 1. Clone & Set Up Virtual Environment

```bash
git clone <repository-url>
cd bicycle-reid

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (cmd):
# .venv\Scripts\activate.bat
```

### 2. Install Project Dependencies

Install the project in editable development mode:

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

This installs:
- Core dependencies: `torch`, `torchvision`, `ultralytics` (YOLO), `opencv-python`, `numpy`, `pyyaml`
- Development & testing tools: `pytest`

---

## 🐳 Running with Docker (Containerized)

The application is fully containerized with self-contained model weights and runtime dependencies (OpenCV, PyTorch, FFmpeg, Ultralytics, and Streamlit).

### 1. Quick Start with Docker Compose

Launch the interactive Web UI with one command:

```bash
docker compose up -d
```

Open your browser and navigate to **[http://localhost:8501](http://localhost:8501)**.

To stop the application:
```bash
docker compose down
```

### 2. Run Built-In Demo via Docker Compose

```bash
docker compose run --rm cli
```

### 3. Run Custom CLI Commands with Docker

Analyze your own CCTV footage mounted from the host's `data/` directory:

```bash
docker run --rm -v $(pwd)/data:/app/data bicycle-reid \
    --source-a data/cctv_camera_a.mp4 \
    --source-b data/cctv_camera_b.mp4
```

Or run the built-in demo directly:
```bash
docker run --rm bicycle-reid --demo
```

### 4. Run Offline Test Suite in Container

```bash
docker run --rm bicycle-reid pytest
```

### 5. GPU Acceleration (Optional)

If your host has an NVIDIA GPU and the NVIDIA Container Toolkit configured, you can enable GPU inference:

```bash
docker run --rm --gpus all -p 8501:8501 -v $(pwd)/data:/app/data bicycle-reid
```

---

## 🚀 How to Run (Local Python Environment)

### 1. Launch the Interactive Web Interface (Recommended)

Start the web application supporting video footage inspection, within-camera tracking, similarity matching, and one-click CSV export:

```bash
python src/main.py --ui
```
*or directly with Streamlit:*
```bash
streamlit run src/bicycle_reid/app/streamlit_app.py
```
*or via the CLI shortcut:*
```bash
bicycle-reid --ui
```

The web app allows you to:
- 📹 **View CCTV Video Footage**: Play videos or scrub frame-by-frame with bounding box & track overlays.
- 🚴 **Within-Camera Tracking**: Automatically tracks bicycles across frames with camera-scoped track IDs (`CAM_A:track_0001`).
- 🔍 **Similarity Detection & Re-ID**: Compares visual features with cosine similarity matrices, competitor margins, and review decisions (`ACCEPTED`, `UNCERTAIN`, `NO_MATCH`).
- 📥 **Export to CSV**: Instantly download frame-level detection records or track summaries formatted for spreadsheet and analytics tools.

---

### 2. Run the Built-In Console Demo

Run the out-of-the-box demo using sample dual-camera images:

```bash
python src/main.py --demo
```
*or using the installed CLI entrypoint:*
```bash
bicycle-reid --demo
```

### 3. Compare Two Existing CCTV Sources via CLI

Run cross-camera re-identification between two video files or still images:

```bash
python src/main.py --source-a data/cctv_camera_a.mp4 --source-b data/cctv_camera_b.mp4
```

### 4. Command-Line Options

| Flag | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `--ui` / `--web` | Flag | Launch interactive Streamlit Web Interface | `False` |
| `--source-a` | Path | Video/image source for Camera A | `None` |
| `--source-b` | Path | Video/image source for Camera B | `None` |
| `--sources` | List[Path] | Arbitrary list of sources to analyze | `[]` |
| `--demo` | Flag | Run with sample demo imagery | `False` |
| `--conf` | Float | Detection confidence threshold | `0.30` |
| `--encoder` | String | Vision encoder (`resnet50`, `mobilenet_v3_small`) | `resnet50` |
| `--config` | Path | Path to custom YAML configuration file | `configs/default.yaml` |

---

## 🧪 Running Tests

The test suite runs offline and does not require external network access:

```bash
pytest tests/
```

---

## 📂 Repository Structure

```text
bicycle-reid/
├── AGENTS.md                  # Working agreement, data contracts, and design principles
├── README.md                  # Project overview and documentation
├── pyproject.toml             # Build configuration and dependencies
├── configs/
│   └── default.yaml           # Default detection, encoder, and retrieval settings
├── assets/
│   ├── cover.png              # Project cover image (placeholder)
│   └── demo.mp4               # Demo video file (placeholder)
├── data/
│   ├── demo/                  # Cached sample inputs for demo execution
│   └── input/                 # User-provided CCTV videos (ignored by Git)
├── src/
│   ├── main.py                # Primary script entry point
│   └── bicycle_reid/
│       ├── config/            # YAML configuration loader
│       ├── contracts/         # Typed data models (Detection, Track, EvidenceItem, etc.)
│       ├── ingestion/         # Image & video decoding and metadata validation
│       ├── detection/         # YOLO detector (zero-training transfer learning)
│       ├── tracking/          # Within-camera tracking with camera-scoped track IDs
│       ├── representation/    # Deep visual encoders (ResNet-50 / MobileNet-V3)
│       ├── retrieval/         # Cosine similarity matrix & candidate margin ranking
│       ├── reporting/         # Honest console reporting & explanation generation
│       ├── app/               # Streamlit web UI, video processor, and CSV export
│       ├── pipeline.py        # Pipeline orchestrator
│       └── cli.py             # CLI command parser
└── tests/
    └── unit/                  # Fast, offline pytest unit tests
```

---

## 📋 License & Attribution

Developed according to the multi-agent evidence-fusion architecture outlined in [`AGENTS.md`](AGENTS.md).
