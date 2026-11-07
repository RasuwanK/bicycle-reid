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

## 🚀 How to Run

### 1. Run the Built-In Demo

Run the out-of-the-box demo using sample dual-camera images:

```bash
python src/main.py --demo
```
*or using the installed CLI entrypoint:*
```bash
bicycle-reid --demo
```

### 2. Compare Two Existing CCTV Sources

Run cross-camera re-identification between two video files or still images:

```bash
python src/main.py --source-a data/cctv_camera_a.mp4 --source-b data/cctv_camera_b.mp4
```

### 3. Command-Line Options

| Flag | Type | Description | Default |
| :--- | :--- | :--- | :--- |
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
│       ├── representation/    # Deep visual encoders (ResNet-50 / MobileNet-V3)
│       ├── retrieval/         # Cosine similarity matrix & candidate margin ranking
│       ├── reporting/         # Honest console reporting & explanation generation
│       ├── pipeline.py        # Pipeline orchestrator
│       └── cli.py             # CLI command parser
└── tests/
    └── unit/                  # Fast, offline pytest unit tests
```

---

## 📋 License & Attribution

Developed according to the multi-agent evidence-fusion architecture outlined in [`AGENTS.md`](AGENTS.md).
