# AGENTS.md — Cross-Camera Motorcycle Re-Identification

This file is the shared working agreement for AI coding agents and human contributors working on this project. Read it before changing code.

## 1. Mission

Build a **standalone computer-vision application** that determines whether a motorcycle observed in CCTV Video A is the same physical motorcycle observed in CCTV Video B.

The application must not simply choose the most visually similar candidate. It must independently review the evidence and be able to:

- accept a sufficiently supported identity match;
- report uncertainty when evidence is incomplete, ambiguous, or unreliable; and
- reject a candidate when evidence contradicts the proposed identity.

The application must provide an explanation for every decision.

**Core workflow**

`Video ingestion → Motorcycle detection → Within-camera tracking → Track representation → Cross-camera candidate retrieval → Spatiotemporal checks → Independent evidence review → Decision → Explanation`

The project is an **evidence-fusion and decision-support system**, not just a motorcycle classifier or an image-similarity demo.

## 2. Non-negotiable principles

1. **Track-level comparison:** Compare tracklets / track representations, not just one arbitrary still image.
2. **Local tracking IDs are not global identities:** Track ID `7` in Camera A has no relationship to Track ID `7` in Camera B.
3. **No forced match:** The system must support `NO_MATCH` and `UNCERTAIN`; never select the top candidate by default.
4. **Independent review:** The visual retrieval component proposes candidates. A separate verification component evaluates the evidence and decides whether the candidate is supportable.
5. **Unknown is not contradictory:** Missing timestamps, unusable views, or unavailable camera metadata must be represented as `UNKNOWN`, not as negative evidence.
6. **Contradictions matter:** Strong contradictions (for example, an impossible travel interval under trusted timing assumptions) must not be hidden by a high visual score.
7. **Candidate competition matters:** Retain and display alternatives. A very small gap between the best and second-best candidate is evidence of ambiguity.
8. **Confidence must be honest:** Do not label raw cosine similarity or an arbitrary weighted score as a calibrated probability.
9. **Explain every outcome:** Record the supporting, contradicting, and unknown evidence used by the verifier.
10. **Build an end-to-end baseline first:** Prefer simple, testable components before training a custom deep model or adding distributed infrastructure.

## 3. Scope and expected user workflow

The initial application is a local, standalone prototype:

1. User selects two CCTV video files.
2. User supplies or confirms camera metadata where available.
3. The application processes each video independently.
4. It displays detected motorcycles and within-camera tracks.
5. It constructs track-level visual representations.
6. It retrieves the top candidate tracks in the other video for each source track.
7. It evaluates time, movement, track quality, missing evidence, and competing candidates.
8. It shows `ACCEPTED`, `UNCERTAIN`, `REJECTED`, or `NO_MATCH` with an evidence report.

Do not add cloud services, remote inference, user accounts, distributed queues, or complex deployment infrastructure unless explicitly requested.

## 4. Architecture and module boundaries

Keep modules loosely coupled. Model implementations should be replaceable without rewriting the decision engine.

### 4.1 Video ingestion

Responsibilities:

- Open video files and validate that they can be decoded.
- Read frame rate, frame count, dimensions, and duration when available.
- Preserve source frame indices and video timestamps.
- Support camera-specific timestamp offsets or uncertainty settings.
- Report missing, malformed, or unreliable metadata explicitly.

Do not assume that a video file's creation time is the time of the recorded event. Do not silently assume both camera clocks are synchronized.

### 4.2 Motorcycle detection

Responsibilities:

- Detect the `motorcycle` class in each frame.
- Return bounding boxes, detector confidence, frame index, and timestamp.
- Apply documented confidence and image-size filtering.
- Keep detection configuration explicit and reproducible.

A detector answers **where is a motorcycle?** It does not establish its identity.

### 4.3 Within-camera tracking

Responsibilities:

- Associate detections across frames within one camera.
- Produce tracklets with local track IDs.
- Preserve per-frame bounding boxes and detection confidence.
- Record track start/end observations and track termination reasons when available.
- Detect or flag tracking fragmentation and possible ID switches where possible.

Use a tracker such as ByteTrack or BoT-SORT as an initial baseline. Tracking IDs must remain camera-scoped, e.g. `CAM_A:track_0007`.

A tracker answers **which detections appear to belong to the same moving object in this video?** It does not solve cross-camera re-identification.

### 4.4 Track quality and representative-frame selection

For each track, select a limited set of useful frames rather than embedding every frame by default.

Consider:

- bounding-box size and visible area;
- image sharpness / blur;
- occlusion or truncation;
- detector confidence;
- image exposure and visibility;
- diversity of viewpoints across the track.

Store why frames were selected or excluded. Do not let a large number of near-identical frames dominate the representation.

### 4.5 Track representation / visual feature extraction

Responsibilities:

- Crop the motorcycle from selected frames, preserving enough context when useful.
- Extract embeddings using a configurable feature-extraction backend.
- Aggregate frame-level features into a track-level representation.
- Retain per-frame features and quality metadata for later inspection.

Start with a pretrained feature extractor as a **baseline**, not as a claim of reliable motorcycle identity recognition. Generic image embeddings (including CLIP or DINO-family features) may capture semantic similarity rather than fine-grained instance identity. Evaluate them on the project's own footage.

Keep the embedding backend behind an interface so a motorcycle-specific Re-ID model can replace it later. Do not assume a person-ReID model transfers reliably to motorcycles without testing and adaptation.

### 4.6 Cross-camera candidate retrieval

Responsibilities:

- Compare tracks from different cameras.
- Calculate and store visual similarity metrics.
- Retrieve Top-K candidates, not just the top-1 result.
- Store the best score, second-best score, and candidate margin.
- Allow a configurable no-match threshold / retrieval policy.
- Preserve all candidate scores for the explanation.

Cosine similarity is a useful baseline. Its value is a similarity measure, not automatically a probability of identity.

### 4.7 Temporal and movement consistency

Responsibilities:

- Compare observation intervals using camera timestamps and configured clock-offset uncertainty.
- Check plausible camera-to-camera travel intervals when camera geometry or route assumptions are available.
- Check movement direction / trajectory compatibility only when it can be estimated reliably.
- Distinguish `SUPPORTED`, `CONTRADICTED`, and `UNKNOWN`.
- Include assumptions and uncertainty ranges in the evidence record.

Do not invent camera distances, routes, travel times, clock offsets, or direction mappings. If these are not configured or inferable with adequate reliability, mark the relevant evidence `UNKNOWN`.

For an initial prototype, camera metadata can be supplied manually in a configuration file or UI. Keep the model simple and transparent; avoid introducing road-network routing unless the project has reliable camera location data.

### 4.8 Independent evidence verifier

This module is separate from the embedding / candidate retrieval module.

Input:

- source track and candidate track;
- visual similarity and candidate ranking context;
- track quality measurements;
- temporal evidence;
- movement evidence;
- camera metadata and its reliability;
- missing information and contradictions.

Output:

- evidence items with state, value, explanation, and provenance;
- decision outcome;
- decision rationale;
- confidence / confidence band, clearly distinguished from calibrated probability.

Start with a deterministic, testable, rule-based verifier. Keep thresholds and weights in configuration, not scattered throughout the code. Do not use an LLM as the sole verifier or as a substitute for explicit evidence checks.

An optional language model may later convert an already-computed structured evidence report into readable prose. It must not invent evidence, override hard constraints, or independently declare an identity match.

### 4.9 Decision and explanation

Supported outcomes:

- `ACCEPTED`: evidence is sufficiently strong, reliable, and not materially contradicted.
- `UNCERTAIN`: a match is plausible, but evidence is weak, incomplete, ambiguous, or conflicting.
- `REJECTED`: the candidate is contradicted or sufficiently dissimilar under the configured policy.
- `NO_MATCH`: no candidate meets the minimum retrieval / verification requirements.

Do not collapse `UNCERTAIN`, `REJECTED`, and `NO_MATCH` into one state. They mean different things:

- `UNCERTAIN`: cannot safely decide.
- `REJECTED`: this proposed candidate is not supported / is contradicted.
- `NO_MATCH`: no candidate was found that should be proposed as a reliable match.

A decision must include:
- evidence supporting the candidate;
- evidence contradicting the candidate;
- unknown / missing evidence;
- top competing candidates;
- relevant thresholds and assumptions;
- the reasons for the final outcome.

## 5. Suggested data contracts

Use typed models (for example, Python dataclasses or Pydantic models). Keep contracts stable between modules.

### Camera metadata

```python
CameraMetadata(
    camera_id: str,
    video_id: str,
    timezone: str | None,
    clock_offset_seconds: float | None,
    clock_offset_uncertainty_seconds: float | None,
    location_description: str | None,
    latitude: float | None,
    longitude: float | None,
    viewing_direction: str | None,
    metadata_quality: str,
)
```

Do not require coordinates if they are unavailable. `metadata_quality` should make the reliability of the supplied values clear.

### Detection

```python
Detection(
    camera_id: str,
    frame_index: int,
    timestamp_seconds: float | None,
    bbox_xyxy: tuple[float, float, float, float],
    class_name: str,
    confidence: float,
)
```

### Track

```python
MotorcycleTrack(
    track_id: str,                 # camera-scoped ID
    camera_id: str,
    start_time_seconds: float | None,
    end_time_seconds: float | None,
    detections: list[Detection],
    representative_frame_refs: list[str],
    trajectory: list[tuple[float, float]] | None,
    direction: str | None,
    quality: dict,
)
```

### Track representation

```python
TrackRepresentation(
    track_id: str,
    embedding: list[float],
    frame_embeddings: list[list[float]],
    frame_quality: list[dict],
    aggregation_method: str,
    model_name: str,
    model_version: str | None,
)
```

### Evidence item

```python
EvidenceItem(
    evidence_type: str,
    state: str,                    # SUPPORTED | CONTRADICTED | UNKNOWN | WEAK
    observed_value: object | None,
    expected_value: object | None,
    reliability: str,               # HIGH | MEDIUM | LOW | UNKNOWN
    explanation: str,
    source_refs: list[str],
)
```

### Candidate review

```python
CandidateReview(
    source_track_id: str,
    candidate_track_id: str | None,
    visual_similarity: float | None,
    candidate_rank: int | None,
    candidate_margin: float | None,
    evidence: list[EvidenceItem],
    outcome: str,                   # ACCEPTED | UNCERTAIN | REJECTED | NO_MATCH
    confidence_label: str,
    calibrated_probability: float | None,
    explanation: str,
)
```

Only populate `calibrated_probability` after a calibration procedure has been implemented and evaluated. Otherwise leave it `None`.

## 6. Initial technology direction

Use the following as a practical starting point; verify compatibility and document selected versions in the repository.

| Area | Initial option |
|---|---|
| Language | Python |
| Video processing | OpenCV |
| Detection | Ultralytics YOLO-family detector with motorcycle class |
| Tracking | ByteTrack or BoT-SORT |
| Deep learning | PyTorch |
| Visual features | Configurable pretrained vision encoder baseline |
| Similarity | NumPy / scikit-learn cosine similarity |
| Large-scale retrieval (optional) | FAISS |
| Typed data contracts | Pydantic or dataclasses |
| MVP interface | Streamlit or a simple local web UI |
| Tests | pytest |

Avoid adding FAISS until ordinary in-memory comparison is too slow or the candidate collection is large enough to justify it.

Use a local processing path by default. CCTV videos may be sensitive; do not upload footage to external services or send it to third-party APIs unless the project owner explicitly authorizes that behavior.

## 7. Suggested repository structure

Adapt this layout to the existing repository rather than restructuring everything unnecessarily.

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   └── cameras.example.yaml
├── data/
│   ├── input/                 # ignored by Git
│   ├── processed/             # ignored by Git
│   └── annotations/          # keep only approved / non-sensitive data
├── artifacts/                 # ignored by Git
├── src/
│   └── motorcycle_reid/
│       ├── config/
│       ├── ingestion/
│       ├── detection/
│       ├── tracking/
│       ├── quality/
│       ├── representation/
│       ├── retrieval/
│       ├── temporal/
│       ├── movement/
│       ├── verification/
│       ├── reporting/
│       └── app/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── scripts/
```

Keep raw videos, generated crops, model weights, embeddings, and large evaluation files out of Git unless there is an explicit reason to version them. Provide a `.gitignore` accordingly.

## 8. Delivery plan — build in this order

### Milestone 1: Validate video input and detection

Deliver:
- two videos can be loaded;
- metadata is reported;
- motorcycles are detected;
- annotated preview frames or videos can be exported.

Acceptance checks:
- invalid video input produces a useful error;
- detections include class, box, frame index, timestamp, and confidence;
- the detector's limitations are documented.

### Milestone 2: Track motorcycles within each camera

Deliver:
- stable within-camera tracklets;
- camera-scoped track IDs;
- saved track metadata and representative crops.

Acceptance checks:
- track IDs are never treated as cross-camera identities;
- track records contain per-frame observations;
- fragmented / short / low-quality tracks can be flagged.

### Milestone 3: Build track representations and visual retrieval

Deliver:
- quality-aware representative-frame selection;
- embeddings per selected frame;
- track-level aggregation;
- cross-camera similarity matrix;
- Top-K candidates and candidate margin.

Acceptance checks:
- embedding backend is configurable;
- missing / failed crops do not crash the entire run;
- similarity is not presented as a probability;
- no automatic match is made solely because a candidate ranks first.

### Milestone 4: Add temporal and movement evidence

Deliver:
- configurable camera metadata;
- timestamp offset and uncertainty handling;
- plausible travel interval checks where inputs permit;
- movement consistency checks where direction can be estimated.

Acceptance checks:
- unavailable metadata results in `UNKNOWN`;
- clock offsets are explicit and testable;
- impossible-travel rules are only applied when their assumptions are valid;
- evidence records describe assumptions and source values.

### Milestone 5: Implement independent verification

Deliver:
- separate verifier module;
- deterministic decision policy;
- supporting / contradicting / unknown evidence;
- candidate competition and track quality considered.

Acceptance checks:
- a high visual score cannot override a configured hard contradiction;
- missing evidence does not count as negative evidence;
- close competing candidates can trigger `UNCERTAIN`;
- the system can return `NO_MATCH`;
- thresholds are configurable and tested.

### Milestone 6: Build the review interface and end-to-end demo

Deliver:
- select Camera A and Camera B videos;
- show detected tracks and representative crops;
- show candidate ranking;
- show evidence and final outcome;
- export a machine-readable result (JSON) and a human-readable report.

Acceptance checks:
- every decision can be traced to its evidence;
- the user can inspect competing candidates;
- the UI distinguishes `ACCEPTED`, `UNCERTAIN`, `REJECTED`, and `NO_MATCH`.

## 9. Decision policy guidance

Do not hard-code arbitrary numerical cutoffs as if they are universal truths. Start with configurable provisional thresholds, then tune them on a validation set containing known same-bike pairs and different-bike pairs.

The verifier should reason over the following categories:

| Evidence category | Possible interpretation |
|---|---|
| Visual similarity | Support, weak support, or insufficient visual evidence |
| Distinctive attributes | Consistency / contradiction / not visible |
| Temporal consistency | Plausible / implausible / unknown |
| Movement consistency | Compatible / contradictory / unknown |
| Track quality | Good / weak |
| Candidate competition | Clear winner / ambiguous |
| Missing information | Explicitly recorded |

A weighted evidence score may be useful for ranking or triage, but:
- retain the individual evidence items;
- do not let a score conceal a hard contradiction;
- use minimum evidence requirements before acceptance;
- use an abstention path when evidence quality is inadequate;
- never call a weighted score a probability unless calibrated.

## 10. Evaluation and test strategy

Evaluation must measure both matching ability and the ability to decline unreliable matches.

### Dataset and ground truth

For each observed track, maintain a ground-truth physical motorcycle ID where known. Record camera, recording time, track boundaries, and difficult conditions where possible.

Split training / validation / test data by physical motorcycle and preferably by recording session or camera condition. Do not randomly split adjacent frames from the same track across train and test; that causes severe data leakage.

Include:
- same motorcycles across different cameras;
- different but visually similar motorcycles;
- no-match examples;
- different viewpoints and distances;
- day/night or lighting variation when available;
- occlusion, blur, and partial visibility;
- timestamp uncertainty and camera-clock offset scenarios.

### Metrics

Report metrics appropriate to each module:

- Detection: precision, recall, mAP where labels support it.
- Tracking: IDF1, HOTA, or other suitable tracking metrics when ground truth is available.
- Retrieval: Top-1 / Top-K accuracy, mean average precision, and rank statistics.
- Decision: precision, recall, F1, false-match rate, and confusion matrix for accepted / uncertain / rejected outcomes.
- Abstention: coverage versus risk (how often the system decides, and the error rate among decided cases).
- Reliability: calibration metrics only if confidence is explicitly calibrated.
- Explanation: evidence completeness and whether each stated reason is traceable to source data.

Always report false matches. In this problem, an incorrect confident identity match is a serious failure.

## 11. Testing requirements

Add unit tests for at least:

- invalid / missing video metadata;
- timestamp conversion and clock-offset calculations;
- track aggregation with missing or low-quality frames;
- cosine similarity and candidate ranking;
- candidate margin calculation;
- temporal intervals with exact, uncertain, and missing timestamps;
- evidence state transitions;
- hard contradictions;
- `UNKNOWN` evidence handling;
- close-candidate ambiguity;
- `NO_MATCH` behavior;
- explanation generation.

Add integration tests using short, synthetic or approved sample videos. Tests must not depend on external network access or download model weights at runtime. Mock model boundaries for deterministic unit tests.

## 12. Engineering conventions

- Use type hints for public functions and data models.
- Keep functions small and responsibilities explicit.
- Put thresholds, class IDs, sampling rates, and model settings in configuration.
- Make processing deterministic where practical; record model names, versions, and config with each run.
- Log useful progress and failures without dumping entire video frames or binary payloads.
- Handle corrupt videos, empty detections, short tracks, missing timestamps, and model errors gracefully.
- Do not silently swallow exceptions.
- Avoid premature abstractions, distributed systems, and unnecessary databases.
- Keep model inference separate from business / decision logic.
- Persist intermediate artifacts enough to reproduce and debug a decision.
- Update README documentation when commands, configuration, or data formats change.
- Do not add dependencies without explaining why they are needed.
- Never commit secrets, private CCTV footage, credentials, or unapproved personal data.

## 13. Agent collaboration rules

If multiple coding agents are working in parallel, split work by module boundaries:

- **Detection/tracking agent:** video ingestion, motorcycle detection, track generation.
- **Representation agent:** quality filtering, crop generation, feature extraction, track embeddings.
- **Matching agent:** similarity computation, Top-K retrieval, candidate ranking.
- **Evidence agent:** camera metadata, temporal/movement checks, independent verifier, decision policy.
- **Application agent:** UI, reports, run orchestration, integration.

Coordination rules:

1. Agree on the shared data contracts before parallel implementation.
2. Each agent should work within its assigned module unless an interface change is agreed.
3. Do not create duplicate versions of shared models or configuration.
4. Add tests for every public interface.
5. Communicate assumptions, limitations, dependencies, and changed contracts in the handoff.
6. Integrate incrementally; do not wait until every module is complete.
7. The integration owner must run the full test suite and one end-to-end demo before declaring completion.

## 14. Required output for each candidate review

A result should be serializable and understandable without reading application logs. Example shape:

```json
{
  "source_track_id": "CAM_A:track_0012",
  "candidate_track_id": "CAM_B:track_0007",
  "outcome": "UNCERTAIN",
  "visual_similarity": 0.91,
  "candidate_rank": 1,
  "candidate_margin": 0.02,
  "evidence": [
    {
      "type": "visual_similarity",
      "state": "SUPPORTED",
      "reliability": "MEDIUM",
      "explanation": "The selected track representations are visually similar."
    },
    {
      "type": "temporal_consistency",
      "state": "UNKNOWN",
      "reliability": "UNKNOWN",
      "explanation": "Camera B clock offset has not been established."
    },
    {
      "type": "candidate_competition",
      "state": "WEAK",
      "reliability": "HIGH",
      "explanation": "The second-ranked candidate has a very similar score."
    }
  ],
  "competing_candidates": [
    {"track_id": "CAM_B:track_0009", "visual_similarity": 0.89}
  ],
  "supporting_evidence": [
    "Visual representations are similar."
  ],
  "contradicting_evidence": [],
  "unknown_evidence": [
    "Camera clock offset is not configured."
  ],
  "decision_explanation": "The visual evidence is promising, but the leading candidates are too close and temporal evidence is unavailable."
}
```

This is an illustrative schema, not a fixed score policy. Keep the outcome consistent with the actual evidence.

## 15. Definition of done for the first usable MVP

The MVP is complete only when all of the following are true:

- [ ] Two CCTV videos can be processed locally.
- [ ] Motorcycles are detected in both videos.
- [ ] Each camera produces its own tracklets.
- [ ] Track-level visual representations are generated.
- [ ] Top-K cross-camera candidates are retrieved.
- [ ] The system can identify an ambiguous lookalike case.
- [ ] Temporal / movement checks run when metadata supports them and return `UNKNOWN` otherwise.
- [ ] A separate verifier reviews all evidence.
- [ ] `ACCEPTED`, `UNCERTAIN`, `REJECTED`, and `NO_MATCH` are handled distinctly.
- [ ] Every decision includes supporting, contradicting, and unknown evidence.
- [ ] Competing candidates are visible in the result.
- [ ] Tests cover decision rules and missing-data cases.
- [ ] A short end-to-end demo and setup instructions are documented.
- [ ] No unsupported claim is made that the prototype is production-grade or has calibrated identity probabilities.

## 16. How to start when entering an existing repository

Before coding:

1. Inspect the existing repository structure, README, dependencies, and tests.
2. Identify what is already implemented; do not overwrite working code without a reason.
3. Check the local Python / GPU environment and whether model weights are already available.
4. Propose the smallest implementation slice that produces a visible end-to-end result.
5. Confirm or introduce the shared data contracts.
6. Implement and test one milestone at a time.

When reporting progress, state:
- files changed;
- behavior implemented;
- tests run and results;
- known limitations;
- next concrete task.

**Final instruction:** Optimize for a defensible, inspectable decision. A system that correctly says “uncertain” is better than one that confidently invents a motorcycle identity.
