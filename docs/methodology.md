# Methodology: Tennis Serve Analysis via Markerless Human Pose Estimation

Working document for the thesis chapter "Approach / Implementation"
(Vorgehensweise). Sections 1–4 describe work that is implemented and
validated; sections 5–7 describe the planned procedure for the remaining
stages and mark open decision points as such.

---

## 1. Methodological Approach

### 1.1 Research approach

The project follows a **constructive research approach** (design science):
an artifact — a modular analysis pipeline — is designed, implemented and
evaluated against defined quality criteria. The central methodological
commitment is made *before* implementation:

> The pipeline is decomposed into strictly separated stages, each of which
> persists its complete output before the next stage begins.

This commitment is what makes the later evaluation scientifically useful.
Markerless pose estimation on fast sports motion fails in known ways
(occlusion, motion blur, tracking loss). If extraction, signal processing
and evaluation were interleaved, a wrong joint angle at ball contact could
not be attributed to any single cause. With staged persistence, every error
observed in the final output can be traced backwards through the stored
intermediate representations to the stage that introduced it
(**per-stage error attribution**).

### 1.2 Design principles

Three principles were fixed at project start and act as acceptance criteria
for every stage:

1. **Modularity.** Each stage is an independent module with a defined
   input/output contract, testable in isolation. Orchestration code contains
   no domain logic.
2. **Reproducibility.** Raw data is persisted immediately (CSV/JSON) and is
   never modified afterwards. No stage recomputes what an earlier stage
   produced; no on-the-fly calculations bypass a persisted boundary. All
   parameters (model, thresholds, versions) are recorded in metadata files
   alongside the data.
3. **Stage-wise validation.** No stage is built on top of an unvalidated
   predecessor. Each stage requires (a) automated unit tests for its logic
   and (b) a visual sanity check of its output on real data before the next
   stage begins.

### 1.3 Pipeline overview

| Stage | Name | Input | Persisted output | Validation |
|------:|------|-------|------------------|------------|
| 1 | Ingestion & pose extraction | video file | landmark time series (CSV), metadata (JSON), overlay video (MP4), contact sheet (PNG) | unit tests, overlay inspection, detection statistics |
| 2 | Signal processing & kinematics | Stage 1 CSV + JSON | cleaned/filtered series, joint angles & angular velocities (CSV), processing metadata (JSON) | unit tests, raw-vs-filtered plots, angle plausibility |
| 3 | Phase segmentation & evaluation | Stage 2 output | serve phase boundaries, kinematic metrics per phase (CSV/JSON), annotated video | unit tests, event markers rendered into video, comparison with literature values |

---

## 2. Technology Selection

### 2.1 Pose estimation model

**Requirement profile:** 3D landmark output (joint angles in space),
per-landmark confidence scores (occlusion detection), single-person
temporal tracking mode (video, not independent images), CPU-viable
inference (no dedicated GPU available), maintained API.

**Chosen: MediaPipe BlazePose** (Bazarevsky et al., 2020), accessed through
the MediaPipe *Tasks API* (`PoseLandmarker`), model variant **heavy**
(highest accuracy of the three published variants; real-time performance is
not a requirement for offline analysis).

Decisive properties over alternatives (OpenPose, MoveNet, MMPose):

- **33 landmarks** including hands and feet, sufficient for serve
  kinematics (shoulder–elbow–wrist chain, hip, knee, ankle).
- **Two coordinate systems per landmark**: normalized image coordinates
  *and* metric 3D **world landmarks** (origin at hip center, unit meters,
  free of camera projection). The rule angles are computed in the image
  plane (the coordinate decision is argued in `angle_definitions.md`);
  persisting both systems additionally enables the empirical 2D-versus-3D
  cross-check of that choice without re-running the model.
- **Per-landmark `visibility` and `presence` scores** — the basis for the
  occlusion analysis in Stage 2's quality gating.
- **VIDEO running mode** with temporal tracking between frames, relevant
  for the fast phases of the serve.
- Lightweight installation (single pip package + one model file), no
  compiled dependencies (OpenPose) and no config-heavy framework (MMPose).

Known limitation, accepted consciously: BlazePose is a single-person
model trained mostly on everyday motion; extreme serve poses (full
overhead extension) and racket-arm self-occlusion are expected failure
modes. This is precisely why Stage 1 includes diagnostic instrumentation
rather than assuming clean output.

**Implementation note (documented as a finding):** the legacy
`mp.solutions.pose` API used in most tutorials and older literature was
removed in mediapipe ≥ 0.10.x. The implementation therefore uses the Tasks
API; this affects reproducibility of older published code.

### 2.2 Environment

| Component | Version / choice | Reason |
|---|---|---|
| Python | 3.10 | mediapipe compatibility |
| mediapipe | 0.10.35 | current Tasks API |
| Model file | `pose_landmarker_heavy.task` (float16) | accuracy over speed |
| OpenCV | 5.x | video decoding/encoding, drawing |
| numpy | 2.x | numerics |
| pytest | 9.x | automated tests |

All runtime dependencies are pinned in `requirements.txt`; development and
test tooling (`pytest`, `mypy`, `flake8`) is pinned in `requirements-dev.txt`.
The model file is identified by its download URL and excluded from version
control.

---

## 3. Data Acquisition and Preparation

### 3.1 Material selection criteria

Input videos (sourced from publicly available footage, e.g. YouTube) are
selected against explicit criteria, since input quality bounds everything
downstream:

- **One serve per clip**, trimmed manually (simplifies Stage 3 segmentation
  and makes clips directly comparable).
- **Frame rate ≥ 50 fps preferred**; at 30 fps the racket arm travels a
  large image distance between frames near contact, causing motion blur and
  tracking loss. Slow-motion recordings are ideal.
- **Camera angle**: side-on or behind-the-baseline, player large in frame;
  a single, fully visible player (single-person model).
- **Static camera preferred**; pans/zooms are tolerated by the model but
  add apparent motion to the normalized coordinates.

### 3.2 Preparation procedure

1. Download video-only stream (`yt-dlp`, best MP4 video track).
2. Trim to a single serve with re-encoding (`ffmpeg -c:v libx264`), *not*
   stream copy: stream copy cuts only at keyframes and can corrupt container
   timestamps, which Stage 1 relies on for the time axis.
3. Store under `data/` with a documented naming scheme; record source URL,
   original resolution/fps and trim points in a data manifest.

**Note for the thesis:** licensing/fair-use status of the footage and the
fact that no personal data beyond the public recording is processed should
be addressed in an ethics/data note.

---

## 4. Stage 1: Ingestion and Pose Extraction (implemented)

### 4.1 Architecture

Stage 1 is decomposed into four independently testable modules plus a shared
constants module and an orchestrator (package `serve_pipeline/`):

| Module | Responsibility | Key abstraction |
|---|---|---|
| `landmarks.py` | shared BlazePose 33-landmark topology | landmark ids/names and skeleton edges in one place, so every stage refers to the same constants without importing mediapipe |
| `ingestion.py` | video decoding, frame iteration, video metadata | `VideoReader` yields `(index, time_s, frame)`; rejects files with invalid FPS |
| `pose_extraction.py` | model inference | `PoseExtractor`, one instance per video (VIDEO mode requires strictly increasing timestamps; a guard enforces monotonicity even at sub-millisecond frame spacing) |
| `persistence.py` | raw data storage | streaming CSV writer (row-level flush: a crash mid-video leaves a valid, truncated file), schema-validating reader, statistics aggregation |
| `visualization.py` | diagnostic rendering | pure image-in/image-out functions (unit-testable without video I/O) |
| `stage1_extract.py` | orchestration & CLI | wires modules; contains no domain logic |

### 4.2 Data model and persistence schema

Per frame and landmark, the following is persisted **unmodified** (raw model
output; no smoothing, interpolation or unit conversion — those are Stage 2
decisions that must remain attributable to Stage 2):

```
frame, time_s, landmark_id, landmark_name,
x, y, z,                     # normalized image coordinates
visibility, presence,        # confidence scores in [0,1]
world_x, world_y, world_z    # meters, origin at hip center
```

Design decisions:

- **Long format** (one row per frame × landmark): schema-stable, trivially
  filterable per landmark, standard for time-series tooling.
- **Dense frame index**: frames *without* a detection still contribute their
  33 rows with empty value fields. Detection gaps are therefore explicit in
  the data instead of silent, and frame↔row alignment is guaranteed.
- **Metadata JSON** per run: video properties, model configuration and
  thresholds, mediapipe/pipeline versions, timestamp, detection statistics
  (detection rate, mean visibility per landmark). Every artifact is thereby
  self-describing.

### 4.3 Diagnostic visualization

The overlay video is not presentation output but a **measuring instrument**
for input quality:

- Skeleton drawn per frame from the extracted landmarks.
- Landmark color encodes the visibility score continuously from green
  (visible) to red (likely occluded) — racket-arm occlusion during trophy
  position and contact becomes directly observable.
- Frames without detection carry a "NO POSE" banner; every frame carries
  index and timestamp (HUD) so observations in the video can be mapped back
  to CSV rows exactly.
- A contact sheet (evenly spaced overlay frames as one PNG) supports quick
  triage of many clips without playing videos.

### 4.4 Validation of Stage 1

1. **Unit tests (19)** — ingestion on synthetic videos (frame count, order,
   timestamps, metadata), CSV round-trip (write → read → equality),
   schema rejection, undetected-frame handling, color-mapping endpoints,
   drawing robustness against out-of-frame coordinates, contact-sheet
   geometry.
2. **Integration tests** — extractor on random noise (must yield
   `detected=False`), on a real person image (33 landmarks, value ranges of
   all scores and coordinates), timestamp monotonicity at extreme frame
   rates. Model-dependent tests skip automatically if the model file is
   absent, keeping the pure unit suite runnable everywhere.
3. **End-to-end smoke test** — synthetic pan/zoom video generated from a
   real person photograph; expected result (100% detection, high visibility,
   stable skeleton under camera motion) was met.
4. **Acceptance on real material** — Stage 1 counts as validated for a given
   clip only after overlay inspection of that clip (detection rate,
   racket-arm visibility around contact).

---

## 5. Stage 2: Signal Processing and Kinematics (planned)

Input: exclusively the persisted Stage 1 CSV/JSON. The model is never
re-run; if Stage 1 output is insufficient, Stage 1 is re-executed with
different parameters and re-validated first.

### 5.1 Quality gating

- Per-landmark time series are masked where `visibility` falls below a
  threshold (to be calibrated against overlay observations, initial
  estimate 0.5): low-visibility values are treated as missing, not as data.
- Detection gaps (empty frames) and masked samples form explicit gap
  intervals; gap statistics per landmark are persisted.

### 5.2 Gap handling and filtering

- **Short gaps** (≤ 3 frames = 120 ms at 25 fps): interpolated by **linear**
  interpolation of the spatial coordinates; every interpolated sample is
  **flagged as interpolated** in the output so downstream metrics can be
  qualified. Linear is used over cubic spline because on ≤ 3-frame spans a
  spline offers no benefit and can overshoot.
- **Long gaps** (> 3 frames) and **edge gaps** (no valid neighbour on one
  side): not interpolated; affected samples are flagged unreliable and the
  phases that contain them are treated as unreliable.
- **Smoothing** *(decided)*: **zero-phase Butterworth, order 4, cut-off 5 Hz**,
  applied dual-pass via `filtfilt` (no phase shift) per contiguous reliable
  segment. Chosen empirically with a **filter-selection diagnostic**: a raw
  coordinate-velocity view (`filtering_velocity_compare.png`, central-difference
  d(y)/dt of the racket wrist), used only to break the tie because the position
  traces were visually indistinguishable across cut-offs. This diagnostic is a
  selection aid; it is *not* the joint angular velocity that Stage 2c computes
  and persists (§5.3), and it is not persisted. From it: at 25 fps a 5 Hz
  cut-off preserves the racket-arm velocity peaks while removing the > 5 Hz
  jitter that an 8 Hz cut-off leaves in, whereas a 3 Hz cut-off clipped the
  peaks by ~30–40 %. Savitzky–Golay remains available as a documented
  alternative. Filtering precedes differentiation, so the jitter it removes does
  not compound into the angular velocities (§5.3).

### 5.3 Kinematic computation

- **Joint angles** computed in the two-dimensional image plane from the
  projected in-plane coordinates, following the operational definitions in
  `angle_definitions.md`: trunk inclination against the image vertical, elbow
  flexion in the ISB convention (180° minus the geometric shoulder–elbow–wrist
  angle) and the optional front-knee flexion; hitting arm identified per clip.
  The 2D plane is chosen deliberately — it avoids the depth-estimation noise
  of the world landmarks during the fast, self-occluded phases around contact,
  at the cost of a projection error that is quantified rather than assumed
  (next point).
- **2D-versus-3D control**: each rule angle is additionally computed from the
  3D world landmarks at the same event frame, and the agreement is reported
  per angle and event (mean and maximum absolute difference in degrees). This
  makes the coordinate choice an examined decision; elbow flexion at impact is
  the primary case, being most exposed to projection error.
- **Angular velocities** by numerical differentiation of filtered series
  (differentiation amplifies noise — hence filtering *before*
  differentiation, order documented).
- Output: tidy CSV of derived series + JSON with all processing parameters
  (thresholds, filter type/order/cut-off, interpolation method) for full
  reproducibility.

### 5.4 Validation of Stage 2

Unit tests on synthetic signals with known ground truth (e.g. a synthetic
90° elbow must yield 90°; a known sine must survive filtering with
documented attenuation); visual validation via per-landmark plots
(raw vs. gated vs. filtered) and angle-over-time plots cross-checked
against the Stage 1 overlay video at selected frames.

---

## 6. Stage 3: Phase Segmentation and Evaluation (planned)

Input: exclusively persisted Stage 2 output.

### 6.1 Serve phase model

Segmentation of each clip into the phases established in tennis
biomechanics literature (e.g. start, ball toss/wind-up, trophy position,
acceleration, contact, follow-through). Phase boundaries are detected from
kinematic events, candidates:

- ball-toss onset: vertical velocity reversal of the tossing wrist,
- trophy position: maximal knee flexion / racket-arm configuration,
- contact: peak height and peak velocity of the hitting wrist,
- follow-through end: velocity decay below threshold.

⚠ *Open decision point: exact event definitions and thresholds, to be
fixed against annotated example clips.*

### 6.2 Metrics and evaluation

Per phase, kinematic metrics (peak angular velocities, key angles at
events, timing ratios between phases) are computed and compared against
reference values from literature. Ground-truth phase boundaries are
obtained by **manual frame-accurate annotation** of the evaluation clips;
segmentation quality is reported as temporal deviation (frames / ms) per
boundary.

### 6.3 Per-stage error attribution (thesis core)

For every observed deviation in the final output the persisted
intermediate representations are inspected in reverse order:

1. Is the deviation already present in the raw Stage 1 series
   (→ extraction error: occlusion/tracking, visible in overlay)?
2. Introduced by gating/interpolation/filtering (→ Stage 2 parameters)?
3. Introduced by event definition (→ Stage 3 logic)?

This analysis is only possible because every boundary is persisted — it
operationalizes the architectural commitment from section 1.1 and should be
presented in the thesis as the methodological contribution of the pipeline
design.

---

## 7. Cross-Cutting Quality Assurance

- **Version control:** git; a single linear mainline with each validated
  milestone tagged on it (`stage1-validated`, …), so any reported result is
  reproducible from a named commit; short-lived branches are used only for
  discardable experiments. Artifacts in `results/` and model binaries are
  not versioned (reproducible / re-downloadable). The producing commit
  hash — with a `-dirty` marker when the working tree has uncommitted
  changes — is embedded into each metadata JSON.
- **Automated tests and static checks:** pytest (model-dependent tests
  isolated and auto-skipping), plus `flake8` and a strict `mypy` run over
  `serve_pipeline` (complete type hints enforced via `disallow_untyped_defs`);
  all three must pass on `main` at every commit.
- **Parameter transparency:** no "magic numbers" in code paths that affect
  results — every threshold and filter parameter surfaces in the persisted
  metadata.
- **Limitations to report (threats to validity):** monocular 3D estimation
  (world landmarks are model estimates, not measurements; no absolute
  scale validation), single-person model, training-domain mismatch for
  extreme serve poses, video material heterogeneity (fps, angle,
  compression), no marker-based reference system available for absolute
  accuracy claims — accuracy statements are therefore made *relative*
  (plausibility, internal consistency, literature comparison), not
  absolute.
