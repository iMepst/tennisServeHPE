# Processing Pipeline - Implementation Spec

The pipeline transforms a video recording into technical deviation indicators across five stages:

```
recording -> [1] pose extraction -> [2] preprocessing -> [3] key-event detection
          -> [4] angle computation -> [5] rule evaluation -> deviation indicators
```

Per-clip parameters (recorded manually and passed at runtime): **serving arm**, **front leg**, **camera plane** (frontal / sagittal), **view direction**, and **frame rate (fps)**.

**Viewpoints and camera orientations.** The frontal plane can be recorded from the front or from behind the player; both front and posterior (back) views support the trunk inclination criterion. Posterior views are accepted directly from scraped recordings:
- `camera_plane = "frontal"` encompasses both anterior and posterior views; `"sagittal"` specifies the lateral view (knee flexion). Oblique viewpoints are excluded.
- `view_direction` records the facing orientation for provenance (`"front"`/`"back"` for frontal, `"left"`/`"right"` for sagittal) without altering criterion availability.
- **No mirroring correction required.** All angles are calculated as unsigned planar magnitudes (`atan2(|u x v|, u.v)` in [0, 180] deg). The left-right reflection between anterior and posterior views does not alter the resulting angle. Trunk inclination uses a two-sided band, making the sign of lateral lean irrelevant.
- `serving_arm` and `front_leg` denote anatomical (body-relative) sides. MediaPipe consistently labels anatomical landmarks across viewpoints. These parameters refer to the player's anatomical limbs rather than screen-space positions.

**Frame rate (`fps`) and untagged slow-motion.** The container/playback fps of the video file is recorded directly without attempting to infer an unrecoverable capture rate. Online footage often contains slow-motion sequences re-encoded to nominal playback rates; sampling at the container rate provides a consistent temporal baseline for the 120 ms interpolation bound and the 8 Hz filter:
- **Core outputs are invariant to fps**: the four joint angles reflect per-frame geometry and key events correspond to frame-index extrema.
- Events are identified by **frame index** rather than absolute physical time; event detection accuracy is evaluated as an offset distribution and move rate. Higher recording rates improve the temporal resolution of the brief impact phase.
- In slow-motion recordings, physical motion frequencies are scaled downward, meaning the fixed 8 Hz filter provides light smoothing relative to real-time recordings. This preserves peak kinematics without rescaling the filter cut-off.
- **QC diagnostic flag:** Under real-time conditions, the trophy-to-impact interval spans approximately 0.5 to 1.0 s. If `(impact_frame - trophy_frame) / fps` exceeds 1.0 s, the clip is flagged as `likely_slow_motion` for quality control without modifying the processing rate.

---

## Stage 1 - Pose extraction

- Video decoding is performed frame-by-frame using OpenCV.
- **Estimator**: MediaPipe PoseLandmarker (Tasks API, `heavy` model).
  - Mode: `RunningMode.VIDEO`, tracking a single person (`num_poses = 1`).
  - Confidence thresholds: detection = 0.5, tracking = 0.5, presence = 0.5.
  - Segmentation masks: disabled.
  - No internal temporal smoothing is applied by the Tasks API; Stage 2 provides the single filtering stage.
- **Per-frame output**: 33 landmarks in normalized image coordinates `(x, y)` in [0, 1] with an associated `visibility` score.
  - Depth (`z`) and world coordinates are discarded (2D operating point).
  - Frames without a detected pose are retained with unpopulated fields to maintain a contiguous, fixed-length time series.
- **Persisted output**: 2D landmark trajectories `[T frames x 33 landmarks x (x, y, visibility)]`.

---

## Stage 2 - Landmark preprocessing

Conditioning is applied sequentially using NumPy and SciPy. Parameters follow standard biomechanical conventions:

**Processing sequence: (a) visibility gating -> (b) short-gap interpolation -> (c) low-pass filtering.**

### (a) Visibility gating
- At each frame, a landmark sample is classified as reliable if `visibility >= 0.5`, otherwise unreliable.
- Samples are retained and marked with their gating state (`ok`, `undetected`, `low_visibility`) to maintain traceability.
- Downstream rules evaluate angles only when required landmarks meet the reliability threshold at the respective key frame; occluded landmarks result in an `unavailable` status.

### (b) Short-gap interpolation
- Interior gaps bounded by reliable samples on both ends and spanning at most 120 ms are filled via linear interpolation.
- The temporal threshold is converted to frames per clip: `max_gap_frames = round(0.120 * fps)`.
- Boundary gaps and gaps exceeding 120 ms remain unpopulated and are marked as unreliable.
- Interpolated samples are flagged explicitly (`interpolated = True`) to distinguish them from original measurements.

### (c) Low-pass filtering
- Filter: 2nd-order Butterworth low-pass with an 8 Hz cut-off frequency, applied zero-phase (`scipy.signal.filtfilt`, yielding an effective 4th-order response without phase shift).
- The 8 Hz physical cut-off is fixed; normalized cut-off coefficients `wn = 8 / (fps / 2)` are computed per clip. The 8 Hz threshold preserves rapid limb motion prior to impact while attenuating high-frequency landmark jitter.
- Filtering is executed independently across each contiguous segment of valid/interpolated samples. Unfilled gaps isolate separate segments; runs below the minimum filter length (`3 * (order + 1) + 1`) are retained without smoothing.

---

## Stage 3 - Key-event detection

Two kinematic events are detected from body landmarks without requiring ball or racket tracking:

### 3.1 Ball impact
- **Kinematic proxy**: frame of minimum vertical coordinate ($y$-minimum, highest image position) for the racket-arm wrist across the complete sequence.
- Identifies the instant of maximum upward extension.

### 3.2 Trophy position
- **Kinematic proxy**: frame of maximum vertical coordinate ($y$-maximum, lowest image position) for the pelvis midpoint (mean of left and right hip landmarks), evaluated strictly prior to the detected ball impact.
- The vertical pelvis trajectory remains invariant under orthographic projection across frontal and sagittal viewpoints.

### Guard conditions (both required for valid event localization)
1. **Originally reliable sample required**: The located extrema must fall on originally measured samples (`valid = True`, `interpolated = False`) and cannot border an unpopulated gap.
2. **Kinematic consistency and temporal separation**: The wrist position at ball impact must be strictly higher in the image plane than at the trophy position (`wrist_y[impact] < wrist_y[trophy]`), separated by a non-degenerate window of at least two frames.
- If either condition is violated, the event is marked as `not locatable`.

**Input dependence note:** Pelvis position, trunk inclination, and front knee flexion all incorporate hip landmarks; errors in hip estimation influence both event localization and angle calculation simultaneously.

---

## Stage 4 - Angle computation

Planar angles are computed from filtered landmark coordinates at the detected key frames:
- Trophy frame: trunk inclination and front knee flexion.
- Ball impact frame: elbow flexion and shoulder elevation.

Calculation conventions:
- **Coordinate scaling**: Normalized coordinates are converted to pixel space (`x_px = x * width`, `y_px = y * height`) to eliminate aspect ratio distortion.
- **Planar vector angle formula**:
  ```
  theta = atan2( |u_x * v_y - u_y * v_x| , u_x * v_x + u_y * v_y )
  ```
  Evaluated in [0, 180] degrees using the cross product magnitude against the dot product for numerical stability.

Angle definitions (detailed in `rule_base_spec.md`):
- **Front knee flexion**: Turning angle between hip-to-knee and knee-to-ankle vectors on the front-leg side (extended = 0 deg).
- **Elbow flexion**: Turning angle between shoulder-to-elbow and elbow-to-wrist vectors on the serving-arm side (extended = 0 deg).
- **Shoulder elevation**: Enclosed angle between upper arm (shoulder-to-elbow) and trunk axis (shoulder-to-hip) on the serving side (arm along torso = 0 deg).
- **Trunk inclination**: Enclosed angle between trunk axis (mid-hip to mid-shoulder) and the vertical upward reference `(0, -1)`.

---

## Stage 5 - Rule evaluation and indicator generation

- Evaluated angles are compared against reference ranges from Jacquier-Bret et al. (2024):
  - Symmetric band (`mean +/- 1 SD`): Trunk inclination, elbow flexion, shoulder elevation.
  - One-sided lower bound (`mean - 1 SD`): Front knee flexion (flagging insufficient flexion; deeper flexion remains unpenalized).
- Status classifications: `inside`, `outside`, or `unavailable`.
- **Availability gate**: An indicator is generated only when:
  1. The camera plane matches the criterion (frontal for trunk inclination, sagittal for knee flexion).
  2. The corresponding key frame is successfully localized.
  3. All constituent landmarks are reliable at that frame.
  Otherwise, the indicator is assigned `unavailable`.

---

## Module layout

```
serve_pipeline/
  extract.py         # Stage 1: Video decoding, MediaPipe pose extraction, artifact persistence
  process.py         # Stage 2: Orchestration for gating (2a) and interpolation/filtering (2b)
  gating.py          # Stage 2a: Visibility thresholding and gap statistics
  interpolation.py   # Stage 2b: Linear gap filling under 120 ms
  filtering.py       # Stage 2b: Butterworth zero-phase filtering
  keyevents.py       # Stage 3: Impact and trophy event detection with guard conditions
  angles.py          # Stage 4: Pixel coordinate scaling and planar vector angle functions
  rules.py           # Stage 5: Reference band definitions and indicator evaluation
  run.py             # Pipeline orchestrator processing a single clip end to end
```

Runtime parameters: `serving_arm`, `front_leg`, `camera_plane`, `view_direction`, `fps`, `frame_width`, `frame_height`.
