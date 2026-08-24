# Processing Pipeline — Implementation Spec

Extracted from methodology.tex, Section 3.5 (`sec:meth_pipeline`). Pairs with
`rule_base_spec.md` (Stage 4/5 use those rules).

The pipeline turns **one recording** into deviation indicators in **five stages**:

```
recording -> [1] pose extraction -> [2] preprocessing -> [3] key-event detection
          -> [4] angle computation -> [5] rule evaluation -> deviation indicators
```

Per-clip parameters (recorded manually, passed in): **serving arm**, **front leg**,
**camera plane** (frontal / sagittal), **view direction**, **frame rate (fps)**.

**Viewpoints, incl. back view.** The frontal plane can be faced from **in front of OR behind** the
player; both a front view and a **back (posterior) view** support the trunk-inclination criterion.
Back-view clips are explicitly accepted (they are common in online-scraped footage). So:
- `camera_plane = "frontal"` covers both front-facing and back-facing views; `"sagittal"` is the
  side view (knee flexion). An oblique view is rejected (faces no plane cleanly).
- `view_direction` records the actual facing for provenance: `"front"`/`"back"` when frontal,
  `"left"`/`"right"` when sagittal. It does **not** change which criterion is available.
- **No mirroring correction is needed.** All four angles are computed as unsigned magnitudes
  (`atan2(|u x v|, u.v)`, 0..180), so the left-right flip between a front and a back view does not
  change any angle. Trunk inclination is two-sided, so its lean direction is irrelevant either way.
- `serving_arm` and `front_leg` are **anatomical** (body-relative): MediaPipe labels left/right by
  the person's body, consistently across views. Record the player's actual serving arm / front leg,
  not "the side on the left of the screen".

**Frame rate (`fps`), incl. untagged slow-motion.** Record the **container/playback fps** of the
file you actually have, and do **not** try to guess a true capture rate. Scraped footage (e.g.
YouTube) is often slow-motion re-encoded to a normal playback rate, so its capture rate is not
recoverable — the frames you process are sampled at the container fps, which is the correct
operating rate for the 120 ms gap bound and the 8 Hz filter (both stay internally consistent with
that signal). Consequences to keep in mind:
- The **core outputs do not depend on fps**: the four angles are per-frame geometry and the key
  events are frame-index extrema. Untagged slow-motion does not corrupt the angles or the verdicts.
- Do **not** rely on absolute real-world seconds. Report events by **frame index**; the assessment
  reports event error as a **rate**. Slow-motion actually *helps* localize the brief ball impact,
  which is why the methodology prefers it.
- The one real effect: slow-motion pushes true motion frequencies down, so the fixed 8 Hz filter
  under-smooths a slowed clip relative to a normal-speed one. This is acceptable and conservative
  (it keeps the peaks); do **not** rescale the cut-off for untagged footage. Note it under the
  differing-frame-rate limitation.
- **Optional QC flag (diagnostic, not a correction):** a real trophy-to-contact spans ~0.5-1.0 s of
  real time, so if `(impact_frame - trophy_frame) / fps` is much larger than ~1 s, the clip is
  likely slow-motion. Log it as a flag; never convert or "fix" the fps from it.

---

## Stage 1 — Pose extraction

- **Decode** the video frame by frame with **OpenCV**.
- **Estimator**: MediaPipe **Tasks API PoseLandmarker**, **heavy** model.
  - mode = **VIDEO**, track a **single pose**.
  - detection / tracking / presence thresholds = **0.5** (defaults).
  - segmentation output = **disabled**.
  - **No temporal smoothing** — the Tasks API offers none, and video mode only reuses the
    previous detection to place the tracking region; it does not filter coordinates.
    → Stage 2 is therefore the *only* temporal filter.
- **Per frame** the model returns **33 landmarks** in **normalized** image coordinates.
  - Keep for each landmark: `x`, `y` (image plane) and `visibility`.
  - **Discard** depth (`z`) and world coordinates.
  - Frames with **no detected pose**: mark as such but **still carry** them, so the series stays
    dense across frames and landmarks (no reindexing).
- **Output**: 2D landmark trajectories `[T frames x 33 landmarks x (x, y, visibility)]`.

---

## Stage 2 — Landmark preprocessing

Conditions each trajectory in a short, deliberately simple sequence. Implemented with **NumPy /
SciPy**. Parameters are conventional, not per-recording tuned (feasibility study, not an optimised
measurement chain).

**Order: (a) visibility gating → (b) short-gap interpolation → (c) low-pass smoothing.**

### (a) Visibility gating
- At every frame, a landmark is **reliable** iff `visibility >= 0.5`, else **unreliable**.
- **Do not discard** samples — store the reliability mark next to the coordinate, so every
  rejection stays traceable (undetected pose vs. below-threshold landmark).
- A criterion is later evaluated only where its landmarks are reliable **at the key frame**;
  an occluded landmark makes that criterion `unavailable`, never a wrong angle.

### (b) Short-gap interpolation
- Bridge an **interior gap** only if it is bounded by reliable samples on **both** sides **and**
  is **no longer than 120 ms**.
- Fill by **linear interpolation** between the two reliable endpoints.
- The **120 ms** bound is defined **in time**, converted **per clip** to a frame count:
  `max_gap_frames = round(0.120 * fps)`. Same physical gap length holds from 30 to 240 fps.
- Over-length gaps and edge gaps are **never filled** (a reliable run simply ends there).
- Mark interpolated samples as **interpolated** (distinct from originally reliable) — Stage 3
  needs this distinction.

### (c) Low-pass smoothing
- Filter: **2nd-order Butterworth low-pass**, **8 Hz** cut-off, applied **zero-phase**
  (forward + backward, e.g. `scipy.signal.filtfilt`) so trajectories are not shifted in time.
  - Two passes double the effective order to 4th and lower the half-power point slightly below
    8 Hz — a small fixed offset the feasibility setting accepts.
- The **8 Hz physical cut-off is fixed** across recordings; **redesign the coefficients per clip**
  because the normalized cut-off = `8 / (fps/2)` depends on the clip's Nyquist frequency
  (`fps/2`). 8 Hz stays below Nyquist at every admitted fps (min 30 → Nyquist 15 Hz).
  - (8 Hz, not the lower locomotion cut-offs, because the fast racket-arm motion near impact
    holds higher-frequency content.)
- Run the filter over **each maximal run of reliable + interpolated samples separately**:
  - An unfilled gap never bridges two runs.
  - An unreliable sample outside a run cannot leak into a reliable neighbour across the window.
  - A run **too short** for the filter is carried **unsmoothed**.

---

## Stage 3 — Key-event detection

Locate two frames from **body landmarks only** (no racket / ball detector — the racket only names
the instant, it never enters a measured angle). **Run in this order:**

### 3.1 Ball impact (first)
- **Proxy**: frame where the **racket-arm wrist is highest** = **minimum y** of the racket-arm
  wrist over the whole clip (image y grows downward). Racket arm = serving arm (per-clip param).
- Coincides with the extended reach at contact.

### 3.2 Trophy position (second)
- **Proxy**: frame where the **pelvis is lowest** = **maximum y** of the **mid-hip** (midpoint of
  the two hip landmarks), searched only over frames **before ball impact**.
- Pelvis is used (not the knee angle or racket) because its **vertical image component is
  preserved** under orthographic projection, so it is recoverable in both frontal and sagittal
  views, whereas the knee angle is foreshortened frontally.
- Both events are **extrema of a single coordinate series**, computed with NumPy over the
  **filtered** trajectory.

### Guard conditions (both must hold, else event = `not locatable`)
1. **Originally reliable sample only.** The event frame must sit on an *originally reliable*
   sample — **not** an interpolated one (a linear fill is monotonic, holds no interior extremum;
   matters most for the brief impact peak that a ≤120 ms gap can hide). A boundary extremum
   sitting at the edge of an unfilled/over-length gap is **not admitted** (the true extremum may
   lie inside the unobserved gap).
2. **Wrist-above-trophy check.** Ball impact is accepted only when the wrist there lies **above
   its trophy height** *and* a **non-degenerate window** separates the two frames. (Guards against
   incomplete extension / low contact putting the global wrist minimum in the loading region and
   collapsing the trophy search window.)
- On failure, report the event as **not locatable** rather than returning it from an interpolated
  sample, a run edge beside an unfilled gap, or a collapsed window.

**Note (shared-input dependence, by design):** pelvis, trunk inclination, and front knee flexion
all derive from the hip landmarks, so the trophy frame and the two angles read at it are not
independent — a hip-landmark error shifts both. Documented, not corrected here.

---

## Stage 4 — Angle computation

Read the four candidate angles from the **filtered** trajectories at the located key frames:
trunk inclination + front knee flexion at the **trophy** frame; elbow flexion + shoulder
elevation at the **ball-impact** frame. Compute an angle **only** when its landmarks are reliable
at that frame, else `unavailable`.

Two conventions before any angle is formed:
- **Rescale normalized coords to pixels**: `x_px = x * width`, `y_px = y * height` (else the frame
  aspect ratio distorts every angle). Only the two pixel coords enter — depth is discarded.
- **Angle between two planar vectors** `u`, `v`:
  ```
  theta = atan2( |u_x*v_y - u_y*v_x| , u_x*v_x + u_y*v_y )   # 0..180 deg, stable
  ```

The four angles (full detail in `rule_base_spec.md`):
- **Front knee flexion** — turning angle: `u = hip->knee`, `v = knee->ankle` (straight = 0). Front-leg side.
- **Elbow flexion** — turning angle: `u = shoulder->elbow`, `v = elbow->wrist`. Serving-arm side.
- **Shoulder elevation** — spanned at serving shoulder: `shoulder->elbow` vs. `shoulder->hip` (same side). Arm along trunk = 0.
- **Trunk inclination** — trunk axis `mid-hip -> mid-shoulder` vs. image **vertical upward (0,-1)**
  (upward because normalized y grows downward; recovers the 25.0° reference, not its 180° complement).
  Mid-hip / mid-shoulder = midpoints of left+right landmarks.

---

## Stage 5 — Rule evaluation & indicator generation

- Compare each computed angle against its band (from `rule_base_spec.md`):
  - Trunk inclination, elbow flexion, shoulder elevation → **symmetric** band `mean ± 1·SD`.
  - Front knee flexion → **one-sided lower bound** at `mean - 1·SD`.
- Each criterion returns **inside** or **outside** its reference range.
- **Output** = the set of **deviation indicators**, each naming the criterion (and, for the knee,
  the flagged direction = insufficient flexion).
- **Availability**: an indicator is produced only where **all** conditions hold together —
  (1) the **camera plane supports** the criterion (trophy criteria are plane-bound: trunk =
  frontal, front knee = sagittal → only one per recording), (2) the **key frame is locatable**,
  (3) the **landmarks are reliable** at that frame. Otherwise the criterion is **unavailable**,
  never forced to a verdict.
- Indicators are **attention flags for a coach**, not an automated verdict on the serve.

---

## Suggested module layout

```
pipeline/
  extract.py     # Stage 1: OpenCV decode + MediaPipe Tasks PoseLandmarker (heavy, VIDEO)
  preprocess.py  # Stage 2: gating -> interpolate_short_gaps -> butterworth_filtfilt (per-run)
  keyevents.py   # Stage 3: detect_impact, detect_trophy + guard conditions
  angles.py      # Stage 4: rescale_to_px, angle_between, four angle readers
  rules.py       # Stage 5: RULES table + evaluate() (see rule_base_spec.md)
  run.py         # orchestrates the five stages for one clip; takes per-clip params
```

Per-clip params to thread through: `serving_arm`, `front_leg`, `camera_plane`, `view_direction`,
`fps`, `frame_width`, `frame_height`.
