# Stage 1 Validation Report: Pose Extraction

Validation of the Stage 1 implementation (`serve_pipeline/`, ingestion +
BlazePose extraction + persistence + diagnostic overlay) according to the
stage-wise validation requirement defined in `methodology.md`, section 1.2.

Validation date: 2026-07-10
Pipeline version: 0.1.0 · mediapipe 0.10.35 · model: `pose_landmarker_heavy.task` (float16)

Stage 1 was validated on three levels: automated tests (4.1), a synthetic
end-to-end smoke test (4.2), and acceptance on real serve footage (4.3).

---

## 1. Automated tests

19 pytest tests, all passing. Coverage by module:

| Module | Tests | Verified properties |
|---|---|---|
| `ingestion.py` | 3 | frame count, ordering and timestamps on a synthetic video with frame-encoded pixel values; metadata correctness (fps, dimensions); error on missing file |
| `persistence.py` | 6 | CSV write→read round-trip equality (detected and undetected frames, all 8 value fields, 1e-6 tolerance); dense schema: exactly 33 rows per frame with empty values for undetected frames; rejection of foreign CSV schemas; metadata JSON round-trip; statistics aggregation incl. empty-input edge case |
| `visualization.py` | 7 | visibility→color mapping endpoints (1.0→green, 0.0→red) and out-of-range clipping; connection table validity (all indices < 33); drawing operates on a copy, never mutates input; "NO POSE" warning rendering; robustness against out-of-frame coordinates; contact-sheet tiling geometry |
| `pose_extraction.py` (integration, model required) | 3 | random-noise input yields `detected=False`; real person image yields 33 landmarks with all scores/coordinates in valid ranges; internal timestamp monotonicity guard at simulated 10 000 fps |

Model-dependent tests skip automatically when the `.task` file is absent,
keeping the pure unit suite environment-independent.

## 2. Synthetic end-to-end smoke test

Input: 60-frame video (640×480, 30 fps) synthesized by panning/zooming over
a photograph of a person (`data/smoke_test.mp4`), so that pose content is
real but camera motion is controlled.

Result: detection rate **100%** (60/60 frames), mean visibility **0.96**;
CSV contains exactly 1 + 60 × 33 rows; skeleton visually stable under the
artificial camera motion (contact sheet inspected). All four output
artifacts produced.

## 3. Acceptance on real serve footage

### 3.1 Material

`data/serve_01.mp4`: professional player, practice serve, side-on camera,
1920×1080, **25 fps**, 838 frames (33.5 s). The clip contains the full
serve plus preparation and post-serve movement (turning away from camera).

### 3.2 Quantitative results

- Detection rate: **100%** (838/838 frames)
- Mean visibility over all landmarks: **0.95**
- Mean visibility, key landmarks: shoulders/hips 1.00, right (racket) arm
  0.94–0.96, left arm 0.77–0.87, legs 0.93–0.98
- Contact window (identified via wrist-height maximum in the raw CSV,
  frames ~660–690, t ≈ 26.4–27.6 s): racket-arm landmarks
  (shoulder/elbow/wrist) hold visibility **≥ 0.97 throughout** — no
  racket-arm occlusion breakdown at contact on this material/camera angle.

### 3.3 Qualitative inspection

Overlay video and zoomed crops of the contact window
(`results/serve_01_contact_zoom*.png`) show the skeleton correctly locked
onto the player through leg drive, extension, contact and follow-through
(including the characteristic back-leg kick). No identity switches to
spectators, no frame-level tracking loss.

### 3.4 Findings

1. **Low-visibility episodes lie outside the serve.** The global minima
   (left elbow ≈ 0.04) cluster at t ≈ 30 s and t ≈ 33 s — after the serve,
   when the player turns away from the camera. Expected model behavior;
   demonstrates that the visibility instrumentation localizes unreliable
   segments as intended.
2. **Clip framing.** The 33.5 s clip contains substantial non-serve motion;
   whole-clip statistics are therefore diluted. Consequence: trim analysis
   clips to the serve, or restrict later-stage analysis to the segmented
   serve window.
3. **Temporal resolution is the binding constraint, not tracking.** At
   25 fps the acceleration phase between racket drop and contact spans only
   a few samples; angular-velocity estimates near contact will be coarse.
   Consequence for material selection: prefer ≥ 50 fps or slow-motion
   footage for the main analysis; `serve_01` remains as validation
   material.

## 4. Verdict

All three validation levels passed. **Stage 1 is accepted**; Stage 2
(signal processing and kinematics) may consume its persisted output.
Constraint carried forward: metric quality near contact is limited by
source frame rate, not by extraction quality — to be revisited in material
selection (methodology, section 3.1) and in the thesis' limitations
discussion.
