# Feasibility Assessment - Implementation Spec

## Framing

- The assessment evaluates the internal validity of the **measurement chain**, examining whether the pipeline recovers the target criteria under monocular estimation. Serving quality itself is not evaluated.
- Geometric ground truth is prescribed analytically rather than captured through an external 3D optical system. Key-event detection is evaluated against manual video annotations.
- Alignment with investigatory questions:
  - **Q1 (extractable criteria)**: Defined by criteria selection and pipeline stages; outcomes are summarized directly.
  - **Q2 (kinematic stability)**: Evaluated across landmark noise, camera viewpoint, and key-event detection in Sections 2 and 3.
  - **Q3 (reliability boundary)**: Evaluated via the decidability criterion in Section 3 and the flags the pipeline returns on the corpus (`serve_pipeline/report.py` -> `indicators.csv`).

---

## 1. Error budget

Total measurement error (reported angle minus true angle) is structured into four components:

| # | Error source | Definition | Methodological treatment | Quantifiable | Primary output artifact |
|---|--------------|------------|--------------------------|--------------|-------------------------|
| E1 | **Pose estimation error** | Estimated landmark minus true image position | Not measurable: the clips carry no hand-annotated landmarks and the model card reports only a torso-normalised detection rate, no pixel-level error. Sigma is therefore swept over a plausible range | Swept, not measured | Induced angular spread over sigma band |
| E2 | **Projection error** | True spatial angle minus monocular projected angle | Computed from the projection geometry alone, without video dependencies | Yes | Projected-angle curves over theta |
| E3 | **Event error** | Detected frame minus true event instant, including the offset built into the pelvis and wrist proxies | Measured against manual key-frame video annotations | Yes | Tolerance shares, large errors and offset distribution |
| E4 | **Definitional mismatch** | Estimator keypoint definition vs. anatomical joint centres behind the reference values | Documented qualitatively under limitations; would need 3D motion capture of the same serves. Trunk inclination carries a second offset (read against the vertical, reference taken against the pelvis) | No | Qualitative discussion |

E1-E3 enter the analysis numerically. E4 is treated as a documented, unquantified offset and is not simulated.

---

## 2. Projection and noise propagation (E2 + E1)

Projection distortion and landmark noise propagation are evaluated synthetically via Monte Carlo simulation without video dependencies.

### 2a. Projection (E2, analytic and numeric)
- Camera is assumed level and turned about the vertical axis; perspective is approximated by scaled orthographic projection. The approximation is close only when the body's depth relief is small against the camera distance and the player stays near the image centre. At the admitted distances of at most about 4 m the arm's relief is not small, so the results indicate the size of the projection effect rather than fix it exactly.
- **Single inclination (trunk)**: Evaluated via the closed-form equation `tan(a_proj) = tan(a_true) * cos(theta)`.
- **Two-segment joints (knee, elbow, shoulder)**: The segments are placed symmetrically about the vertical at the prescribed angle, their shared plane is turned by theta and the projected angle is re-read numerically.
- The viewpoint angle theta (between motion plane and image plane) is swept across the range `[0, 45]` deg in increments of 5 deg. 0 deg (camera facing the motion plane head-on) cannot be held in practice, since the player's lean direction is not known before the serve. The upper bound stays just below the 46 deg at which projection alone moves a trunk inclination at its reference mean out of its band.

### 2b. Landmark noise propagation (E1, Monte Carlo)
- Each 2D landmark is perturbed independently of the others by isotropic zero-mean Gaussian noise with standard deviation sigma in pixels per image axis.
- The resulting angular spread is estimated via Monte Carlo sampling (N = 10,000 draws, fixed seed 42) per criterion, theta and sigma.
- Landmarks are placed at representative Winter body-segment proportions on a stature of 600 px, and each criterion is evaluated separately at its own reference mean. Shorter arm segments (elbow, shoulder) exhibit higher angular sensitivity to pixel perturbations than longer leg and trunk segments.
- Sigma is evaluated across a parameter sweep (`config.sigma_sweep = (2.0, 3.0, 4.0, 5.0, 6.0)` px, about 0.3-1 % of stature). 2 px stands for clean footage; 6 px is the largest error still plausible under the admission criteria (whole body framed at a size that resolves the limbs).

---

## 3. Decision stability and decidability criterion (Q2 + Q3)

Because the pipeline outputs qualitative indicators rather than continuous angles, the assessment measures whether the induced angular spread remains sufficiently small to maintain reliable classification.

### 3a. Decidability criterion (Q3 threshold)
- For each criterion and each (theta, sigma) cell of the sweep, the induced angular standard deviation is held against the reference band half-width (`1.0 * rule.sd`, the distance from reference mean to band edge; this also covers the one-sided knee band).
- **Decidable cell**: The induced standard deviation stays strictly below the band half-width.
- **Unreliable cell**: The induced standard deviation equals or exceeds the band half-width.
- A criterion counts as decidable only when every cell of the swept range is decidable.
- The threshold uses the same factor of one as the bands. The noise level and viewpoint at which a criterion first turns unreliable (onset sigma and breakdown theta) define the Q3 reliability boundary.

### 3b. Event-detection stability (E3)
- A single observer steps frame by frame through each clip's pose overlay and marks one trophy and one impact frame, without seeing the detected frames. Frames are counted with the pipeline's frame index.
- Manual definitions are the racket-based ones the landmark proxies approximate: trophy at the racket's first vertical peak over the loaded legs, ball impact at visible racket-ball contact.
- The event error is the offset between detected and manual frame, reported as the share of locatable clips whose |offset| exceeds 1, 3 or 5 frames and the share of large errors of 30 frames or more. `event_error.json` carries these counts and shares plus the median, IQR, mean and maximum offset.
- Detected frames are not corrected; the angles are read at them, so the event error passes into the flags.

---

## 4. Assessment artifacts

Outputs are structured as machine-readable tables and reproducible figures:

| Artifact | Error source | Content |
|----------|--------------|---------|
| `projection_curves.csv` | E2 | Projected angle as a function of theta per criterion |
| `noise_propagation.csv` | E1+E2 | Induced angular standard deviation over theta and sigma sweeps |
| `event_error.json` | E3 | Tolerance shares, large errors and offset distribution from manual annotations |
| `decidability.csv` | 3a | Induced SD vs. band half-width ratio, decidable status, and onset points |
| `run_meta.json` | Metadata | Complete configuration parameters and provenance for reproduction |
| `figures/` | All | Rendered projection curves, spread vs. theta plots, and decidability maps |

---

## Module layout

```
assessment/
  annotation.py    # E3: Manual annotation ingestion and event-error statistics
  projection.py    # E2: Analytic and numerical projection modeling over theta
  propagation.py   # E1+E2: Monte Carlo landmark noise propagation over (theta, sigma)
  decidability.py  # 3a: Decidability ratio evaluation and breakdown localization
  report.py        # Orchestrator: E3 plus the synthetic sweep, artifacts and figures under results/assessment/
```

Shared configuration parameters (`theta_range`, `sigma_sweep`, `mc_samples`, `seed`) are loaded centrally from `PipelineConfig`.

