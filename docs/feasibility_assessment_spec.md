# Feasibility Assessment - Implementation Spec

## Framing

- The assessment evaluates the internal validity of the **measurement chain**, examining whether the pipeline recovers the target criteria under monocular estimation. Serving quality itself is not evaluated.
- Geometric ground truth is prescribed analytically rather than captured through an external 3D optical system. Key-event detection is evaluated against manual video annotations.
- Alignment with investigatory questions:
  - **Q1 (extractable criteria)**: Defined by criteria selection and pipeline stages; outcomes are summarized directly.
  - **Q2 (kinematic stability)**: Evaluated across landmark noise, camera viewpoint, and key-event detection in Sections 2 and 3.
  - **Q3 (reliability boundary)**: Evaluated via the decidability criterion in Section 3.

---

## 1. Error budget

Total measurement error (reported angle minus true angle) is structured into four components:

| # | Error source | Definition | Methodological treatment | Quantifiable | Primary output artifact |
|---|--------------|------------|--------------------------|--------------|-------------------------|
| E1 | **Pose estimation error** | Estimated landmark minus true image position | Noise parameter sigma informed by model card accuracy (BlazePose PDJ) and evaluated across a sensitivity sweep | Yes | Induced angular spread over sigma band |
| E2 | **Projection error** | True spatial angle minus monocular projected angle | Computed analytically from the projection equations without video dependencies | Yes | Projected-angle curves over theta |
| E3 | **Event error** | Detected frame minus true event instant | Measured against manual key-frame video annotations | Yes | Offset distribution and move rates |
| E4 | **Definitional mismatch** | Surface landmarks vs. internal joint centers | Documented qualitatively under limitations (most pronounced for trunk inclination) | No | Qualitative discussion |

E1-E3 are quantified numerically. E4 is treated as a documented, unquantified offset and is not simulated.

---

## 2. Projection and noise propagation (E2 + E1)

Projection distortion and landmark noise propagation are evaluated synthetically via Monte Carlo simulation without video dependencies.

### 2a. Projection (E2, analytic and numeric)
- Camera orientation is assumed level with orthographic projection (applicable for a distant subject relative to body height).
- **Single inclination (trunk)**: Evaluated via the closed-form equation `tan(a_proj) = tan(a_true) * cos(theta)`.
- **Two-segment joints (knee, elbow, shoulder)**: Evaluated numerically by rotating segment vectors out of plane by theta and recomputing the enclosed projected angle.
- The viewpoint angle theta (between motion plane and image plane) is swept across the range `[0, 45]` deg in increments of 5 deg to evaluate sensitivity to camera positioning and torso rotation.

### 2b. Landmark noise propagation (E1, Monte Carlo)
- Each 2D landmark is perturbed by isotropic zero-mean Gaussian noise with standard deviation sigma in pixels.
- The resulting angular spread is estimated via Monte Carlo sampling (N = 10,000 draws).
- Each criterion is evaluated independently using representative segment lengths based on Winter body proportions. Shorter arm segments (elbow, shoulder) exhibit higher angular sensitivity to pixel perturbations than longer leg and trunk segments.
- Sigma is evaluated across a parameter sweep (`config.sigma_sweep = (2.0, 3.0, 4.0, 5.0, 6.0)` px) to provide a sensitivity profile across noise levels.

---

## 3. Decision stability and decidability criterion (Q2 + Q3)

Because the pipeline outputs qualitative indicators rather than continuous angles, the assessment measures whether the induced angular spread remains sufficiently small to maintain reliable classification.

### 3b. Event-detection stability (E3)
- Evaluated from the empirical offset between automatically detected and manually annotated key frames (trophy position and ball impact).
- Reported as move rates across multiple frame tolerances and robust offset distributions (median, IQR, max offset, large-failure counts).

### 3c. Decidability criterion (Q3 threshold)
- The induced angular standard deviation (from projection and landmark noise) is held directly against the reference band half-width (`1.0 * rule.sd`).
- **Decidable**: The induced standard deviation remains strictly below the band half-width across the viewpoint (theta) and noise (sigma) sweep.
- **Unreliable**: The induced standard deviation equals or exceeds the band half-width (where measurement scatter reaches the distance from band center to edge).
- The transition point (onset sigma and breakdown theta) defines the Q3 reliability boundary.

---

## 4. Assessment artifacts

Outputs are structured as machine-readable tables and reproducible figures:

| Artifact | Error source | Content |
|----------|--------------|---------|
| `projection_curves.csv` | E2 | Projected angle as a function of theta per criterion |
| `noise_propagation.csv` | E1+E2 | Induced angular standard deviation over theta and sigma sweeps |
| `event_error.json` | E3 | Frame-move rates and offset distribution statistics from manual annotations |
| `decidability.csv` | 3c | Induced SD vs. band half-width ratio, decidable status, and onset points |
| `run_meta.json` | Metadata | Complete configuration parameters and provenance for reproduction |
| `figures/` | All | Rendered projection curves, spread vs. theta plots, and decidability maps |

---

## Module layout

```
assessment/
  annotation.py    # E3: Manual annotation ingestion and event-error statistics
  projection.py    # E2: Analytic and numerical projection modeling over theta
  propagation.py   # E1+E2: Monte Carlo landmark noise propagation over (theta, sigma)
  decidability.py  # 3c: Decidability ratio evaluation and breakdown localization
  run_measured.py  # Orchestrator linking empirical E3 with the synthetic core
  report.py        # Artifact and figure generation under results/assessment/
```

Shared configuration parameters (`theta_range`, `sigma_sweep`, `mc_samples`, `seed`) are loaded centrally from `PipelineConfig`.

