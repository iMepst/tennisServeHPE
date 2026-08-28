# Rule Base - Implementation Spec

Reference values from Jacquier-Bret et al. (2024).

---

## 0. Global conventions

Preconditions applied across all rules:

- **Input**: BlazePose (`heavy` model) landmarks: 33 points per frame in normalized image coordinates `(x, y)` in [0, 1] with an associated `visibility` score. Depth and world coordinates are discarded (2D operating point).
- **Coordinate scaling**: Normalized coordinates are converted to pixel space (`x_px = x * frame_width`, `y_px = y * frame_height`) before computing angles to prevent aspect ratio distortion.
- **Vertical coordinate orientation**: Image $y$ increases downward. The vertical reference vector pointing upward is defined as `(0, -1)`.
- **Planar vector angle**:
  ```
  theta = atan2( |u_x * v_y - u_y * v_x| , u_x * v_x + u_y * v_y )
  ```
  Calculates the unsigned angle between 2D vectors `u` and `v` in [0, 180] degrees. Evaluated using the 2D cross product magnitude against the dot product for numerical stability.
- **Availability gate**: A rule is evaluated only when every required landmark satisfies the reliability criterion (`visibility >= 0.5`) at the respective key frame. If any landmark is unobserved or low-confidence, the indicator is marked `unavailable`.
- **Reference bands**: Defined as `mean +/- 1 * SD` (symmetric), with the exception of front knee flexion which uses a one-sided lower bound (`mean - 1 * SD`).
- **Output classification**: Each evaluated criterion produces a status of `inside`, `outside`, or `unavailable`. Indicators serve as technical attention flags rather than definitive coaching judgments.

---

## 1. Biomechanical rules

| # | Rule | Key frame | Reference [deg] | Flag threshold [deg] | Band type | Landmark vectors |
|---|------|-----------|-----------------|----------------------|-----------|------------------|
| R1 | Trunk inclination | Trophy position | 25.0 (SD 7.1) | outside 17.9-32.1 | two-sided | mid-hip to mid-shoulder vs. vertical `(0, -1)` |
| R2 | Front knee flexion | Trophy position | 64.5 (SD 9.7) | below 54.8 | one-sided lower | hip to knee vs. knee to ankle |
| R3 | Elbow flexion | Ball impact | 29.2 (SD 9.9)* | outside 19.3-39.1 | two-sided | shoulder to elbow vs. elbow to wrist |
| R4 | Shoulder elevation | Ball impact | 104.6 (SD 6.1)* | outside 98.5-110.7 | two-sided | shoulder to elbow vs. trunk (shoulder to hip) |

* Values derived following the post-hoc exclusion of non-homogeneous reference studies.

---

## 2. Criterion details

### R1 - Trunk inclination (core)
- **Key frame**: Trophy position.
- **Plane**: Frontal (supports anterior and posterior viewpoints; posterior recordings do not require reflection as all angles are unsigned magnitudes).
- **Vectors**: Trunk axis from mid-hip (`(left_hip + right_hip) / 2`) to mid-shoulder (`(left_shoulder + right_shoulder) / 2`); evaluated against the vertical upward vector `(0, -1)`.
- **Interpretation**: An upright torso yields approximately 0 deg; lateral lean corresponds directly to the inclination magnitude.
- **Band**: Two-sided range `[17.9, 32.1]` deg (`25.0 +/- 7.1`).
- **Limitation**: Evaluated along the hip-to-shoulder axis rather than the internal spine segment (treated as a definitional offset under limitations).

### R2 - Front knee flexion (core)
- **Key frame**: Trophy position.
- **Plane**: Sagittal (lateral viewpoint).
- **Vectors**: Turning angle between hip-to-knee and knee-to-ankle vectors along the front leg.
- **Side selection**: Specified by the `front_leg` parameter.
- **Band**: One-sided lower bound at `54.8` deg (`64.5 - 9.7`). Values `< 54.8` deg indicate insufficient knee flexion. Deeper flexion is unpenalized.

### R3 - Elbow flexion (conditional)
- **Key frame**: Ball impact.
- **Vectors**: Turning angle between shoulder-to-elbow and elbow-to-wrist vectors along the serving arm.
- **Side selection**: Specified by the `serving_arm` parameter.
- **Band**: Two-sided range `[19.3, 39.1]` deg (`29.2 +/- 9.9`).
- **Limitation**: Evaluated at impact near full extension, where axial humerus rotation cannot be reconstructed from 2D landmarks.

### R4 - Shoulder elevation (conditional)
- **Key frame**: Ball impact.
- **Vectors**: Enclosed angle at the serving shoulder between upper arm (shoulder to elbow) and ipsilateral torso (shoulder to hip).
- **Side selection**: Specified by the `serving_arm` parameter.
- **Band**: Two-sided range `[98.5, 110.7]` deg (`104.6 +/- 6.1`).

---

## 3. Excluded criteria

- **Back knee flexion**: Excluded due to excessive inter-study dispersion.
- **Shoulder external rotation**: Excluded because axial rotation about the longitudinal bone axis cannot be measured from 2D point landmarks, and the racket low point is not identifiable from body landmarks.

---

## 4. Evaluation constraints

1. **Orthogonal camera planes**: Trunk inclination (frontal) and front knee flexion (sagittal) occupy perpendicular planes. A monocular recording cleanly observes only the in-plane criterion; the orthogonal criterion is subject to foreshortening and marked `unavailable`.
2. **Body landmark localization**: Key frames are identified strictly from body landmarks (ball impact via wrist height peak; trophy position via pelvis depth peak prior to impact).
3. **Reliability gating**: Evaluation requires `visibility >= 0.5` across all required landmarks at the key frame.
4. **Composite availability**: An indicator is computed only when camera plane support, successful event detection, and landmark reliability are simultaneously satisfied.
