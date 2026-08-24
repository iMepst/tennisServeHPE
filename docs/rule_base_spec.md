# Rule Base — Implementation Spec

Extracted from methodology.tex, Section 3.4 (`sec:meth_rule_base`). Reference values
from Jacquier-Bret et al. (2024).

---

## 0. Global conventions (apply to every rule)

These fix the computation before any single rule runs.

- **Input**: BlazePose (heavy variant) landmarks, 33 points per frame, normalized image
  coordinates `(x, y)` in `[0, 1]`, plus a `visibility` score per landmark. Depth / world
  coordinates are discarded (2D operating point only).
- **Rescale to pixels first**: `x_px = x * frame_width`, `y_px = y * frame_height`. Without this
  the frame aspect ratio distorts every angle.
- **y grows downward** (normalized image convention). Relevant for the trunk vertical reference.
- **Angle between two planar vectors `u`, `v`**:
  ```
  theta = atan2( |u_x * v_y - u_y * v_x| , u_x * v_x + u_y * v_y )
  ```
  (2D cross product is the scalar `u_x*v_y - u_y*v_x`; `||u x v||` is its absolute value.)
  Returns `0..180 deg`, numerically stable where `acos` would not be.
- **Availability gate**: a rule is evaluated **only** when every landmark its angle needs is
  reliable (`visibility >= 0.5`) at that rule's key frame. Otherwise the criterion is reported
  `unavailable` — never forced to a value.
- **Band definition**: reference `mean ± 1 * SD`, applied symmetrically, **except** front knee
  flexion (one-sided lower bound). The factor is exactly 1 (minimal non-arbitrary choice).
- **Output per rule**: a binary deviation indicator — `inside` (no flag) or `outside` (flag) —
  or `unavailable`. Indicators are attention flags for a coach, not a verdict on the serve.

---

## 1. The four rules

| # | Rule | Key frame | Reference [deg] | Flag when [deg] | Band type | Angle from landmarks |
|---|------|-----------|-----------------|-----------------|-----------|----------------------|
| R1 | Trunk inclination | Trophy position | 25.0 (SD 7.1) | outside 17.9–32.1 | two-sided | mid-hip→mid-shoulder axis vs. vertical |
| R2 | Front knee flexion | Trophy position | 64.5 (SD 9.7) | below 54.8 | one-sided lower | hip→knee vs. knee→ankle |
| R3 | Elbow flexion | Ball impact | 29.2 (SD 9.9)ᵃ | outside 19.3–39.1 | two-sided | shoulder→elbow vs. elbow→wrist |
| R4 | Shoulder elevation | Ball impact | 104.6 (SD 6.1)ᵃ | outside 98.5–110.7 | two-sided | shoulder→elbow vs. trunk (shoulder→hip) |

ᵃ Value recomputed after the post-hoc exclusion of deviating studies (the all-studies elbow band
~14–46 deg is too wide to discriminate; the excluded value narrows it to ~19–39 deg).

---

## 2. Per-rule detail

### R1 — Trunk inclination (core)
- **Key frame**: trophy position.
- **Plane**: frontal — faced from **in front of OR behind** the player. Both a front view and a
  back (posterior) view support this criterion; back-view clips are explicitly accepted. The angle
  is an unsigned magnitude and the rule is two-sided, so the front/back left-right flip does not
  affect it (no mirroring correction).
- **Vectors**: trunk axis from **mid-hip → mid-shoulder**; reference is the image **vertical
  upward `(0, -1)`** (upward because normalized y grows downward, so the axis points toward
  decreasing y). This recovers the 25.0 deg reference, not its 180-deg complement.
- **mid-hip** = midpoint of left+right hip landmarks; **mid-shoulder** = midpoint of left+right
  shoulder landmarks.
- **Semantics**: upright trunk ≈ 0; a lean reads as the inclination itself.
- **Flag**: angle outside `[17.9, 32.1]` (= 25.0 ± 7.1).
- **Known approximation**: taken over the hip-to-shoulder axis, not the trunk segment itself
  (definitional offset — carried to limitations, not corrected here).

### R2 — Front knee flexion (core)
- **Key frame**: trophy position.
- **Plane**: sagittal (camera to the side).
- **Vectors**: `u` = hip → knee, `v` = knee → ankle. Straight leg ≈ 0, bent leg = positive flexion.
- **Side**: front leg is a **per-clip parameter** — fixes which hip/knee/ankle triplet is used.
- **Band**: **one-sided lower bound** at `mean - SD = 54.8`. Flag **only** when flexion `< 54.8`.
  Deep flexion is left unpenalised (directional evidence: greater flexion → greater racket
  velocity).

### R3 — Elbow flexion (conditional)
- **Key frame**: ball impact.
- **Vectors**: `u` = shoulder → elbow, `v` = elbow → wrist (turning angle, same convention as knee).
- **Side**: serving arm is a **per-clip parameter**.
- **Flag**: angle outside `[19.3, 39.1]` (= 29.2 ± 9.9, post-hoc-excluded value).
- **Reservation**: read at ball impact with the arm near full extension, the configuration where
  axial rotation is least recoverable. Carries this reservation into every verdict.

### R4 — Shoulder elevation (conditional)
- **Key frame**: ball impact.
- **Vectors**: at the **serving shoulder** — upper-arm vector `shoulder → elbow`, and trunk
  vector `shoulder → hip` on the **same side**. Arm along the trunk ≈ 0; raised arm = larger angle.
- **Flag**: angle outside `[98.5, 110.7]` (= 104.6 ± 6.1, post-hoc-excluded value).

---

## 3. Excluded criteria (do NOT implement)

- **Back knee flexion** — too dispersed across studies (fails the dispersion condition).
- **Shoulder external rotation** — an axial rotation about the segment long axis; not recoverable
  from point landmarks in principle. Its key point (racket low point) is also unlocatable from
  body landmarks, so it is dropped entirely.

---

## 4. Constraints the rule evaluation must respect

1. **One core criterion per recording.** Trunk inclination (frontal) and front knee flexion
   (sagittal) occupy orthogonal planes. A single camera faces only one plane, so only that
   criterion is read cleanly. The viewpoint decides which. (Projection foreshortens the other:
   `tan(a_proj) = tan(a_true) * cos(theta)`.)
2. **Key frames come from body landmarks only** (no racket / ball detector):
   - **Ball impact** = frame where the racket-arm **wrist y is highest** (min y in image coords),
     found first over the whole clip.
   - **Trophy position** = frame where the **pelvis (mid-hip) y is lowest** (max y), searched only
     over frames **before** ball impact.
3. **Reliability gate at the key frame** (visibility ≥ 0.5 for all needed landmarks), else
   `unavailable`.
4. An indicator is produced only where camera plane, locatable key frame, and landmark
   reliability all hold together.

---

## 5. Suggested implementation shape

```python
RULES = [
    {
        "id": "trunk_inclination",
        "key_frame": "trophy",
        "plane": "frontal",
        "landmarks": ["hip_l", "hip_r", "shoulder_l", "shoulder_r"],
        "mean": 25.0, "sd": 7.1,
        "band": ("two_sided", 17.9, 32.1),
    },
    {
        "id": "front_knee_flexion",
        "key_frame": "trophy",
        "plane": "sagittal",
        "landmarks": ["hip", "knee", "ankle"],   # front-leg side, per-clip
        "mean": 64.5, "sd": 9.7,
        "band": ("lower_bound", 54.8, None),
    },
    {
        "id": "elbow_flexion",
        "key_frame": "impact",
        "plane": None,
        "landmarks": ["shoulder", "elbow", "wrist"],  # serving-arm side, per-clip
        "mean": 29.2, "sd": 9.9,   # post-hoc-excluded value
        "band": ("two_sided", 19.3, 39.1),
    },
    {
        "id": "shoulder_elevation",
        "key_frame": "impact",
        "plane": None,
        "landmarks": ["shoulder", "elbow", "hip"],   # serving side, per-clip
        "mean": 104.6, "sd": 6.1,  # post-hoc-excluded value
        "band": ("two_sided", 98.5, 110.7),
    },
]

def evaluate(angle, band):
    kind, lo, hi = band
    if kind == "two_sided":
        return "inside" if lo <= angle <= hi else "outside"
    if kind == "lower_bound":
        return "inside" if angle >= lo else "outside"
```
