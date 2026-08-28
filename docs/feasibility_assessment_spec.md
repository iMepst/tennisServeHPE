# Feasibility Assessment — Implementation Spec

Extracted from methodology.tex, Section 3.6 (`sec:meth_evaluation`). Pairs with
`rule_base_spec.md` and `pipeline_spec.md`.

**Goal of this spec:** structure the code so the numbers the Results chapter reports and the
Discussion interprets fall out directly. Every quantity below is a **planned output**; the
methodology states the plan, the Results chapter states the values.

## Framing (drives what the code must and must NOT do)

- The assessment is **internal**: it verifies the **measurement chain**, i.e. whether the
  pipeline recovers the criteria as designed. It does **not** judge whether a flagged serve is a
  genuine coaching fault (that needs an external rater the work does not use).
- **No external 3D reference is required.** The geometric part **prescribes the true angle by
  construction** and computes its projection. The event detection is checked against a **manual
  frame check**, not an external ground truth.
- Maps onto the three investigatory questions:
  - **Q1 (which criteria extractable at all)** — already settled by criteria selection + pipeline;
    only the **outcome** is carried here (no new computation).
  - **Q2 (how stable under landmark noise, viewpoint, key-event detection)** — Sections 2 + 3 below.
  - **Q3 (under which conditions indicators stay reliable enough)** — the **decidability
    criterion** in Section 3 below.

---

## 1. Error budget — the organising decomposition

Reported angle − true angle is decomposed into **four sources**. This ordering structures the
whole assessment (build one analysis module per source).

| # | Error source | Definition | How it is handled | Quantifiable? | Output artifact |
|---|--------------|------------|-------------------|---------------|-----------------|
| E1 | **Pose estimation error** | estimated landmark − true image position | σ **taken from the estimator's reported accuracy** (BlazePose PDJ) and **swept as a sensitivity range**, not measured on the clips | yes | induced spread over a σ band |
| E2 | **Projection error** | true spatial angle − monocular projected angle | computed **analytically** from the projection relation, no recording | yes | projected-angle curves over θ |
| E3 | **Event error** | selected frame − true instant (incl. pelvis-proxy offset) | read from the **manual frame check** | yes | reported as a **rate** |
| E4 | **Definitional mismatch** | surface landmarks vs. joint centres behind the reference values | **not quantifiable** (needs joint-centre ground truth) → limitations | no | qualitative note (worst on trunk inclination) |

Code implication: E1–E3 produce numbers; **E4 is not simulated** — leave it as a documented,
unquantified offset. Do not fabricate a value for it.

---

## 2. Projection and noise propagation (E2 + E1)

Isolates the projection error and then layers landmark noise on top. **No recording used** —
fully synthetic / Monte Carlo.

### 2a. Projection (E2, analytic/numeric)
- Camera **level**, projection **orthographic** (valid when player is distant relative to body scale).
- For a **single inclination** (trunk): projected angle from the closed form
  `tan(a_proj) = tan(a_true) · cos(θ)`.
- For **two-segment joints** (knee, elbow, shoulder): **no closed form** — both segments can tilt
  out of plane independently. Evaluate **numerically**: orient each segment direction, project it,
  recompute the enclosed angle.
- **θ (motion-plane vs. image-plane angle) is NOT a single value** — it comes partly from camera
  placement, partly from the player's lean direction (unknown before the serve, cannot be zeroed).
  → **Sweep θ over a range**; this same range feeds the decidability criterion (Section 3).

### 2b. Landmark noise on top (E1, Monte Carlo)
- Perturb **each landmark** by an **isotropic Gaussian**, standard deviation **σ in pixels**.
- Read the perturbed landmarks into the angle; estimate the **spread of the resulting angle** by
  **Monte Carlo**.
- **Treat each criterion separately**: a fixed pixel error subtends a larger angle across a
  **shorter** segment, so the short arm segments (elbow, shoulder) react more strongly than the
  longer trunk/leg segments.
- **σ is parameterised**, fixed from the estimator's reported accuracy (BlazePose PDJ), and
  **swept over a band** (`config.sigma_sweep`) rather than measured on the clips. The induced
  spread and the decidability verdict are therefore reported as a **function of the noise level**,
  a sensitivity analysis rather than a single operating point.
- **Known simplification (→ limitations):** noise is modelled as isotropic and independent between
  frames, whereas real error is temporally correlated and larger in the fast serve phases. Figures
  are indicative, not exact.

**Outputs for Results:** per-criterion angular spread (SD, deg) as a function of θ over the σ band;
the projected-vs-true angle curves.

---

## 3. Decision stability & the decidability criterion (Q2 + Q3)

The system emits a **decision, not an angle**, so the reported quantity is **whether the induced
spread is small enough to keep the verdict trustworthy**, not the angular error itself. Q3 is
answered by the **decidability criterion (3c)**; there is no separate verdict-flip simulation.

### 3b. Event-detection stability (E3)
- Read from the **rate at which the manual check has to move the selected frame**.
- Reported as a **finding**, not assumed. Carries the event error (E3) into the verdict.

### 3c. Decidability criterion (the Q3 threshold — fix in code as a constant)
- Stated on the **angular spread** that projection + landmark noise induce in a rule's input,
  measured as a **standard deviation** and held against the rule's **own band**.
- Band half-width = **one reference SD** → the comparison needs **no external scale**.
- **Decidable** where the induced spread stays **below the band half-width** across the expected
  range of camera viewpoint (θ) and landmark noise (σ).
- **Unreliable** where the induced spread **reaches** the band half-width (an input scattering as
  far as centre→edge can no longer separate sound from faulty).
- Threshold **factor = 1**, the same minimal non-arbitrary choice as the bands.
- Report per criterion the σ (and θ) at which the induced spread **reaches** the band half-width —
  the point at which the criterion turns unreliable is the Q3 reading.

---

## 4. What the code should emit (so Results/Discussion are easy)

Structure outputs as machine-readable tables + reproducible figures. Suggested artifacts:

| Artifact | Feeds | Content |
|----------|-------|---------|
| `projection_curves.csv` | E2 | per criterion: θ (deg) → projected angle; trunk closed-form, others numeric |
| `noise_propagation.csv` | E1+E2 | per criterion: (θ, σ) → induced angular spread SD (deg), Monte Carlo, over the σ sweep |
| `event_error.json` | E3 | trophy / impact frame-move rate from manual check |
| `decidability.csv` | 3c | per criterion: (θ, σ) → induced SD vs. band half-width → decidable / unreliable flag |
| `figures/` | all | projection curves, spread-vs-θ, decidability map per criterion over the σ band |

Design notes for reproducibility:
- **Parameterise** θ-range, σ sweep, Monte Carlo N, RNG seed; log them with every output.
- Keep **per-criterion** everywhere (segment length differences matter — Section 2b).
- Separate the **synthetic** modules (E2, E1-propagation, 3c — no recordings) from the one
  **empirical** module (E3 frame check — needs the manual annotation).
- Do **not** produce a serve-quality verdict or any E4 number — out of scope by design.

## Suggested module layout (extends the pipeline layout)

```
assessment/
  annotation.py     # E3: load manual frame check -> trophy/impact frame-move rate
  projection.py     # E2: trunk closed-form + numeric two-segment projection over theta
  propagation.py    # E1+E2: Monte Carlo landmark-noise -> per-criterion angular spread
  decidability.py   # 3c: induced SD vs band half-width -> decidable/unreliable
  run_measured.py   # ties E3 to the synthetic core over the sigma sweep
```

Shared config: `theta_range`, `sigma` / `sigma_sweep`, `mc_samples`, `seed`, plus the rule bands
imported from `rules.py`.
