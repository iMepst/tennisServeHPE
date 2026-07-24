# Development Roadmap: From Stage 1 to Coaching Feedback

Milestone-based plan for the remaining implementation and evaluation work.
Planning document, not thesis text. It extends `methodology.md` sections 5 to 7
and turns the planned stages into time-boxed sprints with explicit acceptance
gates. Scope assumptions for this plan are a comfortable time budget (about 12
or more weeks), a balanced scientific weighting of pipeline and empirical study,
and inclusion of the optional front-knee rule.

## 1. Target definition (definition of done)

The code contribution of the thesis is complete when a single documented command
turns a raw serve video into a structured coaching-feedback report, and when the
verdicts of that system have been compared against a human reference rater whose
own reliability has been established. Two contributions are delivered in
parallel. The first is the staged, error-attributable pipeline. The second is
the empirical validation of the system against the human reference.

Everything below serves these two outcomes. Any task that does not advance one
of them is out of scope for the thesis.

## 2. Standing quality gates (every sprint)

These carry over from `methodology.md` section 7 and act as acceptance criteria
for every sprint, not as a one-time checklist.

- A stage reads only the persisted output of its predecessor. The model is never
  re-run downstream. If an earlier stage is insufficient, it is re-executed and
  re-validated first.
- Every stage ships automated unit tests for its logic and a visual sanity check
  of its output on real data before the next sprint starts.
- No result-affecting magic numbers in code. Every threshold, filter parameter
  and reference value surfaces in the persisted metadata.
- The test suite passes on `main` at every commit. The producing commit hash is
  embedded into each metadata JSON.
- Work proceeds on a single linear mainline; each validated milestone is tagged
  on it (`stage1-validated`, `sprint0-hardened`, `stage2-validated`, and so on),
  so any reported result is reproducible from a named commit. A short-lived
  branch is used only for a genuinely discardable experiment (for example the
  filter-choice trial in Sprint 2), then merged or deleted.

## 3. Critical path and parallel work

The manual frame-accurate annotation of the evaluation clips and the recruitment
of the human raters are on the critical path of the empirical study and have long
lead times. They start early and run in parallel with the coding sprints rather
than after them.

- Rater recruitment and the rating protocol are prepared during Sprint 1 and 2,
  so raters are ready when the clip set is fixed.
- Manual boundary annotation (Sprint 4 input) begins as soon as the evaluation
  clip set is selected, independent of the segmentation code.
- The human rating sessions (Sprint 6 input) can proceed while Sprint 5 is being
  finalized, because the raters judge the video, not the system output.

## 4. Sprint plan

Each sprint is about one and a half to two weeks. The gate column is binding. The
next sprint does not start until the gate is met.

### Sprint 0: Foundation hardening

Goal. Pay down the debts from the Stage 1 audit before extending the pipeline,
so Stage 2 is built on a clean base.

Scope.
- Remove the unused `Optional` import in `pose_extraction.py`.
- Add complete type hints across `serve_pipeline` (parameters and return types),
  including the currently untyped functions. Introduce a small typing alias for
  image arrays to keep signatures readable.
- Add a `mypy` configuration and make a clean `mypy` run part of the test gate.
- Reconcile the version mismatch between `methodology.md` section 2.2 and
  `requirements.txt`, and pin the actual runtime versions consistently.
- Replace `print` progress output with the `logging` module while keeping the CLI
  summary readable.
- Embed the producing commit hash into the metadata JSON.

Deliverables. Hardened `serve_pipeline`, `mypy` config, corrected docs and
requirements.

Gate. `flake8` clean, `mypy` clean on `serve_pipeline`, all existing tests green,
documentation and requirements consistent.

### Sprint 1: Stage 2a, quality gating and gap handling

Goal. Turn the raw Stage 1 CSV into a gated, gap-annotated series.

Scope.
- New `stage2` module that loads the landmarks CSV through the existing reader.
- Visibility masking with a threshold calibrated against the overlay
  observations recorded in `stage1-validation.md`, initial estimate 0.5. Masked
  samples are treated as missing, not as data.
- Gap detection over undetected frames and masked samples, producing explicit gap
  intervals and per-landmark gap statistics.
- Persist the gated series and a gap-statistics JSON with the threshold and
  versions.

Deliverables. Gated series CSV, gap-statistics JSON, processing metadata.

Gate. Unit tests on synthetic series with known gaps. Visual raw-versus-gated
plots on `serve_01`.

### Sprint 2: Stage 2b, filtering and interpolation

Goal. Fixed and justified smoothing and interpolation, with every interpolated
sample flagged. This sprint resolves the open decision point in `methodology.md`
section 5.2.

Scope.
- Short-gap interpolation up to about three frames, with the frame threshold
  justified against the source frame rate. Interpolated samples are flagged so
  downstream metrics can be qualified.
- Long gaps are not interpolated. Affected phases are marked unreliable.
- Low-pass filtering, primary candidate a zero-phase Butterworth via `filtfilt`
  with the cutoff chosen against the serve motion bandwidth. Savitzky-Golay is
  the documented alternative. The choice is decided on raw-versus-filtered plots
  and written down.

Deliverables. Filtered series CSV, raw-versus-filtered plots, a filter-choice
note that records type, order and cutoff.

Gate. Unit tests, including a known sine that survives filtering with documented
attenuation and correct interpolation flagging. Visual raw-versus-filtered check
on real clips.

### Sprint 3: Stage 2c, kinematics and the 2D versus 3D control

Goal. Joint angles and angular velocities per `angle_definitions.md`, together
with the empirical 2D-versus-3D control the document now prescribes. This sprint
settles the coordinate decision with evidence.

Scope.
- Compute the rule angles. Trunk inclination against the image vertical, elbow
  flexion in the ISB convention as 180 degrees minus the geometric angle, and the
  optional front-knee flexion.
- Identify the hitting arm per clip.
- Angular velocities by numerical differentiation of the filtered series, with
  filtering applied before differentiation and the order documented.
- Run the 2D-versus-3D control. Compute each rule angle both from the 2D image
  projection and from the 3D world landmarks at the event frames, and report the
  mean and maximum absolute difference per angle and event.

Deliverables. Angle and velocity series CSV, the 2D-versus-3D agreement table,
angle-plausibility plots.

Gate. Unit tests on synthetic geometry, including a synthetic ninety-degree elbow
that returns ninety degrees, a correct ISB conversion, and an upright trunk that
returns zero. Cross-check of angle-over-time against the Stage 1 overlay at
selected frames.

> **Post-Stage-2 checkpoint (deferred refactor).** With Stage 2 complete the
> `serve_pipeline/` module inventory is final, so this is the point to consider
> splitting the flat package into subpackages (e.g. `io/`, `pose/`,
> `processing/`, `viz/`, `stages/`). Do it as its own `[refactor]` commit with
> the test suite proving nothing moved semantically. Deferred until here on
> purpose: the natural boundaries are not fixed while Stage 2b/2c are still
> adding processing modules.

### Sprint 4: Stage 3a, phase segmentation

Goal. Segment each clip into serve phases from kinematic events. Resolves the
open decision point in `methodology.md` section 6.1.

Scope.
- Event proxies. Ball-toss onset from the vertical velocity reversal of the
  tossing wrist, trophy position from maximal knee flexion and peak wrist height,
  contact from peak wrist height and peak velocity, follow-through end from
  velocity decay below a threshold.
- Fix the exact event definitions and thresholds against annotated example clips.
- Manual frame-accurate annotation of the evaluation clips as ground truth for
  the boundaries.
- Report segmentation quality as temporal deviation in frames and milliseconds
  per boundary.

Deliverables. Phase-boundary CSV and JSON, an annotated video with event markers,
a segmentation-accuracy report.

Gate. Unit tests on synthetic kinematic signals with known events. Event markers
rendered into the video and visually checked. Temporal-deviation report against
the manual annotation.

### Sprint 5: Stage 3b, rule evaluation and coaching feedback

Goal. The scientific core. Apply the three rules at their events, classify the
deviation, and generate structured feedback rather than a single score.

Scope.
- Evaluate trunk inclination at the trophy position against 25.0 plus or minus
  7.1 degrees, elbow flexion at contact against 30.1 plus or minus 15.9 degrees,
  and the optional knee flexion against 64.5 plus or minus 9.7 degrees.
- Classify the deviation with direction and magnitude, using a justified band
  based on the reference dispersion and stating the caveat about that dispersion.
- Generate actionable feedback templates keyed to rule, direction and magnitude.
- Qualify or suppress feedback where the event frame fell inside a gap or a
  low-visibility window, linking back to the Stage 2 gating.

Deliverables. Per-clip evaluation JSON, a human-readable feedback report.

Gate. Unit tests on synthetic angle inputs that map to expected verdicts. Manual
check on the validated clips.

### Sprint 6: Validation study, rater reliability and system agreement

Goal. The empirical backbone. Compare system verdicts against a human reference
rater whose own reliability is established first. This addresses the key open
methodological risk.

Scope.
- Define the rating instrument the human rater uses, with per-serve criteria that
  match the system deviation classes on an ordinal scale.
- Inter-rater reliability. At least two raters rate the clip set independently,
  reported with a weighted kappa for the ordinal scale plus percent agreement.
  This establishes that the human reference is itself reliable before it is used
  as a reference.
- Intra-rater reliability. One rater re-rates a subset after a washout interval,
  reported as test-retest agreement.
- System-versus-reference agreement. System verdicts against the human reference
  on the same clips, reported as an agreement metric with a confusion analysis
  and per-stage error attribution for every disagreement.
- Sample size stated realistically, with `baily2020stroke` as the small-sample
  precedent. The comfortable time budget allows a larger clip set.
- Sensitivity analysis. Vary the deviation band and the visibility threshold and
  report how verdict agreement changes.

Deliverables. Rating protocol, reliability results, the agreement analysis,
error-attribution case studies, sensitivity tables.

Gate. The analysis plan is fixed before rating. Reliability is computed before
the system comparison. Every figure is reproducible from the persisted data.

### Sprint 7: Robustness, reproducibility and thesis integration

Goal. Consolidate for defensibility. Enabled by the comfortable time budget.

Scope.
- MediaPipe robustness characterization on the supporting datasets, THETIS, Penn
  Action and SportsPose, to describe extraction failure modes beyond the own
  footage. These datasets stay in a supporting role only.
- Consolidate the threshold sensitivity results.
- End-to-end reproducibility check. A fresh run from raw video to feedback report
  reproduces the reported numbers, driven by one documented command with the
  commit hash recorded.
- Synthesize the limitations into the thesis limitations section, covering
  monocular depth estimation, the single-person model, the frame-rate constraint,
  and the 2D-versus-3D definition mismatch against the reference values.

Gate. The full pipeline reproduces the reported numbers from data and code. The
limitations are documented.

## 5. Milestone-to-methodology mapping

| Sprint | Milestone | methodology.md section |
|--------|-----------|------------------------|
| 0 | Foundation hardening | 7 (cross-cutting QA) |
| 1 to 3 | Stage 2 validated | 5 |
| 4 to 5 | Stage 3 validated | 6 |
| 6 | Rater study complete | 6.2, threats to validity |
| 7 | Robustness and reproducibility | 7 |

## 6. Risk register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Elbow foreshortening at contact biases the primary rule | Wrong verdicts on the most important angle | The 2D-versus-3D control in Sprint 3 quantifies the bias before rules are applied |
| Human reference rater is not internally reliable | The reference the system is judged against is invalid | Inter-rater and intra-rater reliability are established in Sprint 6 before any system comparison |
| Frame rate below 50 fps on some clips | Coarse angular velocity near contact | Material selection prefers high-fps or slow-motion footage, low-fps clips are validation-only |
| Event proxies mis-detect phase boundaries | Rule read at the wrong frame | Manual annotation as ground truth and per-boundary temporal-deviation reporting in Sprint 4 |
| Reference-value dispersion is wide | Deviation bands are lenient | The band choice is justified and a sensitivity analysis reports its effect in Sprint 6 |

## 7. How Stage 3 of the mentoring plan proceeds

Implementation runs sprint by sprint. Each sprint begins with a short design
note, proceeds to code and tests, and closes only when its gate is met and the
visual sanity check on real data has been inspected. Work does not advance to the
next sprint on an unvalidated predecessor.
