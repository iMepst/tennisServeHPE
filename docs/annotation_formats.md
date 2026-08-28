# Manual Annotation Format

One CSV feeds the empirical part of the feasibility assessment
(`feasibility_assessment_spec.md`, Section 3b): the event-error rate (E3). It
is produced by hand and exported as CSV into `data/annotations/`. The
assessment code only reads it; it never writes or generates it.

The landmark noise σ (E1) is **not** annotated: it is taken from the
estimator's reported accuracy and swept as a sensitivity range
(`config.sigma_sweep`), so no blinded landmark ground truth is produced.

## Event annotation (E3)

The true trophy and ball-impact instants, judged by eye from the video, used
to measure the rate at which the detected key frame has to be moved.

**File:** `data/annotations/events.csv` (one row per clip).

**Columns:**

| Column | Type | Meaning |
|--------|------|---------|
| `clip` | string | clip identifier, matching the pipeline `results/<clip>/` name |
| `true_trophy_frame` | int | frame index of the trophy instant, judged by eye |
| `true_impact_frame` | int | frame index of the ball-impact instant, judged by eye |

Integer frame indices only. Judge each instant directly from the video, not
from the detected key frames.
