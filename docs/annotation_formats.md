# Manual Annotation Format

The file is created manually under `data/annotations/` and consumed as read-only data by the assessment modules.

Landmark noise sigma is not annotated; it is informed by the estimator model card accuracy and evaluated across a sensitivity parameter sweep (`config.sigma_sweep`).

---

## Event annotation

Records the visually identified frame indices for trophy position and ball impact, providing reference instants to quantify automatic event detection accuracy.

**File location:** `data/annotations/events.csv` (one row per clip).

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `clip` | string | Clip identifier matching the pipeline output directory `results/<clip>/` |
| `true_trophy_frame` | int | Reference frame index of the trophy position instant |
| `true_impact_frame` | int | Reference frame index of the ball impact instant |

Values are stored strictly as integer frame indices determined directly from the source video recordings independently of pipeline outputs.
