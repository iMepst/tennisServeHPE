# Manual Annotation Format

The file is created manually under `data/annotations/` and consumed as read-only data by the assessment modules.

Landmark noise sigma is not annotated: the clips carry no hand-annotated landmark positions and the model card reports no pixel-level error, so sigma is swept over a plausible range (`config.sigma_sweep`, 2-6 px).

---

## Event annotation

Records the visually identified frame indices for trophy position and ball impact, providing reference instants to quantify automatic event detection accuracy.

**File location:** `data/annotations/events.csv` (one row per clip).

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `clip` | string | Clip identifier matching the pipeline output directory `results/<clip>/` |
| `true_trophy_frame` | int | Reference frame index of the trophy position instant (racket's first vertical peak over the loaded legs) |
| `true_impact_frame` | int | Reference frame index of the ball impact instant (visible racket-ball contact) |

Values are stored strictly as integer frame indices. A single observer steps frame by frame through each clip's pose overlay and marks the frames using the pipeline's frame index, without seeing the detected frames.
