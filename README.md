# tennisServeHPE — Tennis Serve Analysis Pipeline

Modular pipeline for tennis serve analysis based on human pose estimation
(MediaPipe BlazePose, Tasks API). Stages are strictly separated; every stage
persists its raw output so errors can be attributed to a specific stage.

## Pipeline stages

| Stage | Purpose | Input | Persisted output | Status |
|-------|---------|-------|------------------|--------|
| 1 | Ingestion + pose extraction | video file | `*_landmarks.csv`, `*_meta.json`, `*_overlay.mp4`, `*_contact_sheet.png` | done |
| 2 | Processing (filtering, kinematics) | Stage 1 CSV | planned | — |
| 3 | Evaluation (serve phases, metrics) | Stage 2 output | planned | — |

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p models
curl -L -o models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

## Stage 1: extraction

```bash
python -m serve_pipeline.stage1_extract path/to/serve.mp4 --outdir results
# quick test on the first 200 frames:
python -m serve_pipeline.stage1_extract serve.mp4 --max-frames 200
```

Outputs:

- `<name>_landmarks.csv` — raw time series, long format, one row per
  frame × landmark: normalized image coordinates (`x`, `y`, `z`), metric
  world coordinates in meters with hip-center origin (`world_*`), and the
  model's `visibility` / `presence` scores. Frames without a detection keep
  their 33 rows with empty values, so the frame index stays dense.
- `<name>_meta.json` — video properties, model configuration, detection
  statistics (detection rate, mean visibility per landmark).
- `<name>_overlay.mp4` — diagnostic overlay. Landmark colour encodes
  visibility: green = visible, red = likely occluded. Racket-arm occlusion
  and tracking loss during fast phases show up here directly.
- `<name>_contact_sheet.png` — evenly spaced overlay frames as one image
  for a quick look without playing the video.

The CSV stores raw model output only — no smoothing, interpolation or unit
conversion. Those are Stage 2 decisions.

## Package layout

```
serve_pipeline/
  ingestion.py        video decoding, frame iteration (VideoReader)
  pose_extraction.py  BlazePose wrapper, VIDEO mode (PoseExtractor)
  persistence.py      streaming CSV writer + readers, metadata JSON
  visualization.py    overlay drawing, contact sheet
  landmarks.py        BlazePose topology constants
  stage1_extract.py   Stage 1 CLI orchestrator
tests/                unit + integration tests (pytest)
models/               .task model files (downloaded, not committed)
data/                 sample inputs
results/              stage outputs
```

## Tests

```bash
.venv/bin/pytest
```

Model-dependent integration tests skip automatically when
`models/pose_landmarker_heavy.task` is missing.

## Notes

- mediapipe ≥ 0.10.x removed the legacy `mp.solutions.pose` API; this
  project uses the Tasks API (`PoseLandmarker`) in VIDEO running mode,
  which applies temporal tracking across frames.
- For serve videos, prefer ≥ 60 fps material; at 30 fps the racket arm
  moves several hundred pixels between frames near contact.
