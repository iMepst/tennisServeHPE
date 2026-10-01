# tennisServeHPE

Feasibility study (bachelor's thesis): rule-based analysis of the tennis
serve from monocular video using 2D human pose estimation.

One recording passes through pose extraction (MediaPipe BlazePose, heavy
model) → landmark preprocessing → key-event detection → angle computation →
rule evaluation, yielding deviation indicators: attention flags, not a
verdict on the serve. A separate assessment tests how stable they are under
projection, landmark noise and event-detection error.

Specifications:

- `docs/pipeline_spec.md` — the five processing stages
- `docs/rule_base_spec.md` — the four rules and their reference bands
- `docs/feasibility_assessment_spec.md` — error budget and stability analysis
- `docs/annotation_formats.md` — manual key-frame annotation

## Setup

Python 3.11. Run all commands from the repo root.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # tests, lint, type checks
```

Download the pose model (not tracked, ~29 MB):

```bash
curl -L --create-dirs -o models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

Run the checks:

```bash
pytest
flake8
mypy
```

## Usage

**1. Provide a clip.** Any video OpenCV can decode (H.264 MP4
recommended) showing:

- one complete serve by a single player, whole body in frame
- a static camera facing the frontal plane (front or back view) or the
  sagittal plane (side view); oblique views are not supported
- a start at the toss, after any ball bouncing

The clip can live anywhere; `data/` is the ignored default
(`mkdir -p data`). The file name is the clip id: `data/serve_01.mp4` ->
`serve_01`.

**2. Pass the per-clip parameters:**

| Flag | Values | Meaning |
|---|---|---|
| `--serving-arm` | `left` / `right` | Racket arm (anatomical) |
| `--front-leg` | `left` / `right` | Front leg in the stance (anatomical) |
| `--camera-plane` | `frontal` / `sagittal` | `frontal` (front or back view): trunk inclination; `sagittal` (side view): front knee flexion |
| `--view-direction` | `front` / `back` / `left` / `right` | Provenance only |

fps and frame size are read from the container; `--fps`,
`--frame-width` and `--frame-height` override them.

**3. Run:**

```bash
python -m serve_pipeline.run data/serve_01.mp4 \
  --serving-arm right --front-leg left \
  --camera-plane frontal --view-direction back
```

Stages 1-2 are cached on disk; `--no-reuse` recomputes them.

**4. Outputs:**

```
results/serve_01/
├── stage1/         landmarks.csv, meta.json, overlay.mp4, contact_sheet.png
├── stage2/         gated.csv, filtered.csv, *_meta.json, *_qc.png
├── result.json     indicators (inside / outside / unavailable), key frames, angles, provenance
└── key_frames.png  trophy and impact stills with pose overlay and angles
```

### Corpus and assessment

```bash
python -m serve_pipeline.report   # results/_report/indicators.csv across all processed clips
python -m assessment.report       # results/assessment/
```

The assessment reads manual key frames from `data/annotations/events.csv`
(format: `docs/annotation_formats.md`); without it the event error is
skipped.

## Layout

```
serve_pipeline/   pipeline package
assessment/       feasibility assessment
tests/            unit tests
docs/             specifications
data/             input clips and annotations/ (not tracked)
models/           pose model (not tracked)
results/          outputs (not tracked)
```
