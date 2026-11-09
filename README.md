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

Python 3.11. Run all commands from the repo root. Tested on macOS with
Apple Silicon (M1, M1 Pro) only.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # tests, lint, type checks
```

Download the pose model (not tracked, ~29 MB; to reproduce the thesis,
use the model from the data archive instead, see below):

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

## Reproducing the thesis results

The clips are third-party footage and are not part of the repository; they
come as a separate archive, `tennisServeHPE-data.zip`. It holds the 52
admitted clips, their metadata (`videos.csv`), the manual key frames
(`annotations/events.csv`), the exact pose model used for the thesis and a
clip overview (`clip_overview.pdf`).

1. Clone the repository (branch `master`) and complete the setup above,
   skipping the model download.
2. Unpack the archive in the repository root; it fills `data/` and
   `models/`:

   ```bash
   unzip tennisServeHPE-data.zip
   ```

   If the browser already unpacked the download (e.g. Safari), move the
   `data/` and `models/` folders from `tennisServeHPE-data/` into the
   repository root instead.

3. Run all clips, then aggregate the corpus and the assessment:

   ```bash
   python -m serve_pipeline.corpus     # results/<clip>/ for all 52 clips (~10-15 min on a laptop CPU)
   python -m serve_pipeline.report     # results/_report/
   python -m assessment.report         # results/assessment/
   ```

The Results chapter maps to these outputs:

| Thesis | Output |
|---|---|
| Sample composition | `data/clip_overview.pdf` |
| Criterion availability (table) | `results/_report/indicators.csv`, column `status` |
| Key-event detection performance (table) | `results/assessment/event_error.json` |
| Projection curves (figure, foreshortening at 45°) | `results/assessment/figures/projection_curves.png`, `projection_curves.csv` |
| Landmark noise propagation (figure, spread ranges) | `results/assessment/figures/spread_vs_theta.png`, `noise_propagation.csv` |
| Decidability map (figure, peak ratios) | `results/assessment/figures/decidability_map.png`, `decidability.csv` |
| Angles against reference bands (figure) | `results/_report/angles_vs_bands.png` |
| Empirical indicators (table, split by plane) | `results/_report/indicators.csv`, aggregated per criterion |
| Example on a single clip (figure) | `results/video_000003/key_frames.png`, `result.json` |

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
