# tennisServeHPE

Feasibility study (bachelor's thesis): rule-based analysis of the tennis
serve from monocular video using 2D human pose estimation.

One recording is turned into deviation indicators in five stages: pose
extraction (MediaPipe BlazePose, heavy model) → landmark preprocessing →
key-event detection (trophy position, ball impact) → angle computation →
rule evaluation against reference bands from Jacquier-Bret et al. (2024).
The indicators are attention flags for a coach, not a verdict on the
serve. A separate assessment quantifies how stable the indicators are
under projection, landmark noise, and event-detection error.

The binding specifications live in `docs/`:

- `docs/pipeline_spec.md` — the five processing stages
- `docs/rule_base_spec.md` — the four rules and their reference bands
- `docs/feasibility_assessment_spec.md` — error budget and stability analysis

## Setup

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # tests, lint, type checks
```

Download the pose model (not tracked, ~29 MB):

```bash
curl -L -o models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

Run the checks:

```bash
pytest
flake8 serve_pipeline
mypy
```

## Usage

Process one serve clip end to end.

**1. Add the clip.** Drop the video into `data/`. The file
name, without extension, becomes the clip id: `data/serve_01.mp4` ->
clip `serve_01`.

**2. Record the per-clip parameters by hand.** Four facts the pipeline
cannot infer, passed on the command line:

| Flag | Values | Meaning |
|---|---|---|
| `--serving-arm` | `left` / `right` | Racket arm (anatomical, body-relative). |
| `--front-leg` | `left` / `right` | Leg in front in the stance (anatomical). |
| `--camera-plane` | `frontal` / `sagittal` | Body plane the camera faces. `frontal` (front OR back view) reads trunk inclination; `sagittal` (side view) reads knee flexion. |
| `--view-direction` | free text | Actual facing, provenance only: `front`/`back` when frontal, `left`/`right` when sagittal. |

fps and frame size default to the video's container metadata; override
with `--fps`, `--frame-width`, `--frame-height` if the file is wrong
(e.g. untagged slow-motion).

**3. Run.**

```bash
python -m serve_pipeline.run data/serve_01.mp4 \
  --serving-arm right --front-leg left \
  --camera-plane frontal --view-direction back
```

Stages 1-2 persist to disk and are reused on the next run; pass
`--no-reuse` to recompute them (e.g. after changing pipeline code).

**4. Read the outputs**, all under `results/<clip>/` (untracked,
reproducible):

```
results/serve_01/
├── stage1/         raw landmarks.csv, meta.json, overlay.mp4, contact_sheet.png
├── stage2/         gated.csv, filtered.csv, *_meta.json, *_qc.png (QC plots)
├── result.json     the four deviation indicators + key frames, angles, provenance
└── key_frames.png  trophy and impact stills, pose overlay + measured angles
```

`result.json` is the deliverable: one deviation indicator per rule
(`inside` / `outside` / `unavailable`), the located key frames, the
angles read at them, and the producing commit. `key_frames.png` shows
those two instants for a visual check.

## Roadmap

The repository is being refactored step by step from an experimental
state (tag `v0.1-experimental`) into the pipeline the specs prescribe.
Each step lands as its own set of commits.

1. [x] Setup: central config, per-clip parameters, housekeeping
2. [x] Stage 1 pose extraction: restrict to the 2D operating point
       (image x/y + visibility only)
3. [x] Stage 2 preprocessing: visibility gating, 120 ms gap
       interpolation, 8 Hz zero-phase Butterworth filter
4. [x] Stage 3 key-event detection: ball impact and trophy position
       from body landmarks, with guard conditions
5. [x] Stage 4 angle computation: the four candidate angles at the key
       frames
6. [x] Stage 5 rule evaluation and `run.py` orchestrator: indicators
       with availability conditions, plus the key-frame stills figure
7. [ ] Feasibility assessment: landmark accuracy (E1), projection (E2),
       event error (E3), noise propagation, decision stability,
       decidability
8. [ ] Report artifacts: machine-readable tables and figures for the
       Results chapter

## Layout

```
serve_pipeline/   pipeline package (config, stages, QC tooling)
tests/            unit tests per module
docs/             binding specifications
data/             input clips (not tracked)
models/           pose model (not tracked)
results/          per-clip, per-stage outputs (not tracked, reproducible)
```
