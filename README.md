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

## Roadmap

The repository is being refactored step by step from an experimental
state (tag `v0.1-experimental`) into the pipeline the specs prescribe.
Each step lands as its own set of commits.

1. [x] Setup: central config, per-clip parameters, housekeeping
2. [ ] Stage 1 pose extraction: restrict to the 2D operating point
       (image x/y + visibility only)
3. [ ] Stage 2 preprocessing: visibility gating, 120 ms gap
       interpolation, 8 Hz zero-phase Butterworth filter
4. [ ] Stage 3 key-event detection: ball impact and trophy position
       from body landmarks, with guard conditions
5. [ ] Stage 4 angle computation: the four candidate angles at the key
       frames
6. [ ] Stage 5 rule evaluation and `run.py` orchestrator: indicators
       with availability conditions
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
