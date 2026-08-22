"""Central configuration for the rule-based serve pipeline.

All fixed parameters prescribed by the specs live here, each annotated
with its origin. Per-clip parameters (serving arm, front leg, camera
plane, fps) are recorded manually per recording and passed separately;
they do not belong in a global config.

Sources: docs/pipeline_spec.md, docs/rule_base_spec.md,
docs/feasibility_assessment_spec.md.
"""

import os
from dataclasses import dataclass
from typing import Tuple

# Repository root, so the defaults work from any working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PipelineConfig:
    # ------------------------------------------------------------------
    # Paths (matching the results/<clip>/<stage>/ convention in layout.py)
    # ------------------------------------------------------------------
    video_dir: str = os.path.join(_REPO_ROOT, "data")
    model_path: str = os.path.join(
        _REPO_ROOT, "models", "pose_landmarker_heavy.task")
    results_root: str = os.path.join(_REPO_ROOT, "results")

    # ------------------------------------------------------------------
    # Stage 2 preprocessing (pipeline_spec.md, Stage 2)
    # ------------------------------------------------------------------
    # (a) A landmark sample is reliable iff visibility >= this threshold
    # (pipeline_spec.md Stage 2a; same gate as the rule availability
    # condition in rule_base_spec.md Section 0).
    visibility_threshold: float = 0.5

    # (b) Only interior gaps up to this length are linearly interpolated.
    # The bound is defined in time and converted per clip to frames via
    # round(0.120 * fps), so the same physical gap length holds at any
    # admitted frame rate (pipeline_spec.md Stage 2b).
    max_gap_ms: float = 120.0

    # (c) Butterworth low-pass, applied zero-phase with filtfilt.
    # The order is the NOMINAL filter order: the forward+backward pass
    # doubles it to an effective 4th order and shifts the half-power
    # point slightly below 8 Hz, which the spec accepts as a fixed
    # offset (pipeline_spec.md Stage 2c). The 8 Hz physical cut-off is
    # fixed across recordings; only the normalized cut-off 8/(fps/2) is
    # recomputed per clip.
    # Deviation from the experimental code: the old pipeline locked
    # 5 Hz / nominal order 4, justified by a velocity-differentiation
    # stage that was never built. The methodology reads static angles at
    # single key frames and prescribes 8 Hz / nominal order 2 instead.
    # The values here are authoritative; the filter code itself is
    # switched over in roadmap step 3.
    butterworth_order: int = 2
    cutoff_hz: float = 8.0

    # ------------------------------------------------------------------
    # Feasibility assessment (feasibility_assessment_spec.md, Sec. 2-4)
    # Placeholder defaults; every assessment output logs the values it
    # ran with, so results stay reproducible.
    # ------------------------------------------------------------------
    # Sweep range for theta, the angle between motion plane and image
    # plane, in degrees (min, max). Feeds projection (E2) and the
    # decidability criterion (Sec. 3c).
    theta_range: Tuple[float, float] = (0.0, 45.0)
    theta_step: float = 5.0

    # Landmark noise standard deviation in pixels (E1). Placeholder
    # until replaced by the value measured against the blinded manual
    # annotation (feasibility_assessment_spec.md Sec. 2b).
    sigma: float = 3.0

    # Monte Carlo sample count and RNG seed (Sec. 2b / 3a).
    mc_samples: int = 10000
