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
    butterworth_order: int = 2
    cutoff_hz: float = 8.0

    # Feasibility assessment. Placeholder defaults; every output logs the
    # values it ran with, so results stay reproducible.
    # Sweep range (min, max) for theta, the angle between motion and image
    # planes, in degrees. Feeds projection (E2) and decidability.
    theta_range: Tuple[float, float] = (0.0, 45.0)
    theta_step: float = 5.0

    # Landmark noise sd in pixels (E1). Not measured here: taken from the
    # estimator's reported accuracy (BlazePose PDJ) and treated as a range, not
    # one point. sigma is nominal; sigma_sweep is the band propagation and
    # decidability are reported over, so the verdict reads as a function of
    # pose-estimate noise.
    sigma: float = 3.0
    sigma_sweep: Tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0)

    # Monte Carlo sample count and RNG seed.
    mc_samples: int = 10000
    seed: int = 42

    # Frame tolerances for the event-error report (E3). The manual check only
    # moves a detected key frame when the detector is off by more than the
    # tolerance. Reported at several tolerances so the "usually spot-on,
    # occasionally far off" structure stays visible: on slow-motion clips a
    # tight one-frame tolerance alone overstates the move rate.
    event_tolerances_frames: Tuple[int, ...] = (1, 3, 5)

    # An absolute offset at or beyond this many frames counts as a large
    # detection failure (the heavy tail from mistimed slow-motion clips).
    # Counted separately so a few extreme misses do not distort the median/IQR.
    event_large_offset_frames: int = 30


@dataclass
class ClipParams:
    """Manually recorded parameters of one recording.

    Stages 3 and 4 cannot run without them: they decide which wrist
    marks ball impact, which leg the knee angle is read from, and which
    of the two plane-bound trophy criteria the viewpoint supports
    (pipeline_spec.md, header; rule_base_spec.md Section 4).
    """

    # Which arm holds the racket. Selects the wrist whose y-minimum
    # locates ball impact (Stage 3) and the shoulder/elbow/wrist side
    # for elbow flexion and shoulder elevation (Stage 4). Anatomical,
    # i.e. body-relative: "left" means the player's left arm regardless
    # of where the camera stands.
    serving_arm: str  # "left" | "right"

    # Which leg stands in front in the stance. Selects the
    # hip/knee/ankle triplet for front knee flexion (Stage 4).
    # Anatomical, like serving_arm.
    front_leg: str  # "left" | "right"

    # Which body plane the camera faces. Decides the one core trophy
    # criterion that is read cleanly: "frontal" -> trunk inclination,
    # "sagittal" -> front knee flexion (rule_base_spec.md Section 4.1).
    # "frontal" covers both front and back (posterior) views — back-view
    # clips are accepted (common in scraped footage) and need no
    # mirroring, since all angles are unsigned magnitudes.
    camera_plane: str  # "frontal" | "sagittal"

    # Where the camera stands within that plane: "front"/"back" for
    # frontal clips, "left"/"right" for sagittal clips. Provenance only:
    # it documents the recording setup but does not change which
    # criteria are available.
    view_direction: str

    # Frame rate of the clip: the container/playback fps of the file at
    # hand, never a guessed capture rate. Scraped footage may be
    # untagged slow-motion whose capture rate is not recoverable; the
    # frames are sampled at the container fps, which is the correct
    # operating rate for the 120 ms gap bound and the per-clip filter
    # design (Nyquist), and the core outputs (angles, verdicts) do not
    # depend on fps (pipeline_spec.md, per-clip parameters and Stage 2).
    fps: float

    # Frame size in pixels. Rescales normalized landmark coordinates to
    # pixels before any angle is formed, otherwise the aspect ratio
    # distorts every angle (rule_base_spec.md Section 0).
    frame_width: int
    frame_height: int
