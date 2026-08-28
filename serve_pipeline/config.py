"""Central configuration for the serve pipeline.

Fixed parameters prescribed by the specs, each annotated with its origin.
Per-clip parameters (serving arm, front leg, camera plane, fps) are recorded
manually per recording and passed separately.
"""

import os
from dataclasses import dataclass
from typing import Tuple

# Repository root, so the defaults work from any working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PipelineConfig:
    # Paths (results/<clip>/<stage>/ convention, see layout.py)
    video_dir: str = os.path.join(_REPO_ROOT, "data")
    model_path: str = os.path.join(
        _REPO_ROOT, "models", "pose_landmarker_heavy.task")
    results_root: str = os.path.join(_REPO_ROOT, "results")

    # Preprocessing
    # (a) A sample is reliable iff visibility >= this threshold (also the
    # rule availability condition).
    visibility_threshold: float = 0.5

    # (b) Interior gaps up to this length are linearly interpolated. Defined in
    # time, converted per clip via round(0.120 * fps), so the same physical gap
    # holds at any frame rate.
    max_gap_ms: float = 120.0

    # (c) Butterworth low-pass, applied zero-phase with filtfilt. order is
    # nominal: the forward+backward pass doubles it to effective 4th order and
    # shifts the half-power point just below 8 Hz. The 8 Hz cut-off is fixed;
    # only the normalized cut-off 8/(fps/2) is recomputed per clip.
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

    Key-event detection and angle computation need them: which wrist marks
    impact, which leg the knee angle uses, and which plane-bound trophy
    criterion the viewpoint supports.
    """

    # Which arm holds the racket. Selects the wrist whose y-minimum locates
    # impact and the shoulder/elbow/wrist side for elbow and shoulder angles.
    # Anatomical (body-relative), regardless of camera position.
    serving_arm: str  # "left" | "right"

    # Which leg stands in front. Selects the hip/knee/ankle triplet for front
    # knee flexion. Anatomical, like serving_arm.
    front_leg: str  # "left" | "right"

    # Which body plane the camera faces, deciding the one trophy criterion read
    # cleanly: "frontal" -> trunk inclination, "sagittal" -> front knee flexion.
    # "frontal" covers front and back views (back accepted, no mirroring needed
    # since all angles are unsigned magnitudes).
    camera_plane: str  # "frontal" | "sagittal"

    # Where the camera stands within that plane: "front"/"back" (frontal) or
    # "left"/"right" (sagittal). Provenance only; does not change availability.
    view_direction: str

    # Frame rate of the clip: the file's container/playback fps, never a guessed
    # capture rate. Untagged slow-motion is sampled at the container fps, the
    # correct rate for the 120 ms gap bound and the per-clip filter (Nyquist);
    # the core outputs do not depend on fps.
    fps: float

    # Frame size in pixels. Rescales normalized coordinates to pixels before any
    # angle, else the aspect ratio distorts it.
    frame_width: int
    frame_height: int
