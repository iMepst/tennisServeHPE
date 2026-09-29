"""Central configuration for the serve pipeline.

Fixed parameters prescribed by the specs, each annotated with its origin.
Per-clip parameters (serving arm, front leg, camera plane, fps) are recorded
manually per recording and passed separately.
"""

import os
from dataclasses import dataclass
from typing import Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PipelineConfig:
    video_dir: str = os.path.join(_REPO_ROOT, "data")
    model_path: str = os.path.join(
        _REPO_ROOT, "models", "pose_landmarker_heavy.task")
    results_root: str = os.path.join(_REPO_ROOT, "results")

    visibility_threshold: float = 0.5

    max_gap_ms: float = 120.0

    butterworth_order: int = 2
    cutoff_hz: float = 8.0

    theta_range: Tuple[float, float] = (0.0, 45.0)
    theta_step: float = 5.0

    sigma: float = 3.0
    sigma_sweep: Tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0)

    mc_samples: int = 10000
    seed: int = 42

    event_tolerances_frames: Tuple[int, ...] = (1, 3, 5)

    event_large_offset_frames: int = 30


@dataclass
class ClipParams:
    """Manually recorded parameters of one recording.

    Key-event detection and angle computation need them: which wrist marks
    impact, which leg the knee angle uses, and which plane-bound trophy
    criterion the viewpoint supports.
    """

    serving_arm: str

    front_leg: str

    camera_plane: str

    view_direction: str

    fps: float

    frame_width: int
    frame_height: int
