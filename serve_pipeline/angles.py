"""Stage 4: angle computation at the key frames.

Reads the four candidate angles from the filtered trajectories at the
Stage 3 key frames (pipeline_spec.md, Stage 4; rule_base_spec.md,
Sections 0 and 2). Runs in memory; nothing is persisted here.

Two conventions precede every angle: normalized coordinates are rescaled
to pixels (otherwise the frame aspect ratio distorts each angle), and
all angles come from one planar-vector formula.
"""

from typing import Tuple

from .config import ClipParams


def pixel_point(x: float, y: float,
                clip_params: ClipParams) -> Tuple[float, float]:
    """Rescale a normalized landmark position to pixel coordinates.

    Only the two image-plane coordinates enter the angles; depth stays
    discarded (2D operating point).
    """
    return x * clip_params.frame_width, y * clip_params.frame_height
