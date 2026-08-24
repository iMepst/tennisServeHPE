"""Stage 4: angle computation at the key frames.

Reads the four candidate angles from the filtered trajectories at the
Stage 3 key frames (pipeline_spec.md, Stage 4; rule_base_spec.md,
Sections 0 and 2). Runs in memory; nothing is persisted here.

Two conventions precede every angle: normalized coordinates are rescaled
to pixels (otherwise the frame aspect ratio distorts each angle), and
all angles come from one planar-vector formula.
"""

import math
from typing import Tuple

from .config import ClipParams
from .interpolation import ProcessedFrame
from .landmarks import NAME_TO_ID


def pixel_point(x: float, y: float,
                clip_params: ClipParams) -> Tuple[float, float]:
    """Rescale a normalized landmark position to pixel coordinates.

    Only the two image-plane coordinates enter the angles; depth stays
    discarded (2D operating point).
    """
    return x * clip_params.frame_width, y * clip_params.frame_height


def vector_angle(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    """Angle between two planar vectors in degrees, 0..180.

    theta = atan2(|u_x*v_y - u_y*v_x|, u_x*v_x + u_y*v_y): the 2D
    cross-product magnitude against the dot product. Numerically stable
    where an acos of the normalized dot product is not (near 0 and
    180 deg).
    """
    cross = u[0] * v[1] - u[1] * v[0]
    dot = u[0] * v[0] + u[1] * v[1]
    return math.degrees(math.atan2(abs(cross), dot))


def landmark_pixel(frame: ProcessedFrame, landmark_name: str,
                   clip_params: ClipParams) -> Tuple[float, float]:
    """One landmark's pixel position at this frame.

    Raises ValueError when the sample carries no coordinates (an
    unfilled gap or undetected frame); the reliability gate itself is
    applied by the caller before any angle is formed.
    """
    sample = frame.samples[NAME_TO_ID[landmark_name]]
    if sample.x is None or sample.y is None:
        raise ValueError(
            f"landmark {landmark_name} has no coordinates at frame "
            f"{frame.frame_index}")
    return pixel_point(sample.x, sample.y, clip_params)


    for name in landmark_names:
        sample = frame.samples[NAME_TO_ID[name]]
        if not sample.reliable or sample.x is None or sample.y is None:
            return False
    return True


def turning_angle(frame: ProcessedFrame, first: str, middle: str,
                  last: str, clip_params: ClipParams) -> float:
    """Turning angle of the chain first -> middle -> last, 0..180 deg.

    u = first->middle, v = middle->last: a straight chain reads ~0, a
    right-angle bend 90. Knee and elbow flexion share this construction.
    """
    ax, ay = landmark_pixel(frame, first, clip_params)
    bx, by = landmark_pixel(frame, middle, clip_params)
    cx, cy = landmark_pixel(frame, last, clip_params)
    return vector_angle((bx - ax, by - ay), (cx - bx, cy - by))


def _check_side(name: str, side: str) -> None:
    if side not in ("left", "right"):
        raise ValueError(f"{name} must be 'left' or 'right', got {side!r}")


def front_knee_flexion(frame: ProcessedFrame,
                       clip_params: ClipParams) -> float:
    """Front knee flexion at the trophy frame (rule_base_spec.md, R2).

    Turning angle hip->knee vs knee->ankle on the front-leg side:
    straight leg ~0, bent leg = positive flexion.