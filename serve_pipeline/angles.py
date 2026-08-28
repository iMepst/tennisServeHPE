"""Angle computation at the key frames: the four candidate angles read
from the filtered trajectories. Runs in memory; persists nothing.

Two conventions: normalized coordinates are rescaled to pixels (else the
aspect ratio distorts the angle), and all angles use one planar-vector formula.
"""

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

from .config import ClipParams
from .interpolation import ProcessedFrame
from .keyevents import KeyEvents
from .landmarks import NAME_TO_ID


def pixel_point(x: float, y: float,
                clip_params: ClipParams) -> Tuple[float, float]:
    """Rescale a normalized landmark position to pixels; depth is discarded."""
    return x * clip_params.frame_width, y * clip_params.frame_height


def vector_angle(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    """Angle between two planar vectors in degrees, 0..180.

    atan2(|cross|, dot): stable near 0 and 180 deg where acos of the
    normalized dot product is not.
    """
    cross = u[0] * v[1] - u[1] * v[0]
    dot = u[0] * v[0] + u[1] * v[1]
    return math.degrees(math.atan2(abs(cross), dot))


def landmark_pixel(frame: ProcessedFrame, landmark_name: str,
                   clip_params: ClipParams) -> Tuple[float, float]:
    """One landmark's pixel position at this frame.

    Raises ValueError when the sample has no coordinates; the reliability
    gate is applied by the caller.
    """
    sample = frame.samples[NAME_TO_ID[landmark_name]]
    if sample.x is None or sample.y is None:
        raise ValueError(
            f"landmark {landmark_name} has no coordinates at frame "
            f"{frame.frame_index}")
    return pixel_point(sample.x, sample.y, clip_params)


def landmarks_reliable(frame: ProcessedFrame,
                       landmark_names: Iterable[str]) -> bool:
    """True when every named landmark is reliable with coordinates present.

    Availability gate: an angle is read only when all its landmarks pass,
    else it is unavailable rather than computed from a wrong value.
    """
    for name in landmark_names:
        sample = frame.samples[NAME_TO_ID[name]]
        if not sample.reliable or sample.x is None or sample.y is None:
            return False
    return True


def turning_angle(frame: ProcessedFrame, first: str, middle: str,
                  last: str, clip_params: ClipParams) -> float:
    """Turning angle of the chain first -> middle -> last, 0..180 deg.

    Straight chain ~0, right-angle bend 90. Knee and elbow flexion share this.
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
    """Front knee flexion at the trophy frame (R2).

    Turning angle hip->knee vs knee->ankle, front leg: straight ~0.
    """
    side = clip_params.front_leg
    _check_side("front_leg", side)
    return turning_angle(frame, f"{side}_hip", f"{side}_knee",
                         f"{side}_ankle", clip_params)


def elbow_flexion(frame: ProcessedFrame,
                  clip_params: ClipParams) -> float:
    """Elbow flexion at the impact frame (R3).

    Turning angle shoulder->elbow vs elbow->wrist, serving arm: straight ~0.
    """
    side = clip_params.serving_arm
    _check_side("serving_arm", side)
    return turning_angle(frame, f"{side}_shoulder", f"{side}_elbow",
                         f"{side}_wrist", clip_params)


def shoulder_elevation(frame: ProcessedFrame,
                       clip_params: ClipParams) -> float:
    """Shoulder elevation at the impact frame (R4).

    Upper-arm shoulder->elbow against trunk shoulder->hip, same side:
    arm along the trunk ~0, raised = larger.
    """
    side = clip_params.serving_arm
    _check_side("serving_arm", side)
    sx, sy = landmark_pixel(frame, f"{side}_shoulder", clip_params)
    ex, ey = landmark_pixel(frame, f"{side}_elbow", clip_params)
    hx, hy = landmark_pixel(frame, f"{side}_hip", clip_params)
    return vector_angle((ex - sx, ey - sy), (hx - sx, hy - sy))


def body_midpoint(frame: ProcessedFrame, left_name: str, right_name: str,
                  clip_params: ClipParams) -> Tuple[float, float]:
    """Pixel midpoint of a left/right landmark pair (trunk-axis ends, R1)."""
    lx, ly = landmark_pixel(frame, left_name, clip_params)
    rx, ry = landmark_pixel(frame, right_name, clip_params)
    return (lx + rx) / 2.0, (ly + ry) / 2.0


def trunk_inclination(frame: ProcessedFrame,
                      clip_params: ClipParams) -> float:
    """Trunk inclination at the trophy frame (R1).

    Trunk axis mid-hip -> mid-shoulder against image-up (0, -1); up because
    image y grows downward, recovering the reference angle not its complement.
    Upright ~0.
    """
    hx, hy = body_midpoint(frame, "left_hip", "right_hip", clip_params)
    sx, sy = body_midpoint(frame, "left_shoulder", "right_shoulder",
                           clip_params)
    return vector_angle((sx - hx, sy - hy), (0.0, -1.0))


@dataclass
class AngleReadings:
    """The four candidate angles (None when unavailable) with the key frame
    each was read at.

    Trunk and knee at the trophy frame; elbow and shoulder at impact. A frame
    is None when its event is not locatable.
    """
    trophy_frame: Optional[int]
    impact_frame: Optional[int]
    trunk_inclination: Optional[float]
    front_knee_flexion: Optional[float]
    elbow_flexion: Optional[float]
    shoulder_elevation: Optional[float]


def _frame_at(frames: List[ProcessedFrame],
              frame_index: int) -> ProcessedFrame:
    """The frame carrying frame_index (looked up by index, not positional)."""
    for frame in frames:
        if frame.frame_index == frame_index:
            return frame
    raise ValueError(f"no frame with index {frame_index}")


def _gated(frame: ProcessedFrame, names: List[str],
           reader: Callable[[ProcessedFrame], float]) -> Optional[float]:
    """Read the angle only when its landmarks pass the availability gate, else None."""
    if not landmarks_reliable(frame, names):
        return None
    return reader(frame)


def compute_angles(frames: List[ProcessedFrame], key_events: KeyEvents,
                   clip_params: ClipParams) -> AngleReadings:
    """Read the four candidate angles at the located key frames.

    Each angle is gated on its own landmarks; an unavailable one stays None,
    and a non-locatable event leaves both of its angles None.

    Shared-input dependence, by design: trunk inclination and front knee
    flexion both read at the trophy frame and both from the hips, so a hip
    error shifts the trophy frame and both angles together. Not corrected.
    """
    arm = clip_params.serving_arm
    leg = clip_params.front_leg

    trunk = knee = elbow = shoulder = None

    if key_events.trophy_locatable and key_events.trophy_frame is not None:
        frame = _frame_at(frames, key_events.trophy_frame)
        trunk = _gated(
            frame,
            ["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
            lambda f: trunk_inclination(f, clip_params))
        knee = _gated(
            frame, [f"{leg}_hip", f"{leg}_knee", f"{leg}_ankle"],
            lambda f: front_knee_flexion(f, clip_params))

    if key_events.impact_locatable and key_events.impact_frame is not None:
        frame = _frame_at(frames, key_events.impact_frame)
        elbow = _gated(
            frame, [f"{arm}_shoulder", f"{arm}_elbow", f"{arm}_wrist"],
            lambda f: elbow_flexion(f, clip_params))
        shoulder = _gated(
            frame, [f"{arm}_shoulder", f"{arm}_elbow", f"{arm}_hip"],
            lambda f: shoulder_elevation(f, clip_params))

    return AngleReadings(
        trophy_frame=key_events.trophy_frame if key_events.trophy_locatable
        else None,
        impact_frame=key_events.impact_frame if key_events.impact_locatable
        else None,
        trunk_inclination=trunk, front_knee_flexion=knee,
        elbow_flexion=elbow, shoulder_elevation=shoulder)
