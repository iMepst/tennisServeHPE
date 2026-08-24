"""Stage 4: angle computation at the key frames.

Reads the four candidate angles from the filtered trajectories at the
Stage 3 key frames (pipeline_spec.md, Stage 4; rule_base_spec.md,
Sections 0 and 2). Runs in memory; nothing is persisted here.

Two conventions precede every angle: normalized coordinates are rescaled
to pixels (otherwise the frame aspect ratio distorts each angle), and
all angles come from one planar-vector formula.
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


def landmarks_reliable(frame: ProcessedFrame,
                       landmark_names: Iterable[str]) -> bool:
    """True when every named landmark is usable at this frame.

    Usable = the Stage 2 reliability (valid, or filled within a short
    gap) with coordinates present. The availability gate: an angle is
    read only when all its landmarks pass here, else it is unavailable
    rather than computed from a wrong value.
    """
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
    """
    side = clip_params.front_leg
    _check_side("front_leg", side)
    return turning_angle(frame, f"{side}_hip", f"{side}_knee",
                         f"{side}_ankle", clip_params)


def elbow_flexion(frame: ProcessedFrame,
                  clip_params: ClipParams) -> float:
    """Elbow flexion at the ball-impact frame (rule_base_spec.md, R3).

    Turning angle shoulder->elbow vs elbow->wrist on the serving-arm
    side: straight arm ~0.
    """
    side = clip_params.serving_arm
    _check_side("serving_arm", side)
    return turning_angle(frame, f"{side}_shoulder", f"{side}_elbow",
                         f"{side}_wrist", clip_params)


def shoulder_elevation(frame: ProcessedFrame,
                       clip_params: ClipParams) -> float:
    """Shoulder elevation at the ball-impact frame (rule_base_spec.md, R4).

    Spanned at the serving shoulder: upper-arm vector shoulder->elbow
    against the trunk vector shoulder->hip on the same side. Arm along
    the trunk ~0, raised arm = larger angle.
    """
    side = clip_params.serving_arm
    _check_side("serving_arm", side)
    sx, sy = landmark_pixel(frame, f"{side}_shoulder", clip_params)
    ex, ey = landmark_pixel(frame, f"{side}_elbow", clip_params)
    hx, hy = landmark_pixel(frame, f"{side}_hip", clip_params)
    return vector_angle((ex - sx, ey - sy), (hx - sx, hy - sy))


def body_midpoint(frame: ProcessedFrame, left_name: str, right_name: str,
                  clip_params: ClipParams) -> Tuple[float, float]:
    """Pixel midpoint of a left/right landmark pair.

    Builds the mid-hip and mid-shoulder points of the trunk axis
    (rule_base_spec.md, R1).
    """
    lx, ly = landmark_pixel(frame, left_name, clip_params)
    rx, ry = landmark_pixel(frame, right_name, clip_params)
    return (lx + rx) / 2.0, (ly + ry) / 2.0


def trunk_inclination(frame: ProcessedFrame,
                      clip_params: ClipParams) -> float:
    """Trunk inclination at the trophy frame (rule_base_spec.md, R1).

    Trunk axis mid-hip -> mid-shoulder against the image vertical
    upward (0, -1): upward because image y grows downward, so an
    upright axis points toward decreasing y — this recovers the
    reference angle, not its 180-deg complement. Upright trunk ~0, a
    lean reads as the inclination itself.
    """
    hx, hy = body_midpoint(frame, "left_hip", "right_hip", clip_params)
    sx, sy = body_midpoint(frame, "left_shoulder", "right_shoulder",
                           clip_params)
    return vector_angle((sx - hx, sy - hy), (0.0, -1.0))


@dataclass
class AngleReadings:
    """Stage 4 result: the four candidate angles, each None when
    unavailable, with the key frame each was read at.

    Trunk inclination and front knee flexion are read at the trophy
    frame, elbow flexion and shoulder elevation at the ball-impact
    frame. A frame is None when its event is not locatable.
    """
    trophy_frame: Optional[int]
    impact_frame: Optional[int]
    trunk_inclination: Optional[float]
    front_knee_flexion: Optional[float]
    elbow_flexion: Optional[float]
    shoulder_elevation: Optional[float]


def _frame_at(frames: List[ProcessedFrame],
              frame_index: int) -> ProcessedFrame:
    """The ProcessedFrame carrying frame_index (frames are dense but keyed
    by their own index, so this is looked up, not positionally assumed)."""
    for frame in frames:
        if frame.frame_index == frame_index:
            return frame
    raise ValueError(f"no frame with index {frame_index}")


def _gated(frame: ProcessedFrame, names: List[str],
           reader: Callable[[ProcessedFrame], float]) -> Optional[float]:
    """Read an angle only when its landmarks pass the availability gate,
    else None (never a value computed from an unreliable landmark)."""
    if not landmarks_reliable(frame, names):
        return None
    return reader(frame)


def compute_angles(frames: List[ProcessedFrame], key_events: KeyEvents,
                   clip_params: ClipParams) -> AngleReadings:
    """Read the four candidate angles at the located key frames.

    Each angle is gated on its own landmarks; an unavailable one stays
    None. An event that is not locatable leaves both of its angles None.

    Shared-input dependence, by design: trunk inclination and front knee
    flexion are both read at the trophy frame and both derive from the
    hip landmarks, so a hip-landmark error shifts the trophy frame and
    these two angles together — documented, not corrected.
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
