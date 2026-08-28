"""Key-event detection: ball impact and the trophy position as extrema
of single body-landmark coordinate series. Runs in memory; persists nothing.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import ClipParams
from .interpolation import ProcessedFrame
from .landmarks import NAME_TO_ID


def landmark_y_series(frames: List[ProcessedFrame],
                      lm_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """One landmark's filtered y per frame plus an `original` mask.

    y is NaN on unreliable samples; original marks samples valid at gating
    and not interpolated. Only these may carry an event: a linear fill holds
    no interior extremum.
    """
    y = np.full(len(frames), np.nan)
    original = np.zeros(len(frames), dtype=bool)
    for i, f in enumerate(frames):
        s = f.samples[lm_id]
        if s.reliable and s.y is not None:
            y[i] = s.y
        original[i] = s.valid and not s.interpolated
    return y, original


def guarded_extremum(y: np.ndarray, original: np.ndarray,
                     kind: str) -> Tuple[Optional[int], str]:
    """Position of the min/max over non-NaN values, or None with a reason.

    Rejects an extremum on an interpolated sample or beside an unfilled gap,
    where the true extremum may differ from what was observed.
    """
    if not np.isfinite(y).any():
        return None, "no reliable samples in the series"
    pos = int(np.nanargmin(y) if kind == "min" else np.nanargmax(y))
    if not original[pos]:
        return None, "extremum sits on an interpolated sample"
    for neighbour in (pos - 1, pos + 1):
        if 0 <= neighbour < len(y) and np.isnan(y[neighbour]):
            return None, "extremum sits at the edge of an unfilled gap"
    return pos, "ok"


def detect_ball_impact(frames: List[ProcessedFrame],
                       serving_arm: str) -> Tuple[Optional[int], str]:
    """Ball-impact proxy: racket-arm wrist y-minimum, its highest point
    (image y grows downward), the extended reach at contact.
    """
    if serving_arm not in ("left", "right"):
        raise ValueError(
            f"serving_arm must be 'left' or 'right', got {serving_arm!r}")
    wrist_id = NAME_TO_ID[f"{serving_arm}_wrist"]
    y, original = landmark_y_series(frames, wrist_id)
    return guarded_extremum(y, original, "min")


def midhip_y_series(
        frames: List[ProcessedFrame]) -> Tuple[np.ndarray, np.ndarray]:
    """Pelvis proxy: mean y of the two hips; NaN or non-original unless
    both hips are reliable and original.
    """
    y_left, orig_left = landmark_y_series(frames, NAME_TO_ID["left_hip"])
    y_right, orig_right = landmark_y_series(frames, NAME_TO_ID["right_hip"])
    return (y_left + y_right) / 2.0, orig_left & orig_right


def detect_trophy(frames: List[ProcessedFrame],
                  impact_pos: int) -> Tuple[Optional[int], str]:
    """Trophy proxy: mid-hip y-maximum among frames before impact, the
    pelvis's lowest point (deepest leg drive; image y grows downward).

    Shared-input dependence, by design: trunk inclination and front-knee
    flexion also derive from the hips, so the trophy frame and the angles
    read at it move together under a hip error. Not corrected.
    """
    if impact_pos <= 0:
        return None, "no frames before impact"
    y, original = midhip_y_series(frames)
    return guarded_extremum(y[:impact_pos], original[:impact_pos], "max")


@dataclass
class KeyEvents:
    """The two key frames, or why they are not locatable."""
    trophy_frame: Optional[int]
    impact_frame: Optional[int]
    trophy_locatable: bool
    impact_locatable: bool
    reason: str


def detect_key_events(frames: List[ProcessedFrame],
                      clip_params: ClipParams) -> KeyEvents:
    """Detect impact first, then trophy.

    Accepts impact only when the wrist sits above its trophy height with a
    non-degenerate window between the events; on any failure both events are
    reported not locatable, since each depends on the other.
    """
    impact_pos, reason = detect_ball_impact(frames, clip_params.serving_arm)
    if impact_pos is None:
        return KeyEvents(None, None, False, False, f"impact: {reason}")
    trophy_pos, reason = detect_trophy(frames, impact_pos)
    if trophy_pos is None:
        return KeyEvents(None, None, False, False, f"trophy: {reason}")

    wrist_y, _ = landmark_y_series(
        frames, NAME_TO_ID[f"{clip_params.serving_arm}_wrist"])
    # NaN at trophy makes the comparison False, rejecting the impact too.
    if not wrist_y[impact_pos] < wrist_y[trophy_pos]:
        return KeyEvents(None, None, False, False,
                         "impact: wrist not above its trophy height")
    if impact_pos - trophy_pos < 2:
        return KeyEvents(None, None, False, False,
                         "impact: degenerate window between trophy "
                         "and impact")
    return KeyEvents(trophy_frame=frames[trophy_pos].frame_index,
                     impact_frame=frames[impact_pos].frame_index,
                     trophy_locatable=True, impact_locatable=True,
                     reason="ok")


@dataclass
class SlowMotionFlag:
    """QC diagnostic result; assessable is False when an event is missing."""
    assessable: bool
    likely_slow_motion: bool
    trophy_to_impact_s: Optional[float]


def flag_possible_slow_motion(key_events: KeyEvents, fps: float,
                              max_real_seconds: float = 1.0
                              ) -> SlowMotionFlag:
    """QC flag: a trophy-to-contact span well beyond max_real_seconds
    (real is ~0.5-1.0 s) suggests untagged slow-motion footage.

    Diagnostic only: never touches fps or the detection; not assessable
    when either event is missing.
    """
    if not (key_events.trophy_locatable and key_events.impact_locatable):
        return SlowMotionFlag(assessable=False, likely_slow_motion=False,
                              trophy_to_impact_s=None)
    assert key_events.impact_frame is not None
    assert key_events.trophy_frame is not None
    span_s = (key_events.impact_frame - key_events.trophy_frame) / fps
    return SlowMotionFlag(assessable=True,
                          likely_slow_motion=span_s > max_real_seconds,
                          trophy_to_impact_s=span_s)
