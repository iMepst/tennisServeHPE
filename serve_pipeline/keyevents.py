"""Stage 3: key-event detection on the filtered trajectory.

Locates ball impact and the trophy position as extrema of single
coordinate series, from body landmarks only (pipeline_spec.md, Stage 3).
Runs in memory on the Stage 2 output; nothing is persisted here.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import ClipParams
from .interpolation import ProcessedFrame
from .landmarks import NAME_TO_ID


def landmark_y_series(frames: List[ProcessedFrame],
                      lm_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """One landmark's filtered y per frame, with its reliability history.

    Returns (y, original): y holds the filtered y value and is NaN where
    the sample is unreliable (an unfilled gap); original marks samples
    that are *originally* reliable — valid at gating and not filled by
    interpolation. Only such samples may carry an event frame (guard
    condition 1: a linear fill is monotonic and holds no interior
    extremum).
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
    """Position of the series extremum, or None with the rejection reason.

    kind is "min" or "max". The search runs over the reliable (non-NaN)
    values only; guard condition 1 then rejects an extremum that sits on
    an interpolated sample (the true extremum may differ from the linear
    fill) or beside an unfilled gap (the true extremum may lie inside
    the unobserved gap).
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
    """Ball-impact proxy: frame of the racket-arm wrist's y-minimum.

    Image y grows downward, so the minimum is the wrist's highest point,
    coinciding with the extended reach at contact. Returns the position
    in frames and "ok", or None and the guard's rejection reason.
    """
    if serving_arm not in ("left", "right"):
        raise ValueError(
            f"serving_arm must be 'left' or 'right', got {serving_arm!r}")
    wrist_id = NAME_TO_ID[f"{serving_arm}_wrist"]
    y, original = landmark_y_series(frames, wrist_id)
    return guarded_extremum(y, original, "min")


def midhip_y_series(
        frames: List[ProcessedFrame]) -> Tuple[np.ndarray, np.ndarray]:
    """Pelvis proxy: mean y of the two hip landmarks per frame.

    The mid-hip exists only where both hips are reliable (one NaN makes
    the mean NaN), and counts as originally reliable only where both
    hips are.
    """
    y_left, orig_left = landmark_y_series(frames, NAME_TO_ID["left_hip"])
    y_right, orig_right = landmark_y_series(frames, NAME_TO_ID["right_hip"])
    return (y_left + y_right) / 2.0, orig_left & orig_right


def detect_trophy(frames: List[ProcessedFrame],
                  impact_pos: int) -> Tuple[Optional[int], str]:
    """Trophy proxy: frame of the mid-hip y-maximum before ball impact.

    Image y grows downward, so the maximum is the pelvis's lowest point
    (the deepest leg drive). Searched strictly over the frames before
    the impact position. Returns the position in frames and "ok", or
    None and the rejection reason.

    Shared-input dependence, by design: the pelvis proxy, trunk
    inclination, and front knee flexion all derive from the hip
    landmarks, so the trophy frame and the two angles read at it are
    not independent — a hip-landmark error shifts both. Documented
    here, not corrected.
    """
    if impact_pos <= 0:
        return None, "no frames before impact"
    y, original = midhip_y_series(frames)
    return guarded_extremum(y[:impact_pos], original[:impact_pos], "max")


@dataclass
class KeyEvents:
    """Stage 3 result: the two key frames, or why they are not locatable."""
    trophy_frame: Optional[int]
    impact_frame: Optional[int]
    trophy_locatable: bool
    impact_locatable: bool
    reason: str


def detect_key_events(frames: List[ProcessedFrame],
                      clip_params: ClipParams) -> KeyEvents:
    """Run the Stage 3 detection in spec order: impact first, then trophy.

    Guard condition 2 then accepts ball impact only when the wrist at
    impact lies above its trophy height and a non-degenerate window (at
    least one frame) separates the two events. Any failure reports both
    events as not locatable rather than returning a wrong instant, since
    each event's validity depends on the other.
    """
    impact_pos, reason = detect_ball_impact(frames, clip_params.serving_arm)
    if impact_pos is None:
        return KeyEvents(None, None, False, False, f"impact: {reason}")
    trophy_pos, reason = detect_trophy(frames, impact_pos)
    if trophy_pos is None:
        return KeyEvents(None, None, False, False, f"trophy: {reason}")

    wrist_y, _ = landmark_y_series(
        frames, NAME_TO_ID[f"{clip_params.serving_arm}_wrist"])
    # NaN at the trophy frame makes the comparison False, so an
    # unreadable wrist height also rejects the impact.
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
    """QC flag: does trophy-to-contact take unrealistically long?

    A real trophy-to-contact spans roughly 0.5-1.0 s, so a span well
    beyond max_real_seconds suggests untagged slow-motion footage.
    Diagnostic only: it never converts, scales, or fixes the fps, and
    it does not feed back into the detection. When either event is not
    locatable it reports not assessable instead of a number.
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
