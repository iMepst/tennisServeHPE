"""Stage 3: key-event detection on the filtered trajectory.

Locates ball impact and the trophy position as extrema of single
coordinate series, from body landmarks only (pipeline_spec.md, Stage 3).
Runs in memory on the Stage 2 output; nothing is persisted here.
"""

from typing import List, Optional, Tuple

import numpy as np

from .interpolation import ProcessedFrame


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
