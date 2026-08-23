"""Stage 3: key-event detection on the filtered trajectory.

Locates ball impact and the trophy position as extrema of single
coordinate series, from body landmarks only (pipeline_spec.md, Stage 3).
Runs in memory on the Stage 2 output; nothing is persisted here.
"""

from typing import List, Tuple

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
