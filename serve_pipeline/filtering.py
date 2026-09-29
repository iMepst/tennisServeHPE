from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.signal import butter, filtfilt

from .config import PipelineConfig
from .interpolation import COORD_FIELDS, ProcessedFrame
from .landmarks import NUM_LANDMARKS

_DEFAULTS = PipelineConfig()


@dataclass
class FilterConfig:
    order: int = _DEFAULTS.butterworth_order
    cutoff_hz: float = _DEFAULTS.cutoff_hz

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "butterworth",
            "order": self.order,
            "cutoff_hz": self.cutoff_hz,
        }


def _reliable_segments(reliable: List[bool]) -> List[tuple]:
    segs: List[tuple] = []
    start: Optional[int] = None
    for i, r in enumerate(reliable):
        if r and start is None:
            start = i
        elif not r and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(reliable) - 1))
    return segs


def filter_series(frames: List[ProcessedFrame], fps: float,
                  cfg: FilterConfig) -> Dict[str, Any]:
    """Filter reliable segments in place; flag filtered samples."""
    # filtfilt's default padlen is 3 * max(len(a), len(b)) = 3*(order+1);
    # the segment must be strictly longer than that.
    min_len = 3 * (cfg.order + 1) + 1
    nyquist = 0.5 * fps
    wn = cfg.cutoff_hz / nyquist
    if not 0.0 < wn < 1.0:
        raise ValueError(
            f"cutoff_hz {cfg.cutoff_hz} must be in (0, {nyquist}) at "
            f"fps {fps}")
    b, a = butter(cfg.order, wn, btype="low")
    n_filtered = 0
    n_reliable = 0
    n_short_segments = 0
    for lm_id in range(NUM_LANDMARKS):
        reliable = [f.samples[lm_id].reliable for f in frames]
        n_reliable += sum(reliable)
        for start, end in _reliable_segments(reliable):
            if (end - start + 1) < min_len:
                n_short_segments += 1
                continue
            for field in COORD_FIELDS:
                raw = np.array(
                    [getattr(frames[p].samples[lm_id], field)
                     for p in range(start, end + 1)], dtype=float)
                smoothed = filtfilt(b, a, raw)
                for k, p in enumerate(range(start, end + 1)):
                    setattr(frames[p].samples[lm_id], field,
                            float(smoothed[k]))
            for p in range(start, end + 1):
                frames[p].samples[lm_id].filtered = True
                n_filtered += 1
    return {
        "filter": cfg.to_dict(),
        "fps": fps,
        "min_segment_length": min_len,
        "n_reliable_samples": n_reliable,
        "n_filtered_samples": n_filtered,
        "n_reliable_unfiltered_samples": n_reliable - n_filtered,
        "n_segments_too_short": n_short_segments,
    }
