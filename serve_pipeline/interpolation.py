from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .gating import GatedFrame
from .landmarks import LANDMARK_NAMES, NUM_LANDMARKS

# Spatial channels that get interpolated (and later filtered).
COORD_FIELDS = ["x", "y"]
# Quality channels carried through untouched.
PASS_FIELDS = ["visibility"]


@dataclass
class ProcessedSample:
    landmark_id: int
    valid: bool          # original gating decision
    mask_reason: str     # original reason
    interpolated: bool   # coordinates were filled by interpolation
    reliable: bool       # usable downstream (valid or short-gap interpolated)
    filtered: bool       # low-pass filter was applied (set in filtering.py)
    x: Optional[float]
    y: Optional[float]
    visibility: Optional[float]


@dataclass
class ProcessedFrame:
    frame_index: int
    time_s: float
    samples: List[ProcessedSample]  # dense: NUM_LANDMARKS, ordered by id


def _invalid_runs(valid: List[bool]) -> List[tuple]:
    """Maximal runs of consecutive invalid positions, as (start, end) pairs."""
    runs: List[tuple] = []
    start: Optional[int] = None
    for i, ok in enumerate(valid):
        if not ok and start is None:
            start = i
        elif ok and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(valid) - 1))
    return runs


def interpolate_gaps(gated: List[GatedFrame],
                     max_gap_frames: int) -> List[ProcessedFrame]:
    """Fill short interior gaps; flag interpolated and unreliable samples."""
    n = len(gated)
    # Seed each sample from its gated counterpart; invalid ones start unreliable until filled below.
    out: List[ProcessedFrame] = []
    for g in gated:
        samples = [
            ProcessedSample(
                landmark_id=s.landmark_id, valid=s.valid,
                mask_reason=s.mask_reason, interpolated=False,
                reliable=s.valid, filtered=False,
                x=s.x, y=s.y,
                visibility=s.visibility,
            )
            for s in g.samples
        ]
        out.append(ProcessedFrame(frame_index=g.frame_index,
                                  time_s=g.time_s, samples=samples))

    for lm_id in range(NUM_LANDMARKS):
        valid = [f.samples[lm_id].valid for f in out]
        for start, end in _invalid_runs(valid):
            length = end - start + 1
            interior = start > 0 and end < n - 1
            if not interior or length > max_gap_frames:
                continue  # edge or long gap: stays unreliable, untouched
            left, right = start - 1, end + 1
            for field in COORD_FIELDS:
                lv = getattr(out[left].samples[lm_id], field)
                rv = getattr(out[right].samples[lm_id], field)
                span = right - left
                for pos in range(start, end + 1):
                    frac = (pos - left) / span
                    setattr(out[pos].samples[lm_id], field,
                            lv + (rv - lv) * frac)
            for pos in range(start, end + 1):
                out[pos].samples[lm_id].interpolated = True
                out[pos].samples[lm_id].reliable = True

    return out


def summarize_interpolation(frames: List[ProcessedFrame]) -> Dict[str, Any]:
    """Interpolation / reliability counts for the Stage 2b metadata JSON."""
    n = len(frames)
    per_landmark: Dict[str, Any] = {}
    total_interpolated = 0
    total_unreliable = 0
    for lm_id in range(NUM_LANDMARKS):
        n_interp = sum(f.samples[lm_id].interpolated for f in frames)
        n_unreliable = sum(not f.samples[lm_id].reliable for f in frames)
        total_interpolated += n_interp
        total_unreliable += n_unreliable
        per_landmark[LANDMARK_NAMES[lm_id]] = {
            "n_interpolated": n_interp,
            "n_unreliable": n_unreliable,
        }
    worst = sorted(per_landmark.items(),
                   key=lambda kv: kv[1]["n_unreliable"], reverse=True)[:5]
    return {
        "num_frames": n,
        "total_interpolated_samples": total_interpolated,
        "total_unreliable_samples": total_unreliable,
        "most_unreliable": {name: v["n_unreliable"] for name, v in worst},
        "per_landmark": per_landmark,
    }
