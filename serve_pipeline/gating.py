from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .landmarks import LANDMARK_NAMES, NUM_LANDMARKS
from .pose_extraction import FramePose

MASK_OK = "ok"
MASK_UNDETECTED = "undetected"
MASK_LOW_VISIBILITY = "low_visibility"

# Value fields carried through from the raw series (kept even when masked).
_VALUE_FIELDS = ["x", "y", "z", "visibility", "presence",
                 "world_x", "world_y", "world_z"]


@dataclass
class GatedSample:
    landmark_id: int
    valid: bool
    mask_reason: str
    # None only for undetected frames, where Stage 1 has no values at all.
    x: Optional[float]
    y: Optional[float]
    z: Optional[float]
    visibility: Optional[float]
    presence: Optional[float]
    world_x: Optional[float]
    world_y: Optional[float]
    world_z: Optional[float]


@dataclass
class GatedFrame:
    frame_index: int
    time_s: float
    samples: List[GatedSample]  # dense: length NUM_LANDMARKS, ordered by id


def gate_frames(frames: List[FramePose],
                visibility_threshold: float) -> List[GatedFrame]:
    """Apply visibility gating to a raw landmark series."""
    gated: List[GatedFrame] = []
    for fp in frames:
        by_id = {obs.landmark_id: obs for obs in fp.landmarks}
        samples: List[GatedSample] = []
        for lm_id in range(NUM_LANDMARKS):
            obs = by_id.get(lm_id)
            if obs is None:
                samples.append(GatedSample(
                    landmark_id=lm_id, valid=False,
                    mask_reason=MASK_UNDETECTED,
                    x=None, y=None, z=None, visibility=None, presence=None,
                    world_x=None, world_y=None, world_z=None,
                ))
                continue
            valid = obs.visibility >= visibility_threshold
            samples.append(GatedSample(
                landmark_id=lm_id, valid=valid,
                mask_reason=MASK_OK if valid else MASK_LOW_VISIBILITY,
                x=obs.x, y=obs.y, z=obs.z,
                visibility=obs.visibility, presence=obs.presence,
                world_x=obs.world_x, world_y=obs.world_y, world_z=obs.world_z,
            ))
        gated.append(GatedFrame(frame_index=fp.frame_index,
                                time_s=fp.time_s, samples=samples))
    return gated


def _gap_record(start_pos: int, end_pos: int,
                frame_indices: List[int], fps: float) -> Dict[str, Any]:
    length = end_pos - start_pos + 1
    return {
        "start_frame": frame_indices[start_pos],
        "end_frame": frame_indices[end_pos],
        "length_frames": length,
        "length_ms": length / fps * 1000.0 if fps else None,
    }


def _find_gaps(valid_flags: List[bool], frame_indices: List[int],
               fps: float) -> List[Dict[str, Any]]:
    """Maximal runs of consecutive invalid samples, as gap records."""
    gaps: List[Dict[str, Any]] = []
    start_pos: Optional[int] = None
    for i, ok in enumerate(valid_flags):
        if not ok:
            if start_pos is None:
                start_pos = i
        elif start_pos is not None:
            gaps.append(_gap_record(start_pos, i - 1, frame_indices, fps))
            start_pos = None
    if start_pos is not None:
        gaps.append(_gap_record(start_pos, len(valid_flags) - 1,
                                frame_indices, fps))
    return gaps


def compute_gap_statistics(gated: List[GatedFrame],
                           fps: float) -> Dict[str, Any]:
    """Per-landmark validity and gap statistics for the metadata JSON."""
    frame_indices = [g.frame_index for g in gated]
    n = len(gated)
    per_landmark: Dict[str, Any] = {}
    for lm_id in range(NUM_LANDMARKS):
        name = LANDMARK_NAMES[lm_id]
        valid_flags = [g.samples[lm_id].valid for g in gated]
        reasons = [g.samples[lm_id].mask_reason for g in gated]
        n_valid = sum(valid_flags)
        gaps = _find_gaps(valid_flags, frame_indices, fps)
        longest = max((gp["length_frames"] for gp in gaps), default=0)
        per_landmark[name] = {
            "valid_rate": n_valid / n if n else 0.0,
            "n_valid": n_valid,
            "n_undetected": reasons.count(MASK_UNDETECTED),
            "n_low_visibility": reasons.count(MASK_LOW_VISIBILITY),
            "num_gaps": len(gaps),
            "longest_gap_frames": longest,
            "longest_gap_ms": longest / fps * 1000.0 if fps else None,
            "gaps": gaps,
        }
    worst = sorted(per_landmark.items(),
                   key=lambda kv: kv[1]["valid_rate"])[:5]
    return {
        "num_frames": n,
        "lowest_valid_rate": {name: round(v["valid_rate"], 4)
                              for name, v in worst},
        "per_landmark": per_landmark,
    }
