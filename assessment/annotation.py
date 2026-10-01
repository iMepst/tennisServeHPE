"""E3 event error from the manual frame check.

The synthetic core (propagation.py, decidability.py) needs no recordings. This
module measures the event error on them: the offset between the detected key
frames and the manually read ones (trophy position and ball impact).
"""

import csv
import math
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from serve_pipeline.config import PipelineConfig
from serve_pipeline.persistence import read_metadata

_EVENT_HEADER = ["clip", "true_trophy_frame", "true_impact_frame"]


@dataclass
class EventAnnotation:
    """The manually judged key frames of one clip: integer frame indices judged
    by eye from the video (docs/annotation_formats.md)."""

    clip: str
    true_trophy_frame: int
    true_impact_frame: int


def read_event_annotations(path: str) -> List[EventAnnotation]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != _EVENT_HEADER:
            raise ValueError(
                f"Unexpected event annotation schema in {path}: "
                f"{reader.fieldnames}")
        return [EventAnnotation(
                    clip=row["clip"],
                    true_trophy_frame=int(row["true_trophy_frame"]),
                    true_impact_frame=int(row["true_impact_frame"]))
                for row in reader]


@dataclass
class EventStats:
    """Offset statistics for one event type (trophy or impact) across clips.

    Offsets are detected - manual, in frames, over the locatable clips.
    n_moved_by_tolerance[t] counts the clips with |offset| > t and
    share_by_tolerance[t] is that count over n_locatable; n_large_failures
    and share_large_failures do the same for |offset| >= large_offset_frames.
    """

    event: str
    n_clips: int
    n_locatable: int
    n_not_locatable: int
    tolerances: Tuple[int, ...]
    n_moved_by_tolerance: Dict[int, int]
    share_by_tolerance: Dict[int, float]
    median_offset: float
    iqr_offset: float
    max_abs_offset: float
    large_offset_frames: int
    n_large_failures: int
    share_large_failures: float
    mean_offset: float


def _event_stats(event: str, offsets: List[Optional[int]],
                 tolerances: Tuple[int, ...],
                 large_offset_frames: int) -> EventStats:
    located = [o for o in offsets if o is not None]
    n_clips = len(offsets)
    n_locatable = len(located)
    n_moved_by_tolerance = {tol: sum(1 for o in located if abs(o) > tol)
                            for tol in tolerances}
    n_large = sum(1 for o in located if abs(o) >= large_offset_frames)

    iqr_offset = math.nan
    if len(located) >= 2:
        q1, _, q3 = statistics.quantiles(located, n=4)
        iqr_offset = q3 - q1

    return EventStats(
        event=event, n_clips=n_clips, n_locatable=n_locatable,
        n_not_locatable=n_clips - n_locatable,
        tolerances=tolerances,
        n_moved_by_tolerance=n_moved_by_tolerance,
        share_by_tolerance={
            tol: n / n_locatable if n_locatable else math.nan
            for tol, n in n_moved_by_tolerance.items()},
        median_offset=statistics.median(located) if located else math.nan,
        iqr_offset=iqr_offset,
        max_abs_offset=float(max(abs(o) for o in located))
        if located else math.nan,
        large_offset_frames=large_offset_frames,
        n_large_failures=n_large,
        share_large_failures=(n_large / n_locatable if n_locatable
                              else math.nan),
        mean_offset=statistics.fmean(located) if located else math.nan)


@dataclass
class EventError:
    n_clips: int
    trophy: EventStats
    impact: EventStats


def estimate_event_error(annotations: List[EventAnnotation],
                         results_root: str,
                         tolerances: Optional[Tuple[int, ...]] = None,
                         large_offset_frames: Optional[int] = None
                         ) -> EventError:
    config = PipelineConfig()
    if tolerances is None:
        tolerances = config.event_tolerances_frames
    if large_offset_frames is None:
        large_offset_frames = config.event_large_offset_frames

    trophy_offsets: List[Optional[int]] = []
    impact_offsets: List[Optional[int]] = []
    for ann in annotations:
        events = read_metadata(os.path.join(
            results_root, ann.clip, "result.json"))["key_events"]
        trophy, impact = events["trophy_frame"], events["impact_frame"]
        trophy_offsets.append(
            None if trophy is None else trophy - ann.true_trophy_frame)
        impact_offsets.append(
            None if impact is None else impact - ann.true_impact_frame)

    return EventError(
        n_clips=len(annotations),
        trophy=_event_stats("trophy", trophy_offsets, tolerances,
                            large_offset_frames),
        impact=_event_stats("impact", impact_offsets, tolerances,
                            large_offset_frames))
