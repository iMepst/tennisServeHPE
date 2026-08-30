"""E3 event error from the manual frame check (feasibility_assessment_spec.md,
Section 3b).

The synthetic core (propagation.py, decidability.py) runs over a sigma taken
from the estimator's reported accuracy and swept as a sensitivity range; it
needs no recordings. This module measures the one empirical number the
recordings actually supply:

- E3: the event-error rate, from the offset between the detected key frames
  and the manually judged ones (trophy position and ball impact).

The event annotation is produced externally and exported as CSV
(docs/annotation_formats.md). Nothing here writes or generates it.
"""

import csv
import math
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from serve_pipeline.config import ClipParams, PipelineConfig
from serve_pipeline.keyevents import detect_key_events
from serve_pipeline.persistence import read_filtered_csv, read_metadata

# --------------------------------------------------------------------------
# E3: event-detection stability (feasibility_assessment_spec.md, Section 3b).
# --------------------------------------------------------------------------

_EVENT_HEADER = ["clip", "true_trophy_frame", "true_impact_frame"]


@dataclass
class EventAnnotation:
    """The manually judged key frames of one clip (docs/annotation_formats.md,
    E3): integer frame indices judged by eye from the video."""

    clip: str
    true_trophy_frame: int
    true_impact_frame: int


def read_event_annotations(path: str) -> List[EventAnnotation]:
    """Read an event annotation CSV (docs/annotation_formats.md, E3)."""
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


def _detect_events(clip: str, results_root: str
                   ) -> Tuple[Optional[int], Optional[int]]:
    """Detected (trophy_frame, impact_frame) for a clip, each None when the
    event is not locatable. Runs Stage 3 on the clip's filtered trajectory
    with the clip's manually recorded parameters."""
    meta = read_metadata(os.path.join(results_root, clip, "result.json"))
    clip_params = ClipParams(**meta["clip_params"])
    frames = read_filtered_csv(
        os.path.join(results_root, clip, "stage2", "filtered.csv"))
    events = detect_key_events(frames, clip_params)
    trophy = events.trophy_frame if events.trophy_locatable else None
    impact = events.impact_frame if events.impact_locatable else None
    return trophy, impact


def _robust_spread(values: List[int]) -> float:
    """Interquartile range (Q3 - Q1) of the offsets, in frames.

    A robust spread: unlike the standard deviation it ignores the few
    extreme slow-motion misses in the tail, so it describes where the bulk
    of the offsets sit. Needs at least two points; nan below that.
    """
    if len(values) < 2:
        return math.nan
    q1, _median, q3 = statistics.quantiles(values, n=4)
    return q3 - q1


@dataclass
class EventTypeError:
    """Offset statistics for one event type (trophy or impact) across clips.

    Robust-first: the headline is the median offset and the interquartile
    spread (iqr_offset), which the heavy offset tail (a few mistimed
    slow-motion clips off by well over 100 frames) does not distort;
    mean_offset is kept only as a secondary field. Offsets are
    detected - true, in frames; max_abs_offset is the largest correction
    over the locatable events, and n_large_failures counts the locatable
    events off by at least large_offset_frames.

    A not-locatable event carries no offset but still needs the manual check
    to supply the frame, so it counts toward every move rate and is reported
    separately as n_not_locatable. move_rate_by_tolerance[t] is the share of
    all annotated clips the check has to move at tolerance t frames
    (|offset| > t, or not locatable); reporting several tolerances exposes
    the usually-accurate, rarely-far-off structure a single tolerance hides.
    """

    event: str
    n_clips: int
    n_locatable: int
    n_not_locatable: int
    tolerances: Tuple[int, ...]
    n_moved_by_tolerance: Dict[int, int]
    move_rate_by_tolerance: Dict[int, float]
    median_offset: float
    iqr_offset: float
    max_abs_offset: float
    large_offset_frames: int
    n_large_failures: int
    mean_offset: float


def _event_type_error(event: str, offsets: List[Optional[int]],
                      tolerances: Tuple[int, ...],
                      large_offset_frames: int) -> EventTypeError:
    """Aggregate one event type's per-clip offsets (None = not locatable)."""
    located = [o for o in offsets if o is not None]
    n_clips = len(offsets)
    n_not_locatable = n_clips - len(located)

    # Move rate at each tolerance: a not-locatable event always needs a move.
    n_moved_by_tolerance: Dict[int, int] = {}
    move_rate_by_tolerance: Dict[int, float] = {}
    for tol in tolerances:
        n_moved = sum(1 for o in located if abs(o) > tol)
        n_needs_move = n_moved + n_not_locatable
        n_moved_by_tolerance[tol] = n_moved
        move_rate_by_tolerance[tol] = (
            n_needs_move / n_clips if n_clips else math.nan)

    return EventTypeError(
        event=event, n_clips=n_clips, n_locatable=len(located),
        n_not_locatable=n_not_locatable,
        tolerances=tuple(tolerances),
        n_moved_by_tolerance=n_moved_by_tolerance,
        move_rate_by_tolerance=move_rate_by_tolerance,
        median_offset=statistics.median(located) if located else math.nan,
        iqr_offset=_robust_spread(located),
        max_abs_offset=float(max(abs(o) for o in located))
        if located else math.nan,
        large_offset_frames=large_offset_frames,
        n_large_failures=sum(1 for o in located
                             if abs(o) >= large_offset_frames),
        mean_offset=statistics.fmean(located) if located else math.nan)


@dataclass
class EventError:
    """Event-error rate (E3) over the annotated clips, per event type."""

    n_clips: int
    trophy: EventTypeError
    impact: EventTypeError


def estimate_event_error(annotations: List[EventAnnotation],
                         results_root: Optional[str] = None,
                         tolerances: Optional[Tuple[int, ...]] = None,
                         large_offset_frames: Optional[int] = None
                         ) -> EventError:
    """Measure the event-error rate (E3) from the manual frame check.

    For each annotated clip, detect the key events and record the frame
    offset detected - true for trophy and impact. Reported as move rates at
    a set of tolerances (share of events the manual check has to shift beyond
    each tolerance, with not-locatable events counted as needing a move) plus
    the robust offset distribution (median / IQR / max-abs, large-failure
    count, mean secondary). Tolerances and the large-failure threshold default
    to the config values.
    """
    config = PipelineConfig()
    if results_root is None:
        results_root = config.results_root
    if tolerances is None:
        tolerances = config.event_tolerances_frames
    if large_offset_frames is None:
        large_offset_frames = config.event_large_offset_frames

    trophy_offsets: List[Optional[int]] = []
    impact_offsets: List[Optional[int]] = []
    for ann in annotations:
        det_trophy, det_impact = _detect_events(ann.clip, results_root)
        trophy_offsets.append(
            None if det_trophy is None
            else det_trophy - ann.true_trophy_frame)
        impact_offsets.append(
            None if det_impact is None
            else det_impact - ann.true_impact_frame)

    return EventError(
        n_clips=len(annotations),
        trophy=_event_type_error("trophy", trophy_offsets, tolerances,
                                 large_offset_frames),
        impact=_event_type_error("impact", impact_offsets, tolerances,
                                 large_offset_frames))
