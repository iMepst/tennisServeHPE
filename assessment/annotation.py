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

