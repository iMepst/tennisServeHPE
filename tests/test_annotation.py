import csv
import math
import os

import pytest

from assessment.annotation import (
    EventAnnotation, estimate_event_error, read_event_annotations,
    _event_type_error)
from serve_pipeline.config import PipelineConfig
from serve_pipeline.interpolation import ProcessedFrame, ProcessedSample
from serve_pipeline.landmarks import NAME_TO_ID, NUM_LANDMARKS
from serve_pipeline.persistence import write_filtered_csv, write_metadata

_W, _H = 1000, 1000  # frame size chosen so px = normalized * 1000.


# --- E3: event error -------------------------------------------------------

_CLIP_PARAMS = {
    "serving_arm": "right", "front_leg": "left", "camera_plane": "frontal",
    "view_direction": "back", "fps": 25.0,
    "frame_width": _W, "frame_height": _H,
}
_WRIST = NAME_TO_ID["right_wrist"]
_HIPS = (NAME_TO_ID["left_hip"], NAME_TO_ID["right_hip"])


def _event_frame(i, wrist_y, hip_y, wrist_reliable=True):
    """One dense filtered frame; serving wrist and both hips carry the given
    y, everything else a constant reliable 0.5."""
    samples = []
    for lm in range(NUM_LANDMARKS):
        if lm == _WRIST and not wrist_reliable:
            samples.append(ProcessedSample(
                lm, valid=False, mask_reason="low_visibility",
                interpolated=False, reliable=False, filtered=False,
                x=None, y=None, visibility=0.1))
            continue
        y = wrist_y if lm == _WRIST else hip_y if lm in _HIPS else 0.5
        samples.append(ProcessedSample(
            lm, valid=True, mask_reason="ok", interpolated=False,
            reliable=True, filtered=True, x=0.5, y=y, visibility=0.9))
    return ProcessedFrame(frame_index=i, time_s=i / 25.0, samples=samples)


def _make_event_clip(results_root, clip, locatable=True):
    """A clip whose detection gives trophy at frame 2, impact at frame 7,
    or an unlocatable clip (serving wrist unreliable throughout)."""
    # Mid-hip max (pelvis low point) at frame 2; wrist min (contact) at 7.
    hip_ys = [0.5, 0.6, 0.9, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    wrist_ys = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.5, 0.5]
    frames = [_event_frame(i, wrist_ys[i], hip_ys[i],
                           wrist_reliable=locatable)
              for i in range(len(hip_ys))]
    os.makedirs(os.path.join(results_root, clip, "stage2"))
    write_filtered_csv(
        os.path.join(results_root, clip, "stage2", "filtered.csv"), frames)
    write_metadata(os.path.join(results_root, clip, "result.json"),
                   {"clip_params": _CLIP_PARAMS})

