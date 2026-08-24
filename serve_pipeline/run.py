"""Orchestrator: run Stages 3-5 in memory for one clip.

Stages 1-2 persist to disk (extraction, then gating/filtering). This
orchestrator picks up the Stage 2 filtered trajectory, runs key-event
detection (Stage 3), angle computation (Stage 4) and rule evaluation
(Stage 5) in memory, and writes a single result JSON for the clip
(pipeline_spec.md, Stages 3-5).

The four core outputs do not depend on fps; fps only sets the slow-motion
QC flag and (upstream) the Stage 2 filter design.
"""

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
from .angles import AngleReadings, compute_angles
from .config import ClipParams
from .interpolation import ProcessedFrame
from .keyevents import (
    KeyEvents,
    SlowMotionFlag,
    detect_key_events,
    flag_possible_slow_motion,
)
from .layout import clip_from_stage_file
from .persistence import (
    git_commit_hash,
    read_filtered_csv,
    read_metadata,
    write_metadata,
)


def _resolve_video_meta(filtered_csv: str, stage1_meta: Optional[str],
                        fps: Optional[float], frame_width: Optional[int],
                        frame_height: Optional[int]
                        ) -> Tuple[float, int, int]:
    """fps and frame size, from the Stage 1 meta JSON unless overridden.

    The Stage 1 meta records the container fps and frame dimensions of the
    decoded video; explicit arguments win over it (e.g. to record a
    manually corrected container fps). Falls back to auto-detecting the
    Stage 1 meta next to the clip's stage1 folder.
    """
    if stage1_meta is None:
        clip_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(filtered_csv)))
        candidate = os.path.join(clip_dir, "stage1", "meta.json")
        stage1_meta = candidate if os.path.isfile(candidate) else None

    video: dict = {}
    if stage1_meta and os.path.isfile(stage1_meta):
        video = read_metadata(stage1_meta).get("video", {})

    fps = fps if fps is not None else video.get("fps")
    frame_width = (frame_width if frame_width is not None
                   else video.get("width"))
    frame_height = (frame_height if frame_height is not None
                    else video.get("height"))
    if fps is None or frame_width is None or frame_height is None:
        raise ValueError(
            "fps, frame_width and frame_height must come from the Stage 1 "
            "meta JSON or be passed explicitly.")
    return float(fps), int(frame_width), int(frame_height)


@dataclass
class ClipResult:
    """In-memory result of Stages 3-4 for one clip (Stage 5 added later)."""
    clip: str
    clip_params: ClipParams
    frames: List[ProcessedFrame]
    key_events: KeyEvents
    slow_motion: SlowMotionFlag
    angles: AngleReadings


def run_clip(filtered_csv: str, serving_arm: str, front_leg: str,
             camera_plane: str, view_direction: str,
             stage1_meta: Optional[str] = None,
             fps: Optional[float] = None,
             frame_width: Optional[int] = None,
             frame_height: Optional[int] = None) -> ClipResult:
    """Run Stages 3-4 in memory on a Stage 2 filtered trajectory.

    The manual per-clip parameters (anatomical serving arm / front leg,
    camera plane, view direction) are recorded by hand; fps and frame
    size default to the Stage 1 meta. Returns the located key events, the
    slow-motion QC flag and the four angle readings.
    """
    clip = clip_from_stage_file(filtered_csv)
    fps, frame_width, frame_height = _resolve_video_meta(
        filtered_csv, stage1_meta, fps, frame_width, frame_height)
    clip_params = ClipParams(
        serving_arm=serving_arm, front_leg=front_leg,
        camera_plane=camera_plane, view_direction=view_direction,
        fps=fps, frame_width=frame_width, frame_height=frame_height)

    frames = read_filtered_csv(filtered_csv)
    key_events = detect_key_events(frames, clip_params)
    slow_motion = flag_possible_slow_motion(key_events, fps)
    angles = compute_angles(frames, key_events, clip_params)

    return ClipResult(clip=clip, clip_params=clip_params, frames=frames,
                      key_events=key_events, slow_motion=slow_motion,
                      angles=angles)