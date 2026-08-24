"""Orchestrator: run Stages 3-5 in memory for one clip.

Stages 1-2 persist to disk (extraction, then gating/filtering). This
orchestrator picks up the Stage 2 filtered trajectory, runs key-event
detection (Stage 3), angle computation (Stage 4) and rule evaluation
(Stage 5) in memory, and writes a single result JSON for the clip
(pipeline_spec.md, Stages 3-5).

The four core outputs do not depend on fps; fps only sets the slow-motion
QC flag and (upstream) the Stage 2 filter design.
"""

import argparse
import datetime
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
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
    read_landmarks_csv,
    read_metadata,
    write_metadata,
)
from .rules import Indicator, evaluate_all
from .visualization import save_key_frame_stills

logger = logging.getLogger(__name__)


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


def assemble_result(result: ClipResult,
                    filtered_csv: str) -> Dict[str, Any]:
    """Stage 5 plus provenance, as the single result dict for the clip.

    Runs rule evaluation on the angle readings and gathers the whole
    in-memory chain (key frames, slow-motion flag, angles, indicators)
    into one JSON-serialisable record. The dense per-frame trajectory is
    deliberately left out: only the located instants and their readings
    are reported.
    """
    indicators: List[Indicator] = evaluate_all(result.angles,
                                               result.clip_params)
    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        "clip": result.clip,
        "pipeline_version": __version__,
        "commit": git_commit_hash(),
        "created_utc": now.isoformat(),
        "input_filtered_csv": os.path.abspath(filtered_csv),
        "clip_params": asdict(result.clip_params),
        "key_events": asdict(result.key_events),
        "slow_motion": asdict(result.slow_motion),
        "angles": asdict(result.angles),
        "indicators": [asdict(i) for i in indicators],
    }


def write_result(filtered_csv: str,
                 result_dict: Dict[str, Any]) -> str:
    """Write the result dict to results/<clip>/result.json.

    The single JSON sits in the clip's results folder beside the
    persisted stage1/stage2 subfolders; Stages 3-5 hold nothing else on
    disk.
    """
    clip_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(filtered_csv)))
    os.makedirs(clip_dir, exist_ok=True)
    out_path = os.path.join(clip_dir, "result.json")
    write_metadata(out_path, result_dict)
    return out_path


def _angle_line(name: str, value: Optional[float]) -> str:
    return f"{name}: {value:.1f} deg" if value is not None else f"{name}: n/a"


def write_key_frame_stills(video_path: str, stage1_meta: str,
                           filtered_csv: str,
                           result: ClipResult) -> Optional[str]:
    """Render results/<clip>/key_frames.png: the trophy and impact stills.

    Reads the raw landmarks persisted by Stage 1 for the pose overlay and
    the located key frames from the Stage 3 result, labels each with the
    angles read at it (Stage 4), and tiles them side by side. Returns the
    PNG path, or None when neither key frame was locatable.
    """
    ev = result.key_events
    ang = result.angles
    specs: List[Tuple[int, List[str]]] = []
    if ev.trophy_locatable and ev.trophy_frame is not None:
        specs.append((ev.trophy_frame, [
            f"TROPHY  frame {ev.trophy_frame}",
            _angle_line("trunk incl.", ang.trunk_inclination),
            _angle_line("knee flex", ang.front_knee_flexion)]))
    if ev.impact_locatable and ev.impact_frame is not None:
        specs.append((ev.impact_frame, [
            f"IMPACT  frame {ev.impact_frame}",
            _angle_line("elbow flex", ang.elbow_flexion),
            _angle_line("shoulder elev", ang.shoulder_elevation)]))
    if not specs:
        return None
    # The stills are an optional QC figure: they need the source video and
    # the Stage 1 raw landmarks, neither of which Stages 3-5 otherwise
    # require. Skip (rather than fail the run) when either is unavailable.
    landmarks_csv = os.path.join(os.path.dirname(stage1_meta), "landmarks.csv")
    if not (os.path.isfile(video_path) and os.path.isfile(landmarks_csv)
            and os.path.getsize(landmarks_csv) > 0):
        logger.info("  key stills:  skipped (source video or raw landmarks "
                    "unavailable)")
        return None
    frame_poses = read_landmarks_csv(landmarks_csv)
    clip_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(filtered_csv)))
    out_path = os.path.join(clip_dir, "key_frames.png")
    return save_key_frame_stills(video_path, frame_poses, specs, out_path)
def _log_summary(result: ClipResult, result_dict: Dict[str, Any],
                 out_path: str, stills_path: Optional[str] = None) -> None:
    ev = result.key_events
    logger.info("Run complete for clip %s", result.clip)
    logger.info("  result JSON: %s", out_path)
    if stills_path is not None:
        logger.info("  key stills:  %s", stills_path)
    if ev.trophy_locatable and ev.impact_locatable:
        logger.info("  key frames:  trophy %d, impact %d",
                    ev.trophy_frame, ev.impact_frame)
    else:
        logger.info("  key frames:  not locatable (%s)", ev.reason)
    sm = result.slow_motion
    if sm.assessable:
        logger.info("  slow-motion: %s (trophy->impact %.2f s)",
                    "likely" if sm.likely_slow_motion else "no",
                    sm.trophy_to_impact_s)
    for ind in result_dict["indicators"]:
        angle = ind["angle"]
        logger.info("  %-18s %-11s %s", ind["criterion"], ind["status"],
                    "" if angle is None else f"{angle:.1f} deg")


def process_clip(filtered_csv: str, serving_arm: str, front_leg: str,
                 camera_plane: str, view_direction: str,
                 stage1_meta: Optional[str] = None,
                 fps: Optional[float] = None,
                 frame_width: Optional[int] = None,
                 frame_height: Optional[int] = None) -> str:
    """Full Stage 3-5 run for one clip; returns the result JSON path."""
    result = run_clip(filtered_csv, serving_arm, front_leg, camera_plane,
                      view_direction, stage1_meta, fps, frame_width,
                      frame_height)
    result_dict = assemble_result(result, filtered_csv)
    out_path = write_result(filtered_csv, result_dict)
    _log_summary(result, result_dict, out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Stages 3-5: key events, angles and rule evaluation "
                    "for one clip; writes results/<clip>/run/result.json.")
    parser.add_argument("filtered_csv",
                        help="path to a Stage 2 stage2/filtered.csv")
    # Manually recorded per-clip parameters (anatomical, body-relative).
    parser.add_argument("--serving-arm", required=True,
                        choices=("left", "right"))
    parser.add_argument("--front-leg", required=True,
                        choices=("left", "right"))
    parser.add_argument("--camera-plane", required=True,
                        choices=("frontal", "sagittal"))
    parser.add_argument("--view-direction", required=True,
                        help="front/back for frontal, left/right for "
                             "sagittal (provenance only)")
    # fps and frame size default to the Stage 1 meta; override if needed.
    parser.add_argument("--stage1-meta", default=None,
                        help="Stage 1 meta.json; auto-detected under the "
                             "clip's stage1/ folder if omitted")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--frame-width", type=int, default=None)
    parser.add_argument("--frame-height", type=int, default=None)
    args = parser.parse_args()
    process_clip(
        filtered_csv=args.filtered_csv,
        serving_arm=args.serving_arm,
        front_leg=args.front_leg,
        camera_plane=args.camera_plane,
        view_direction=args.view_direction,
        stage1_meta=args.stage1_meta,
        fps=args.fps,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )


if __name__ == "__main__":
    main()
