"""Orchestrator: run key-event detection, angles and rule evaluation in
memory for one clip.

Extraction, gating and filtering persist to disk; this orchestrator picks up
the filtered trajectory, runs the in-memory steps, and writes a single result
JSON for the clip.

The four core outputs do not depend on fps; fps only sets the slow-motion QC
flag and (upstream) the filter design.
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
from .layout import STAGE1, STAGE2, clip_from_stage_file, clip_from_video, \
    stage_dir
from .persistence import (
    git_commit_hash,
    read_filtered_csv,
    read_landmarks_csv,
    read_metadata,
    write_metadata,
)
from .rules import Indicator, evaluate_all
from .extract import DEFAULT_MODEL, run_extraction
from .process import GATING_META_JSON, run_filtering, run_gating
from .visualization import save_key_frame_stills

logger = logging.getLogger(__name__)

def ensure_filtered(video_path: str, outdir: str = "results",
                    reuse: bool = True,
                    model_path: Optional[str] = None) -> Tuple[str, str]:
    """Run (or reuse) extraction + gating + filtering; return (filtered_csv, meta).

    These persist to disk; when reuse is set, a step whose output already exists
    is skipped (extraction is the slow one). Returns the filtered trajectory and
    the extraction meta JSON, the two inputs the in-memory steps need.
    """
    clip = clip_from_video(video_path)
    stage1_dir = stage_dir(outdir, clip, STAGE1)
    stage2_dir = stage_dir(outdir, clip, STAGE2)
    landmarks_csv = os.path.join(stage1_dir, "landmarks.csv")
    stage1_meta = os.path.join(stage1_dir, "meta.json")
    gated_csv = os.path.join(stage2_dir, "gated.csv")
    gating_meta = os.path.join(stage2_dir, GATING_META_JSON)
    filtered_csv = os.path.join(stage2_dir, "filtered.csv")

    # Pose extraction (the slow step).
    if reuse and os.path.isfile(landmarks_csv):
        logger.info("extraction: reusing %s", landmarks_csv)
    else:
        run_extraction(video_path, outdir=outdir,
                       model_path=model_path or DEFAULT_MODEL)

    # Visibility gating.
    if reuse and os.path.isfile(gated_csv):
        logger.info("gating: reusing %s", gated_csv)
    else:
        run_gating(landmarks_csv, meta_path=stage1_meta)

    # Interpolation + low-pass filtering.
    if reuse and os.path.isfile(filtered_csv):
        logger.info("filtering: reusing %s", filtered_csv)
    else:
        run_filtering(gated_csv, meta_path=gating_meta)

    return filtered_csv, stage1_meta


def _resolve_video_meta(filtered_csv: str, stage1_meta: Optional[str],
                        fps: Optional[float], frame_width: Optional[int],
                        frame_height: Optional[int]
                        ) -> Tuple[float, int, int]:
    """fps and frame size, from the extraction meta JSON unless overridden.

    The meta records the container fps and frame dimensions; explicit arguments
    win (e.g. a manually corrected fps). Falls back to auto-detecting the meta
    in the clip's stage1 folder.
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
            "fps, frame_width and frame_height must come from the extraction "
            "meta JSON or be passed explicitly.")
    return float(fps), int(frame_width), int(frame_height)


@dataclass
class ClipResult:
    """In-memory result for one clip (indicators added later)."""
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
    """Run key-event detection and angle computation on a filtered trajectory.

    The manual per-clip parameters (anatomical serving arm / front leg, camera
    plane, view direction) are recorded by hand; fps and frame size default to
    the extraction meta. Returns the key events, slow-motion flag and angles.
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
    """Rule evaluation plus provenance, as the single result dict for the clip.

    Runs rule evaluation on the angle readings and gathers the in-memory chain
    (key frames, slow-motion flag, angles, indicators) into one JSON record.
    The dense per-frame trajectory is left out: only the located instants are
    reported.
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

    Sits in the clip's results folder beside the persisted stage1/stage2
    subfolders; the in-memory steps hold nothing else on disk.
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

    Reads the raw landmarks for the pose overlay and the located key frames
    from the result, labels each with the angles read at it, and tiles them
    side by side. Returns None when neither key frame was locatable.
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
    # Optional QC figure: needs the source video and the raw landmarks, neither
    # of which the in-memory steps otherwise require. Skip (rather than fail the
    # run) when either is unavailable.
    landmarks_csv = os.path.join(os.path.dirname(stage1_meta), "landmarks.csv")
    if not (os.path.isfile(video_path) and os.path.isfile(landmarks_csv)
            and os.path.getsize(landmarks_csv) > 0):
        logger.info("  key stills:  skipped (source video or raw landmarks "
                    "unavailable)")
        return None
    frame_poses = read_landmarks_csv(
        landmarks_csv, frame_indices={idx for idx, _ in specs})
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


def process_clip(video_path: str, serving_arm: str, front_leg: str,
                 camera_plane: str, view_direction: str,
                 outdir: str = "results", reuse: bool = True,
                 model_path: Optional[str] = None,
                 fps: Optional[float] = None,
                 frame_width: Optional[int] = None,
                 frame_height: Optional[int] = None) -> str:
    """Run one clip end to end; returns the result JSON path.

    Extraction/gating/filtering run (or reuse) on disk; the rest runs in memory.
    fps and frame size default to the extraction meta and can be overridden.
    """
    filtered_csv, stage1_meta = ensure_filtered(
        video_path, outdir=outdir, reuse=reuse, model_path=model_path)
    result = run_clip(filtered_csv, serving_arm, front_leg, camera_plane,
                      view_direction, stage1_meta, fps, frame_width,
                      frame_height)
    result_dict = assemble_result(result, filtered_csv)
    out_path = write_result(filtered_csv, result_dict)
    stills_path = write_key_frame_stills(
        video_path, stage1_meta, filtered_csv, result)
    _log_summary(result, result_dict, out_path, stills_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Run one serve clip end to end; writes "
                    "results/<clip>/result.json.")
    parser.add_argument("video", help="path to the input serve video")
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
    parser.add_argument("--outdir", default="results",
                        help="results root (default: results)")
    parser.add_argument("--model", default=None,
                        help="path to a pose_landmarker .task file "
                             "(default: the heavy model)")
    parser.add_argument("--no-reuse", dest="reuse", action="store_false",
                        help="recompute extraction/gating/filtering even if "
                             "outputs exist")
    # fps and frame size default to the extraction meta; override if needed.
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--frame-width", type=int, default=None)
    parser.add_argument("--frame-height", type=int, default=None)
    args = parser.parse_args()
    process_clip(
        video_path=args.video,
        serving_arm=args.serving_arm,
        front_leg=args.front_leg,
        camera_plane=args.camera_plane,
        view_direction=args.view_direction,
        outdir=args.outdir,
        reuse=args.reuse,
        model_path=args.model,
        fps=args.fps,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )


if __name__ == "__main__":
    main()
