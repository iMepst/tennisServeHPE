"""Stage 1 orchestrator: video -> raw landmark CSV + meta JSON + overlay MP4."""

import argparse
import datetime
import logging
import os
from typing import Any, Dict, List, Optional

import mediapipe

from . import __version__
from .ingestion import BgrImage, VideoReader
from .layout import STAGE1, clip_from_video, stage_dir
from .persistence import (
    LandmarkCsvWriter,
    git_commit_hash,
    summarize_extraction,
    write_metadata,
)
from .pose_extraction import FramePose, PoseExtractor
from .visualization import OverlayVideoWriter, draw_pose, save_contact_sheet

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "pose_landmarker_heavy.task",
)

COORDINATE_NOTE = (
    "x,y normalized by image width/height (multiply to get pixels); "
    "z is relative image-space depth. world_x/y/z are meters with origin "
    "at the hip center and are the preferred input for joint angles."
)


def run_stage1(video_path: str, outdir: str = "results",
               model_path: str = DEFAULT_MODEL,
               min_detection_confidence: float = 0.5,
               min_tracking_confidence: float = 0.5,
               max_frames: Optional[int] = None,
               contact_sheet_frames: int = 8,
               progress_every: int = 25) -> Dict[str, Any]:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Pose model not found: {model_path}\nDownload it with:\n"
            "curl -L -o models/pose_landmarker_heavy.task "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        )
    clip = clip_from_video(video_path)
    out_dir = stage_dir(outdir, clip, STAGE1)
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "landmarks_csv": os.path.join(out_dir, "landmarks.csv"),
        "meta_json": os.path.join(out_dir, "meta.json"),
        "overlay_mp4": os.path.join(out_dir, "overlay.mp4"),
        "contact_sheet_png": os.path.join(out_dir, "contact_sheet.png"),
    }

    frame_poses: List[FramePose] = []
    sheet_frames: List[BgrImage] = []

    with VideoReader(video_path) as reader:
        meta_video = reader.metadata
        # Pick evenly spaced frames for the contact sheet up front.
        n_expected = meta_video.frame_count_reported
        if max_frames is not None and n_expected > 0:
            n_expected = min(n_expected, max_frames)
        sheet_indices = set()
        if n_expected > 0:
            step = max(1, n_expected // contact_sheet_frames)
            sheet_indices = set(range(0, n_expected, step))

        with PoseExtractor(
            model_path=model_path,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) as extractor, \
                LandmarkCsvWriter(paths["landmarks_csv"]) as csv_out, \
                OverlayVideoWriter(
                    paths["overlay_mp4"], meta_video.fps,
                    meta_video.width, meta_video.height) as vid_out:
            for frame in reader:
                if max_frames is not None and frame.index >= max_frames:
                    break
                frame_pose = extractor.process(frame.index, frame.time_s,
                                               frame.image_bgr)
                # persist before anything else can fail
                csv_out.write_frame(frame_pose)
                overlay = draw_pose(frame.image_bgr, frame_pose)
                vid_out.write(overlay)
                frame_poses.append(frame_pose)
                if frame.index in sheet_indices:
                    sheet_frames.append(overlay)
                if frame.index % progress_every == 0:
                    logger.info(
                        "  frame %d%s", frame.index,
                        "" if frame_pose.detected else "  [no pose]")
            extractor_config = extractor.config

    if sheet_frames:
        save_contact_sheet(paths["contact_sheet_png"], sheet_frames)

    stats = summarize_extraction(frame_poses)
    now = datetime.datetime.now(datetime.timezone.utc)
    meta: Dict[str, Any] = {
        "stage": 1,
        "clip": clip,
        "pipeline_version": __version__,
        "commit": git_commit_hash(),
        "mediapipe_version": mediapipe.__version__,
        "created_utc": now.isoformat(),
        "video": meta_video.to_dict(),
        "extractor": extractor_config,
        "statistics": stats,
        "outputs": {k: os.path.abspath(v) for k, v in paths.items()},
        "coordinate_note": COORDINATE_NOTE,
    }
    write_metadata(paths["meta_json"], meta)

    logger.info("")
    logger.info("Stage 1 complete")
    logger.info("  landmarks:     %s", paths["landmarks_csv"])
    logger.info("  metadata:      %s", paths["meta_json"])
    logger.info("  overlay video: %s", paths["overlay_mp4"])
    logger.info("  contact sheet: %s", paths["contact_sheet_png"])
    logger.info(
        "  detection rate: %.1f%% (%d/%d frames), mean visibility %.2f",
        stats["detection_rate"] * 100.0,
        stats["frames_with_pose"], stats["frames_processed"],
        stats["mean_visibility"])
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Stage 1: BlazePose extraction + diagnostic overlay.")
    parser.add_argument("video", help="path to the input serve video")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="path to a pose_landmarker .task file")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="limit frames for quick tests")
    args = parser.parse_args()
    run_stage1(
        video_path=args.video,
        outdir=args.outdir,
        model_path=args.model,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
