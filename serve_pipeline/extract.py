"""Stage 1 orchestrator: video -> landmark CSV, meta JSON, overlay MP4."""

import datetime
import logging
import os
from typing import Any, Dict, List

import mediapipe

from . import __version__
from .config import PipelineConfig
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

DEFAULT_MODEL = PipelineConfig().model_path

CONTACT_SHEET_FRAMES = 8
PROGRESS_EVERY = 25


def run_extraction(video_path: str, outdir: str = "results",
                   model_path: str = DEFAULT_MODEL) -> Dict[str, Any]:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Pose model not found: {model_path}\nDownload it with:\n"
            "curl -L --create-dirs -o models/pose_landmarker_heavy.task "
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
        video = reader.metadata
        n_expected = video.frame_count_reported
        sheet_indices = set(range(
            0, n_expected, max(1, n_expected // CONTACT_SHEET_FRAMES)))

        with PoseExtractor(model_path=model_path) as extractor, \
                LandmarkCsvWriter(paths["landmarks_csv"]) as csv_out, \
                OverlayVideoWriter(
                    paths["overlay_mp4"], video.fps,
                    video.width, video.height) as vid_out:
            for frame in reader:
                frame_pose = extractor.process(frame.index, frame.time_s,
                                               frame.image_bgr)
                # Persist first, before overlay/sheet work can fail.
                csv_out.write_frame(frame_pose)
                overlay = draw_pose(frame.image_bgr, frame_pose)
                vid_out.write(overlay)
                frame_poses.append(frame_pose)
                if frame.index in sheet_indices:
                    sheet_frames.append(overlay)
                if frame.index % PROGRESS_EVERY == 0:
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
        "video": video.to_dict(),
        "extractor": extractor_config,
        "statistics": stats,
        "outputs": {k: os.path.abspath(v) for k, v in paths.items()},
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
