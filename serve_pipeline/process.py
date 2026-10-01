"""Stage 2 orchestrator: gating (2a), interpolation (2b), filtering (2c)."""

import datetime
import logging
import os
from typing import Any, Dict, Optional

from . import __version__
from .config import PipelineConfig
from .filtering import FilterConfig, filter_series
from .gating import compute_gap_statistics, gate_frames
from .interpolation import interpolate_gaps, summarize_interpolation
from .layout import STAGE2, clip_from_stage_file, sibling_stage_dir
from .persistence import (
    git_commit_hash,
    read_gated_csv,
    read_landmarks_csv,
    read_metadata,
    write_filtered_csv,
    write_gated_csv,
    write_metadata,
)
from .plotting import (
    DEFAULT_QC_LANDMARKS,
    plot_raw_vs_filtered,
    plot_raw_vs_gated,
)

logger = logging.getLogger(__name__)

_DEFAULTS = PipelineConfig()

GATING_META_JSON = "gating_meta.json"
FILTERING_META_JSON = "filtering_meta.json"


def run_gating(landmarks_csv: str,
               meta_path: Optional[str] = None) -> Dict[str, Any]:
    clip = clip_from_stage_file(landmarks_csv)
    outdir = sibling_stage_dir(landmarks_csv, STAGE2)
    os.makedirs(outdir, exist_ok=True)
    if meta_path is None:
        meta_path = os.path.join(os.path.dirname(landmarks_csv), "meta.json")
    visibility_threshold = _DEFAULTS.visibility_threshold

    frames = read_landmarks_csv(landmarks_csv)
    fps = float(read_metadata(meta_path)["video"]["fps"])

    gated = gate_frames(frames, visibility_threshold)
    gap_stats = compute_gap_statistics(gated, fps)

    paths = {
        "gated_csv": os.path.join(outdir, "gated.csv"),
        "gating_meta_json": os.path.join(outdir, GATING_META_JSON),
        "gating_qc_png": os.path.join(outdir, "gating_qc.png"),
    }
    write_gated_csv(paths["gated_csv"], gated)

    now = datetime.datetime.now(datetime.timezone.utc)
    meta: Dict[str, Any] = {
        "stage": "2a",
        "clip": clip,
        "step": "gating",
        "pipeline_version": __version__,
        "commit": git_commit_hash(),
        "created_utc": now.isoformat(),
        "input_landmarks_csv": os.path.abspath(landmarks_csv),
        "input_meta_json": os.path.abspath(meta_path),
        "parameters": {
            "visibility_threshold": visibility_threshold,
            "fps": fps,
        },
        "gap_statistics": gap_stats,
        "outputs": {k: os.path.abspath(v) for k, v in paths.items()},
    }
    write_metadata(paths["gating_meta_json"], meta)

    plot_raw_vs_gated(gated, DEFAULT_QC_LANDMARKS, visibility_threshold,
                      paths["gating_qc_png"])

    per_landmark = gap_stats["per_landmark"]
    overall = (sum(v["valid_rate"] for v in per_landmark.values())
               / len(per_landmark))
    logger.info("Stage 2a (gating) complete")
    logger.info("  gated series: %s", paths["gated_csv"])
    logger.info("  metadata:     %s", paths["gating_meta_json"])
    logger.info("  QC plot:      %s", paths["gating_qc_png"])
    logger.info("  rule: visibility >= %.2f  (fps %.3g)",
                visibility_threshold, fps)
    logger.info("  overall valid rate: %.1f%% across %d landmarks",
                overall * 100.0, len(per_landmark))
    for name, rate in gap_stats["lowest_valid_rate"].items():
        logger.info("    lowest: %-14s %.1f%%", name, rate * 100.0)
    return meta


def run_filtering(gated_csv: str,
                  meta_path: Optional[str] = None) -> Dict[str, Any]:
    clip = clip_from_stage_file(gated_csv)
    outdir = os.path.dirname(os.path.abspath(gated_csv))
    if meta_path is None:
        meta_path = os.path.join(os.path.dirname(gated_csv), GATING_META_JSON)
    max_gap_ms = _DEFAULTS.max_gap_ms
    filter_cfg = FilterConfig()

    gated = read_gated_csv(gated_csv)
    fps = float(read_metadata(meta_path)["parameters"]["fps"])
    # The gap bound is defined in time; convert it to this clip's frames so
    # the same physical gap length holds at any frame rate.
    max_gap_frames = round(max_gap_ms / 1000.0 * fps)

    pre_filter = interpolate_gaps(gated, max_gap_frames)
    interp_stats = summarize_interpolation(pre_filter)

    # Fresh interpolation pass to filter in place, leaving pre_filter
    # untouched for QC.
    filtered = interpolate_gaps(gated, max_gap_frames)
    filter_stats = filter_series(filtered, fps, filter_cfg)

    paths = {
        "filtered_csv": os.path.join(outdir, "filtered.csv"),
        "filtering_meta_json": os.path.join(outdir, FILTERING_META_JSON),
        "filtering_qc_png": os.path.join(outdir, "filtering_qc.png"),
    }
    write_filtered_csv(paths["filtered_csv"], filtered)

    now = datetime.datetime.now(datetime.timezone.utc)
    meta: Dict[str, Any] = {
        "stage": "2b-2c",
        "clip": clip,
        "step": "interpolation, filtering",
        "pipeline_version": __version__,
        "commit": git_commit_hash(),
        "created_utc": now.isoformat(),
        "input_gated_csv": os.path.abspath(gated_csv),
        "input_gating_meta_json": os.path.abspath(meta_path),
        "parameters": {
            "max_gap_ms": max_gap_ms,
            "max_gap_frames": max_gap_frames,
            "fps": fps,
            "filter": filter_cfg.to_dict(),
        },
        "interpolation": interp_stats,
        "filtering": filter_stats,
        "outputs": {k: os.path.abspath(v) for k, v in paths.items()},
    }
    write_metadata(paths["filtering_meta_json"], meta)

    plot_raw_vs_filtered(
        pre_filter, filtered, f"butterworth {filter_cfg.cutoff_hz:g} Hz",
        DEFAULT_QC_LANDMARKS, "y", paths["filtering_qc_png"],
        title=f"filtered (y) - {clip}")

    logger.info("Stage 2b-2c (interpolation, filtering) complete")
    logger.info("  filtered series: %s", paths["filtered_csv"])
    logger.info("  metadata:        %s", paths["filtering_meta_json"])
    logger.info("  QC plot:         %s", paths["filtering_qc_png"])
    logger.info("  interpolation: max gap %.0f ms (%d frames), "
                "%d samples filled",
                max_gap_ms, max_gap_frames,
                interp_stats["total_interpolated_samples"])
    logger.info("  unreliable (long/edge gaps): %d samples",
                interp_stats["total_unreliable_samples"])
    logger.info("  filter: butterworth order %d cutoff %s Hz  (fps %.3g)",
                filter_cfg.order, filter_cfg.cutoff_hz, fps)
    logger.info("  filtered %d of %d reliable samples",
                filter_stats["n_filtered_samples"],
                filter_stats["n_reliable_samples"])
    return meta
