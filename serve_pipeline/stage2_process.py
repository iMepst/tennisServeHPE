"""Stage 2a orchestrator: raw landmark CSV -> gated CSV + gap stats + QC plot.

Reads only the persisted Stage 1 outputs (the landmark CSV, and the meta JSON
for fps); the model is never re-run. Applies visibility gating, detects
per-landmark gaps, and writes the gated series, a Stage 2 metadata JSON
(parameters + gap statistics + provenance) and a raw-vs-gated QC image.

This is the Stage 2a step only. Filtering, interpolation and kinematics are
later sprints and are not performed here.

Outputs go to the sibling  <clip>/stage2/  folder next to the input CSV:
    gated.csv       gated time series (keep-and-flag; see persistence)
    meta.json       parameters + gap statistics + provenance
    gating_qc.png   raw-vs-gated visual sanity check

Usage:
    python -m serve_pipeline.stage2_process \
        results/serve_01/stage1/landmarks.csv
    python -m serve_pipeline.stage2_process \
        results/serve_01/stage1/landmarks.csv --visibility-threshold 0.5
"""

import argparse
import datetime
import logging
import os
from typing import Any, Dict, List, Optional

from . import __version__
from .gating import compute_gap_statistics, gate_frames
from .layout import (
    META_JSON,
    STAGE2,
    clip_from_stage_file,
    sibling_stage_dir,
)
from .persistence import (
    git_commit_hash,
    read_landmarks_csv,
    read_metadata,
    write_gated_csv,
    write_metadata,
)
from .plotting import DEFAULT_QC_LANDMARKS, plot_raw_vs_gated
from .pose_extraction import FramePose

logger = logging.getLogger(__name__)

DEFAULT_VISIBILITY_THRESHOLD = 0.5


def _resolve_fps(meta_path: Optional[str], frames: List[FramePose]) -> float:
    """fps from the Stage 1 meta JSON, falling back to frame timestamps."""
    if meta_path and os.path.isfile(meta_path):
        fps = read_metadata(meta_path).get("video", {}).get("fps")
        if fps:
            return float(fps)
    if len(frames) >= 2:
        dt = frames[1].time_s - frames[0].time_s
        if dt > 0:
            return 1.0 / dt
    raise ValueError(
        "Could not determine fps from meta JSON or frame timestamps; "
        "pass the Stage 1 --meta explicitly."
    )


def run_stage2a(csv_path: str, outdir: Optional[str] = None,
                meta_path: Optional[str] = None,
                visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD,
                qc_landmarks: Optional[List[str]] = None) -> Dict[str, Any]:
    clip = clip_from_stage_file(csv_path)
    if outdir is None:
        outdir = sibling_stage_dir(csv_path, STAGE2)
    os.makedirs(outdir, exist_ok=True)
    if meta_path is None:
        candidate = os.path.join(os.path.dirname(csv_path), META_JSON)
        meta_path = candidate if os.path.isfile(candidate) else None

    frames = read_landmarks_csv(csv_path)
    fps = _resolve_fps(meta_path, frames)

    gated = gate_frames(frames, visibility_threshold)
    gap_stats = compute_gap_statistics(gated, fps)

    paths = {
        "gated_csv": os.path.join(outdir, "gated.csv"),
        "stage2_meta_json": os.path.join(outdir, META_JSON),
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
        "input_landmarks_csv": os.path.abspath(csv_path),
        "input_meta_json": os.path.abspath(meta_path) if meta_path else None,
        "parameters": {
            "visibility_threshold": visibility_threshold,
            "fps": fps,
        },
        "gap_statistics": gap_stats,
        "outputs": {k: os.path.abspath(v) for k, v in paths.items()},
    }
    write_metadata(paths["stage2_meta_json"], meta)

    landmarks = qc_landmarks or DEFAULT_QC_LANDMARKS
    plot_raw_vs_gated(gated, landmarks, visibility_threshold,
                      paths["gating_qc_png"])

    per_lm = gap_stats["per_landmark"]
    overall = sum(v["valid_rate"] for v in per_lm.values()) / len(per_lm)
    logger.info("Stage 2a (gating) complete")
    logger.info("  gated series: %s", paths["gated_csv"])
    logger.info("  metadata:     %s", paths["stage2_meta_json"])
    logger.info("  QC plot:      %s", paths["gating_qc_png"])
    logger.info("  rule: visibility >= %.2f  (fps %.3g)",
                visibility_threshold, fps)
    logger.info("  overall valid rate: %.1f%% across %d landmarks",
                overall * 100.0, len(per_lm))
    for name, rate in gap_stats["lowest_valid_rate"].items():
        logger.info("    lowest: %-14s %.1f%%", name, rate * 100.0)
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Stage 2a: visibility gating and gap handling.")
    parser.add_argument("landmarks_csv",
                        help="path to a Stage 1 stage1/landmarks.csv")
    parser.add_argument("--outdir", default=None,
                        help="override output dir (default: the sibling "
                             "<clip>/stage2/ next to the input)")
    parser.add_argument("--meta", default=None,
                        help="Stage 1 meta.json (for fps); auto-detected "
                             "next to the CSV if omitted")
    parser.add_argument("--visibility-threshold", type=float,
                        default=DEFAULT_VISIBILITY_THRESHOLD)
    parser.add_argument("--qc-landmarks", nargs="*", default=None,
                        help="landmark names to plot (default: serving-arm "
                             "elbows and wrists)")
    args = parser.parse_args()
    run_stage2a(
        csv_path=args.landmarks_csv,
        outdir=args.outdir,
        meta_path=args.meta,
        visibility_threshold=args.visibility_threshold,
        qc_landmarks=args.qc_landmarks,
    )


if __name__ == "__main__":
    main()
