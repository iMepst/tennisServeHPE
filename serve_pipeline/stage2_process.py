"""Stage 2 orchestrator: gating (2a) and filtering/interpolation (2b).

Each sub-step reads only its predecessor's persisted output (the model is
never re-run) and writes into the clip's ``stage2/`` folder with sub-step
named artifacts so the steps never collide:

    2a  gating:     gated.csv     gating_meta.json     gating_qc.png
    2b  filtering:  filtered.csv  filtering_meta.json  filtering_qc.png
                    filtering_compare.png  (candidate-cut-off comparison)

Stage 2a applies visibility gating and gap statistics. Stage 2b interpolates
short gaps (flagging every filled sample), marks long/edge gaps unreliable,
and low-pass filters the reliable segments. The filter type and cut-off are an
empirical decision made from ``filtering_compare.png`` (methodology 5.2); the
CLI defaults produce a provisional filtered series plus that comparison.

Usage:
    python -m serve_pipeline.stage2_process 2a \
        results/serve_01/stage1/landmarks.csv
    python -m serve_pipeline.stage2_process 2b \
        results/serve_01/stage2/gated.csv --cutoff-hz 6
"""

import argparse
import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .filtering import (
    KIND_BUTTERWORTH,
    KIND_SAVGOL,
    FilterConfig,
    filter_series,
)
from .gating import GatedFrame, compute_gap_statistics, gate_frames
from .interpolation import (
    ProcessedFrame,
    interpolate_gaps,
    summarize_interpolation,
)
from .landmarks import LANDMARK_NAMES
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
    plot_velocity_compare,
)
from .pose_extraction import FramePose

logger = logging.getLogger(__name__)

DEFAULT_VISIBILITY_THRESHOLD = 0.5
DEFAULT_MAX_GAP_FRAMES = 3
DEFAULT_FILTER_ORDER = 4
# Decided in Sprint 2 from filtering_velocity_compare.png (methodology 5.2):
# 5 Hz preserves the racket-arm velocity peaks while removing the >5 Hz jitter
# that 8 Hz leaves in, ahead of Stage 3 differentiation (3 Hz clipped peaks).
DEFAULT_CUTOFF_HZ = 5.0
# Butterworth cut-offs overlaid in the comparison plot for the filter decision.
CANDIDATE_CUTOFFS_HZ = [3.0, 5.0, 8.0]
# Channel shown in the raw-vs-filtered plot (image vertical: the serve swing).
DEFAULT_QC_COORD = "y"
# Half-width of the auto-zoom window centred on peak racket motion.
QC_WINDOW_PAD_S = 2.5

GATING_META_JSON = "gating_meta.json"
FILTERING_META_JSON = "filtering_meta.json"

# The filter settled on in Sprint 2; a run matching it is recorded "decided".
DECIDED_FILTER = FilterConfig(
    kind=KIND_BUTTERWORTH, order=DEFAULT_FILTER_ORDER,
    cutoff_hz=DEFAULT_CUTOFF_HZ)


def _is_decided_filter(cfg: FilterConfig) -> bool:
    return (cfg.kind == DECIDED_FILTER.kind
            and cfg.order == DECIDED_FILTER.order
            and cfg.cutoff_hz == DECIDED_FILTER.cutoff_hz)


# --------------------------------------------------------------------------- #
# Stage 2a: gating
# --------------------------------------------------------------------------- #
def _resolve_fps_stage1(meta_path: Optional[str],
                        frames: List[FramePose]) -> float:
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
        candidate = os.path.join(os.path.dirname(csv_path), "meta.json")
        meta_path = candidate if os.path.isfile(candidate) else None

    frames = read_landmarks_csv(csv_path)
    fps = _resolve_fps_stage1(meta_path, frames)

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
        "input_landmarks_csv": os.path.abspath(csv_path),
        "input_meta_json": os.path.abspath(meta_path) if meta_path else None,
        "parameters": {
            "visibility_threshold": visibility_threshold,
            "fps": fps,
        },
        "gap_statistics": gap_stats,
        "outputs": {k: os.path.abspath(v) for k, v in paths.items()},
    }
    write_metadata(paths["gating_meta_json"], meta)

    landmarks = qc_landmarks or DEFAULT_QC_LANDMARKS
    plot_raw_vs_gated(gated, landmarks, visibility_threshold,
                      paths["gating_qc_png"])

    per_lm = gap_stats["per_landmark"]
    overall = sum(v["valid_rate"] for v in per_lm.values()) / len(per_lm)
    logger.info("Stage 2a (gating) complete")
    logger.info("  gated series: %s", paths["gated_csv"])
    logger.info("  metadata:     %s", paths["gating_meta_json"])
    logger.info("  QC plot:      %s", paths["gating_qc_png"])
    logger.info("  rule: visibility >= %.2f  (fps %.3g)",
                visibility_threshold, fps)
    logger.info("  overall valid rate: %.1f%% across %d landmarks",
                overall * 100.0, len(per_lm))
    for name, rate in gap_stats["lowest_valid_rate"].items():
        logger.info("    lowest: %-14s %.1f%%", name, rate * 100.0)
    return meta


# --------------------------------------------------------------------------- #
# Stage 2b: interpolation + filtering
# --------------------------------------------------------------------------- #
def _resolve_fps_stage2a(meta_path: Optional[str],
                         gated: List[GatedFrame]) -> float:
    """fps from the Stage 2a meta JSON, falling back to gated timestamps."""
    if meta_path and os.path.isfile(meta_path):
        fps = read_metadata(meta_path).get("parameters", {}).get("fps")
        if fps:
            return float(fps)
    if len(gated) >= 2:
        dt = gated[1].time_s - gated[0].time_s
        if dt > 0:
            return 1.0 / dt
    raise ValueError(
        "Could not determine fps from Stage 2a meta or gated timestamps.")


def _peak_motion_window(
        frames: List[ProcessedFrame], landmark_names: List[str], coord: str,
        pad_s: float = QC_WINDOW_PAD_S) -> Optional[Tuple[float, float]]:
    """Time window centred on the swing, as a serve-proxy for the plot zoom.

    Picks the QC landmark with the largest *reliable* excursion in ``coord``
    (the serving wrist travels farthest), then centres on that landmark's peak
    reliable frame-to-frame velocity. Restricting to reliable samples keeps the
    window off the post-serve turn-away, where masked landmarks jump around.

    A heuristic for plotting only; true phase segmentation is Stage 3.
    """
    def _reliable_vals(lm_id: int) -> List[Tuple[float, float]]:
        out = []
        for f in frames:
            s = f.samples[lm_id]
            v = getattr(s, coord)
            if s.reliable and v is not None:
                out.append((f.time_s, v))
        return out

    # Landmark with the widest reliable excursion.
    best_lm: Optional[int] = None
    best_range = -1.0
    for name in landmark_names:
        vals = _reliable_vals(LANDMARK_NAMES.index(name))
        if len(vals) < 2:
            continue
        rng = max(v for _, v in vals) - min(v for _, v in vals)
        if rng > best_range:
            best_range, best_lm = rng, LANDMARK_NAMES.index(name)
    if best_lm is None:
        return None

    # Peak velocity between *adjacent* reliable frames (reset across gaps, so a
    # jump either side of a hole is never mistaken for fast motion).
    best_t: Optional[float] = None
    best_speed = -1.0
    prev_val: Optional[float] = None
    prev_pos = -2
    for pos, f in enumerate(frames):
        s = f.samples[best_lm]
        v = getattr(s, coord)
        if not s.reliable or v is None:
            prev_val = None
            continue
        if prev_val is not None and pos == prev_pos + 1:
            speed = abs(v - prev_val)
            if speed > best_speed:
                best_speed, best_t = speed, f.time_s
        prev_val, prev_pos = v, pos
    if best_t is None:
        return None
    return (best_t - pad_s, best_t + pad_s)


def run_stage2b(gated_csv_path: str, outdir: Optional[str] = None,
                meta_path: Optional[str] = None,
                max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
                filter_cfg: Optional[FilterConfig] = None,
                qc_landmarks: Optional[List[str]] = None,
                qc_coord: str = DEFAULT_QC_COORD) -> Dict[str, Any]:
    clip = clip_from_stage_file(gated_csv_path)
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(gated_csv_path))
    os.makedirs(outdir, exist_ok=True)
    if meta_path is None:
        candidate = os.path.join(os.path.dirname(gated_csv_path),
                                 GATING_META_JSON)
        meta_path = candidate if os.path.isfile(candidate) else None
    if filter_cfg is None:
        filter_cfg = FilterConfig(kind=KIND_BUTTERWORTH,
                                  order=DEFAULT_FILTER_ORDER,
                                  cutoff_hz=DEFAULT_CUTOFF_HZ)

    gated = read_gated_csv(gated_csv_path)
    fps = _resolve_fps_stage2a(meta_path, gated)

    pre_filter = interpolate_gaps(gated, max_gap_frames)
    interp_stats = summarize_interpolation(pre_filter)

    # Chosen filter -> the persisted filtered series.
    filtered = interpolate_gaps(gated, max_gap_frames)
    filter_stats = filter_series(filtered, fps, filter_cfg)

    paths = {
        "filtered_csv": os.path.join(outdir, "filtered.csv"),
        "filtering_meta_json": os.path.join(outdir, FILTERING_META_JSON),
        "filtering_qc_png": os.path.join(outdir, "filtering_qc.png"),
        "filtering_compare_png": os.path.join(outdir,
                                              "filtering_compare.png"),
        "filtering_velocity_compare_png": os.path.join(
            outdir, "filtering_velocity_compare.png"),
    }
    write_filtered_csv(paths["filtered_csv"], filtered)

    now = datetime.datetime.now(datetime.timezone.utc)
    meta: Dict[str, Any] = {
        "stage": "2b",
        "clip": clip,
        "step": "filtering",
        "pipeline_version": __version__,
        "commit": git_commit_hash(),
        "created_utc": now.isoformat(),
        "input_gated_csv": os.path.abspath(gated_csv_path),
        "input_gating_meta_json": (
            os.path.abspath(meta_path) if meta_path else None),
        "parameters": {
            "max_gap_frames": max_gap_frames,
            "max_gap_ms": max_gap_frames / fps * 1000.0 if fps else None,
            "fps": fps,
            "filter": filter_cfg.to_dict(),
            "candidate_cutoffs_hz": CANDIDATE_CUTOFFS_HZ,
        },
        "interpolation": interp_stats,
        "filtering": filter_stats,
        "filter_choice": {
            "status": ("decided" if _is_decided_filter(filter_cfg)
                       else "custom"),
            "rationale": (
                "5 Hz Butterworth (order 4) preserves the racket-arm velocity "
                "peaks while removing >5 Hz jitter ahead of Stage 3 "
                "differentiation; 3 Hz clipped the peaks and 8 Hz barely "
                "filtered."),
            "evidence": os.path.basename(
                paths["filtering_velocity_compare_png"]),
            "evidence_kind": ("coordinate-velocity diagnostic (not the "
                              "Stage 2c joint angular velocity)"),
            "methodology_ref": "5.2",
        },
        "outputs": {k: os.path.abspath(v) for k, v in paths.items()},
    }
    write_metadata(paths["filtering_meta_json"], meta)

    landmarks = qc_landmarks or DEFAULT_QC_LANDMARKS
    window = _peak_motion_window(filtered, landmarks, qc_coord)
    # Chosen-filter sanity check.
    plot_raw_vs_filtered(
        pre_filter,
        [(f"{filter_cfg.kind} {filter_cfg.cutoff_hz:g} Hz", filtered)],
        landmarks, qc_coord, paths["filtering_qc_png"],
        title=f"Stage 2b filtered ({qc_coord}) - {clip}",
        time_window=window)
    # Candidate-cut-off comparison for the empirical filter decision.
    variants = []
    for cut in CANDIDATE_CUTOFFS_HZ:
        cfg = FilterConfig(kind=KIND_BUTTERWORTH,
                           order=filter_cfg.order, cutoff_hz=cut)
        fv = interpolate_gaps(gated, max_gap_frames)
        filter_series(fv, fps, cfg)
        variants.append((f"{cut:g} Hz", fv))
    plot_raw_vs_filtered(
        pre_filter, variants, landmarks, qc_coord,
        paths["filtering_compare_png"],
        title=f"Stage 2b Butterworth cut-off comparison ({qc_coord}) - {clip}",
        time_window=window)
    # Filter-selection diagnostic (coordinate velocity, not the Stage 2c
    # angular velocity): the discriminating view for the cut-off decision.
    plot_velocity_compare(
        pre_filter, variants, landmarks, qc_coord, fps,
        paths["filtering_velocity_compare_png"],
        title=f"Stage 2b filter diagnostic (d{qc_coord}/dt) - {clip}",
        time_window=window)

    logger.info("Stage 2b (filtering) complete")
    logger.info("  filtered series: %s", paths["filtered_csv"])
    logger.info("  metadata:        %s", paths["filtering_meta_json"])
    logger.info("  QC plot:         %s", paths["filtering_qc_png"])
    logger.info("  comparison plot: %s", paths["filtering_compare_png"])
    logger.info("  interpolation: max gap %d frames (%.0f ms), "
                "%d samples filled",
                max_gap_frames, max_gap_frames / fps * 1000.0,
                interp_stats["total_interpolated_samples"])
    logger.info("  unreliable (long/edge gaps): %d samples",
                interp_stats["total_unreliable_samples"])
    logger.info("  filter: %s order %d cutoff %s Hz  (fps %.3g)",
                filter_cfg.kind, filter_cfg.order,
                filter_cfg.cutoff_hz, fps)
    logger.info("  filtered %d of %d reliable samples",
                filter_stats["n_filtered_samples"],
                filter_stats["n_reliable_samples"])
    return meta


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_common_qc(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--outdir", default=None,
                     help="override output dir (default: the clip's "
                          "stage2/ folder)")
    sub.add_argument("--qc-landmarks", nargs="*", default=None,
                     help="landmark names to plot (default: serving-arm "
                          "elbows and wrists)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Stage 2: gating (2a) and filtering (2b).")
    sub = parser.add_subparsers(dest="step", required=True)

    p2a = sub.add_parser("2a", help="visibility gating and gap handling")
    p2a.add_argument("landmarks_csv",
                     help="path to a Stage 1 stage1/landmarks.csv")
    p2a.add_argument("--meta", default=None,
                     help="Stage 1 meta.json (for fps); auto-detected "
                          "next to the CSV if omitted")
    p2a.add_argument("--visibility-threshold", type=float,
                     default=DEFAULT_VISIBILITY_THRESHOLD)
    _add_common_qc(p2a)

    p2b = sub.add_parser("2b", help="interpolation and low-pass filtering")
    p2b.add_argument("gated_csv", help="path to a Stage 2a stage2/gated.csv")
    p2b.add_argument("--meta", default=None,
                     help="Stage 2a gating_meta.json (for fps); auto-detected "
                          "next to the CSV if omitted")
    p2b.add_argument("--max-gap-frames", type=int,
                     default=DEFAULT_MAX_GAP_FRAMES)
    p2b.add_argument("--filter", dest="filter_kind",
                     choices=[KIND_BUTTERWORTH, KIND_SAVGOL],
                     default=KIND_BUTTERWORTH)
    p2b.add_argument("--order", type=int, default=DEFAULT_FILTER_ORDER,
                     help="Butterworth order or Savgol polyorder")
    p2b.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ,
                     help="Butterworth cut-off frequency")
    p2b.add_argument("--window-length", type=int, default=None,
                     help="Savgol window length (odd)")
    p2b.add_argument("--qc-coord", default=DEFAULT_QC_COORD,
                     help="coordinate channel to plot (default: y)")
    _add_common_qc(p2b)

    args = parser.parse_args()
    if args.step == "2a":
        run_stage2a(
            csv_path=args.landmarks_csv,
            outdir=args.outdir,
            meta_path=args.meta,
            visibility_threshold=args.visibility_threshold,
            qc_landmarks=args.qc_landmarks,
        )
    else:
        cfg = FilterConfig(
            kind=args.filter_kind, order=args.order,
            cutoff_hz=args.cutoff_hz, window_length=args.window_length)
        run_stage2b(
            gated_csv_path=args.gated_csv,
            outdir=args.outdir,
            meta_path=args.meta,
            max_gap_frames=args.max_gap_frames,
            filter_cfg=cfg,
            qc_landmarks=args.qc_landmarks,
            qc_coord=args.qc_coord,
        )


if __name__ == "__main__":
    main()
