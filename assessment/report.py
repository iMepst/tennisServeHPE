"""Assemble the feasibility assessment into machine-readable artifacts.

Written to results/assessment/:

- projection_curves.csv   E2:    per criterion, theta -> projected angle
- noise_propagation.csv   E1+E2: per criterion, (theta, sigma) -> induced SD
- decidability.csv        3a:    per criterion, (theta, sigma) -> SD vs band
- event_error.json        E3:    tolerance shares, large errors, offsets
- run_meta.json                  every parameter, so a run reproduces exactly

    python -m assessment.report [--annotations DIR] [--results-root DIR]
                                [--out DIR] [--no-figures]
"""

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from assessment.annotation import (EventError, EventStats,
                                   estimate_event_error,
                                   read_event_annotations)
from assessment.decidability import Decidability, decidability
from assessment.projection import (ProjectionCurve, projection_curves,
                                   theta_values)
from assessment.propagation import (REP_STATURE_PX, NoisePropagation,
                                    noise_propagation)
from serve_pipeline.config import PipelineConfig
from serve_pipeline.persistence import write_metadata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_SUBDIR = "assessment"
FIGURE_SUBDIR = "figures"

_DEFAULT_ANNOTATIONS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "annotations")


@dataclass
class SigmaPoint:
    sigma: float
    propagation: List[NoisePropagation]
    decidability: List[Decidability]


@dataclass
class MeasuredAssessment:
    event_error: Optional[EventError]
    sweep: List[SigmaPoint]


def measured_assessment(config: PipelineConfig, annotations_dir: str,
                        results_root: Optional[str] = None
                        ) -> MeasuredAssessment:
    """E3 from the manual frame check, then E1/E2 over the sigma sweep."""
    if results_root is None:
        results_root = config.results_root

    events_path = os.path.join(annotations_dir, "events.csv")
    event_error = (estimate_event_error(
        read_event_annotations(events_path), results_root)
        if os.path.isfile(events_path) else None)

    sweep = [
        SigmaPoint(sigma=s,
                   propagation=noise_propagation(config, s),
                   decidability=decidability(config, s))
        for s in config.sigma_sweep]

    return MeasuredAssessment(event_error=event_error, sweep=sweep)


def _write_csv(path: str, header: List[str],
               rows: List[Dict[str, Any]]) -> str:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path


_PROJECTION_HEADER = ["criterion", "kind", "a_true", "theta",
                      "projected_angle"]


def projection_rows(curves: List[ProjectionCurve]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for c in curves:
        for theta, projected in zip(c.thetas, c.projected):
            rows.append({
                "criterion": c.criterion, "kind": c.kind, "a_true": c.a_true,
                "theta": theta, "projected_angle": projected})
    return rows


_NOISE_HEADER = ["criterion", "a_true", "sigma", "mc_samples", "seed",
                 "theta", "sd_deg"]


def noise_rows(sweep: List[SigmaPoint]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for point in sweep:
        for prop in point.propagation:
            for theta, sd in zip(prop.thetas, prop.sd_deg):
                rows.append({
                    "criterion": prop.criterion, "a_true": prop.a_true,
                    "sigma": prop.sigma, "mc_samples": prop.mc_samples,
                    "seed": prop.seed, "theta": theta, "sd_deg": sd})
    return rows


_DECIDABILITY_HEADER = ["criterion", "sigma", "mc_samples", "seed", "theta",
                        "induced_sd", "half_width", "ratio", "decidable",
                        "verdict", "onset_sigma", "onset_theta"]


def _unreliable_onset(sweep: List[SigmaPoint]
                      ) -> Dict[str, Dict[str, Optional[float]]]:
    """First (sigma, theta) at which each criterion turns unreliable.

    Walks the sigma band in ascending order (the sweep order) and takes the
    first sigma whose verdict is "unreliable"; the theta is that verdict's
    breakdown viewpoint. This pair is the Q3 reading. Both None for a
    criterion that stays decidable across the whole grid.
    """
    onset: Dict[str, Dict[str, Optional[float]]] = {}
    for point in sweep:
        for d in point.decidability:
            if d.criterion in onset:
                continue
            if d.verdict == "unreliable":
                onset[d.criterion] = {"sigma": point.sigma,
                                      "theta": d.breakdown_theta}
    return onset


def decidability_rows(sweep: List[SigmaPoint]) -> List[Dict[str, Any]]:
    onset = _unreliable_onset(sweep)
    rows: List[Dict[str, Any]] = []
    for point in sweep:
        for d in point.decidability:
            crit_onset = onset.get(d.criterion, {})
            for theta, sd, ratio, ok in zip(
                    d.thetas, d.induced_sd, d.ratio, d.decidable):
                rows.append({
                    "criterion": d.criterion, "sigma": d.sigma,
                    "mc_samples": d.mc_samples, "seed": d.seed, "theta": theta,
                    "induced_sd": sd, "half_width": d.half_width,
                    "ratio": ratio, "decidable": ok, "verdict": d.verdict,
                    "onset_sigma": crit_onset.get("sigma"),
                    "onset_theta": crit_onset.get("theta")})
    return rows


def _event_stats_dict(e: EventStats) -> Dict[str, Any]:
    return {
        "n_clips": e.n_clips,
        "n_locatable": e.n_locatable,
        "n_not_locatable": e.n_not_locatable,
        "tolerances": list(e.tolerances),
        "n_moved_by_tolerance": {str(t): e.n_moved_by_tolerance[t]
                                 for t in e.tolerances},
        "share_by_tolerance": {str(t): e.share_by_tolerance[t]
                               for t in e.tolerances},
        "median_offset": e.median_offset,
        "iqr_offset": e.iqr_offset,
        "max_abs_offset": e.max_abs_offset,
        "large_offset_frames": e.large_offset_frames,
        "n_large_failures": e.n_large_failures,
        "share_large_failures": e.share_large_failures,
        "mean_offset": e.mean_offset,
    }


def event_error_dict(event_error: Optional[EventError]) -> Dict[str, Any]:
    if event_error is None:
        return {"available": False}
    return {"available": True, "n_clips": event_error.n_clips,
            "trophy": _event_stats_dict(event_error.trophy),
            "impact": _event_stats_dict(event_error.impact)}


def run_meta(config: PipelineConfig, outputs: Dict[str, str],
             out_dir: str) -> Dict[str, Any]:
    return {
        "theta_range": list(config.theta_range),
        "theta_step": config.theta_step,
        "thetas": theta_values(config),
        "sigma_sweep": list(config.sigma_sweep),
        "mc_samples": config.mc_samples,
        "seed": config.seed,
        "reference_stature_px": REP_STATURE_PX,
        "event_tolerances_frames": list(config.event_tolerances_frames),
        "event_large_offset_frames": config.event_large_offset_frames,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {name: os.path.relpath(path, out_dir)
                    for name, path in outputs.items()},
    }


_CRITERION_LABEL = {
    "trunk_inclination": "Trunk inclination",
    "front_knee_flexion": "Front knee flexion",
    "elbow_flexion": "Elbow flexion",
    "shoulder_elevation": "Shoulder elevation",
}

# Colour-scale bounds for the decidability heatmaps, shared across panels so
# they stay comparable. The upper limit sits just above the largest ratio the
# grid reaches, so the full colour range spans the values that occur
# and the contrast around the ratio = 1 boundary is visible.
_DECIDABILITY_VMIN = 0.0
_DECIDABILITY_VMAX = 1.1


def _plot_projection_curves(curves: List[ProjectionCurve], path: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in curves:
        ax.plot(c.thetas, c.projected, marker="o", ms=3,
                label=f"{_CRITERION_LABEL[c.criterion]} "
                      f"({c.kind.replace('_', ' ')})")
    ax.set_xlabel("viewpoint angle theta (deg)")
    ax.set_ylabel("projected angle (deg)")
    ax.set_title("Projected angle over viewpoint")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_spread_vs_theta(sweep: List[SigmaPoint], path: str) -> str:
    criteria = [d.criterion for d in sweep[0].decidability]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, criterion in zip(axes.flat, criteria):
        for point in sweep:
            prop = {p.criterion: p for p in point.propagation}[criterion]
            ax.plot(prop.thetas, prop.sd_deg, marker="o", ms=3,
                    label=f"sigma = {point.sigma:g} px")
        ax.set_title(_CRITERION_LABEL[criterion])
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel("induced SD (deg)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(sweep),
               fontsize=8)
    fig.suptitle("Induced angular spread over viewpoint and noise level")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _cell_edges(centers: List[float]) -> np.ndarray:
    """Cell-boundary coordinates for centers, matching pcolormesh 'nearest':
    midpoints between centers, half a step beyond at each end."""
    c = np.asarray(centers, dtype=float)
    mid = (c[:-1] + c[1:]) / 2.0
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid,
                           [c[-1] + (c[-1] - mid[-1])]])


def _draw_threshold_boundary(ax: Any, thetas: List[float],
                             sigmas: List[float], grid: np.ndarray,
                             level: float) -> None:
    """Outline where grid crosses level, along the pcolormesh cell edges.

    Draws only the interior edges separating a below-level cell from an
    at/above-level one, giving a crisp stair-step boundary that follows the
    grid instead of an interpolated diagonal.
    """
    g = np.asarray(grid, dtype=float)
    over = g >= level
    xe = _cell_edges(thetas)
    ye = _cell_edges(sigmas)
    kw = dict(color="k", lw=1.1, zorder=4)
    rows, cols = g.shape
    for j in range(rows):
        for i in range(cols - 1):
            if over[j, i] != over[j, i + 1]:
                ax.plot([xe[i + 1], xe[i + 1]], [ye[j], ye[j + 1]], **kw)
    for j in range(rows - 1):
        for i in range(cols):
            if over[j, i] != over[j + 1, i]:
                ax.plot([xe[i], xe[i + 1]], [ye[j + 1], ye[j + 1]], **kw)


def _plot_decidability_map(sweep: List[SigmaPoint], path: str) -> str:
    onset = _unreliable_onset(sweep)
    sigmas = [p.sigma for p in sweep]
    thetas = sweep[0].decidability[0].thetas
    criteria = [d.criterion for d in sweep[0].decidability]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, criterion in zip(axes.flat, criteria):
        # Ratio grid: rows are sigma (ascending), columns theta.
        grid = np.array([
            {d.criterion: d for d in point.decidability}[criterion].ratio
            for point in sweep])
        mesh = ax.pcolormesh(thetas, sigmas, grid, shading="nearest",
                             cmap="cividis", vmin=_DECIDABILITY_VMIN,
                             vmax=_DECIDABILITY_VMAX)
        _draw_threshold_boundary(ax, thetas, sigmas, grid, 1.0)
        crit_onset = onset.get(criterion)
        title = _CRITERION_LABEL[criterion]
        if crit_onset:
            ax.plot(crit_onset["theta"], crit_onset["sigma"], marker="o",
                    ms=9, markerfacecolor="white", markeredgecolor="black",
                    markeredgewidth=1.4, clip_on=True, zorder=5)
            title += (f"  (unreliable from sigma = "
                      f"{crit_onset['sigma']:g} px, "
                      f"theta = {crit_onset['theta']:g} deg)")
        else:
            title += "  (decidable across grid)"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel("sigma (px)")
        fig.colorbar(mesh, ax=ax, label="induced SD / half-width")
    fig.suptitle("Decidability over viewpoint and noise level")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_figures(curves: List[ProjectionCurve], sweep: List[SigmaPoint],
                  fig_dir: str) -> Dict[str, str]:
    os.makedirs(fig_dir, exist_ok=True)
    return {
        "projection_figure": _plot_projection_curves(
            curves, os.path.join(fig_dir, "projection_curves.png")),
        "spread_figure": _plot_spread_vs_theta(
            sweep, os.path.join(fig_dir, "spread_vs_theta.png")),
        "decidability_figure": _plot_decidability_map(
            sweep, os.path.join(fig_dir, "decidability_map.png")),
    }


def build_assessment_report(config: PipelineConfig, annotations_dir: str,
                            results_root: Optional[str] = None,
                            out_dir: Optional[str] = None,
                            make_figures: bool = True) -> Dict[str, Any]:
    if results_root is None:
        results_root = config.results_root
    if out_dir is None:
        out_dir = os.path.join(results_root, DEFAULT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    curves = projection_curves(config)
    measured = measured_assessment(config, annotations_dir, results_root)

    outputs: Dict[str, str] = {
        "projection_curves": _write_csv(
            os.path.join(out_dir, "projection_curves.csv"),
            _PROJECTION_HEADER, projection_rows(curves)),
        "noise_propagation": _write_csv(
            os.path.join(out_dir, "noise_propagation.csv"),
            _NOISE_HEADER, noise_rows(measured.sweep)),
        "decidability": _write_csv(
            os.path.join(out_dir, "decidability.csv"),
            _DECIDABILITY_HEADER, decidability_rows(measured.sweep)),
    }

    event_path = os.path.join(out_dir, "event_error.json")
    write_metadata(event_path, event_error_dict(measured.event_error))
    outputs["event_error"] = event_path

    if make_figures:
        outputs.update(write_figures(
            curves, measured.sweep, os.path.join(out_dir, FIGURE_SUBDIR)))

    # run_meta last, so its output manifest lists everything already written.
    meta_path = os.path.join(out_dir, "run_meta.json")
    write_metadata(meta_path, run_meta(config, outputs, out_dir))
    outputs["run_meta"] = meta_path

    return {"out_dir": out_dir, "outputs": outputs}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble the feasibility assessment artifacts: run the "
                    "projection, noise-propagation, decidability and event-"
                    "error modules and write their tables into "
                    "results/assessment/.")
    parser.add_argument("--annotations", default=_DEFAULT_ANNOTATIONS,
                        help="annotation directory holding events.csv "
                             "(default: data/annotations)")
    parser.add_argument("--results-root", default=None,
                        help="pipeline results root (default: config)")
    parser.add_argument("--out", default=None,
                        help="output dir (default: <results>/assessment)")
    parser.add_argument("--no-figures", dest="make_figures",
                        action="store_false",
                        help="write the CSV/JSON tables only, "
                             "skip the figures")
    args = parser.parse_args()

    config = PipelineConfig()
    report = build_assessment_report(
        config, args.annotations, args.results_root, args.out,
        make_figures=args.make_figures)

    print(f"assessment written to {report['out_dir']}")
    for name, path in report["outputs"].items():
        print(f"  {name:<20} {os.path.basename(path)}")


if __name__ == "__main__":
    main()
