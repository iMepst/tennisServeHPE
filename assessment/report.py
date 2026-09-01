"""Assemble the feasibility assessment into machine-readable artifacts.

Step 8, the reporter: it only *runs* the existing assessment modules over the
configured theta sweep (0deg-45deg) and sigma band (2-6 px) and serialises what
they return. No analysis lives here -- the projection (E2), the Monte-Carlo
landmark-noise spread (E1), the decidability criterion (3b/3c) and the event
error (E3) are all computed in their own modules; this file just orchestrates
the calls and writes the tables and metadata the Results and Discussion
chapters read.

Written to results/assessment/:

- projection_curves.csv   E2:    per criterion, theta -> projected angle
- noise_propagation.csv   E1+E2: per criterion, (theta, sigma) -> induced SD
- decidability.csv        3b/3c: per criterion, (theta, sigma) -> SD vs band
- event_error.json        E3:    frame-move rate + robust offset distribution
- run_meta.json                  every parameter, so a run reproduces exactly

    python -m assessment.report [--annotations DIR] [--results-root DIR]
                                [--out DIR]
"""

import argparse
import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from assessment.annotation import EventError, EventTypeError
from assessment.projection import ProjectionCurve, projection_curves
from assessment.propagation import REP_STATURE_PX
from assessment.run_measured import SigmaPoint, measured_assessment
from serve_pipeline.config import PipelineConfig
from serve_pipeline.persistence import write_metadata

# Everything the reporter produces lands under results/<this>/.
DEFAULT_SUBDIR = "assessment"


# --------------------------------------------------------------------------
# CSV writers -- one per artifact, each a flat table of the module's output.
# --------------------------------------------------------------------------

def _write_csv(path: str, header: List[str],
               rows: List[Dict[str, Any]]) -> str:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path


_PROJECTION_HEADER = ["criterion", "kind", "a_true", "theta", "projected_angle"]


def projection_rows(curves: List[ProjectionCurve]) -> List[Dict[str, Any]]:
    """One row per (criterion, theta): the projected angle E2 predicts."""
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
    """One row per (criterion, sigma, theta): the Monte-Carlo induced SD.

    The whole sigma band is unrolled, so the induced spread is available as a
    function of both viewpoint and noise level.
    """
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
    breakdown viewpoint. This pair is the Q3 reading -- the operating point at
    which the criterion stops separating sound from faulty. Both None for a
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
    """One row per (criterion, sigma, theta): induced SD held against the band.

    Each row carries the induced SD, the rule's band half-width, their ratio
    and the per-theta decidable flag; the onset columns repeat the criterion's
    first-unreliable (sigma, theta) so the Q3 reading is on every row.
    """
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


# --------------------------------------------------------------------------
# JSON writers -- the empirical event error and the run parameters.
# --------------------------------------------------------------------------

def _event_type_dict(e: EventTypeError) -> Dict[str, Any]:
    """Serialise one event type, robust statistics first (task 1b ordering)."""
    return {
        "n_clips": e.n_clips,
        "n_locatable": e.n_locatable,
        "n_not_locatable": e.n_not_locatable,
        "tolerances": list(e.tolerances),
        "n_moved_by_tolerance": {str(t): e.n_moved_by_tolerance[t]
                                 for t in e.tolerances},
        "move_rate_by_tolerance": {str(t): e.move_rate_by_tolerance[t]
                                   for t in e.tolerances},
        "median_offset": e.median_offset,
        "iqr_offset": e.iqr_offset,
        "max_abs_offset": e.max_abs_offset,
        "large_offset_frames": e.large_offset_frames,
        "n_large_failures": e.n_large_failures,
        "mean_offset": e.mean_offset,
    }


def event_error_dict(event_error: Optional[EventError],
                     annotations_path: str) -> Dict[str, Any]:
    """The E3 record, or a clearly-marked placeholder when no CSV was found.

    ``available`` is the flag the Results chapter keys on: False means the
    event error was not measured (no events.csv), never that it was zero.
    """
    if event_error is None:
        # No events.csv: mark the record as a placeholder and leave every rate
        # unset. E3 is the one input that can be absent (sigma is always the
        # swept band), and an absent rate is never fabricated as zero.
        return {"available": False, "placeholder": True,
                "note": f"no event annotation at {annotations_path}; "
                        "E3 not measured"}
    return {"available": True, "n_clips": event_error.n_clips,
            "trophy": _event_type_dict(event_error.trophy),
            "impact": _event_type_dict(event_error.impact)}


# E4 (definitional mismatch: surface landmarks vs the joint centres behind the
# reference values) is out of scope by construction -- quantifying it needs
# joint-centre ground truth the study does not have. It is recorded here as a
# documented, unquantified offset so the artifacts show it was set aside on
# purpose, never simply overlooked; it is never assigned a number.
_E4_NOTE = (
    "E4 definitional mismatch is not quantified by design: the gap between "
    "surface landmarks and the joint centres behind the reference values "
    "needs joint-centre ground truth this study does not have. It is left as "
    "a documented, unquantified offset (worst on trunk inclination), never "
    "simulated and never assigned a number.")


def run_meta(config: PipelineConfig, outputs: Dict[str, str],
             out_dir: str) -> Dict[str, Any]:
    """Every parameter the run used, so a later run reproduces it exactly.

    Output paths are logged relative to out_dir (figures sit in a subdir), and
    the E4 note records the one error source deliberately left unquantified.
    """
    from assessment.projection import theta_values
    return {
        "theta_range": list(config.theta_range),
        "theta_step": config.theta_step,
        "thetas": theta_values(config),
        "sigma": config.sigma,
        "sigma_sweep": list(config.sigma_sweep),
        "mc_samples": config.mc_samples,
        "seed": config.seed,
        "reference_stature_px": REP_STATURE_PX,
        "event_tolerances_frames": list(config.event_tolerances_frames),
        "event_large_offset_frames": config.event_large_offset_frames,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "outputs": {name: os.path.relpath(path, out_dir)
                    for name, path in outputs.items()},
        "notes": {"e4_definitional_mismatch": _E4_NOTE},
    }


# --------------------------------------------------------------------------
# Figures -- reproducible views of the same numbers the CSVs carry. Matplotlib
# is imported lazily (Agg, headless) so a numbers-only run needs no display.
# --------------------------------------------------------------------------

FIGURE_SUBDIR = "figures"

_CRITERION_LABEL = {
    "trunk_inclination": "Trunk inclination",
    "front_knee_flexion": "Front knee flexion",
    "elbow_flexion": "Elbow flexion",
    "shoulder_elevation": "Shoulder elevation",
}


def _dec_by_criterion(point: SigmaPoint) -> Dict[str, Any]:
    """Index one sweep point's decidability records by criterion id."""
    return {d.criterion: d for d in point.decidability}


def _plot_projection_curves(curves: List[ProjectionCurve], path: str) -> str:
    """E2: projected angle of every criterion over the theta sweep (one axes).

    Each curve starts at its true angle (theta = 0) and foreshortens as the
    viewpoint tilts; the trunk closed form and the numeric joints sit together.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in curves:
        ax.plot(c.thetas, c.projected, marker="o", ms=3,
                label=f"{_CRITERION_LABEL.get(c.criterion, c.criterion)} "
                      f"({c.kind.replace('_', ' ')})")
    ax.set_xlabel("viewpoint angle theta (deg)")
    ax.set_ylabel("projected angle (deg)")
    ax.set_title("Projection error over viewpoint")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_spread_vs_theta(sweep: List[SigmaPoint], path: str) -> str:
    """E1+E2: induced angular spread over theta, one panel per criterion.

    A line per swept sigma, with the rule's band half-width drawn as the
    dashed threshold: where a line crosses it the criterion turns unreliable,
    which is exactly the decidability reading rendered as a curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    criteria = [d.criterion for d in sweep[0].decidability]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, criterion in zip(axes.flat, criteria):
        for point in sweep:
            prop = {p.criterion: p for p in point.propagation}[criterion]
            ax.plot(prop.thetas, prop.sd_deg, marker="o", ms=3,
                    label=f"sigma = {point.sigma:g} px")
        half = _dec_by_criterion(sweep[0])[criterion].half_width
        ax.axhline(half, ls="--", color="k", lw=1.0,
                   label="band half-width")
        ax.set_title(_CRITERION_LABEL.get(criterion, criterion))
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel("induced SD (deg)")
        ax.legend(fontsize=7)
    fig.suptitle("Induced angular spread over viewpoint and noise level")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_decidability_map(sweep: List[SigmaPoint], path: str) -> str:
    """The summary figure: per-criterion decidability over the (theta, sigma)
    grid, so the headline (sigma, theta) onset reads at a glance.

    Each panel colours the ratio induced_SD / band half-width (green below 1,
    red above), draws the reliability boundary at ratio = 1, and marks the
    onset -- the first (sigma, theta) at which the criterion turns unreliable,
    the Q3 reading. Panels with no marker stay decidable across the whole grid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    onset = _unreliable_onset(sweep)
    sigmas = [p.sigma for p in sweep]
    thetas = sweep[0].decidability[0].thetas
    criteria = [d.criterion for d in sweep[0].decidability]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, criterion in zip(axes.flat, criteria):
        # Ratio grid: rows are sigma (ascending), columns theta.
        grid = np.array([
            _dec_by_criterion(point)[criterion].ratio for point in sweep])
        mesh = ax.pcolormesh(thetas, sigmas, grid, shading="nearest",
                             cmap="RdYlGn_r", vmin=0.0, vmax=2.0)
        # The reliability boundary: induced spread equal to the half-width.
        if len(thetas) > 1 and len(sigmas) > 1:
            ax.contour(thetas, sigmas, grid, levels=[1.0], colors="k",
                       linewidths=1.2)
        crit_onset = onset.get(criterion)
        title = _CRITERION_LABEL.get(criterion, criterion)
        if crit_onset and crit_onset["theta"] is not None:
            ax.plot(crit_onset["theta"], crit_onset["sigma"], marker="*",
                    ms=16, color="black")
            title += (f"  (unreliable from sigma = {crit_onset['sigma']:g} px, "
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
    """Write the three assessment figures into fig_dir, returning their paths."""
    os.makedirs(fig_dir, exist_ok=True)
    return {
        "projection_figure": _plot_projection_curves(
            curves, os.path.join(fig_dir, "projection_curves.png")),
        "spread_figure": _plot_spread_vs_theta(
            sweep, os.path.join(fig_dir, "spread_vs_theta.png")),
        "decidability_figure": _plot_decidability_map(
            sweep, os.path.join(fig_dir, "decidability_map.png")),
    }

