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

