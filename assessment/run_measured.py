"""Produce the feasibility result.

Ties the empirical annotation (Step 7b) to the synthetic core (Step 7a):
measure the event-error rate (E3) from the manual frame-check CSV, then run
the per-criterion spread and decidability table over the sigma sweep (E1).

Sigma is not measured on the clips. It is taken from the estimator's reported
accuracy and swept over config.sigma_sweep, so the induced spread and the
decidability verdict are reported as a function of how noisy the pose estimate
is, rather than at one hand-labelled operating point.

Still numeric only -- the report artifacts and figures are Step 8.

    python -m assessment.run_measured [--annotations DIR] [--results-root DIR]
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

from assessment.annotation import (
    EventError, estimate_event_error, read_event_annotations)
from assessment.decidability import Decidability, decidability
from assessment.propagation import NoisePropagation, noise_propagation
from serve_pipeline.config import PipelineConfig

_DEFAULT_ANNOTATIONS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "annotations")


@dataclass
class SigmaPoint:
    """The synthetic core at one swept sigma (px): the per-criterion induced
    spread and its decidability verdict."""

    sigma: float
    propagation: List[NoisePropagation]
    decidability: List[Decidability]


@dataclass
class MeasuredAssessment:
    """The feasibility result: the sigma sweep the synthetic core is reported
    over, the per-sigma spread/decidability, and the event error (None when no
    event annotation was found)."""

    sigma_sweep: List[float]
    event_error: Optional[EventError]
    sweep: List[SigmaPoint]


def measured_assessment(config: PipelineConfig, annotations_dir: str,
                        results_root: Optional[str] = None
                        ) -> MeasuredAssessment:
    """Gather the assessment from the annotation directory.

    Reads ``events.csv`` for the event rate (E3), then runs the synthetic core
    over config.sigma_sweep. The annotation directory now supplies only the
    event annotation; sigma comes from the config sweep, not from the clips.
    """
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

    return MeasuredAssessment(
        sigma_sweep=list(config.sigma_sweep),
        event_error=event_error, sweep=sweep)


def _print_sigma_sweep(m: MeasuredAssessment) -> None:
    band = ", ".join(f"{s:g}" for s in m.sigma_sweep)
    print(f"sigma (E1): swept over [{band}] px  "
          f"[from the estimator's reported accuracy, not measured on these "
          f"clips; reported as a sensitivity range]")


def _print_event_error(m: MeasuredAssessment) -> None:
    if m.event_error is None:
        print("event error (E3): no event annotation found")
        return
    print(f"event error (E3): {m.event_error.n_clips} clips")
    for e in (m.event_error.trophy, m.event_error.impact):
        rates = "  ".join(
            f">{t}f {e.move_rate_by_tolerance[t]:.0%}" for t in e.tolerances)
        print(f"    {e.event:<8} move rate  {rates}  "
              f"({e.n_not_locatable} not locatable)")
        if e.n_locatable:
            print(f"             offset median {e.median_offset:+.2f}, "
                  f"IQR {e.iqr_offset:.1f}, max |.| {e.max_abs_offset:.0f}, "
                  f"{e.n_large_failures} large (>= {e.large_offset_frames}f), "
                  f"mean {e.mean_offset:+.2f} frames")


def _print_core(m: MeasuredAssessment) -> None:
    """Per-criterion induced spread and decidability verdict at each swept
    sigma."""
    thetas = m.sweep[0].decidability[0].thetas
    for point in m.sweep:
        print(f"\nper-criterion induced SD (deg) over theta, sigma = "
              f"{point.sigma:g} px")
        header = "criterion".ljust(20) \
            + "".join(f"{th:7.0f}" for th in thetas) + "   verdict"
        print(header)
        for d in point.decidability:
            row = d.criterion.ljust(20) + "".join(
                f"{sd:7.2f}" for sd in d.induced_sd)
            print(f"{row}   {d.verdict}")
    print("\n(decidable while induced SD < band half-width; the sigma at "
          "which each criterion turns unreliable is the Q3 reading)")


def print_report(m: MeasuredAssessment) -> None:
    _print_sigma_sweep(m)
    print()
    _print_event_error(m)
    _print_core(m)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce the feasibility result: event rate (E3) from the "
                    "manual frame check, then the synthetic spread / "
                    "decidability table over the sigma sweep (E1).")
    parser.add_argument("--annotations", default=_DEFAULT_ANNOTATIONS,
                        help="annotation directory holding events.csv "
                             "(default: data/annotations)")
    parser.add_argument("--results-root", default=None,
                        help="pipeline results root (default: config)")
    args = parser.parse_args()

    config = PipelineConfig()
    m = measured_assessment(config, args.annotations, args.results_root)
    print_report(m)


if __name__ == "__main__":
    main()
