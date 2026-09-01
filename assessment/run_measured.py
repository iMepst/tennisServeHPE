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

