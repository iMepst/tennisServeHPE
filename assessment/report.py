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
