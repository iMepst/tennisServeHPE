"""Decidability criterion 3c.

The question Q3: under which conditions does a criterion stay reliable
enough? Answered by holding the induced angular spread (the SD that
projection and landmark noise put into a rule's input) against the rule's
own band half-width.

Band half-width is one reference SD, so the comparison needs no external
scale: it asks whether the noise-driven scatter is smaller than the very
spread the band is drawn from. Decidable where the induced spread stays
below the half-width across the expected viewpoint and noise range;
unreliable where it reaches it -- an input scattering as far as
centre-to-edge can no longer separate sound from faulty.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from assessment.propagation import noise_propagation
from serve_pipeline.config import PipelineConfig
from serve_pipeline.rules import RULES, Rule

# The band half-width is factor * SD with factor exactly 1, the same
# minimal non-arbitrary choice the rule bands themselves use. It is not a
# claim that reliability ends precisely at one SD.
THRESHOLD_FACTOR = 1.0


def band_half_width(rule: Rule) -> float:
    """The rule's band half-width in degrees: one reference SD (factor 1).

    Both two-sided and one-sided bands are set one SD from the mean, so
    this half-width is the natural, scale-free yardstick for the induced
    spread.
    """
    return THRESHOLD_FACTOR * rule.sd


def assess_series(induced_sd: List[float], thetas: List[float],
                  half_width: float) -> Tuple[List[float], List[bool],
                                              Optional[float], str]:
    """Turn an induced-SD series into a decidability verdict.

    Returns the per-theta ratio (induced SD / half-width), the per-theta
    decidable flags (True while the spread stays below the half-width), the
    breakdown theta (first viewpoint where the spread reaches the
    half-width, or None if it never does), and the overall verdict --
    "decidable" only if it holds across the whole range.
    """
    ratio = [sd / half_width for sd in induced_sd]
    decidable = [sd < half_width for sd in induced_sd]
    breakdown = next((th for th, ok in zip(thetas, decidable) if not ok), None)
    verdict = "decidable" if all(decidable) else "unreliable"
    return ratio, decidable, breakdown, verdict


@dataclass
class Decidability:
    """Per-criterion decidability across the theta sweep at a fixed sigma.

    ratio[i] = induced_sd[i] / half_width and decidable[i] are the values at
    thetas[i]; verdict holds across the whole range and breakdown_theta is
    the first viewpoint that reaches the half-width (None if none does).
    """

    criterion: str
    sigma: float
    mc_samples: int
    seed: int
    half_width: float
    thetas: List[float]
    induced_sd: List[float]
    ratio: List[float]
    decidable: List[bool]
    verdict: str
    breakdown_theta: Optional[float]

