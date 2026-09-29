"""Decidability criterion 3c.

The question Q3: under which conditions does a criterion stay reliable
enough? Answered by holding the induced angular spread (the SD that
projection and landmark noise put into a rule's input) against the rule's
own band half-width.

Band half-width is one reference SD, so the comparison needs no external
scale: it asks whether the noise-driven scatter is smaller than the
spread the band is drawn from. Decidable where the induced spread stays
below the half-width across the expected viewpoint and noise range;
unreliable where it reaches it.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from assessment.propagation import noise_propagation
from serve_pipeline.config import PipelineConfig
from serve_pipeline.rules import RULES, Rule

# The band half-width is factor * SD with factor exactly 1, the same
# minimal non-arbitrary choice the rule bands themselves use.
THRESHOLD_FACTOR = 1.0


def band_half_width(rule: Rule) -> float:
    return THRESHOLD_FACTOR * rule.sd


def assess_series(induced_sd: List[float], thetas: List[float],
                  half_width: float) -> Tuple[List[float], List[bool],
                                              Optional[float], str]:
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


def decidability(config: PipelineConfig,
                 sigma: Optional[float] = None) -> List[Decidability]:
    if sigma is None:
        sigma = config.sigma
    props = {p.criterion: p for p in noise_propagation(config, sigma)}
    results: List[Decidability] = []
    for rule in RULES:
        prop = props[rule.id]
        half = band_half_width(rule)
        ratio, decidable, breakdown, verdict = assess_series(
            prop.sd_deg, prop.thetas, half)
        results.append(Decidability(
            criterion=rule.id, sigma=sigma, mc_samples=config.mc_samples,
            seed=config.seed, half_width=half,
            thetas=prop.thetas, induced_sd=prop.sd_deg, ratio=ratio,
            decidable=decidable, verdict=verdict, breakdown_theta=breakdown))
    return results
