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
