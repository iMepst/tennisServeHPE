"""Stage 5: rule evaluation and deviation indicators.

Compares each computed angle against its reference band and returns a
binary deviation indicator (rule_base_spec.md, Sections 1-2;
pipeline_spec.md, Stage 5). Runs in memory; nothing is persisted here.

Reference values are from Jacquier-Bret et al. (2024).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Rule:
    """One deviation rule: its key frame, plane, and reference statistics.

    The band is not stored as literal bounds but derived from mean and sd
    (see below), so the flag thresholds are never hard-coded magic
    numbers. band_kind selects how the band is applied:
    "two_sided" flags outside [mean-sd, mean+sd]; "lower_bound" flags
    only below mean-sd (insufficient value), leaving the upper side
    unpenalised.
    """

    id: str
    key_frame: str    # "trophy" | "impact"
    plane: Optional[str]  # "frontal" | "sagittal" | None (plane-independent)
    mean: float
    sd: float
    band_kind: str    # "two_sided" | "lower_bound"

    # The band is mean +/- 1*sd: the factor is exactly 1, the minimal
    # non-arbitrary choice. It operationalises the reference spread of the
    # Jacquier-Bret sample; it is NOT a claim that correct technique ends
    # at its edge. An "outside" reading is an attention flag for a coach,
    # not a verdict on the serve.
    @property
    def lo(self) -> float:
        """Lower band bound (mean - sd); the flag threshold both kinds use."""
        return self.mean - self.sd

    @property
    def hi(self) -> float:
        """Upper band bound (mean + sd); only the two-sided kind uses it."""
        return self.mean + self.sd


# The four rules in one readable table. Reference mean/sd from
# Jacquier-Bret et al. (2024); bands are derived (Rule.lo/Rule.hi), never
# written out.
RULES = [
    # Trunk inclination: mid-hip -> mid-shoulder axis vs vertical, read at
    # the trophy frame. Frontal plane (front OR back view). Band [17.9, 32.1].
    Rule(id="trunk_inclination", key_frame="trophy", plane="frontal",
         mean=25.0, sd=7.1, band_kind="two_sided"),
]
