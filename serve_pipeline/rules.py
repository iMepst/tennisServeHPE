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
