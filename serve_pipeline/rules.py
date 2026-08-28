"""Rule evaluation: compares each computed angle against its reference band
and returns a binary deviation indicator. Runs in memory; persists nothing.

Reference values are from Jacquier-Bret et al. (2024).
"""

from dataclasses import dataclass
from typing import List, Optional

from .angles import AngleReadings
from .config import ClipParams


@dataclass(frozen=True)
class Rule:
    """One deviation rule: its key frame, plane, and reference statistics.

    The band derives from mean and sd, not literal bounds, so thresholds are
    never hard-coded. band_kind selects how it applies: "two_sided" flags
    outside [mean-sd, mean+sd]; "lower_bound" flags only below mean-sd,
    leaving the upper side unpenalised.
    """

    id: str
    key_frame: str    # "trophy" | "impact"
    plane: Optional[str]  # "frontal" | "sagittal" | None (plane-independent)
    mean: float
    sd: float
    band_kind: str    # "two_sided" | "lower_bound"

    # Band is mean +/- 1*sd: factor exactly 1, the minimal non-arbitrary
    # choice. It reflects the reference spread, not a claim that correct
    # technique ends at its edge; "outside" is an attention flag, not a verdict.
    @property
    def lo(self) -> float:
        """Lower band bound (mean - sd); the flag threshold both kinds use."""
        return self.mean - self.sd

    @property
    def hi(self) -> float:
        """Upper band bound (mean + sd); only the two-sided kind uses it."""
        return self.mean + self.sd


# The four rules in one table. Reference mean/sd from Jacquier-Bret et al.
# (2024); bands are derived (Rule.lo/Rule.hi), never written out.
RULES = [
    # Trunk inclination vs vertical, trophy frame. Frontal plane. Band [17.9, 32.1].
    Rule(id="trunk_inclination", key_frame="trophy", plane="frontal",
         mean=25.0, sd=7.1, band_kind="two_sided"),
    # Front-leg hip->knee->ankle, trophy frame. Sagittal. Lower bound 54.8:
    # flag only insufficient flexion; deep flexion is unpenalised.
    Rule(id="front_knee_flexion", key_frame="trophy", plane="sagittal",
         mean=64.5, sd=9.7, band_kind="lower_bound"),
    # Serving-arm shoulder->elbow->wrist, impact. Plane-independent. Band [19.3, 39.1].
    Rule(id="elbow_flexion", key_frame="impact", plane=None,
         mean=29.2, sd=9.9, band_kind="two_sided"),
    # Serving-side shoulder->elbow vs shoulder->hip, impact. Plane-independent. Band [98.5, 110.7].
    Rule(id="shoulder_elevation", key_frame="impact", plane=None,
         mean=104.6, sd=6.1, band_kind="two_sided"),
]


def evaluate(angle: Optional[float], rule: Rule) -> str:
    """Deviation indicator for one angle: inside / outside / unavailable.

    A None angle (event not locatable, or landmark unreliable) is reported
    unavailable, never forced to a value.
    """
    if angle is None:
        return "unavailable"
    if rule.band_kind == "two_sided":
        # Inside iff within [mean-sd, mean+sd].
        return "inside" if rule.lo <= angle <= rule.hi else "outside"
    if rule.band_kind == "lower_bound":
        # Inside iff at least the lower bound; the upper side is unpenalised.
        return "inside" if angle >= rule.lo else "outside"
    raise ValueError(f"unknown band_kind: {rule.band_kind!r}")


def plane_supported(rule: Rule, camera_plane: str) -> bool:
    """Whether the camera plane reads this rule's angle cleanly.

    The two trophy criteria are plane-bound and orthogonal: one camera faces
    one plane, so only that criterion reads cleanly. Trunk needs "frontal",
    front knee "sagittal". Impact criteria (plane None) are always supported.
    """
    if rule.plane is None:
        return True
    return rule.plane == camera_plane


@dataclass
class Indicator:
    """One criterion's result: a deviation flag, or why there is none.

    status is "inside", "outside", or "unavailable". angle is the value flagged
    (None when unavailable). detail carries the skip reason, or a flagged knee's
    direction ("insufficient_flexion"); the one-sided band only flags too little.
    """

    criterion: str
    status: str
    angle: Optional[float]
    detail: Optional[str] = None


def angle_for(readings: AngleReadings, rule: Rule) -> Optional[float]:
    """The computed angle a rule evaluates, or None when unavailable.

    AngleReadings names its angles exactly like the rule ids, so the id
    selects the field directly.
    """
    return getattr(readings, rule.id)


def evaluate_all(readings: AngleReadings,
                 clip_params: ClipParams) -> List[Indicator]:
    """One Indicator per rule.

    A genuine flag needs all three: the camera plane supports the criterion,
    the key frame was locatable, and the landmarks were reliable there. An
    unsupported plane is reported unavailable before evaluation; a None angle
    by evaluate. A flagged knee names its direction (insufficient flexion only).
    """
    indicators: List[Indicator] = []
    for rule in RULES:
        if not plane_supported(rule, clip_params.camera_plane):
            indicators.append(Indicator(
                criterion=rule.id, status="unavailable", angle=None,
                detail=f"camera plane {clip_params.camera_plane!r} does not "
                       f"support this criterion (needs {rule.plane!r})"))
            continue
        angle = angle_for(readings, rule)
        status = evaluate(angle, rule)
        detail: Optional[str] = None
        if status == "unavailable":
            detail = "key frame not locatable or landmark unreliable"
        elif status == "outside" and rule.band_kind == "lower_bound":
            detail = "insufficient_flexion"
        indicators.append(Indicator(
            criterion=rule.id, status=status, angle=angle, detail=detail))
    return indicators
