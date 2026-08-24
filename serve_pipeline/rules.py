"""Stage 5: rule evaluation and deviation indicators.

Compares each computed angle against its reference band and returns a
binary deviation indicator (rule_base_spec.md, Sections 1-2;
pipeline_spec.md, Stage 5). Runs in memory; nothing is persisted here.

Reference values are from Jacquier-Bret et al. (2024).
"""

from dataclasses import dataclass
from typing import List, Optional

from .angles import AngleReadings
from .config import ClipParams


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
    # Front knee flexion: hip->knee vs knee->ankle on the front leg, read
    # at the trophy frame. Sagittal plane. One-sided lower bound at 54.8:
    # flag only insufficient flexion; deep flexion (greater racket
    # velocity) is left unpenalised.
    Rule(id="front_knee_flexion", key_frame="trophy", plane="sagittal",
         mean=64.5, sd=9.7, band_kind="lower_bound"),
    # Elbow flexion: shoulder->elbow vs elbow->wrist on the serving arm,
    # read at ball impact. Plane-independent. Post-hoc-excluded reference
    # value; band [19.3, 39.1].
    Rule(id="elbow_flexion", key_frame="impact", plane=None,
         mean=29.2, sd=9.9, band_kind="two_sided"),
    # Shoulder elevation: shoulder->elbow vs shoulder->hip on the serving
    # side, read at ball impact. Plane-independent. Post-hoc-excluded
    # reference value; band [98.5, 110.7].
    Rule(id="shoulder_elevation", key_frame="impact", plane=None,
         mean=104.6, sd=6.1, band_kind="two_sided"),
]


def evaluate(angle: Optional[float], rule: Rule) -> str:
    """Deviation indicator for one angle: inside / outside / unavailable.

    A None angle means the criterion could not be read (event not
    locatable, or a landmark unreliable at the key frame): it is reported
    "unavailable", never forced to a value.
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

    The two trophy criteria are plane-bound and orthogonal: a single
    camera faces only one plane, so only that criterion is read cleanly
    (the other is foreshortened). Trunk inclination needs "frontal",
    front knee flexion needs "sagittal". The impact criteria (plane
    None) are plane-independent and always supported.
    """
    if rule.plane is None:
        return True
    return rule.plane == camera_plane


@dataclass
class Indicator:
    """One criterion's Stage 5 result: a deviation flag, or why there is none.

    status is "inside", "outside", or "unavailable". angle is the value the
    flag was read from (None when unavailable). detail carries the reason
    an unavailable criterion was skipped, or, for a flagged knee, the
    direction of the deviation ("insufficient_flexion") — the one-sided
    band only ever flags too little flexion.
    """

    criterion: str
    status: str
    angle: Optional[float]
    detail: Optional[str] = None


def angle_for(readings: AngleReadings, rule: Rule) -> Optional[float]:
    """The computed angle a rule evaluates, or None when unavailable.

    AngleReadings names its four angles exactly like the rule ids, so the
    rule id selects the matching field directly.
    """
    return getattr(readings, rule.id)


def evaluate_all(readings: AngleReadings,
                 clip_params: ClipParams) -> List[Indicator]:
    """The Stage 5 indicator set: one Indicator per rule.

    An indicator is a genuine flag only where all three conditions hold
    together (rule_base_spec.md, Section 4): the camera plane supports the
    criterion, the key frame was locatable, and the landmarks were
    reliable there. A plane the camera cannot read is reported unavailable
    before evaluation; an angle that could not be read (None) is reported
    unavailable by evaluate itself. A flagged knee names its direction —
    the one-sided band only ever flags insufficient flexion.
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
        indicators.append(Indicator(criterion=rule.id, status=status,
                                     angle=angle, detail=detail))
    return indicators
