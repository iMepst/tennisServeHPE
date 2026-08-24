import pytest

from serve_pipeline.rules import (
    RULES,
    Rule,
    evaluate,
)


def _rule(rule_id: str) -> Rule:
    return next(r for r in RULES if r.id == rule_id)


# --------------------------------------------------------------------------- #
# Band derivation: lo/hi are mean -/+ sd, not literal magic numbers.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rule_id, lo, hi", [
    ("trunk_inclination", 17.9, 32.1),
    ("front_knee_flexion", 54.8, 74.2),
    ("elbow_flexion", 19.3, 39.1),
    ("shoulder_elevation", 98.5, 110.7),
])
def test_band_bounds_match_reference(rule_id: str, lo: float,
                                     hi: float) -> None:
    rule = _rule(rule_id)
    assert rule.lo == pytest.approx(lo)
    assert rule.hi == pytest.approx(hi)


# --------------------------------------------------------------------------- #
# evaluate: two-sided band flags on either side.
# --------------------------------------------------------------------------- #
def test_two_sided_inside() -> None:
    rule = _rule("trunk_inclination")
    assert evaluate(25.0, rule) == "inside"
    assert evaluate(rule.lo, rule) == "inside"   # boundary is inside
    assert evaluate(rule.hi, rule) == "inside"


def test_two_sided_outside() -> None:
    rule = _rule("trunk_inclination")
    assert evaluate(10.0, rule) == "outside"     # below the band
    assert evaluate(40.0, rule) == "outside"     # above the band


# --------------------------------------------------------------------------- #
# evaluate: one-sided lower bound flags only insufficient value.
# --------------------------------------------------------------------------- #
def test_lower_bound_inside() -> None:
    rule = _rule("front_knee_flexion")
    assert evaluate(rule.lo, rule) == "inside"   # boundary is inside
    assert evaluate(90.0, rule) == "inside"      # deep flexion unpenalised


def test_lower_bound_outside() -> None:
    rule = _rule("front_knee_flexion")
    assert evaluate(50.0, rule) == "outside"     # below the lower bound


# --------------------------------------------------------------------------- #
# evaluate: a missing angle is never forced to a verdict.
# --------------------------------------------------------------------------- #
def test_none_angle_is_unavailable() -> None:
    for rule in RULES:
        assert evaluate(None, rule) == "unavailable"