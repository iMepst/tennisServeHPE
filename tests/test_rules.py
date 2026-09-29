import pytest

from serve_pipeline.angles import AngleReadings
from serve_pipeline.config import ClipParams
from serve_pipeline.rules import (
    RULES,
    Rule,
    evaluate,
    evaluate_all,
    plane_supported,
)


def _rule(rule_id: str) -> Rule:
    return next(r for r in RULES if r.id == rule_id)


def _params(camera_plane: str) -> ClipParams:
    return ClipParams(serving_arm="right", front_leg="left",
                      camera_plane=camera_plane, view_direction="front",
                      fps=30.0, frame_width=1920, frame_height=1080)


def _indicator(indicators, criterion):
    return next(i for i in indicators if i.criterion == criterion)


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


def test_two_sided_inside() -> None:
    rule = _rule("trunk_inclination")
    assert evaluate(25.0, rule) == "inside"
    assert evaluate(rule.lo, rule) == "inside"
    assert evaluate(rule.hi, rule) == "inside"


def test_two_sided_outside() -> None:
    rule = _rule("trunk_inclination")
    assert evaluate(10.0, rule) == "outside"
    assert evaluate(40.0, rule) == "outside"


def test_lower_bound_inside() -> None:
    rule = _rule("front_knee_flexion")
    assert evaluate(rule.lo, rule) == "inside"
    assert evaluate(90.0, rule) == "inside"


def test_lower_bound_outside() -> None:
    rule = _rule("front_knee_flexion")
    assert evaluate(50.0, rule) == "outside"


def test_none_angle_is_unavailable() -> None:
    for rule in RULES:
        assert evaluate(None, rule) == "unavailable"


def test_plane_support() -> None:
    assert plane_supported(_rule("trunk_inclination"), "frontal")
    assert not plane_supported(_rule("trunk_inclination"), "sagittal")
    assert plane_supported(_rule("front_knee_flexion"), "sagittal")
    assert not plane_supported(_rule("front_knee_flexion"), "frontal")
    for rule_id in ("elbow_flexion", "shoulder_elevation"):
        assert plane_supported(_rule(rule_id), "frontal")
        assert plane_supported(_rule(rule_id), "sagittal")


def test_evaluate_all_frontal_reads_trunk_not_knee() -> None:
    readings = AngleReadings(
        trophy_frame=5, impact_frame=10,
        trunk_inclination=40.0, front_knee_flexion=60.0,
        elbow_flexion=30.0, shoulder_elevation=105.0)
    ind = evaluate_all(readings, _params("frontal"))

    trunk = _indicator(ind, "trunk_inclination")
    assert trunk.status == "outside" and trunk.angle == 40.0

    knee = _indicator(ind, "front_knee_flexion")
    assert knee.status == "unavailable" and knee.angle is None
    assert "sagittal" in knee.detail

    assert _indicator(ind, "elbow_flexion").status == "inside"
    assert _indicator(ind, "shoulder_elevation").status == "inside"


def test_evaluate_all_sagittal_reads_knee_not_trunk() -> None:
    readings = AngleReadings(
        trophy_frame=5, impact_frame=10,
        trunk_inclination=25.0, front_knee_flexion=50.0,
        elbow_flexion=30.0, shoulder_elevation=105.0)
    ind = evaluate_all(readings, _params("sagittal"))

    assert _indicator(ind, "trunk_inclination").status == "unavailable"
    knee = _indicator(ind, "front_knee_flexion")
    assert knee.status == "outside"
    assert knee.detail == "insufficient_flexion"


def test_evaluate_all_unreadable_angle_is_unavailable() -> None:
    readings = AngleReadings(
        trophy_frame=5, impact_frame=None,
        trunk_inclination=25.0, front_knee_flexion=None,
        elbow_flexion=None, shoulder_elevation=None)
    ind = evaluate_all(readings, _params("frontal"))

    elbow = _indicator(ind, "elbow_flexion")
    assert elbow.status == "unavailable" and elbow.angle is None
    assert "locatable" in elbow.detail or "unreliable" in elbow.detail
