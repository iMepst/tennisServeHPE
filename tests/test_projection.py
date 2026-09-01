import math

import pytest

from assessment.projection import (_tilt_about_vertical, numeric_projected_angle,
                                    project_orthographic, projection_curves,
                                    theta_values, trunk_projected_angle)
from serve_pipeline.angles import vector_angle
from serve_pipeline.config import PipelineConfig


def _direct_trunk_projection(a_true: float, theta: float) -> float:
    """Independent numeric projection of a single trunk segment.

    Builds the trunk axis at a_true from the vertical, tilts it by theta and
    projects it, then reads the angle back against the vertical -- an
    independent path to compare the closed form against.
    """
    a = math.radians(a_true)
    axis = (math.sin(a), math.cos(a), 0.0)
    px, py = project_orthographic(_tilt_about_vertical(axis, theta))
    return vector_angle((px, py), (0.0, 1.0))


@pytest.mark.parametrize("a_true", [10.0, 25.0, 40.0])
@pytest.mark.parametrize("theta", [0.0, 15.0, 30.0, 45.0])
def test_trunk_closed_form_matches_direct_numeric(a_true, theta):
    assert trunk_projected_angle(a_true, theta) == pytest.approx(
        _direct_trunk_projection(a_true, theta), abs=1e-9)


def test_known_theta_foreshortening():
    # tan(a_proj) = tan(a_true) * cos(theta); check the 45 deg value directly.
    expected = math.degrees(math.atan(math.tan(math.radians(25.0))
                                      * math.cos(math.radians(45.0))))
    assert trunk_projected_angle(25.0, 45.0) == pytest.approx(expected)
