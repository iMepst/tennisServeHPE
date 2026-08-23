import math

import pytest

from serve_pipeline.angles import pixel_point, vector_angle
from serve_pipeline.config import ClipParams


def _params(width: int = 1920, height: int = 1080) -> ClipParams:
    return ClipParams(serving_arm="right", front_leg="left",
                      camera_plane="frontal", view_direction="front",
                      fps=25.0, frame_width=width, frame_height=height)


def test_pixel_point_rescales_by_frame_size() -> None:
    assert pixel_point(0.5, 0.5, _params()) == (960.0, 540.0)
    assert pixel_point(0.0, 1.0, _params()) == (0.0, 1080.0)


def test_equal_normalized_offsets_differ_in_pixels_on_wide_frames() -> None:
    # On a 16:9 frame the same normalized offset spans different pixel
    # lengths per axis — the reason rescaling precedes every angle.
    x0, y0 = pixel_point(0.4, 0.4, _params(1920, 1080))
    x1, y1 = pixel_point(0.5, 0.5, _params(1920, 1080))
    assert (x1 - x0, y1 - y0) == (192.0, 108.0)
    # only a square frame keeps the offsets equal
    x0, y0 = pixel_point(0.4, 0.4, _params(1000, 1000))
    x1, y1 = pixel_point(0.5, 0.5, _params(1000, 1000))
    assert x1 - x0 == y1 - y0


def test_vector_angle_cardinal_cases() -> None:
    assert vector_angle((1.0, 0.0), (2.0, 0.0)) == 0.0
    assert vector_angle((1.0, 0.0), (0.0, 1.0)) == 90.0
    assert vector_angle((1.0, 0.0), (-3.0, 0.0)) == 180.0


def test_vector_angle_oblique_and_scale_invariant() -> None:
    assert vector_angle((1.0, 0.0), (1.0, 1.0)) == pytest.approx(45.0)
    assert vector_angle((10.0, 0.0), (0.5, 0.5)) == pytest.approx(45.0)


TINY = 1e-8  # slope where cos(theta) rounds to exactly +-1


def test_vector_angle_stable_near_parallel() -> None:
    ang = vector_angle((1.0, 0.0), (1.0, TINY))
    assert ang == pytest.approx(math.degrees(TINY), rel=1e-6)
    # the acos route collapses to exactly 0 for the same vectors
    cos = 1.0 / math.hypot(1.0, TINY)
    assert math.degrees(math.acos(min(1.0, cos))) == 0.0


def test_vector_angle_stable_near_antiparallel() -> None:
    ang = vector_angle((1.0, 0.0), (-1.0, TINY))
    assert ang == pytest.approx(180.0 - math.degrees(TINY), rel=1e-12)
    assert ang < 180.0
    # the acos route collapses to exactly 180 for the same vectors
    cos = -1.0 / math.hypot(1.0, TINY)
    assert math.degrees(math.acos(max(-1.0, cos))) == 180.0
