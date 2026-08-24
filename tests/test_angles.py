import math
from typing import Dict, Tuple

import pytest

from serve_pipeline.angles import (
    compute_angles,
    landmark_pixel,
    pixel_point,
    vector_angle,
)
from serve_pipeline.config import ClipParams
from serve_pipeline.interpolation import ProcessedFrame, ProcessedSample
from serve_pipeline.landmarks import LANDMARK_NAMES, NAME_TO_ID, NUM_LANDMARKS


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


def _frame(positions: Dict[str, Tuple[float, float]]) -> ProcessedFrame:
    """One frame; named landmarks at normalized (x, y), the rest at 0.5."""
    samples = []
    for lm in range(NUM_LANDMARKS):
        x, y = positions.get(LANDMARK_NAMES[lm], (0.5, 0.5))
        samples.append(ProcessedSample(
            landmark_id=lm, valid=True, mask_reason="ok",
            interpolated=False, reliable=True, filtered=True,
            x=x, y=y, visibility=0.9))
    return ProcessedFrame(frame_index=0, time_s=0.0, samples=samples)


def test_landmark_pixel_returns_rescaled_position() -> None:
    frame = _frame({"right_wrist": (0.25, 0.5)})
    assert landmark_pixel(frame, "right_wrist", _params()) == (480.0, 540.0)


def test_landmark_pixel_rejects_missing_coordinates() -> None:
    frame = _frame({"right_wrist": (0.25, 0.5)})
    sample = frame.samples[16]          # right_wrist
    sample.x = None
    sample.y = None
    with pytest.raises(ValueError, match="right_wrist"):
        landmark_pixel(frame, "right_wrist", _params())

def test_landmarks_reliable_false_on_missing_coordinates() -> None:
    frame = _frame({"right_shoulder": (0.6, 0.2)})
    frame.samples[NAME_TO_ID["right_elbow"]].x = None
    assert not landmarks_reliable(frame, ["right_shoulder", "right_elbow"])


def test_turning_angle_straight_chain_is_zero() -> None:
    frame = _frame({"right_hip": (0.5, 0.2), "right_knee": (0.5, 0.5),
                    "right_ankle": (0.5, 0.8)})
    ang = turning_angle(frame, "right_hip", "right_knee", "right_ankle",
                        _params(1000, 1000))
    assert ang == pytest.approx(0.0)


def test_pixel_rescaling_changes_a_non_square_angle() -> None:
    # Same geometry, read on a 16:9 frame: the correct angle uses the
    # pixel-rescaled vectors and differs from the un-rescaled one that a
    # square-frame assumption would wrongly produce.
    frame = _frame({"right_hip": (0.2, 0.4), "right_knee": (0.5, 0.5),
                    "right_ankle": (0.6, 0.9)})
    w, h = 1920, 1080
    rescaled = turning_angle(frame, "right_hip", "right_knee",
                             "right_ankle", _params(w, h))
    assert rescaled == pytest.approx(
        vector_angle((0.3 * w, 0.1 * h), (0.1 * w, 0.4 * h)))