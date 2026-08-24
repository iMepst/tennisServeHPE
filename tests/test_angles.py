import math
from typing import Dict, Tuple

import pytest

from serve_pipeline.angles import (
    AngleReadings,
    body_midpoint,
    compute_angles,
    elbow_flexion,
    front_knee_flexion,
    landmark_pixel,
    landmarks_reliable,
    pixel_point,
    shoulder_elevation,
    trunk_inclination,
    turning_angle,
    vector_angle,
)
from serve_pipeline.config import ClipParams
from serve_pipeline.keyevents import KeyEvents
from serve_pipeline.interpolation import ProcessedFrame, ProcessedSample
from serve_pipeline.landmarks import LANDMARK_NAMES, NAME_TO_ID, NUM_LANDMARKS


def _params(width: int = 1920, height: int = 1080,
            serving_arm: str = "right",
            front_leg: str = "left") -> ClipParams:
    return ClipParams(serving_arm=serving_arm, front_leg=front_leg,
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


def test_landmarks_reliable_true_when_all_pass() -> None:
    frame = _frame({"right_shoulder": (0.6, 0.2)})
    assert landmarks_reliable(frame, ["right_shoulder", "right_elbow"])


def test_landmarks_reliable_false_on_unreliable_sample() -> None:
    frame = _frame({"right_shoulder": (0.6, 0.2)})
    frame.samples[NAME_TO_ID["right_elbow"]].reliable = False
    assert not landmarks_reliable(frame, ["right_shoulder", "right_elbow"])


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
    assert rescaled != pytest.approx(vector_angle((0.3, 0.1), (0.1, 0.4)))


def test_turning_angle_right_angle_bend() -> None:
    frame = _frame({"right_hip": (0.2, 0.5), "right_knee": (0.5, 0.5),
                    "right_ankle": (0.5, 0.8)})
    ang = turning_angle(frame, "right_hip", "right_knee", "right_ankle",
                        _params(1000, 1000))
    assert ang == pytest.approx(90.0)


# A frame with the left leg bent at a right angle and the right leg
# straight, on a square frame so normalized geometry carries to pixels.
_LEGS = {
    "left_hip": (0.2, 0.5), "left_knee": (0.5, 0.5),
    "left_ankle": (0.5, 0.8),
    "right_hip": (0.7, 0.2), "right_knee": (0.7, 0.5),
    "right_ankle": (0.7, 0.8),
}


def test_front_knee_flexion_reads_the_front_leg() -> None:
    frame = _frame(_LEGS)
    bent = front_knee_flexion(frame, _params(1000, 1000, front_leg="left"))
    straight = front_knee_flexion(
        frame, _params(1000, 1000, front_leg="right"))
    assert bent == pytest.approx(90.0)
    assert straight == pytest.approx(0.0)


def test_front_knee_flexion_rejects_unknown_side() -> None:
    with pytest.raises(ValueError, match="front_leg"):
        front_knee_flexion(_frame(_LEGS),
                           _params(1000, 1000, front_leg="both"))


# Right arm bent at a right angle, left arm hanging straight down.
_ARMS = {
    "right_shoulder": (0.6, 0.2), "right_elbow": (0.6, 0.5),
    "right_wrist": (0.9, 0.5),
    "left_shoulder": (0.4, 0.2), "left_elbow": (0.4, 0.5),
    "left_wrist": (0.4, 0.8),
}


def test_elbow_flexion_reads_the_serving_arm() -> None:
    frame = _frame(_ARMS)
    bent = elbow_flexion(frame, _params(1000, 1000, serving_arm="right"))
    straight = elbow_flexion(frame, _params(1000, 1000, serving_arm="left"))
    assert bent == pytest.approx(90.0)
    assert straight == pytest.approx(0.0)


def test_elbow_flexion_rejects_unknown_side() -> None:
    with pytest.raises(ValueError, match="serving_arm"):
        elbow_flexion(_frame(_ARMS),
                      _params(1000, 1000, serving_arm="up"))


def test_shoulder_elevation_zero_with_arm_along_trunk() -> None:
    # elbow straight below the shoulder, hip below the shoulder as well
    frame = _frame({"right_shoulder": (0.6, 0.2),
                    "right_elbow": (0.6, 0.5),
                    "right_hip": (0.6, 0.6)})
    ang = shoulder_elevation(frame, _params(1000, 1000))
    assert ang == pytest.approx(0.0)


def test_shoulder_elevation_raised_arm_and_side_selection() -> None:
    # right arm raised straight overhead (opposite the trunk direction),
    # left arm horizontal (90 deg to its trunk vector)
    frame = _frame({"right_shoulder": (0.6, 0.5),
                    "right_elbow": (0.6, 0.2),
                    "right_hip": (0.6, 0.8),
                    "left_shoulder": (0.4, 0.5),
                    "left_elbow": (0.1, 0.5),
                    "left_hip": (0.4, 0.8)})
    assert shoulder_elevation(
        frame, _params(1000, 1000, serving_arm="right")
    ) == pytest.approx(180.0)
    assert shoulder_elevation(
        frame, _params(1000, 1000, serving_arm="left")
    ) == pytest.approx(90.0)


def test_body_midpoint_averages_both_sides_in_pixels() -> None:
    frame = _frame({"left_hip": (0.4, 0.6), "right_hip": (0.6, 0.7)})
    assert body_midpoint(frame, "left_hip", "right_hip",
                         _params()) == (960.0, 702.0)


def test_trunk_inclination_upright_is_zero_via_midpoints() -> None:
    # individual sides are tilted, but both midpoints share x = 0.5:
    # only the midpoint axis must count, and an upright trunk reads 0
    frame = _frame({"left_hip": (0.4, 0.62), "right_hip": (0.6, 0.58),
                    "left_shoulder": (0.45, 0.31),
                    "right_shoulder": (0.55, 0.29)})
    ang = trunk_inclination(frame, _params(1000, 1000))
    assert ang == pytest.approx(0.0)


def test_trunk_inclination_recovers_the_lean_not_its_complement() -> None:
    lean = math.radians(25.0)
    hip = (0.5, 0.6)
    shoulder = (0.5 + 0.3 * math.sin(lean), 0.6 - 0.3 * math.cos(lean))
    frame = _frame({"left_hip": hip, "right_hip": hip,
                    "left_shoulder": shoulder, "right_shoulder": shoulder})
    ang = trunk_inclination(frame, _params(1000, 1000))
    assert ang == pytest.approx(25.0)     # not 155.0


# Trophy frame: bent left leg (_LEGS) plus a shoulder pair for the trunk.
_TROPHY = {**_LEGS, "left_shoulder": (0.45, 0.31),
           "right_shoulder": (0.55, 0.29)}
# Impact frame: bent right arm (_ARMS); right_hip defaults for shoulder.
_IMPACT = _ARMS


def _frames_with(trophy_idx: int, impact_idx: int,
                 n: int = 5) -> list:
    """Dense frames keyed by index, posed only at the two key frames."""
    frames = []
    for i in range(n):
        pos = _TROPHY if i == trophy_idx else _IMPACT if i == impact_idx \
            else {}
        frame = _frame(pos)
        frame.frame_index = i
        frames.append(frame)
    return frames


def _locatable(trophy: int, impact: int) -> KeyEvents:
    return KeyEvents(trophy_frame=trophy, impact_frame=impact,
                     trophy_locatable=True, impact_locatable=True,
                     reason="ok")


def test_compute_angles_reads_each_at_its_key_frame() -> None:
    p = _params(1000, 1000, serving_arm="right", front_leg="left")
    frames = _frames_with(1, 3)
    r = compute_angles(frames, _locatable(1, 3), p)
    assert (r.trophy_frame, r.impact_frame) == (1, 3)
    assert r.trunk_inclination == pytest.approx(
        trunk_inclination(frames[1], p))
    assert r.front_knee_flexion == pytest.approx(
        front_knee_flexion(frames[1], p))
    assert r.elbow_flexion == pytest.approx(elbow_flexion(frames[3], p))
    assert r.shoulder_elevation == pytest.approx(
        shoulder_elevation(frames[3], p))


def test_compute_angles_gates_one_unreliable_landmark() -> None:
    p = _params(1000, 1000, serving_arm="right", front_leg="left")
    frames = _frames_with(1, 3)
    frames[1].samples[NAME_TO_ID["left_knee"]].reliable = False
    r = compute_angles(frames, _locatable(1, 3), p)
    assert r.front_knee_flexion is None        # gated out
    assert r.trunk_inclination is not None      # unaffected


def test_compute_angles_skips_unlocatable_event() -> None:
    p = _params(1000, 1000, serving_arm="right", front_leg="left")
    frames = _frames_with(1, 3)
    ke = KeyEvents(trophy_frame=None, impact_frame=3,
                   trophy_locatable=False, impact_locatable=True,
                   reason="trophy: no reliable samples in the series")
    r = compute_angles(frames, ke, p)
    assert r.trophy_frame is None
    assert r.trunk_inclination is None and r.front_knee_flexion is None
    assert r.impact_frame == 3 and r.elbow_flexion is not None


def test_angle_readings_defaults_to_none() -> None:
    r = AngleReadings(None, None, None, None, None, None)
    assert r.trunk_inclination is None and r.shoulder_elevation is None
