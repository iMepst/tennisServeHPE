from typing import List, Optional

import numpy as np
import pytest

from serve_pipeline.interpolation import ProcessedFrame, ProcessedSample
from serve_pipeline.keyevents import (
    detect_ball_impact,
    detect_trophy,
    guarded_extremum,
    landmark_y_series,
    midhip_y_series,
)
from serve_pipeline.landmarks import NAME_TO_ID, NUM_LANDMARKS

FPS = 25.0
LM = NAME_TO_ID["right_wrist"]


def _ok_sample(lm_id: int, y: float) -> ProcessedSample:
    return ProcessedSample(
        landmark_id=lm_id, valid=True, mask_reason="ok",
        interpolated=False, reliable=True, filtered=True,
        x=y, y=y, visibility=0.9)


def _series(ys: List[float], states: Optional[List[str]] = None,
            lm_id: int = LM) -> List[ProcessedFrame]:
    """Frames where landmark lm_id follows ys; all others stay constant.

    states[i] sets the sample state: "ok" (originally reliable,
    default), "interp" (short-gap fill: reliable but not original),
    "gap" (unfilled gap: unreliable, keep-and-flag value).
    """
    states = states or ["ok"] * len(ys)
    frames: List[ProcessedFrame] = []
    for i, (y, st) in enumerate(zip(ys, states)):
        samples: List[ProcessedSample] = []
        for lm in range(NUM_LANDMARKS):
            if lm != lm_id:
                samples.append(_ok_sample(lm, 0.5))
            elif st == "gap":
                samples.append(ProcessedSample(
                    lm, valid=False, mask_reason="low_visibility",
                    interpolated=False, reliable=False, filtered=False,
                    x=y, y=y, visibility=0.1))
            elif st == "interp":
                samples.append(ProcessedSample(
                    lm, valid=False, mask_reason="low_visibility",
                    interpolated=True, reliable=True, filtered=True,
                    x=y, y=y, visibility=None))
            else:
                samples.append(_ok_sample(lm, y))
        frames.append(ProcessedFrame(frame_index=i, time_s=i / FPS,
                                     samples=samples))
    return frames


def test_y_series_carries_filtered_values() -> None:
    y, original = landmark_y_series(_series([0.5, 0.4, 0.3]), LM)
    assert np.allclose(y, [0.5, 0.4, 0.3])
    assert original.all()


def test_y_series_masks_gaps_and_flags_interpolated() -> None:
    y, original = landmark_y_series(
        _series([0.5, 0.4, 0.3, 0.2], ["ok", "gap", "interp", "ok"]), LM)
    assert np.isnan(y[1])                 # unfilled gap: no usable value
    assert y[2] == 0.3                    # interpolated value is usable...
    assert list(original) == [True, False, False, True]   # ...not original


def test_extremum_found_on_original_sample() -> None:
    y, original = landmark_y_series(_series([0.5, 0.3, 0.2, 0.4]), LM)
    assert guarded_extremum(y, original, "min") == (2, "ok")
    assert guarded_extremum(y, original, "max") == (0, "ok")


def test_extremum_on_interpolated_sample_is_rejected() -> None:
    y, original = landmark_y_series(
        _series([0.5, 0.3, 0.2, 0.4], ["ok", "ok", "interp", "ok"]), LM)
    pos, reason = guarded_extremum(y, original, "min")
    assert pos is None
    assert "interpolated" in reason


def test_extremum_needs_at_least_one_reliable_sample() -> None:
    y, original = landmark_y_series(
        _series([0.5, 0.3], ["gap", "gap"]), LM)
    pos, reason = guarded_extremum(y, original, "min")
    assert pos is None
    assert "no reliable" in reason


def test_extremum_beside_left_gap_is_rejected() -> None:
    # the minimum at position 2 borders the unfilled gap at position 1
    y, original = landmark_y_series(
        _series([0.5, 0.1, 0.2, 0.4], ["ok", "gap", "ok", "ok"]), LM)
    pos, reason = guarded_extremum(y, original, "min")
    assert pos is None
    assert "unfilled gap" in reason


def test_extremum_beside_right_gap_is_rejected() -> None:
    y, original = landmark_y_series(
        _series([0.5, 0.2, 0.1, 0.4], ["ok", "ok", "gap", "ok"]), LM)
    pos, reason = guarded_extremum(y, original, "min")
    assert pos is None
    assert "unfilled gap" in reason


def test_impact_is_the_right_wrist_minimum() -> None:
    frames = _series([0.8, 0.6, 0.2, 0.5], lm_id=NAME_TO_ID["right_wrist"])
    assert detect_ball_impact(frames, "right") == (2, "ok")


def test_impact_uses_the_serving_arm_wrist() -> None:
    # the left wrist dips at frame 1; the right wrist stays constant
    frames = _series([0.8, 0.2, 0.6], lm_id=NAME_TO_ID["left_wrist"])
    assert detect_ball_impact(frames, "left") == (1, "ok")
    # right wrist is constant 0.5: min is frame 0, but still "ok"
    pos, reason = detect_ball_impact(frames, "right")
    assert reason == "ok"


def test_impact_guard_failure_propagates() -> None:
    frames = _series([0.8, 0.6, 0.2, 0.5],
                     ["ok", "ok", "interp", "ok"],
                     lm_id=NAME_TO_ID["right_wrist"])
    pos, reason = detect_ball_impact(frames, "right")
    assert pos is None
    assert "interpolated" in reason


def test_impact_rejects_unknown_arm() -> None:
    with pytest.raises(ValueError, match="serving_arm"):
        detect_ball_impact(_series([0.5]), "both")


def test_midhip_is_the_mean_of_both_hips() -> None:
    # left hip varies, right hip stays at the builder's constant 0.5
    frames = _series([0.7, 0.9], lm_id=NAME_TO_ID["left_hip"])
    y, original = midhip_y_series(frames)
    assert np.allclose(y, [0.6, 0.7])
    assert original.all()


def test_midhip_needs_both_hips() -> None:
    frames = _series([0.7, 0.9, 0.8], ["ok", "gap", "interp"],
                     lm_id=NAME_TO_ID["left_hip"])
    y, original = midhip_y_series(frames)
    assert np.isnan(y[1])               # one missing hip: no pelvis value
    assert y[2] == 0.65                 # interpolated hip: usable value...
    assert list(original) == [True, False, False]   # ...but not original


def test_trophy_is_the_midhip_maximum_before_impact() -> None:
    frames = _series([0.6, 0.9, 0.7, 0.5, 1.2],
                     lm_id=NAME_TO_ID["left_hip"])
    # the global maximum at position 4 lies outside the search window
    assert detect_trophy(frames, impact_pos=4) == (1, "ok")


def test_trophy_guard_failure_propagates() -> None:
    frames = _series([0.6, 0.9, 0.7, 0.5], ["ok", "interp", "ok", "ok"],
                     lm_id=NAME_TO_ID["left_hip"])
    pos, reason = detect_trophy(frames, impact_pos=4)
    assert pos is None
    assert "interpolated" in reason


def test_trophy_needs_frames_before_impact() -> None:
    frames = _series([0.6, 0.9], lm_id=NAME_TO_ID["left_hip"])
    pos, reason = detect_trophy(frames, impact_pos=0)
    assert pos is None
    assert "before impact" in reason