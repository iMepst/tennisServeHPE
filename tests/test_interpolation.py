from typing import List, Optional

from serve_pipeline.gating import (
    MASK_LOW_VISIBILITY,
    MASK_OK,
    MASK_UNDETECTED,
    GatedFrame,
    GatedSample,
)
from serve_pipeline.interpolation import (
    COORD_FIELDS,
    ProcessedFrame,
    ProcessedSample,
    interpolate_gaps,
    summarize_interpolation,
)
from serve_pipeline.landmarks import LANDMARK_NAMES, NUM_LANDMARKS

LM = 16  # the landmark under test; all others stay valid and constant
FPS = 25.0


def _sample(lm_id: int, valid: bool, val: Optional[float],
            undetected: bool = False) -> GatedSample:
    if undetected:
        return GatedSample(lm_id, False, MASK_UNDETECTED,
                           None, None, None, None, None, None, None)
    return GatedSample(
        lm_id, valid, MASK_OK if valid else MASK_LOW_VISIBILITY,
        x=val, y=val, z=val,
        visibility=1.0 if valid else 0.1,
        world_x=val, world_y=val, world_z=val,
    )


def _series(valid_by_frame: List[bool],
            undetected: bool = False) -> List[GatedFrame]:
    """Build a gated series where landmark LM follows the validity pattern.

    Valid samples carry coordinate value = frame_index * 10 so linear
    interpolation is easy to check; all other landmarks stay valid/constant.
    """
    frames: List[GatedFrame] = []
    for i, ok in enumerate(valid_by_frame):
        samples = []
        for lm_id in range(NUM_LANDMARKS):
            if lm_id == LM:
                samples.append(_sample(
                    lm_id, ok, float(i * 10) if ok else None,
                    undetected=(not ok and undetected)))
            else:
                samples.append(_sample(lm_id, True, 1.0))
        frames.append(GatedFrame(frame_index=i, time_s=i / FPS,
                                 samples=samples))
    return frames


def _lm(frames: List[ProcessedFrame], i: int) -> ProcessedSample:
    return frames[i].samples[LM]


def test_short_interior_gap_is_linearly_interpolated() -> None:
    # frames 0..5, LM invalid on 2,3 (interior 2-frame gap)
    out = interpolate_gaps(_series([True, True, False, False, True, True]),
                           max_gap_frames=3)
    # left valid=10 (f1), right valid=40 (f4): linear over positions 2,3
    assert _lm(out, 2).x == 20.0
    assert _lm(out, 3).x == 30.0
    for f in (2, 3):
        s = _lm(out, f)
        assert s.interpolated is True
        assert s.reliable is True
        # all spatial channels filled consistently
        for field in COORD_FIELDS:
            assert getattr(s, field) is not None


def test_gap_longer_than_threshold_is_not_interpolated() -> None:
    # a 3-frame gap with threshold 2 -> left untouched, unreliable
    out = interpolate_gaps(
        _series([True, False, False, False, True]), max_gap_frames=2)
    for f in (1, 2, 3):
        s = _lm(out, f)
        assert s.interpolated is False
        assert s.reliable is False
        assert s.x is None


def test_edge_gap_is_never_interpolated() -> None:
    # leading gap has no left neighbour, even though it is short
    out = interpolate_gaps(_series([False, True, True]), max_gap_frames=3)
    lead = _lm(out, 0)
    assert lead.interpolated is False
    assert lead.reliable is False


def test_valid_samples_are_reliable_and_unchanged() -> None:
    out = interpolate_gaps(_series([True, True, True]), max_gap_frames=3)
    for f in range(3):
        s = _lm(out, f)
        assert s.valid is True
        assert s.reliable is True
        assert s.interpolated is False
        assert s.x == float(f * 10)


def test_undetected_short_gap_still_interpolates_coordinates() -> None:
    # keep-and-flag left None coords for undetected; a short gap fills them
    out = interpolate_gaps(
        _series([True, False, True], undetected=True), max_gap_frames=3)
    s = _lm(out, 1)
    assert s.interpolated is True
    assert s.x == 10.0           # midpoint of frame 0 (0) and frame 2 (20)
    # the model's confidence is not fabricated
    assert s.visibility is None


def test_summary_counts_interpolated_and_unreliable() -> None:
    out = interpolate_gaps(
        _series([True, False, True, False, False, False, True]),
        max_gap_frames=1)
    stats = summarize_interpolation(out)
    lm = stats["per_landmark"][LANDMARK_NAMES[LM]]
    assert lm["n_interpolated"] == 1     # the single-frame gap at index 1
    assert lm["n_unreliable"] == 3       # the 3-frame gap at 3,4,5
    assert stats["total_interpolated_samples"] == 1
    assert stats["total_unreliable_samples"] == 3
    assert LANDMARK_NAMES[LM] in stats["most_unreliable"]
