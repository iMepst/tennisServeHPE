"""Unit tests for Stage 2a gating and gap handling (no model required)."""

from typing import Dict, Optional

from serve_pipeline.gating import (
    MASK_LOW_VISIBILITY,
    MASK_OK,
    MASK_UNDETECTED,
    compute_gap_statistics,
    gate_frames,
)
from serve_pipeline.landmarks import LANDMARK_NAMES, NUM_LANDMARKS
from serve_pipeline.persistence import read_gated_csv, write_gated_csv
from serve_pipeline.pose_extraction import FramePose, LandmarkObservation

FPS = 25.0
_VALUE_FIELDS = ["x", "y", "z", "visibility", "presence",
                 "world_x", "world_y", "world_z"]


def _detected(idx: int,
              vis_overrides: Optional[Dict[int, float]] = None) -> FramePose:
    """Detected frame with all landmarks visible except given overrides."""
    vis_overrides = vis_overrides or {}
    lms = [
        LandmarkObservation(
            landmark_id=i,
            x=0.1 + i, y=0.2 + i, z=0.3 + i,
            visibility=vis_overrides.get(i, 1.0), presence=0.9,
            world_x=0.4 + i, world_y=0.5 + i, world_z=0.6 + i,
        )
        for i in range(NUM_LANDMARKS)
    ]
    return FramePose(frame_index=idx, time_s=idx / FPS,
                     detected=True, landmarks=lms)


def _undetected(idx: int) -> FramePose:
    return FramePose(frame_index=idx, time_s=idx / FPS,
                     detected=False, landmarks=[])


def _lm_stats(gated_stats: dict, name: str) -> dict:
    return gated_stats["per_landmark"][name]


def test_low_visibility_is_masked_others_untouched() -> None:
    gated = gate_frames([_detected(0, {14: 0.2})], visibility_threshold=0.5)
    samples = gated[0].samples
    assert samples[14].valid is False
    assert samples[14].mask_reason == MASK_LOW_VISIBILITY
    # every other landmark stays valid
    assert all(samples[i].valid for i in range(NUM_LANDMARKS) if i != 14)
    assert samples[13].mask_reason == MASK_OK


def test_threshold_is_inclusive_lower_bound() -> None:
    gated = gate_frames([_detected(0, {0: 0.5, 1: 0.4999})],
                        visibility_threshold=0.5)
    samples = gated[0].samples
    assert samples[0].valid is True          # exactly at threshold -> valid
    assert samples[1].valid is False         # just below -> invalid


def test_keep_and_flag_preserves_masked_values() -> None:
    gated = gate_frames([_detected(0, {16: 0.1})], visibility_threshold=0.5)
    masked = gated[0].samples[16]
    assert masked.valid is False
    # values are kept, not blanked
    assert masked.visibility == 0.1
    assert masked.x == 0.1 + 16
    assert masked.world_z == 0.6 + 16


def test_undetected_frame_masks_all_with_none_values() -> None:
    gated = gate_frames([_undetected(0)], visibility_threshold=0.5)
    samples = gated[0].samples
    assert len(samples) == NUM_LANDMARKS
    for s in samples:
        assert s.valid is False
        assert s.mask_reason == MASK_UNDETECTED
        for fld in _VALUE_FIELDS:
            assert getattr(s, fld) is None


def test_output_is_dense_and_ordered() -> None:
    gated = gate_frames([_detected(0), _undetected(1)],
                        visibility_threshold=0.5)
    for g in gated:
        assert [s.landmark_id for s in g.samples] == list(range(NUM_LANDMARKS))


def test_single_gap_length_and_bounds() -> None:
    # landmark 16 invalid on frames 3,4,5 out of 0..9
    frames = [
        _detected(i, {16: 0.1}) if i in (3, 4, 5) else _detected(i)
        for i in range(10)
    ]
    stats = compute_gap_statistics(gate_frames(frames, 0.5), FPS)
    lm = _lm_stats(stats, LANDMARK_NAMES[16])
    assert lm["n_valid"] == 7
    assert lm["valid_rate"] == 7 / 10
    assert lm["num_gaps"] == 1
    gap = lm["gaps"][0]
    assert gap["start_frame"] == 3
    assert gap["end_frame"] == 5
    assert gap["length_frames"] == 3
    assert gap["length_ms"] == 3 / FPS * 1000.0
    assert lm["longest_gap_frames"] == 3


def test_multiple_gaps_and_longest() -> None:
    invalid = {2, 5, 6}  # a length-1 gap and a length-2 gap
    frames = [
        _detected(i, {16: 0.1}) if i in invalid else _detected(i)
        for i in range(8)
    ]
    lm = _lm_stats(compute_gap_statistics(gate_frames(frames, 0.5), FPS),
                   LANDMARK_NAMES[16])
    assert lm["num_gaps"] == 2
    assert lm["longest_gap_frames"] == 2


def test_no_gaps_when_all_valid() -> None:
    frames = [_detected(i) for i in range(5)]
    lm = _lm_stats(compute_gap_statistics(gate_frames(frames, 0.5), FPS),
                   LANDMARK_NAMES[16])
    assert lm["num_gaps"] == 0
    assert lm["valid_rate"] == 1.0
    assert lm["longest_gap_frames"] == 0


def test_all_invalid_is_one_full_length_gap() -> None:
    frames = [_undetected(i) for i in range(4)]
    lm = _lm_stats(compute_gap_statistics(gate_frames(frames, 0.5), FPS),
                   LANDMARK_NAMES[16])
    assert lm["valid_rate"] == 0.0
    assert lm["num_gaps"] == 1
    assert lm["gaps"][0]["length_frames"] == 4


def test_reason_counts_split_undetected_and_low_vis() -> None:
    frames = [
        _detected(0, {16: 0.1}),   # low vis
        _undetected(1),            # undetected
        _detected(2),              # valid
        _detected(3, {16: 0.2}),   # low vis
    ]
    lm = _lm_stats(compute_gap_statistics(gate_frames(frames, 0.5), FPS),
                   LANDMARK_NAMES[16])
    assert lm["n_undetected"] == 1
    assert lm["n_low_visibility"] == 2
    assert lm["n_valid"] == 1
    # frames 0,1 form one gap (mixed reasons), frame 3 another
    assert lm["num_gaps"] == 2


def test_gated_csv_roundtrip(tmp_path) -> None:
    frames = [_detected(0, {14: 0.2}), _undetected(1), _detected(2)]
    gated = gate_frames(frames, 0.5)
    path = str(tmp_path / "gated.csv")
    write_gated_csv(path, gated)
    back = read_gated_csv(path)

    assert len(back) == len(gated)
    for gi, go in zip(gated, back):
        assert gi.frame_index == go.frame_index
        assert abs(gi.time_s - go.time_s) < 1e-6
        assert len(go.samples) == NUM_LANDMARKS
        for si, so in zip(gi.samples, go.samples):
            assert si.landmark_id == so.landmark_id
            assert si.valid == so.valid
            assert si.mask_reason == so.mask_reason
            for fld in _VALUE_FIELDS:
                a, b = getattr(si, fld), getattr(so, fld)
                if a is None:
                    assert b is None
                else:
                    assert abs(a - b) < 1e-6
