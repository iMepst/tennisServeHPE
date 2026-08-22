import math
from pathlib import Path
from typing import List

from serve_pipeline.filtering import (
    KIND_BUTTERWORTH,
    FilterConfig,
    filter_series,
)
from serve_pipeline.interpolation import ProcessedFrame, ProcessedSample
from serve_pipeline.landmarks import NUM_LANDMARKS
from serve_pipeline.persistence import read_filtered_csv, write_filtered_csv

LM = 16
FPS = 25.0
CUTOFF_HZ = 6.0
ORDER = 4
_VALUE_FIELDS = ["x", "y", "visibility"]


def _psample(lm_id: int, val: float, reliable: bool = True) -> ProcessedSample:
    return ProcessedSample(
        landmark_id=lm_id, valid=True, mask_reason="ok",
        interpolated=False, reliable=reliable, filtered=False,
        x=val, y=val, visibility=1.0)


def _sine_series(freq_hz: float, n: int = 200,
                 amp: float = 1.0) -> List[ProcessedFrame]:
    frames: List[ProcessedFrame] = []
    for i in range(n):
        t = i / FPS
        val = amp * math.sin(2.0 * math.pi * freq_hz * t)
        samples = [_psample(lm, val if lm == LM else 0.0)
                   for lm in range(NUM_LANDMARKS)]
        frames.append(ProcessedFrame(frame_index=i, time_s=t, samples=samples))
    return frames


def _amplitude(frames: List[ProcessedFrame], margin: int = 20) -> float:
    # trim the ends: filtfilt padding leaves edge transients that would
    # otherwise dominate max-min for a strongly attenuated high-freq signal.
    xs = [f.samples[LM].x for f in frames[margin:len(frames) - margin]]
    return (max(xs) - min(xs)) / 2.0  # type: ignore[operator]


def _cfg() -> FilterConfig:
    return FilterConfig(
        kind=KIND_BUTTERWORTH, order=ORDER, cutoff_hz=CUTOFF_HZ)


def test_low_frequency_sine_survives_filtering() -> None:
    frames = _sine_series(1.0)          # 1 Hz, well below the 6 Hz cut-off
    filter_series(frames, FPS, _cfg())
    assert _amplitude(frames) > 0.9     # amplitude essentially preserved


def test_high_frequency_sine_is_strongly_attenuated() -> None:
    frames = _sine_series(10.0)         # 10 Hz, well above the cut-off
    filter_series(frames, FPS, _cfg())
    # Butterworth |H| at 10 Hz, order 4, applied twice (filtfilt):
    #   |H|^2 = 1 / (1 + (10/6)^8)^... -> ~0.02, so far below 0.3.
    assert _amplitude(frames) < 0.3


def test_filtered_flag_set_on_long_reliable_segment() -> None:
    frames = _sine_series(1.0, n=100)
    stats = filter_series(frames, FPS, _cfg())
    assert all(f.samples[LM].filtered for f in frames)
    assert stats["n_filtered_samples"] == 100 * NUM_LANDMARKS
    assert stats["n_segments_too_short"] == 0


def test_short_segment_is_left_unfiltered() -> None:
    # min segment length for order 4 is 3*(4+1)+1 = 16; 12 is too short
    frames = _sine_series(1.0, n=12)
    stats = filter_series(frames, FPS, _cfg())
    assert not any(f.samples[LM].filtered for f in frames)
    assert stats["n_filtered_samples"] == 0
    assert stats["n_segments_too_short"] == NUM_LANDMARKS


def test_unreliable_sample_splits_segments() -> None:
    frames = _sine_series(1.0, n=40)
    frames[20].samples[LM].reliable = False   # break LM into two short halves
    filter_series(frames, FPS, _cfg())
    # both halves (~20 < ... ) still filter; the break itself stays unfiltered
    assert frames[20].samples[LM].filtered is False


def test_filtered_csv_roundtrip(tmp_path: Path) -> None:
    frames: List[ProcessedFrame] = []
    for i in range(3):
        samples = []
        for lm in range(NUM_LANDMARKS):
            s = _psample(lm, float(i + lm))
            if lm == 5 and i == 1:           # an unreliable, interpolated one
                s.reliable = False
                s.interpolated = True
                s.filtered = False
            if lm == 7 and i == 2:           # an undetected-style hole
                s = ProcessedSample(
                    lm, valid=False, mask_reason="undetected",
                    interpolated=False, reliable=False, filtered=False,
                    x=None, y=None, visibility=None)
            samples.append(s)
        frames.append(ProcessedFrame(i, i / FPS, samples))

    path = str(tmp_path / "filtered.csv")
    write_filtered_csv(path, frames)
    back = read_filtered_csv(path)

    assert len(back) == len(frames)
    for fi, fo in zip(frames, back):
        assert fi.frame_index == fo.frame_index
        assert len(fo.samples) == NUM_LANDMARKS
        for si, so in zip(fi.samples, fo.samples):
            assert si.landmark_id == so.landmark_id
            assert si.valid == so.valid
            assert si.mask_reason == so.mask_reason
            assert si.interpolated == so.interpolated
            assert si.reliable == so.reliable
            assert si.filtered == so.filtered
            for fld in _VALUE_FIELDS:
                a, b = getattr(si, fld), getattr(so, fld)
                if a is None:
                    assert b is None
                else:
                    assert abs(a - b) < 1e-6
