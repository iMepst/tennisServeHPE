import csv

import pytest

from serve_pipeline.landmarks import NUM_LANDMARKS
from serve_pipeline.persistence import (
    CSV_HEADER,
    LandmarkCsvWriter,
    read_landmarks_csv,
    read_metadata,
    summarize_extraction,
    write_metadata,
)

from conftest import make_frame_pose


def test_csv_roundtrip(tmp_path):
    """What goes in must come out: detected and undetected frames."""
    path = str(tmp_path / "landmarks.csv")
    original = [
        make_frame_pose(0, 0.0, detected=True),
        make_frame_pose(1, 1 / 30, detected=False),
        make_frame_pose(2, 2 / 30, detected=True),
    ]
    with LandmarkCsvWriter(path) as w:
        for fp in original:
            w.write_frame(fp)

    restored = read_landmarks_csv(path)
    assert len(restored) == len(original)
    for orig, rest in zip(original, restored):
        assert rest.frame_index == orig.frame_index
        assert rest.time_s == pytest.approx(orig.time_s, abs=1e-6)
        assert rest.detected == orig.detected
        assert len(rest.landmarks) == len(orig.landmarks)
        for o, r in zip(orig.landmarks, rest.landmarks):
            assert r.landmark_id == o.landmark_id
            for field in ("x", "y", "z", "visibility",
                          "world_x", "world_y", "world_z"):
                assert getattr(r, field) == pytest.approx(
                    getattr(o, field), abs=1e-6)


def test_csv_schema_and_density(tmp_path):
    """Every frame contributes exactly NUM_LANDMARKS rows, even without pose."""
    path = str(tmp_path / "landmarks.csv")
    with LandmarkCsvWriter(path) as w:
        w.write_frame(make_frame_pose(0, 0.0, detected=True))
        w.write_frame(make_frame_pose(1, 0.033, detected=False))

    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_HEADER
    assert len(rows) == 1 + 2 * NUM_LANDMARKS
    undetected = [r for r in rows[1:] if r[0] == "1"]
    assert len(undetected) == NUM_LANDMARKS
    assert all(r[4] == "" for r in undetected)  # x column empty


def test_read_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("frame,x,y\n0,0.1,0.2\n")
    with pytest.raises(ValueError, match="schema"):
        read_landmarks_csv(str(path))


def test_metadata_roundtrip(tmp_path):
    path = str(tmp_path / "meta.json")
    meta = {"stage": 1, "video": {"fps": 30.0}, "nested": {"a": [1, 2]}}
    write_metadata(path, meta)
    assert read_metadata(path) == meta


def test_summarize_extraction():
    frames = [
        make_frame_pose(0, detected=True),
        make_frame_pose(1, detected=False),
        make_frame_pose(2, detected=True),
        make_frame_pose(3, detected=True),
    ]
    stats = summarize_extraction(frames)
    assert stats["frames_processed"] == 4
    assert stats["frames_with_pose"] == 3
    assert stats["detection_rate"] == pytest.approx(0.75)
    assert 0.0 <= stats["mean_visibility"] <= 1.0
    assert len(stats["mean_visibility_per_landmark"]) == NUM_LANDMARKS


def test_summarize_extraction_empty():
    stats = summarize_extraction([])
    assert stats["frames_processed"] == 0
    assert stats["detection_rate"] == 0.0
