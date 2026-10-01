import csv
import math
import os

import pytest

from assessment.annotation import (
    EventAnnotation, estimate_event_error, read_event_annotations,
    _event_stats)
from serve_pipeline.config import PipelineConfig
from serve_pipeline.persistence import write_metadata


def _make_event_clip(results_root, clip, locatable=True):
    os.makedirs(os.path.join(results_root, clip))
    write_metadata(os.path.join(results_root, clip, "result.json"),
                   {"key_events": {"trophy_frame": 2 if locatable else None,
                                   "impact_frame": 7 if locatable else None}})


def test_read_event_annotations_rejects_bad_schema(tmp_path):
    path = tmp_path / "e.csv"
    with open(path, "w", newline="") as f:
        f.write("clip,trophy,impact\nc,1,2\n")
    with pytest.raises(ValueError):
        read_event_annotations(str(path))


def test_event_stats_rate_and_distribution():
    err = _event_stats("trophy", [0, 2, -3, None],
                       tolerances=(1, 3), large_offset_frames=30)
    assert err.n_clips == 4
    assert err.n_locatable == 3
    assert err.n_not_locatable == 1
    assert err.n_moved_by_tolerance[1] == 2
    assert err.n_moved_by_tolerance[3] == 0
    assert err.share_by_tolerance[1] == pytest.approx(2 / 3)
    assert err.share_by_tolerance[3] == 0.0
    assert err.max_abs_offset == 3.0
    assert err.median_offset == 0.0
    assert err.n_large_failures == 0


def test_event_stats_robust_to_heavy_tail():
    err = _event_stats("impact", [0, -1, 1, 0, 200],
                       tolerances=(1,), large_offset_frames=30)
    assert err.n_large_failures == 1
    assert err.median_offset == 0.0
    assert err.max_abs_offset == 200.0
    assert err.mean_offset == pytest.approx(40.0)
    assert not math.isnan(err.iqr_offset)
    assert err.iqr_offset < err.max_abs_offset


def test_estimate_event_error_offsets(tmp_path):
    results_root = str(tmp_path / "results")
    _make_event_clip(results_root, "clipA")
    _make_event_clip(results_root, "clipB")
    annotations = [
        EventAnnotation("clipA", true_trophy_frame=2, true_impact_frame=6),
        EventAnnotation("clipB", true_trophy_frame=2, true_impact_frame=3),
    ]
    err = estimate_event_error(annotations, results_root, tolerances=(1,))

    assert err.trophy.n_moved_by_tolerance[1] == 0
    assert err.trophy.share_by_tolerance[1] == 0.0
    assert err.impact.n_moved_by_tolerance[1] == 1
    assert err.impact.max_abs_offset == 4.0
    assert err.impact.n_not_locatable == 0


def test_estimate_event_error_handles_not_locatable(tmp_path):
    results_root = str(tmp_path / "results")
    _make_event_clip(results_root, "clipX", locatable=False)
    annotations = [EventAnnotation("clipX", true_trophy_frame=2,
                                   true_impact_frame=7)]
    err = estimate_event_error(annotations, results_root)

    assert err.trophy.n_not_locatable == 1
    assert err.impact.n_not_locatable == 1
    assert math.isnan(err.impact.share_by_tolerance[1])
    assert math.isnan(err.impact.mean_offset)


def test_measured_assessment_runs_event_error_and_sigma_sweep(tmp_path):
    from assessment.report import measured_assessment
    from serve_pipeline.rules import RULES

    config = PipelineConfig()
    results_root = str(tmp_path / "results")
    _make_event_clip(results_root, "clipA")

    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    with open(ann_dir / "events.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "true_trophy_frame", "true_impact_frame"])
        writer.writerow(["clipA", 2, 6])

    measured = measured_assessment(config, str(ann_dir), results_root)

    assert measured.event_error is not None
    assert measured.event_error.impact.n_locatable == 1
    assert [p.sigma for p in measured.sweep] == list(config.sigma_sweep)
    for point in measured.sweep:
        assert len(point.decidability) == len(RULES)
        assert point.propagation[0].sigma == pytest.approx(point.sigma)


def test_measured_assessment_without_event_annotation(tmp_path):
    from assessment.report import measured_assessment

    config = PipelineConfig()
    measured = measured_assessment(config, str(tmp_path / "missing"))

    assert measured.event_error is None
    assert [p.sigma for p in measured.sweep] == list(config.sigma_sweep)


def test_swept_sigma_changes_spread():
    from assessment.propagation import noise_propagation

    config = PipelineConfig()
    base = noise_propagation(config, sigma=3.0)
    doubled = noise_propagation(config, sigma=6.0)
    for a, b in zip(base, doubled):
        assert b.sd_deg != a.sd_deg
