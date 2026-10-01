import csv
import json
import os

from assessment.annotation import EventError, _event_stats
from assessment.decidability import Decidability
from assessment.report import (_DECIDABILITY_HEADER, _NOISE_HEADER,
                               _PROJECTION_HEADER, build_assessment_report,
                               decidability_rows, event_error_dict,
                               _unreliable_onset)
from assessment.run_measured import SigmaPoint
from serve_pipeline.config import PipelineConfig
from serve_pipeline.rules import RULES


def _fast_config() -> PipelineConfig:
    """A coarse, cheap config: the report logic is what is under test, not the
    Monte-Carlo precision, so shrink the samples and the two sweeps."""
    config = PipelineConfig()
    config.mc_samples = 200
    config.theta_step = 15.0
    config.sigma_sweep = (2.0, 4.0)
    return config


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _header(path):
    with open(path) as f:
        return next(csv.reader(f))


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def test_build_report_writes_all_artifacts_without_events(tmp_path):
    config = _fast_config()
    out_dir = str(tmp_path / "assessment")
    report = build_assessment_report(
        config, annotations_dir=str(tmp_path / "missing"),
        results_root=str(tmp_path / "results"), out_dir=out_dir,
        make_figures=False)

    names = set(os.listdir(out_dir))
    assert names == {"projection_curves.csv", "noise_propagation.csv",
                     "decidability.csv", "event_error.json", "run_meta.json"}

    n_crit = len(RULES)
    n_theta = 4
    n_sigma = 2

    proj = _read_csv(report["outputs"]["projection_curves"])
    assert len(proj) == n_crit * n_theta
    kinds = {r["criterion"]: r["kind"] for r in proj}
    assert kinds["trunk_inclination"] == "closed_form"
    assert kinds["elbow_flexion"] == "numeric"

    noise = _read_csv(report["outputs"]["noise_propagation"])
    assert len(noise) == n_crit * n_theta * n_sigma
    assert {r["sigma"] for r in noise} == {"2.0", "4.0"}

    dec = _read_csv(report["outputs"]["decidability"])
    assert len(dec) == n_crit * n_theta * n_sigma
    assert {r["verdict"] for r in dec} <= {"decidable", "unreliable"}

    assert (_header(report["outputs"]["projection_curves"])
            == _PROJECTION_HEADER)
    assert _header(report["outputs"]["noise_propagation"]) == _NOISE_HEADER
    assert _header(report["outputs"]["decidability"]) == _DECIDABILITY_HEADER


def test_removed_artifacts_are_not_written(tmp_path):
    config = _fast_config()
    out_dir = str(tmp_path / "assessment")
    build_assessment_report(config, annotations_dir=str(tmp_path / "none"),
                            results_root=str(tmp_path / "results"),
                            out_dir=out_dir, make_figures=True)
    written = set()
    for _root, _dirs, files in os.walk(out_dir):
        written |= set(files)
    assert "sigma_estimate.json" not in written
    assert "decision_instability.csv" not in written


def test_runs_headless(tmp_path):
    import matplotlib
    config = _fast_config()
    build_assessment_report(config, annotations_dir=str(tmp_path / "none"),
                            results_root=str(tmp_path / "results"),
                            out_dir=str(tmp_path / "assessment"),
                            make_figures=True)
    assert matplotlib.get_backend().lower() == "agg"


def test_build_report_writes_figures(tmp_path):
    config = _fast_config()
    out_dir = str(tmp_path / "assessment")
    report = build_assessment_report(
        config, annotations_dir=str(tmp_path / "missing"),
        results_root=str(tmp_path / "results"), out_dir=out_dir,
        make_figures=True)

    fig_dir = os.path.join(out_dir, "figures")
    figs = set(os.listdir(fig_dir))
    assert figs == {"projection_curves.png", "spread_vs_theta.png",
                    "decidability_map.png"}
    assert "decidability_figure" in report["outputs"]
    for name in ("projection_curves.png", "spread_vs_theta.png",
                 "decidability_map.png"):
        assert os.path.getsize(os.path.join(fig_dir, name)) > 0
    event_error = _read_json(os.path.join(out_dir, "event_error.json"))
    assert event_error["placeholder"] is True


def test_reproducible_numbers_across_runs(tmp_path):
    config = _fast_config()
    a = build_assessment_report(config, annotations_dir=str(tmp_path / "none"),
                                results_root=str(tmp_path / "r"),
                                out_dir=str(tmp_path / "a"),
                                make_figures=False)
    b = build_assessment_report(config, annotations_dir=str(tmp_path / "none"),
                                results_root=str(tmp_path / "r"),
                                out_dir=str(tmp_path / "b"),
                                make_figures=False)
    for key in ("projection_curves", "noise_propagation", "decidability"):
        with open(a["outputs"][key]) as fa, open(b["outputs"][key]) as fb:
            assert fa.read() == fb.read()


def test_event_error_placeholder_when_missing(tmp_path):
    config = _fast_config()
    out_dir = str(tmp_path / "assessment")
    build_assessment_report(config, annotations_dir=str(tmp_path / "none"),
                            results_root=str(tmp_path / "results"),
                            out_dir=out_dir, make_figures=False)
    event_error = _read_json(os.path.join(out_dir, "event_error.json"))
    assert event_error["available"] is False
    assert event_error["placeholder"] is True
    assert "note" in event_error
    assert "trophy" not in event_error and "impact" not in event_error


def test_run_meta_logs_every_parameter(tmp_path):
    config = _fast_config()
    out_dir = str(tmp_path / "assessment")
    build_assessment_report(config, annotations_dir=str(tmp_path / "none"),
                            results_root=str(tmp_path / "results"),
                            out_dir=out_dir, make_figures=False)
    meta = _read_json(os.path.join(out_dir, "run_meta.json"))
    assert meta["sigma_sweep"] == [2.0, 4.0]
    assert meta["mc_samples"] == 200
    assert meta["seed"] == config.seed
    assert meta["theta_range"] == [0.0, 45.0]
    assert meta["event_tolerances_frames"] == list(
        config.event_tolerances_frames)
    assert set(meta) == {
        "theta_range", "theta_step", "thetas", "sigma", "sigma_sweep",
        "mc_samples", "seed", "reference_stature_px",
        "event_tolerances_frames", "event_large_offset_frames", "timestamp",
        "outputs", "notes"}
    assert "e4_definitional_mismatch" in meta["notes"]


def test_event_error_dict_serialises_robust_stats():
    impact = _event_stats("impact", [0, -1, 1, 200],
                          tolerances=(1, 3), large_offset_frames=30)
    trophy = _event_stats("trophy", [0, 0, 0, 0],
                          tolerances=(1, 3), large_offset_frames=30)
    record = event_error_dict(
        EventError(n_clips=4, trophy=trophy, impact=impact),
        events_csv="unused")
    assert set(record) == {"available", "n_clips", "trophy", "impact"}
    assert record["available"] is True
    assert set(record["impact"]) == {
        "n_clips", "n_locatable", "n_not_locatable", "tolerances",
        "n_moved_by_tolerance", "move_rate_by_tolerance", "median_offset",
        "iqr_offset", "max_abs_offset", "large_offset_frames",
        "n_large_failures", "mean_offset"}
    assert record["impact"]["n_large_failures"] == 1
    assert set(record["impact"]["move_rate_by_tolerance"]) == {"1", "3"}
    assert record["impact"]["median_offset"] == 0.5


def _decidability(criterion, verdict, breakdown):
    """A minimal Decidability record for the onset logic (only the fields the
    onset reads are meaningful)."""
    return Decidability(
        criterion=criterion, sigma=0.0, mc_samples=0, seed=0, half_width=1.0,
        thetas=[0.0], induced_sd=[0.0], ratio=[0.0], decidable=[True],
        verdict=verdict, breakdown_theta=breakdown)


def test_unreliable_onset_takes_first_ascending_sigma():
    sweep = [
        SigmaPoint(sigma=2.0, propagation=[],
                   decidability=[_decidability("elbow_flexion", "decidable",
                                               None)]),
        SigmaPoint(sigma=4.0, propagation=[],
                   decidability=[_decidability("elbow_flexion", "unreliable",
                                               30.0)]),
        SigmaPoint(sigma=6.0, propagation=[],
                   decidability=[_decidability("elbow_flexion", "unreliable",
                                               15.0)]),
    ]
    onset = _unreliable_onset(sweep)
    assert onset["elbow_flexion"] == {"sigma": 4.0, "theta": 30.0}

    rows = decidability_rows(sweep)
    assert all(r["onset_sigma"] == 4.0 for r in rows)
    assert all(r["onset_theta"] == 30.0 for r in rows)
