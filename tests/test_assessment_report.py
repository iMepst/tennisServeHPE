import csv
import json
import os

from assessment.annotation import EventError, _event_type_error
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
    config.theta_step = 15.0            # thetas 0, 15, 30, 45
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

    # The five tables are the removed-methodology-safe set: no
    # sigma_estimate.json, no decision_instability.csv.
    names = set(os.listdir(out_dir))
    assert names == {"projection_curves.csv", "noise_propagation.csv",
                     "decidability.csv", "event_error.json", "run_meta.json"}

    n_crit = len(RULES)
    n_theta = 4          # 0, 15, 30, 45
    n_sigma = 2          # (2.0, 4.0)

    proj = _read_csv(report["outputs"]["projection_curves"])
    assert len(proj) == n_crit * n_theta
    # Trunk is the closed-form criterion; the joints are numeric.
    kinds = {r["criterion"]: r["kind"] for r in proj}
    assert kinds["trunk_inclination"] == "closed_form"
    assert kinds["elbow_flexion"] == "numeric"

    noise = _read_csv(report["outputs"]["noise_propagation"])
    assert len(noise) == n_crit * n_theta * n_sigma
    assert {r["sigma"] for r in noise} == {"2.0", "4.0"}

    dec = _read_csv(report["outputs"]["decidability"])
    assert len(dec) == n_crit * n_theta * n_sigma
    assert {r["verdict"] for r in dec} <= {"decidable", "unreliable"}

    # Exact column headers, so a schema change is caught here.
    assert _header(report["outputs"]["projection_curves"]) == _PROJECTION_HEADER
    assert _header(report["outputs"]["noise_propagation"]) == _NOISE_HEADER
    assert _header(report["outputs"]["decidability"]) == _DECIDABILITY_HEADER


def test_removed_artifacts_are_not_written(tmp_path):
    # The methodology revision dropped these two; the reporter must never emit
    # them, in the out dir or the figures subdir.
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
    # Building the figures must not need a display: the Agg backend is selected.
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
    # The figure paths are logged in the report outputs and reproducible.
    assert "decidability_figure" in report["outputs"]
    for name in ("projection_curves.png", "spread_vs_theta.png",
                 "decidability_map.png"):
        assert os.path.getsize(os.path.join(fig_dir, name)) > 0
    # Missing events.csv: the placeholder is written, yet every figure is
    # still produced (the figures do not depend on E3).
    ev = _read_json(os.path.join(out_dir, "event_error.json"))
    assert ev["placeholder"] is True

