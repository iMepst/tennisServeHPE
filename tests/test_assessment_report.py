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

