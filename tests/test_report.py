import csv
import os
from typing import Any, Dict, List

from serve_pipeline.persistence import write_metadata
from serve_pipeline.report import build_report
from serve_pipeline.rules import RULES

_RULE_BY_ID = {r.id: r for r in RULES}


def _indicators(trunk: str, knee: str, elbow: str,
                shoulder: str) -> List[Dict[str, Any]]:
    def one(cid: str, status: str) -> Dict[str, Any]:
        angle = None if status == "unavailable" else _RULE_BY_ID[cid].mean
        return {"criterion": cid, "status": status, "angle": angle,
                "detail": None}
    return [one("trunk_inclination", trunk), one("front_knee_flexion", knee),
            one("elbow_flexion", elbow), one("shoulder_elevation", shoulder)]


def _write_clip(root: str, clip: str, plane: str, trophy: bool, impact: bool,
                indicators: List[Dict[str, Any]], det_rate: float) -> None:
    clip_dir = os.path.join(root, clip)
    os.makedirs(os.path.join(clip_dir, "stage1"))
    write_metadata(os.path.join(clip_dir, "stage1", "meta.json"),
                   {"statistics": {"detection_rate": det_rate,
                                   "mean_visibility": 0.8,
                                   "frames_processed": 100}})
    write_metadata(os.path.join(clip_dir, "result.json"), {
        "clip": clip,
        "clip_params": {"camera_plane": plane, "view_direction": "front"},
        "key_events": {"trophy_locatable": trophy, "impact_locatable": impact},
        "indicators": indicators,
    })


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))

def test_build_report_aggregates(tmp_path) -> None:
    root = str(tmp_path)
    # Frontal clip: trunk read, knee unavailable (wrong plane).
    _write_clip(root, "serve_a", "frontal", True, True,
                _indicators("inside", "unavailable", "outside", "inside"),
                det_rate=0.95)
    # Sagittal clip: knee read, trunk unavailable; impact not located.
    _write_clip(root, "serve_b", "sagittal", True, False,
                _indicators("unavailable", "outside", "unavailable",
                            "unavailable"),
                det_rate=0.80)

    report = build_report(root, os.path.join(root, "_report"),
                          make_figure=False)
    assert report["n_clips"] == 2

    ind = {(r["clip"], r["criterion"]): r
           for r in _read_csv(report["outputs"]["indicators_csv"])}
    # One row per (clip, criterion): 2 clips x 4 = 8.
    assert len(ind) == 8
    # Band bounds are joined from RULES, not re-typed.
    trunk = _RULE_BY_ID["trunk_inclination"]
    assert float(ind[("serve_a", "trunk_inclination")]["band_lo"]) == trunk.lo
    assert float(ind[("serve_a", "trunk_inclination")]["band_hi"]) == trunk.hi
    # The one-sided knee has no upper bound.
    assert ind[("serve_b", "front_knee_flexion")]["band_hi"] == ""
    assert ind[("serve_b", "front_knee_flexion")]["band_kind"] == "lower_bound"

    # Only indicators.csv is emitted; no LaTeX fragments, no availability CSV.
    assert set(report["outputs"]) == {"indicators_csv"}
