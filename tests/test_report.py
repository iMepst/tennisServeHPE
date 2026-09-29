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
                indicators: List[Dict[str, Any]]) -> None:
    clip_dir = os.path.join(root, clip)
    os.makedirs(clip_dir)
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
    _write_clip(root, "serve_a", "frontal", True, True,
                _indicators("inside", "unavailable", "outside", "inside"))
    _write_clip(root, "serve_b", "sagittal", True, False,
                _indicators("unavailable", "outside", "unavailable",
                            "unavailable"))

    report = build_report(root, os.path.join(root, "_report"),
                          make_figure=False)
    assert report["n_clips"] == 2

    ind = {(r["clip"], r["criterion"]): r
           for r in _read_csv(report["outputs"]["indicators_csv"])}
    assert len(ind) == 8
    trunk = _RULE_BY_ID["trunk_inclination"]
    assert float(ind[("serve_a", "trunk_inclination")]["band_lo"]) == trunk.lo
    assert float(ind[("serve_a", "trunk_inclination")]["band_hi"]) == trunk.hi
    assert ind[("serve_b", "front_knee_flexion")]["band_hi"] == ""
    assert ind[("serve_b", "front_knee_flexion")]["band_kind"] == "lower_bound"

    assert set(report["outputs"]) == {"indicators_csv"}

def test_key_frame_candidates_need_both_events(tmp_path) -> None:
    root = str(tmp_path)
    _write_clip(root, "serve_a", "frontal", True, True,
                _indicators("inside", "unavailable", "inside", "inside"))
    _write_clip(root, "serve_b", "sagittal", True, False,
                _indicators("unavailable", "inside", "unavailable",
                            "unavailable"))
    open(os.path.join(root, "serve_a", "key_frames.png"), "w").close()

    report = build_report(root, os.path.join(root, "_report"),
                          make_figure=False)
    assert report["key_frame_candidates"] == [
        os.path.join(root, "serve_a", "key_frames.png")]
