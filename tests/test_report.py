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
