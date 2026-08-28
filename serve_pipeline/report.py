"""Aggregate per-clip results into Results-chapter tables and figures.

Post-hoc reporter (not a pipeline stage). Walks results/<clip>/result.json,
joins each criterion against its reference band (rules.py), and writes the
chapter-ready artifacts into results/_report/:

- indicators.csv       one row per (clip, criterion): status, angle, band
- angles_vs_bands.png  (optional) measured angle per clip against each band

The chapter tables are hand-written from indicators.csv.

The raw per-clip result.json is the single source of truth; the diagnostic
artifacts (overlay.mp4, contact sheets, QC plots) are deliberately not read
here; they stay QC-only and never enter the thesis.
"""

import argparse
import csv
import glob
import logging
import os
from typing import Any, Dict, List, Optional

from .persistence import read_metadata
from .rules import RULES, Rule

logger = logging.getLogger(__name__)

RESULT_JSON = "result.json"
DEFAULT_REPORT_DIR = "_report"

# Display labels and column order for the compact tables/figure.
_CRITERION_LABEL = {
    "trunk_inclination": "Trunk",
    "front_knee_flexion": "Knee",
    "elbow_flexion": "Elbow",
    "shoulder_elevation": "Shoulder",
}
_RULE_BY_ID: Dict[str, Rule] = {r.id: r for r in RULES}


def find_result_jsons(results_root: str) -> List[str]:
    """Every results/<clip>/result.json, sorted, skipping the report dir."""
    pattern = os.path.join(results_root, "*", RESULT_JSON)
    return sorted(
        p for p in glob.glob(pattern)
        if os.path.basename(os.path.dirname(p)) != DEFAULT_REPORT_DIR)


def _extraction_stats(result_path: str) -> Dict[str, Any]:
    """Detection statistics for the clip, or empty if unavailable.

    Read from <clip>/stage1/meta.json; missing meta is tolerated (detection
    columns stay blank) so the reporter runs on any results tree.
    """
    clip_dir = os.path.dirname(result_path)
    meta_path = os.path.join(clip_dir, "stage1", "meta.json")
    if not os.path.isfile(meta_path):
        return {}
    stats = read_metadata(meta_path).get("statistics", {})
    return stats if isinstance(stats, dict) else {}


def load_clip(result_path: str) -> Dict[str, Any]:
    """One clip's result.json joined with its detection statistics."""
    data = read_metadata(result_path)
    data["_stats"] = _extraction_stats(result_path)
    return data