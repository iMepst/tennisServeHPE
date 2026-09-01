"""Aggregate per-clip results into Results-chapter tables and figures.

Post-hoc reporter. Walks results/<clip>/result.json,
joins each criterion against its reference band (rules.py), and writes the
chapter-ready artifacts into results/_report/:

- indicators.csv       one row per (clip, criterion): status, angle, band
- angles_vs_bands.png  (optional) measured angle per clip against each band

The chapter tables are hand-written from indicators.csv.
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

# Presentation-only cap for the open (lower-bound) knee zone: an anatomical
# plausibility bound (heel-to-buttock maximum flexion), not a decision
# threshold. It bounds the shaded fill so it does not run to the shoulder-
# driven axis top; the rule stays one-sided with no upper bound.
KNEE_PLAUSIBILITY_CAP_DEG = 150.0


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

def _band_bounds(rule: Rule) -> Dict[str, Optional[float]]:
    """The band bounds a criterion is judged against.

    Two-sided rules use [lo, hi]; the one-sided knee uses only the lower
    bound (hi left None), matching evaluate() in rules.py.
    """
    if rule.band_kind == "lower_bound":
        return {"band_lo": rule.lo, "band_hi": None}
    return {"band_lo": rule.lo, "band_hi": rule.hi}


def indicator_rows(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Long-form rows, one per (clip, criterion), joined with the bands."""
    rows: List[Dict[str, Any]] = []
    for clip in clips:
        params = clip.get("clip_params", {})
        events = clip.get("key_events", {})
        for ind in clip.get("indicators", []):
            rule = _RULE_BY_ID.get(ind["criterion"])
            bounds = (_band_bounds(rule) if rule
                      else {"band_lo": None, "band_hi": None})
            rows.append({
                "clip": clip.get("clip"),
                "camera_plane": params.get("camera_plane"),
                "view_direction": params.get("view_direction"),
                "criterion": ind["criterion"],
                "status": ind["status"],
                "angle": ind.get("angle"),
                "band_lo": bounds["band_lo"],
                "band_hi": bounds["band_hi"],
                "band_kind": rule.band_kind if rule else None,
                "detail": ind.get("detail"),
                "trophy_locatable": events.get("trophy_locatable"),
                "impact_locatable": events.get("impact_locatable"),
            })
    return rows


def write_csv(path: str, header: List[str],
              rows: List[Dict[str, Any]]) -> str:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path

def key_frame_candidates(clips: List[Dict[str, Any]],
                         results_root: str) -> List[str]:
    """key_frames.png paths for clips with both events located.

    Both events means the still shows a trophy and an impact panel. Returned
    for the author to pick one frontal and one sagittal from.
    """
    out: List[str] = []
    for clip in clips:
        events = clip.get("key_events", {})
        if not (events.get("trophy_locatable") and
                events.get("impact_locatable")):
            continue
        png = os.path.join(results_root, str(clip.get("clip")),
                           "key_frames.png")
        if os.path.isfile(png):
            out.append(png)
    return out

def plot_angles_vs_bands(clips: List[Dict[str, Any]], path: str) -> str:
    """Measured angle per clip against each criterion's band (one figure).

    Same matplotlib-Agg pattern as plotting.py. Each criterion is a column;
    its band is shaded (a lower-bound band shades upward from its threshold),
    and every clip's measured angle is a point. Criteria a clip could not
    read contribute no point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    order = list(_CRITERION_LABEL)
    fig, ax = plt.subplots(figsize=(9, 5))
    lower_bound_x = []
    for x, criterion in enumerate(order):
        rule = _RULE_BY_ID[criterion]
        if rule.band_kind != "lower_bound":
            ax.add_patch(plt.Rectangle((x - 0.3, rule.lo), 0.6,
                                       rule.hi - rule.lo,
                                       color="tab:green", alpha=0.15, lw=0))
            ax.hlines(rule.mean, x - 0.3, x + 0.3, color="tab:green", lw=1.0)
        else:
            lower_bound_x.append((x, rule))
        for clip in clips:
            by_crit = {i["criterion"]: i for i in clip.get("indicators", [])}
            ind = by_crit.get(criterion)
            if ind is None or ind.get("angle") is None:
                continue
            color = "tab:blue" if ind["status"] == "inside" else "tab:red"
            ax.plot(x, ind["angle"], "o", color=color, alpha=0.8)
    # One-sided (lower-bound) criteria: everything above the threshold is
    # compliant, so shade upward from the threshold to the top of the plot
    # area and mark only the lower edge, with a distinct solid line that
    # sets it apart from the two-sided bands' edges.
    top = ax.get_ylim()[1]
    ax.autoscale(False)
    for x, rule in lower_bound_x:
        ax.add_patch(plt.Rectangle((x - 0.3, rule.lo), 0.6, top - rule.lo,
                                   color="tab:green", alpha=0.15, lw=0))
        ax.hlines(rule.lo, x - 0.3, x + 0.3, color="tab:green", lw=2.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_CRITERION_LABEL[c] for c in order])
    ax.set_ylabel("angle (deg)")
    ax.set_title("Measured angles against reference bands")
    ax.legend(handles=[
        Patch(color="tab:green", alpha=0.15,
              label="reference band (mean ± SD)"),
        plt.Line2D([], [], marker="o", ls="", color="tab:blue",
                   label="inside"),
        plt.Line2D([], [], marker="o", ls="", color="tab:red",
                   label="outside"),
    ], loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

_INDICATOR_HEADER = [
    "clip", "camera_plane", "view_direction", "criterion", "status", "angle",
    "band_lo", "band_hi", "band_kind", "detail",
    "trophy_locatable", "impact_locatable",
]

def build_report(results_root: str, out_dir: str,
                 make_figure: bool = True) -> Dict[str, Any]:
    """Aggregate every result.json under results_root into out_dir.

    Returns the written paths and the key-frame figure candidates.
    """
    result_paths = find_result_jsons(results_root)
    if not result_paths:
        raise FileNotFoundError(
            f"no {RESULT_JSON} found under {results_root!r} "
            "(run some clips first)")
    clips = [load_clip(p) for p in result_paths]
    os.makedirs(out_dir, exist_ok=True)

    ind_rows = indicator_rows(clips)
    outputs = {
        "indicators_csv": write_csv(
            os.path.join(out_dir, "indicators.csv"),
            _INDICATOR_HEADER, ind_rows),
    }
    if make_figure:
        outputs["angles_figure"] = plot_angles_vs_bands(
            clips, os.path.join(out_dir, "angles_vs_bands.png"))

    candidates = key_frame_candidates(clips, results_root)
    return {"n_clips": len(clips), "outputs": outputs,
            "key_frame_candidates": candidates}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Aggregate per-clip result.json files into Results-"
                    "chapter tables and figures.")
    parser.add_argument("--results", default="results",
                        help="results root to scan (default: results)")
    parser.add_argument("--out", default=None,
                        help="output dir (default: <results>/_report)")
    parser.add_argument("--no-figure", dest="make_figure",
                        action="store_false",
                        help="skip the angles-vs-bands figure")
    args = parser.parse_args()
    out_dir = args.out or os.path.join(args.results, DEFAULT_REPORT_DIR)
    report = build_report(args.results, out_dir, make_figure=args.make_figure)

    logger.info("Aggregated %d clip(s) into %s", report["n_clips"], out_dir)
    for name, path in report["outputs"].items():
        logger.info("  %-18s %s", name, path)
    if report["key_frame_candidates"]:
        logger.info("Key-frame figure candidates (both events located):")
        for png in report["key_frame_candidates"]:
            logger.info("  %s", png)
    else:
        logger.info(
            "No key-frame candidates (no clip had both events located)")


if __name__ == "__main__":
    main()
