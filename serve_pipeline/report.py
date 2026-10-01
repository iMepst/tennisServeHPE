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
from typing import Any, Dict, List

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
    pattern = os.path.join(results_root, "*", RESULT_JSON)
    return sorted(
        p for p in glob.glob(pattern)
        if os.path.basename(os.path.dirname(p)) != DEFAULT_REPORT_DIR)


def indicator_rows(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for clip in clips:
        params = clip["clip_params"]
        events = clip["key_events"]
        for ind in clip["indicators"]:
            rule = _RULE_BY_ID[ind["criterion"]]
            rows.append({
                "clip": clip["clip"],
                "camera_plane": params["camera_plane"],
                "view_direction": params["view_direction"],
                "criterion": ind["criterion"],
                "status": ind["status"],
                "angle": ind["angle"],
                "band_lo": rule.lo,
                "band_hi": (None if rule.band_kind == "lower_bound"
                            else rule.hi),
                "band_kind": rule.band_kind,
                "detail": ind["detail"],
                "trophy_locatable": events["trophy_frame"] is not None,
                "impact_locatable": events["impact_frame"] is not None,
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
    out: List[str] = []
    for clip in clips:
        events = clip["key_events"]
        if events["trophy_frame"] is None or events["impact_frame"] is None:
            continue
        png = os.path.join(results_root, clip["clip"], "key_frames.png")
        if os.path.isfile(png):
            out.append(png)
    return out


def plot_angles_vs_bands(clips: List[Dict[str, Any]], path: str,
                         knee_cap_deg: float = KNEE_PLAUSIBILITY_CAP_DEG
                         ) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    order = list(_CRITERION_LABEL)
    fig, ax = plt.subplots(figsize=(9, 5))
    one_sided = []
    for x, criterion in enumerate(order):
        rule = _RULE_BY_ID[criterion]
        if rule.band_kind != "lower_bound":
            ax.add_patch(plt.Rectangle((x - 0.3, rule.lo), 0.6,
                                       rule.hi - rule.lo,
                                       color="tab:green", alpha=0.15, lw=0))
            ax.hlines(rule.mean, x - 0.3, x + 0.3, color="tab:green", lw=1.0)
        else:
            one_sided.append((x, rule))
        for clip in clips:
            by_crit = {i["criterion"]: i for i in clip["indicators"]}
            ind = by_crit[criterion]
            if ind["angle"] is None:
                continue
            color = "tab:blue" if ind["status"] == "inside" else "tab:red"
            ax.plot(x, ind["angle"], "o", color=color, alpha=0.8)
    # One-sided criteria: shade upward from the threshold to the
    # plausibility cap.
    for x, rule in one_sided:
        ax.add_patch(plt.Rectangle((x - 0.3, rule.lo), 0.6,
                                   knee_cap_deg - rule.lo,
                                   color="tab:green", alpha=0.15, lw=0))
        ax.hlines(rule.lo, x - 0.3, x + 0.3, color="tab:green", lw=2.0)
        ax.hlines(knee_cap_deg, x - 0.3, x + 0.3, color="tab:green",
                  lw=1.0, ls="--")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_CRITERION_LABEL[c] for c in order])
    ax.set_ylabel("angle (deg)")
    ax.set_title("Measured angles against reference bands")
    ax.legend(handles=[
        Patch(color="tab:green", alpha=0.15, label="reference band"),
        plt.Line2D([], [], color="tab:green", lw=1.0,
                   label="reference mean"),
        plt.Line2D([], [], color="tab:green", lw=2.0,
                   label="one-sided threshold"),
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
    result_paths = find_result_jsons(results_root)
    if not result_paths:
        raise FileNotFoundError(
            f"no {RESULT_JSON} found under {results_root!r} "
            "(run some clips first)")
    clips = [read_metadata(p) for p in result_paths]
    os.makedirs(out_dir, exist_ok=True)

    rows = indicator_rows(clips)
    outputs = {
        "indicators_csv": write_csv(
            os.path.join(out_dir, "indicators.csv"),
            _INDICATOR_HEADER, rows),
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
