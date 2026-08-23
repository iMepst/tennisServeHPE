from typing import Any, List, Optional, Sequence, Tuple

from .gating import GatedFrame, MASK_UNDETECTED
from .interpolation import ProcessedFrame
from .landmarks import LANDMARK_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

DEFAULT_QC_LANDMARKS = [
    "left_elbow", "left_wrist", "right_elbow", "right_wrist",
]

_UNDETECTED_COLOR = "0.6"
_LOW_VIS_COLOR = "tab:red"


def _typical_dt(times: List[float]) -> float:
    if len(times) >= 2:
        span = times[-1] - times[0]
        if span > 0:
            return span / (len(times) - 1)
    return 0.04


def _shade_invalid(ax: Any, gated: List[GatedFrame], lm_id: int,
                   times: List[float], pad: float) -> None:
    """Shade contiguous invalid spans, breaking whenever the reason changes."""
    start = None
    reason = ""
    for i, g in enumerate(gated):
        s = g.samples[lm_id]
        if not s.valid:
            if start is None:
                start, reason = i, s.mask_reason
            elif s.mask_reason != reason:
                _span(ax, times, start, i - 1, reason, pad)
                start, reason = i, s.mask_reason
        elif start is not None:
            _span(ax, times, start, i - 1, reason, pad)
            start = None
    if start is not None:
        _span(ax, times, start, len(gated) - 1, reason, pad)


def _span(ax: Any, times: List[float], start: int, end: int,
          reason: str, pad: float) -> None:
    color = _UNDETECTED_COLOR if reason == MASK_UNDETECTED else _LOW_VIS_COLOR
    ax.axvspan(times[start] - pad, times[end] + pad, color=color, alpha=0.3,
               linewidth=0)


def plot_raw_vs_gated(gated: List[GatedFrame], landmark_names: Sequence[str],
                      visibility_threshold: float, path: str) -> str:
    name_to_id = {n: i for i, n in enumerate(LANDMARK_NAMES)}
    times = [g.time_s for g in gated]
    pad = _typical_dt(times) / 2.0

    n = len(landmark_names)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.1 * n),
                             sharex=True, squeeze=False)
    for row, lm_name in enumerate(landmark_names):
        ax = axes[row][0]
        lm_id = name_to_id[lm_name]
        vis = [g.samples[lm_id].visibility
               if g.samples[lm_id].visibility is not None else float("nan")
               for g in gated]
        ax.plot(times, vis, color="tab:blue", lw=1.0)
        ax.axhline(visibility_threshold, color="k", ls="--", lw=0.8)
        _shade_invalid(ax, gated, lm_id, times, pad)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("visibility")
        ax.set_title(lm_name, fontsize=9, loc="left")

    legend = [
        plt.Line2D([], [], color="tab:blue", lw=1.0, label="visibility"),
        plt.Line2D([], [], color="k", ls="--", lw=0.8,
                   label=f"threshold {visibility_threshold:g}"),
        Patch(color=_UNDETECTED_COLOR, alpha=0.3, label="undetected"),
        Patch(color=_LOW_VIS_COLOR, alpha=0.3, label="low visibility"),
    ]
    axes[0][0].legend(handles=legend, loc="lower left", fontsize=7, ncol=2)
    axes[-1][0].set_xlabel("time (s)")
    fig.suptitle("Stage 2a raw-vs-gated: visibility and masked spans")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


_UNRELIABLE_COLOR = "0.8"


def _coord(sample: Any, coord: str) -> float:
    v = getattr(sample, coord)
    return float("nan") if v is None else float(v)


def _shade_unreliable(ax: Any, frames: List[ProcessedFrame], lm_id: int,
                      times: List[float], pad: float) -> None:
    """Shade contiguous spans where the sample is unreliable (a long gap)."""
    start = None
    for i, f in enumerate(frames):
        if not f.samples[lm_id].reliable:
            if start is None:
                start = i
        elif start is not None:
            ax.axvspan(times[start] - pad, times[i - 1] + pad,
                       color=_UNRELIABLE_COLOR, alpha=0.5, linewidth=0)
            start = None
    if start is not None:
        ax.axvspan(times[start] - pad, times[-1] + pad,
                   color=_UNRELIABLE_COLOR, alpha=0.5, linewidth=0)


def plot_raw_vs_filtered(
        pre_filter: List[ProcessedFrame],
        filtered: List[ProcessedFrame], label: str,
        landmark_names: Sequence[str], coord: str, path: str,
        title: Optional[str] = None,
        time_window: Optional[Tuple[float, float]] = None) -> str:
    """Overlay the pre-filter and filtered signals for one channel."""
    name_to_id = {n: i for i, n in enumerate(LANDMARK_NAMES)}
    times = [f.time_s for f in pre_filter]
    pad = _typical_dt(times) / 2.0

    def _in_window(t: float) -> bool:
        return time_window is None or time_window[0] <= t <= time_window[1]

    n = len(landmark_names)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n),
                             sharex=True, squeeze=False)
    for row, lm_name in enumerate(landmark_names):
        ax = axes[row][0]
        lm_id = name_to_id[lm_name]
        raw = [_coord(f.samples[lm_id], coord) for f in pre_filter]
        ax.plot(times, raw, color="0.6", lw=0.8, label="pre-filter")
        vals = [_coord(f.samples[lm_id], coord) for f in filtered]
        ax.plot(times, vals, lw=1.2, label=label)
        _shade_unreliable(ax, pre_filter, lm_id, times, pad)
        ax.set_ylabel(coord)
        ax.set_title(lm_name, fontsize=9, loc="left")
        if time_window is not None:
            ax.set_xlim(*time_window)
            vis = [v for t, v in zip(times, raw)
                   if _in_window(t) and v == v]  # in window, non-nan
            if vis:
                lo, hi = min(vis), max(vis)
                margin = 0.05 * (hi - lo) + 1e-6
                ax.set_ylim(lo - margin, hi + margin)

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(Patch(color=_UNRELIABLE_COLOR, alpha=0.5))
    labels.append("unreliable")
    axes[0][0].legend(handles, labels, loc="best", fontsize=7,
                      ncol=len(labels))
    axes[-1][0].set_xlabel("time (s)")
    fig.suptitle(title or f"Stage 2b raw-vs-filtered: {coord}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
