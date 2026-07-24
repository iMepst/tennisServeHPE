"""Stage 2 diagnostic plotting: the visual sanity checks for gating and
filtering.

Two checks live here:

- ``plot_raw_vs_gated`` (Stage 2a): per landmark, the visibility trace with the
  gating threshold and the masked spans shaded -- grey for undetected, red for
  low visibility. Gating should fire where the Stage 1 overlay looked
  unreliable and nowhere in the serve itself.
- ``plot_raw_vs_filtered`` (Stage 2b): per landmark, one coordinate channel
  before filtering versus one or more filtered variants, with unreliable spans
  shaded. This is the plot the filter type/cut-off decision is made from
  (methodology 5.2).

Uses the Agg backend and writes a PNG; no interactive display.
"""

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
        variants: Sequence[Tuple[str, List[ProcessedFrame]]],
        landmark_names: Sequence[str], coord: str, path: str,
        title: Optional[str] = None,
        time_window: Optional[Tuple[float, float]] = None) -> str:
    """Overlay the pre-filter signal and filtered variants for one channel.

    ``pre_filter`` is the interpolated (un-smoothed) series; ``variants`` maps
    a label (e.g. a candidate cut-off) to a filtered series. Unreliable spans
    are shaded so the reader can see where no variant should be trusted.
    ``time_window`` optionally zooms the x-axis (and auto-scales y to it) so a
    brief event like the serve is not lost inside a long clip.
    """
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
        for label, frames in variants:
            vals = [_coord(f.samples[lm_id], coord) for f in frames]
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


def _velocity(frames: List[ProcessedFrame], lm_id: int, coord: str,
              fps: float) -> List[float]:
    """Central-difference velocity of one channel, NaN across gaps/edges."""
    vals = [getattr(f.samples[lm_id], coord) for f in frames]
    reli = [f.samples[lm_id].reliable for f in frames]
    n = len(vals)
    out = [float("nan")] * n
    for i in range(1, n - 1):
        a, b = vals[i - 1], vals[i + 1]
        if a is not None and b is not None and reli[i - 1] and reli[i + 1]:
            out[i] = (b - a) * fps / 2.0
    return out


def plot_velocity_compare(
        pre_filter: List[ProcessedFrame],
        variants: Sequence[Tuple[str, List[ProcessedFrame]]],
        landmark_names: Sequence[str], coord: str, fps: float, path: str,
        title: Optional[str] = None,
        time_window: Optional[Tuple[float, float]] = None) -> str:
    """Filter-selection diagnostic: velocity of pre-filter vs variants.

    Velocity (central difference of ``coord``, units/s) is where a low-pass
    cut-off actually shows: the pre-filter trace exposes per-frame jitter, and
    a lower cut-off trades a rounded peak for a cleaner signal. This is the
    discriminating view for the filter decision (methodology 5.2).

    Scope note: this is a *coordinate*-velocity diagnostic used only to choose
    the cut-off. It is not the joint angular velocity that Stage 2c computes
    and persists (§5.3), and nothing here is written to disk beyond the plot.
    """
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
        raw = _velocity(pre_filter, lm_id, coord, fps)
        ax.plot(times, raw, color="0.6", lw=0.8, label="pre-filter")
        for label, frames in variants:
            ax.plot(times, _velocity(frames, lm_id, coord, fps),
                    lw=1.2, label=label)
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        _shade_unreliable(ax, pre_filter, lm_id, times, pad)
        ax.set_ylabel(f"d{coord}/dt")
        ax.set_title(lm_name, fontsize=9, loc="left")
        if time_window is not None:
            ax.set_xlim(*time_window)
            vis = [v for t, v in zip(times, raw)
                   if _in_window(t) and v == v]
            if vis:
                lo, hi = min(vis), max(vis)
                margin = 0.1 * (hi - lo) + 1e-6
                ax.set_ylim(lo - margin, hi + margin)

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.append(Patch(color=_UNRELIABLE_COLOR, alpha=0.5))
    labels.append("unreliable")
    axes[0][0].legend(handles, labels, loc="best", fontsize=7,
                      ncol=len(labels))
    axes[-1][0].set_xlabel("time (s)")
    fig.suptitle(
        title or f"Stage 2b filter-selection diagnostic: d{coord}/dt")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
