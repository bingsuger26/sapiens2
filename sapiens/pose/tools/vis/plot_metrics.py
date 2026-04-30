"""Utility to plot per-frame action metrics (velocity ratio, wrist velocities,
bbox velocity, state transitions) as a multi-panel summary figure."""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# State → color mapping (consistent with HUD colours)
_STATE_COLORS = {
    "idle":        "#C8C8C8",
    "raising":     "#FFC800",
    "pointing":    "#00FF00",
    "lowering":    "#FF0000",
    "bbox_change": "#FF00FF",
    "no_bbox":     "#808080",
}


def plot_action_metrics(
    metrics: List[Dict],
    output_path: str,
    ratio_thr: float = 5.0,
    bbox_vel_thr: float = 15.0,
    title: Optional[str] = None,
):
    """Generate a multi-panel summary plot and save to *output_path*.

    Parameters
    ----------
    metrics : list[dict]
        One dict per frame, each containing:
            frame       : int
            vel_ratio   : float
            vel_left    : float
            vel_right   : float
            state       : str
            pos_diff    : float or None
            bbox_vel    : float
            active_side : str or None  ("left" / "right" / None)
    output_path : str
    ratio_thr : float
    bbox_vel_thr : float
    title : str, optional
    """
    if not metrics:
        return

    frames     = np.array([m["frame"] for m in metrics])
    ratios     = np.array([m["vel_ratio"] for m in metrics])
    vel_left   = np.array([m["vel_left"] for m in metrics])
    vel_right  = np.array([m["vel_right"] for m in metrics])
    states     = [m["state"] for m in metrics]
    pos_diffs  = np.array([m.get("pos_diff") or 0.0 for m in metrics])
    bbox_vels  = np.array([m.get("bbox_vel", 0.0) for m in metrics])
    sides      = [m.get("active_side") for m in metrics]

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True,
                             gridspec_kw={"hspace": 0.18})

    # ---- Panel 1: Velocity Ratio + state background ----
    ax = axes[0]
    _draw_state_bands(ax, frames, states)
    ax.plot(frames, ratios, color="royalblue", linewidth=1.0, label="vel_ratio")
    ax.axhline(y=ratio_thr, color="red", linestyle="--", linewidth=0.8,
               label=f"ratio_thr={ratio_thr}")
    ax.set_ylabel("Velocity Ratio")
    ax.set_ylim(bottom=0, top=max(ratios.max() * 1.2, ratio_thr * 1.5))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Left / Right wrist velocities ----
    ax = axes[1]
    _draw_state_bands(ax, frames, states)
    ax.plot(frames, vel_left, color="dodgerblue", linewidth=1.0, label="vel_left")
    ax.plot(frames, vel_right, color="orangered", linewidth=1.0, label="vel_right")
    ax.set_ylabel("Wrist Velocity")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Position diff ----
    ax = axes[2]
    _draw_state_bands(ax, frames, states)
    ax.plot(frames, pos_diffs, color="mediumseagreen", linewidth=1.0, label="pos_diff")
    ax.set_ylabel("Position Diff")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Bbox velocity ----
    ax = axes[3]
    _draw_state_bands(ax, frames, states)
    ax.plot(frames, bbox_vels, color="darkorchid", linewidth=1.0, label="bbox_vel")
    ax.axhline(y=bbox_vel_thr, color="magenta", linestyle="--", linewidth=0.8,
               label=f"bbox_vel_thr={bbox_vel_thr}")
    ax.set_ylabel("Bbox Velocity (px/frame)")
    ax.set_ylim(bottom=0, top=max(bbox_vels.max() * 1.2, bbox_vel_thr * 1.5) if bbox_vels.max() > 0 else bbox_vel_thr * 2)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 5: State timeline with side annotation ----
    ax = axes[4]
    _draw_state_timeline(ax, frames, states, sides)
    ax.set_ylabel("Action State")
    ax.set_xlabel("Frame")
    ax.grid(True, axis="x", alpha=0.3)

    # ---- Legend for state colors ----
    patches = [mpatches.Patch(color=c, label=s, alpha=0.35)
               for s, c in _STATE_COLORS.items()]
    fig.legend(handles=patches, loc="upper left", ncol=len(_STATE_COLORS), fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.01, 0.99))

    if title is None:
        title = os.path.splitext(os.path.basename(output_path))[0]
    fig.suptitle(title, fontsize=11, y=1.01)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helper drawing functions
# ---------------------------------------------------------------------------

def _draw_state_bands(ax, frames, states):
    """Draw coloured vertical bands behind the plot to indicate action state."""
    if len(frames) < 2:
        return
    prev_state = states[0]
    band_start = frames[0]
    for i in range(1, len(frames)):
        if states[i] != prev_state or i == len(frames) - 1:
            band_end = frames[i]
            color = _STATE_COLORS.get(prev_state, "#C8C8C8")
            ax.axvspan(band_start, band_end, alpha=0.15, color=color, linewidth=0)
            band_start = band_end
            prev_state = states[i]


def _draw_state_timeline(ax, frames, states, sides):
    """Draw a categorical state timeline as colored horizontal bars with side labels.

    Each contiguous run of the same state is drawn as a colored bar.
    The active side ("L" / "R") is annotated at the centre of each bar.
    """
    _state_to_y = {
        "idle":        0,
        "raising":     1,
        "pointing":    2,
        "lowering":    3,
        "bbox_change": 4,
        "no_bbox":     5,
    }
    state_names = list(_state_to_y.keys())

    # Identify contiguous runs
    runs = []  # list of (start_frame, end_frame, state, dominant_side)
    run_start = 0
    for i in range(1, len(frames)):
        if states[i] != states[run_start] or i == len(frames) - 1:
            end_i = i if (states[i] != states[run_start]) else i + 1
            # Determine dominant side in this run
            run_sides = [s for s in sides[run_start:end_i] if s is not None]
            if run_sides:
                from collections import Counter
                dom_side = Counter(run_sides).most_common(1)[0][0]
            else:
                dom_side = None
            runs.append((frames[run_start], frames[min(end_i, len(frames) - 1)],
                         states[run_start], dom_side))
            run_start = i

    bar_height = 0.7
    for (fs, fe, st, side) in runs:
        y = _state_to_y.get(st, 0)
        color = _STATE_COLORS.get(st, "#C8C8C8")
        width = max(fe - fs, 1)
        ax.barh(y, width, left=fs, height=bar_height, color=color, alpha=0.6,
                edgecolor="black", linewidth=0.3)
        # Annotate side
        if side and st not in ("idle", "bbox_change", "no_bbox"):
            side_label = "L" if side == "left" else "R"
            mid_x = fs + width / 2.0
            ax.text(mid_x, y, side_label, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="black")

    ax.set_yticks(list(_state_to_y.values()))
    ax.set_yticklabels(state_names, fontsize=8)
    ax.set_ylim(-0.5, len(state_names) - 0.5)
