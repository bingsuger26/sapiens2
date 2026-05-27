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
    "idle":             "#C8C8C8",
    "raising":          "#FFC800",
    "pointing":         "#00FF00",
    "lowering":         "#FF0000",
    "bbox_change":      "#FF00FF",
    "shoulder_change":  "#FF00FF",
    "no_bbox":          "#808080",
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
    ax.plot(frames, bbox_vels, color="darkorchid", linewidth=1.0, label="shoulder_vel")
    ax.axhline(y=bbox_vel_thr, color="magenta", linestyle="--", linewidth=0.8,
               label=f"shoulder_vel_thr={bbox_vel_thr}")
    ax.set_ylabel("Shoulder Velocity (px/frame)")
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
        "idle":             0,
        "raising":          1,
        "pointing":         2,
        "lowering":         3,
        "bbox_change":      4,
        "shoulder_change":  4,
        "no_bbox":          5,
    }
    state_names = ["idle", "raising", "pointing", "lowering", "shoulder_change", "no_bbox"]

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
        if side and st not in ("idle", "bbox_change", "shoulder_change", "no_bbox"):
            side_label = "L" if side == "left" else "R"
            mid_x = fs + width / 2.0
            ax.text(mid_x, y, side_label, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="black")

    ax.set_yticks(list(range(len(state_names))))
    ax.set_yticklabels(state_names, fontsize=8)
    ax.set_ylim(-0.5, len(state_names) - 0.5)


# ---------------------------------------------------------------------------
# Summary distribution plot (aggregated across all clips)
# ---------------------------------------------------------------------------

def plot_summary_distribution(
    all_metrics: List[Dict],
    output_path: str,
    ratio_thr: float = 2.8,
    vel_min: float = 0.025,
    title: Optional[str] = None,
):
    """Generate a summary plot showing arm velocity and ratio distributions
    across ALL processed clips.

    Parameters
    ----------
    all_metrics : list[dict]
        Concatenated per-frame metrics from all clips.  Each dict contains:
            vel_ratio, vel_left, vel_right, state, active_side, etc.
    output_path : str
        Path to save the summary figure.
    ratio_thr : float
        Ratio threshold line to draw on the distribution.
    vel_min : float
        Velocity minimum threshold line.
    title : str, optional
    """
    if not all_metrics:
        return

    ratios = np.array([m["vel_ratio"] for m in all_metrics])
    vel_left = np.array([m["vel_left"] for m in all_metrics])
    vel_right = np.array([m["vel_right"] for m in all_metrics])
    vel_max = np.maximum(vel_left, vel_right)
    states = [m["state"] for m in all_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Panel (0,0): Ratio histogram ----
    ax = axes[0, 0]
    ax.hist(ratios, bins=80, color="royalblue", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.axvline(x=ratio_thr, color="red", linestyle="--", linewidth=1.5,
               label=f"ratio_thr={ratio_thr}")
    ax.set_xlabel("Velocity Ratio (faster / slower)")
    ax.set_ylabel("Frame Count")
    ax.set_title("Velocity Ratio Distribution")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Panel (0,1): Max arm velocity histogram ----
    ax = axes[0, 1]
    ax.hist(vel_max, bins=80, color="darkorange", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.axvline(x=vel_min, color="red", linestyle="--", linewidth=1.5,
               label=f"vel_min={vel_min}")
    ax.set_xlabel("Max Hand Velocity (normalised)")
    ax.set_ylabel("Frame Count")
    ax.set_title("Max Hand Velocity Distribution")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Panel (1,0): Left vs Right velocity scatter (colored by state) ----
    ax = axes[1, 0]
    for state_name, color in _STATE_COLORS.items():
        mask = [s == state_name for s in states]
        if any(mask):
            idx = np.where(mask)[0]
            ax.scatter(vel_left[idx], vel_right[idx], c=color, alpha=0.3, s=8,
                       label=state_name, edgecolors="none")
    ax.set_xlabel("Left Hand Velocity")
    ax.set_ylabel("Right Hand Velocity")
    ax.set_title("Left vs Right Hand Velocity (by state)")
    ax.legend(loc="upper right", fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3)
    # Draw identity line
    lim = max(vel_left.max(), vel_right.max()) * 1.1 if vel_left.max() > 0 else 0.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.5, alpha=0.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # ---- Panel (1,1): Ratio vs Max velocity scatter (colored by state) ----
    ax = axes[1, 1]
    for state_name, color in _STATE_COLORS.items():
        mask = [s == state_name for s in states]
        if any(mask):
            idx = np.where(mask)[0]
            ax.scatter(vel_max[idx], ratios[idx], c=color, alpha=0.3, s=8,
                       label=state_name, edgecolors="none")
    ax.axhline(y=ratio_thr, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(x=vel_min, color="green", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Max Hand Velocity")
    ax.set_ylabel("Velocity Ratio")
    ax.set_title("Ratio vs Max Velocity (by state)")
    ax.legend(loc="upper right", fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3)

    if title is None:
        title = "Summary: Arm Velocity & Ratio Distribution"
    fig.suptitle(title, fontsize=12, y=1.01)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_metrics] wrote summary distribution: {output_path}")
