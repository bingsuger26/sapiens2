"""Utility to plot per-frame action metrics (velocity ratio, wrist velocities,
state transitions) as a single summary figure for a video clip."""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# State → color mapping (consistent with HUD colours)
_STATE_COLORS = {
    "idle":     "#C8C8C8",
    "raising":  "#FFC800",
    "pointing": "#00FF00",
    "lowering": "#FF0000",
}


def plot_action_metrics(
    metrics: List[Dict],
    output_path: str,
    ratio_thr: float = 5.0,
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
            state       : str   ("idle" / "raising" / "pointing" / "lowering")
            pos_diff    : float or None
    output_path : str
        Where to save the PNG.
    ratio_thr : float
        Horizontal threshold line on the ratio subplot.
    title : str, optional
        Figure suptitle (defaults to basename of output_path).
    """
    if not metrics:
        return

    frames    = np.array([m["frame"] for m in metrics])
    ratios    = np.array([m["vel_ratio"] for m in metrics])
    vel_left  = np.array([m["vel_left"] for m in metrics])
    vel_right = np.array([m["vel_right"] for m in metrics])
    states    = [m["state"] for m in metrics]
    pos_diffs = np.array([m.get("pos_diff") or 0.0 for m in metrics])

    state_colors = [_STATE_COLORS.get(s, "#C8C8C8") for s in states]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"hspace": 0.15})

    # --- Panel 1: Velocity Ratio + state background ---
    ax = axes[0]
    # Draw state background bands
    _draw_state_bands(ax, frames, states)
    ax.plot(frames, ratios, color="royalblue", linewidth=1.0, label="vel_ratio")
    ax.axhline(y=ratio_thr, color="red", linestyle="--", linewidth=0.8,
               label=f"threshold={ratio_thr}")
    ax.set_ylabel("Velocity Ratio")
    ax.set_ylim(bottom=0, top=max(ratios.max() * 1.2, ratio_thr * 1.5))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Left / Right wrist velocities ---
    ax = axes[1]
    _draw_state_bands(ax, frames, states)
    ax.plot(frames, vel_left, color="dodgerblue", linewidth=1.0, label="vel_left")
    ax.plot(frames, vel_right, color="orangered", linewidth=1.0, label="vel_right")
    ax.set_ylabel("Wrist Velocity")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Position diff ---
    ax = axes[2]
    _draw_state_bands(ax, frames, states)
    ax.plot(frames, pos_diffs, color="mediumseagreen", linewidth=1.0, label="pos_diff")
    ax.set_ylabel("Position Diff")
    ax.set_xlabel("Frame")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Legend for state colors ---
    patches = [mpatches.Patch(color=c, label=s, alpha=0.25)
               for s, c in _STATE_COLORS.items()]
    fig.legend(handles=patches, loc="upper left", ncol=4, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.01, 0.99))

    if title is None:
        title = os.path.splitext(os.path.basename(output_path))[0]
    fig.suptitle(title, fontsize=11, y=1.01)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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
