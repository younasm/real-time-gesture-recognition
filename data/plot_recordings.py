# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
data/plot_recordings.py — Plot one recording per gesture to inspect quality.

Shows:
  - x, y, z position over time
  - Doppler velocity over time
  - 3D trajectory
  - Detection rate (% frames with points)

Usage:
  python -m data.plot_recordings
  python -m data.plot_recordings --data_dir data/recordings --gesture swipe_left
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def load_one_per_gesture(data_dir: str) -> dict:
    """Load the FIRST recording found for each gesture label."""
    by_gesture = defaultdict(list)
    for fp in sorted(Path(data_dir).glob("*.npy")):
        label = fp.stem.rsplit("_", 1)[0]
        by_gesture[label].append(fp)

    result = {}
    for label, files in by_gesture.items():
        data = np.load(files[0])
        result[label] = (data, files[0].name)
    return result


def plot_gesture(ax_row, data: np.ndarray, label: str, filename: str) -> None:
    """Plot one recording across a row of 4 subplots."""
    n  = len(data)
    t  = np.arange(n)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    v  = data[:, 3]
    pc = data[:, 4]

    active = np.any(data[:, :3] != 0, axis=1)
    rate   = active.mean()

    # ── Position time series ─────────────────────────────
    ax = ax_row[0]
    ax.plot(t, x, label="x (lateral)", color="tab:blue")
    ax.plot(t, y, label="y (depth)",   color="tab:orange")
    ax.plot(t, z, label="z (elev.)",   color="tab:green")
    ax.set_title(f"{label}\ndetection: {rate:.0%}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Frame"); ax.set_ylabel("metres")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5)

    # ── Velocity time series ──────────────────────────────
    ax = ax_row[1]
    ax.plot(t, v, color="tab:red")
    ax.set_title("Doppler velocity", fontsize=9)
    ax.set_xlabel("Frame"); ax.set_ylabel("m/s")
    ax.axhline(0, color="gray", lw=0.5)
    ax.grid(True, alpha=0.3)

    # ── Point count ───────────────────────────────────────
    ax = ax_row[2]
    ax.bar(t, pc, color="tab:purple", alpha=0.7, width=0.8)
    ax.set_title("Detected point count", fontsize=9)
    ax.set_xlabel("Frame"); ax.set_ylabel("# points")
    ax.grid(True, alpha=0.3, axis="y")

    # ── 3D trajectory ─────────────────────────────────────
    ax = ax_row[3]  # this will be a 3D axes
    if active.any():
        xi, yi, zi = x[active], y[active], z[active]
        sc = ax.scatter(xi, yi, zi,
                        c=np.arange(active.sum()), cmap="viridis",
                        s=20, alpha=0.8)
        ax.plot(xi, yi, zi, color="gray", lw=0.5, alpha=0.5)
        ax.set_xlabel("X (m)", fontsize=7)
        ax.set_ylabel("Y (m)", fontsize=7)
        ax.set_zlabel("Z (m)", fontsize=7)
        ax.set_title("3D trajectory\n(colour = time)", fontsize=9)
    else:
        ax.text(0.5, 0.5, 0.5, "NO DETECTIONS",
                ha="center", va="center",
                fontsize=12, color="red", fontweight="bold")
        ax.set_title("3D trajectory", fontsize=9)


def plot_all(data_dir: str, gesture_filter: str = "") -> None:
    recordings = load_one_per_gesture(data_dir)

    if gesture_filter:
        recordings = {k: v for k, v in recordings.items()
                      if k == gesture_filter}
        if not recordings:
            print(f"Gesture '{gesture_filter}' not found.")
            return

    gestures = sorted(recordings.keys())
    n_g = len(gestures)
    if n_g == 0:
        print("No recordings found."); return

    print(f"Plotting {n_g} gesture(s) from {data_dir}\n")
    out_dir = Path(data_dir) / "plots"
    out_dir.mkdir(exist_ok=True)

    for gesture in gestures:
        data, fname = recordings[gesture]
        active = np.any(data[:, :3] != 0, axis=1).mean()
        print(f"  {gesture:<20}  {fname}  detection={active:.0%}  "
              f"frames={len(data)}  max_disp={np.max(np.abs(data[:,:3])):.3f}m")

        # One figure per gesture — fits any screen size
        fig = plt.figure(figsize=(16, 4))
        fig.suptitle(f"Gesture: {gesture}  |  {fname}", fontsize=12, fontweight="bold")

        axes_row = []
        for col in range(4):
            if col == 3:
                ax = fig.add_subplot(1, 4, col + 1, projection="3d")
            else:
                ax = fig.add_subplot(1, 4, col + 1)
            axes_row.append(ax)

        plot_gesture(axes_row, data, gesture, fname)
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        out_path = out_dir / f"{gesture}.png"
        plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
        print(f"    saved → {out_path}")
        plt.close()

    print(f"\nAll plots saved to {out_dir}/")
    print("Open any .png file to view it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/recordings")
    ap.add_argument("--gesture",  default="",
                    help="Plot only this gesture (default: all)")
    args = ap.parse_args()
    plot_all(args.data_dir, args.gesture)