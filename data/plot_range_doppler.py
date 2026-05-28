# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
data/plot_range_doppler.py — Range-Doppler map visualisation from recordings.

Creates pseudo Range-Doppler maps by accumulating detected-point (range, velocity)
pairs across all frames in a recording.  Since our recordings store processed
point-cloud centroids [x, y, z, velocity, count, spread] rather than raw ADC
data, this is an approximation:

  range    = sqrt(x² + y² + z²)   (distance from sensor to centroid)
  velocity = Doppler value stored in recording column 3

The accumulated 2D histogram (range bins × velocity bins) shows the
range-velocity signature of each gesture — directly comparable to a true
Range-Doppler map and useful for research-paper figures.

Usage
─────
  python -m data.plot_range_doppler                    # one map per gesture
  python -m data.plot_range_doppler --gesture swipe_left
  python -m data.plot_range_doppler --samples 5        # average 5 recordings
  python -m data.plot_range_doppler --overlay           # overlay all gestures
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── Config ────────────────────────────────────────────────────────────────────
RANGE_MIN   =  0.0    # metres
RANGE_MAX   =  4.0
VEL_MIN     = -2.0    # m/s
VEL_MAX     =  2.0
RANGE_BINS  = 60
VEL_BINS    = 60


def compute_rdmap(data: np.ndarray) -> np.ndarray:
    """
    Build a (VEL_BINS, RANGE_BINS) range-Doppler heatmap from one recording.
    Active frames only (frames where a centroid was detected).
    """
    active = np.any(data[:, :3] != 0, axis=1)
    if not active.any():
        return np.zeros((VEL_BINS, RANGE_BINS))

    pts  = data[active]
    rng  = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)
    vel  = pts[:, 3]

    h, _, _ = np.histogram2d(
        vel, rng,
        bins=[VEL_BINS, RANGE_BINS],
        range=[[VEL_MIN, VEL_MAX], [RANGE_MIN, RANGE_MAX]],
        weights=pts[:, 4],   # weight by point_count — denser frames contribute more
    )
    return h


def load_first_n(data_dir: str, label: str, n: int) -> list:
    files = sorted(Path(data_dir).glob(f"{label}_*.npy"))[:n]
    return [np.load(f).astype(np.float32) for f in files]


def plot_per_gesture(data_dir: str, gesture_filter: str,
                     samples: int, out_dir: Path) -> None:
    """Save one Range-Doppler map PNG per gesture."""
    by_label: dict = defaultdict(list)
    for fp in sorted(Path(data_dir).glob("*.npy")):
        lbl = fp.stem.rsplit("_", 1)[0]
        if gesture_filter and lbl != gesture_filter:
            continue
        by_label[lbl].append(fp)

    if not by_label:
        print("No recordings found."); return

    out_dir.mkdir(parents=True, exist_ok=True)
    vel_edges   = np.linspace(VEL_MIN, VEL_MAX, VEL_BINS + 1)
    range_edges = np.linspace(RANGE_MIN, RANGE_MAX, RANGE_BINS + 1)

    for label, files in sorted(by_label.items()):
        chosen = files[:samples]
        maps   = [compute_rdmap(np.load(f).astype(np.float32)) for f in chosen]
        avg_map = np.mean(maps, axis=0)   # average across recordings

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.pcolormesh(
            range_edges, vel_edges, avg_map,
            cmap="jet", shading="auto",
            norm=mcolors.PowerNorm(gamma=0.4,
                                   vmin=0, vmax=avg_map.max() + 1e-6)
        )
        plt.colorbar(im, ax=ax, label="Weighted detection count")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Doppler Velocity (m/s)")
        ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlim(RANGE_MIN, RANGE_MAX)
        ax.set_ylim(VEL_MIN, VEL_MAX)

        out_path = out_dir / f"rdmap_{label}.png"
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {label:<20} → {out_path.name}  "
              f"(averaged {len(chosen)} recordings)")


def plot_overlay(data_dir: str, samples: int, out_dir: Path) -> None:
    """Plot all gestures side-by-side in one figure for comparison."""
    by_label: dict = defaultdict(list)
    for fp in sorted(Path(data_dir).glob("*.npy")):
        lbl = fp.stem.rsplit("_", 1)[0]
        if lbl == "static":
            continue
        by_label[lbl].append(fp)

    if not by_label:
        print("No recordings found."); return

    labels  = sorted(by_label.keys())
    n_cols  = 5
    n_rows  = (len(labels) + n_cols - 1) // n_cols
    vel_edges   = np.linspace(VEL_MIN, VEL_MAX, VEL_BINS + 1)
    range_edges = np.linspace(RANGE_MIN, RANGE_MAX, RANGE_BINS + 1)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx, label in enumerate(labels):
        files   = by_label[label][:samples]
        maps    = [compute_rdmap(np.load(f).astype(np.float32)) for f in files]
        avg_map = np.mean(maps, axis=0)

        ax = axes[idx]
        im = ax.pcolormesh(
            range_edges, vel_edges, avg_map,
            cmap="jet", shading="auto",
            norm=mcolors.PowerNorm(gamma=0.4,
                                   vmin=0, vmax=avg_map.max() + 1e-6)
        )
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Range (m)", fontsize=8)
        ax.set_ylabel("Doppler (m/s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlim(RANGE_MIN, RANGE_MAX)
        ax.set_ylim(VEL_MIN, VEL_MAX)
        ax.text(0.03, 0.95, label, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top",
                color="white",
                bbox=dict(facecolor="black", alpha=0.5, pad=2))

    # Hide unused subplots
    for idx in range(len(labels), len(axes)):
        axes[idx].set_visible(False)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rdmap_all_gestures.png"
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  All gestures overlay → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Range-Doppler map visualisation from gesture recordings"
    )
    ap.add_argument("--data_dir", default="data/recordings")
    ap.add_argument("--gesture",  default="",
                    help="Plot only this gesture (default: all)")
    ap.add_argument("--samples",  type=int, default=10,
                    help="Recordings to average per gesture (default: 10)")
    ap.add_argument("--overlay",  action="store_true",
                    help="Save one combined figure with all gestures")
    args = ap.parse_args()

    out_dir = Path(args.data_dir) / "plots" / "range_doppler"
    print(f"\n  Range-Doppler maps  →  {out_dir}/\n")

    if args.overlay:
        plot_overlay(args.data_dir, args.samples, out_dir)
    else:
        plot_per_gesture(args.data_dir, args.gesture, args.samples, out_dir)
        print(f"\n  Saved to {out_dir}/")


if __name__ == "__main__":
    main()
