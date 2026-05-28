# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
data/plot_range_angle.py — Range-Angle map visualisation from recordings.

Creates pseudo Range-Angle maps from processed point-cloud recordings
[x, y, z, velocity, count, spread]:

  range         = sqrt(x² + y² + z²)
  azimuth angle = atan2(x, y)   — horizontal left/right angle (degrees)
  elevation     = atan2(z, sqrt(x²+y²)) — vertical up/down angle (degrees)

Two map types are produced per gesture:
  1. Range vs Azimuth  — top-down view, shows lateral gesture spread
  2. Range vs Elevation — side view, shows vertical gesture spread

Usage
─────
  python -m data.plot_range_angle                    # both maps per gesture
  python -m data.plot_range_angle --gesture swipe_left
  python -m data.plot_range_angle --samples 10
  python -m data.plot_range_angle --overlay          # all gestures, one figure
  python -m data.plot_range_angle --type azimuth     # azimuth maps only
  python -m data.plot_range_angle --type elevation   # elevation maps only
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
RANGE_MIN   =  0.0     # metres
RANGE_MAX   =  4.0
AZ_MIN      = -60.0    # degrees
AZ_MAX      =  60.0
EL_MIN      = -40.0    # degrees
EL_MAX      =  40.0
RANGE_BINS  =  60
ANGLE_BINS  =  60


def compute_ra_maps(data: np.ndarray):
    """
    Compute Range-Azimuth and Range-Elevation heatmaps from one recording.
    Returns (ra_map, re_map) each of shape (ANGLE_BINS, RANGE_BINS).
    """
    active = np.any(data[:, :3] != 0, axis=1)
    if not active.any():
        empty = np.zeros((ANGLE_BINS, RANGE_BINS))
        return empty, empty

    pts   = data[active]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    count = pts[:, 4]

    rng       = np.sqrt(x**2 + y**2 + z**2)
    azimuth   = np.degrees(np.arctan2(x, y))          # horizontal angle
    elevation = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))  # vertical angle

    az_edges    = np.linspace(AZ_MIN,    AZ_MAX,    ANGLE_BINS + 1)
    el_edges    = np.linspace(EL_MIN,    EL_MAX,    ANGLE_BINS + 1)
    range_edges = np.linspace(RANGE_MIN, RANGE_MAX, RANGE_BINS + 1)

    ra_map, _, _ = np.histogram2d(
        azimuth, rng,
        bins=[az_edges, range_edges],
        weights=count,
    )

    re_map, _, _ = np.histogram2d(
        elevation, rng,
        bins=[el_edges, range_edges],
        weights=count,
    )

    return ra_map, re_map


def _pcolormesh(ax, col_edges, row_edges, data, xlabel, ylabel):
    """Helper: plot a range-angle heatmap on ax."""
    im = ax.pcolormesh(
        col_edges, row_edges, data,
        cmap="jet", shading="auto",
        norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=data.max() + 1e-6)
    )
    plt.colorbar(im, ax=ax, label="Weighted detection count")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)


def plot_per_gesture(data_dir: str, gesture_filter: str,
                     samples: int, map_type: str, out_dir: Path) -> None:
    """Save Range-Angle maps for each gesture."""
    by_label: dict = defaultdict(list)
    for fp in sorted(Path(data_dir).glob("*.npy")):
        lbl = fp.stem.rsplit("_", 1)[0]
        if gesture_filter and lbl != gesture_filter:
            continue
        by_label[lbl].append(fp)

    if not by_label:
        print("No recordings found."); return

    out_dir.mkdir(parents=True, exist_ok=True)
    range_edges = np.linspace(RANGE_MIN, RANGE_MAX, RANGE_BINS + 1)
    az_edges    = np.linspace(AZ_MIN,    AZ_MAX,    ANGLE_BINS + 1)
    el_edges    = np.linspace(EL_MIN,    EL_MAX,    ANGLE_BINS + 1)

    for label, files in sorted(by_label.items()):
        chosen = files[:samples]
        all_ra, all_re = [], []
        for f in chosen:
            ra, re = compute_ra_maps(np.load(f).astype(np.float32))
            all_ra.append(ra); all_re.append(re)
        avg_ra = np.mean(all_ra, axis=0)
        avg_re = np.mean(all_re, axis=0)

        if map_type in ("both", "azimuth", "elevation"):
            n_plots = 2 if map_type == "both" else 1
            fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
            if n_plots == 1:
                axes = [axes]

            col = 0
            if map_type in ("both", "azimuth"):
                _pcolormesh(axes[col], range_edges, az_edges, avg_ra,
                            "Range (m)", "Azimuth Angle (°)")
                axes[col].set_xlim(RANGE_MIN, RANGE_MAX)
                axes[col].set_ylim(AZ_MIN, AZ_MAX)
                col += 1

            if map_type in ("both", "elevation"):
                _pcolormesh(axes[col], range_edges, el_edges, avg_re,
                            "Range (m)", "Elevation Angle (°)")
                axes[col].set_xlim(RANGE_MIN, RANGE_MAX)
                axes[col].set_ylim(EL_MIN, EL_MAX)

            plt.tight_layout()
            out_path = out_dir / f"ramap_{label}.png"
            plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  {label:<20} → {out_path.name}  "
                  f"(averaged {len(chosen)} recordings)")


def plot_overlay(data_dir: str, samples: int, map_type: str, out_dir: Path) -> None:
    """One combined figure comparing all gestures."""
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
    range_edges = np.linspace(RANGE_MIN, RANGE_MAX, RANGE_BINS + 1)
    az_edges    = np.linspace(AZ_MIN,    AZ_MAX,    ANGLE_BINS + 1)
    el_edges    = np.linspace(EL_MIN,    EL_MAX,    ANGLE_BINS + 1)

    for mtype, edges, ylabel in [
        ("azimuth",   az_edges, "Azimuth Angle (°)"),
        ("elevation", el_edges, "Elevation Angle (°)"),
    ]:
        if map_type not in ("both", mtype):
            continue

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(4 * n_cols, 3.5 * n_rows))
        axes = np.array(axes).flatten()

        for idx, label in enumerate(labels):
            chosen = by_label[label][:samples]
            all_ra, all_re = [], []
            for f in chosen:
                ra, re = compute_ra_maps(np.load(f).astype(np.float32))
                all_ra.append(ra); all_re.append(re)

            data_map = np.mean(all_ra if mtype == "azimuth" else all_re, axis=0)
            ax = axes[idx]
            im = ax.pcolormesh(
                range_edges, edges, data_map,
                cmap="jet", shading="auto",
                norm=mcolors.PowerNorm(gamma=0.4,
                                       vmin=0, vmax=data_map.max() + 1e-6)
            )
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("Range (m)", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_xlim(RANGE_MIN, RANGE_MAX)
            ax.set_ylim(edges[0], edges[-1])
            ax.text(0.03, 0.95, label, transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top", color="white",
                    bbox=dict(facecolor="black", alpha=0.5, pad=2))

        for idx in range(len(labels), len(axes)):
            axes[idx].set_visible(False)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ramap_all_{mtype}.png"
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {mtype.capitalize()} overlay → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Range-Angle map visualisation from gesture recordings"
    )
    ap.add_argument("--data_dir", default="data/recordings")
    ap.add_argument("--gesture",  default="",
                    help="Plot only this gesture (default: all)")
    ap.add_argument("--samples",  type=int, default=10,
                    help="Recordings to average per gesture (default: 10)")
    ap.add_argument("--overlay",  action="store_true",
                    help="Save one combined figure with all gestures")
    ap.add_argument("--type",     default="both",
                    choices=["both", "azimuth", "elevation"],
                    help="Which angle map to plot (default: both)")
    args = ap.parse_args()

    out_dir = Path(args.data_dir) / "plots" / "range_angle"
    print(f"\n  Range-Angle maps  →  {out_dir}/\n")

    if args.overlay:
        plot_overlay(args.data_dir, args.samples, args.type, out_dir)
    else:
        plot_per_gesture(args.data_dir, args.gesture,
                         args.samples, args.type, out_dir)
        print(f"\n  Saved to {out_dir}/")


if __name__ == "__main__":
    main()
