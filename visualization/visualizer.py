# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
Real-time visualisation for the radar gesture framework.

Layout (2 × 2 figure)
───────────────────────────────────────────────────────────
  [0,0] 3-D scatter   raw point cloud coloured by Doppler
  [0,1] Top-down view  X–Y plane, track tails + IDs
  [1,0] Side view      Y–Z plane, elevation profile
  [1,1] Gesture panel  current gesture label + confidence bar
───────────────────────────────────────────────────────────

All updates are driven by a matplotlib FuncAnimation callback that drains
a thread-safe `VisualisationState` object populated by the main pipeline.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
import sys
if sys.platform == "darwin":
    matplotlib.use("MacOSX")   # native macOS backend — no Tkinter needed
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from config.settings import VisualizationSettings
from processing.tracker import Track


GESTURE_COLORS = {
    "swipe_left":       "#3498db",
    "swipe_right":      "#e74c3c",
    "swipe_up":         "#2ecc71",
    "swipe_down":       "#f39c12",
    "push":             "#9b59b6",
    "pull":             "#1abc9c",
    "wave":             "#e67e22",
    "circle_cw":        "#d35400",
    "circle_ccw":       "#27ae60",
    "static":           "#95a5a6",
    "unknown":          "#bdc3c7",
}

TRACK_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#d35400",
]


@dataclass
class VisualisationState:
    """Thread-safe snapshot consumed by the animation callback."""
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Point cloud
    points_xyz: Optional[np.ndarray] = None    # (N,3)
    points_vel: Optional[np.ndarray] = None    # (N,)  Doppler

    # Tracks: tid → history list of [x,y,z]
    tracks: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    track_positions: Dict[int, np.ndarray] = field(default_factory=dict)

    # Current gesture
    gesture_label: str = "—"
    gesture_confidence: float = 0.0
    gesture_ts: float = 0.0

    # Stats
    fps: float = 0.0
    frame_count: int = 0

    def update(
        self,
        points_xyz,
        points_vel,
        tracks: List[Track],
        gesture_label: str,
        gesture_confidence: float,
        fps: float,
        frame_count: int,
    ) -> None:
        with self._lock:
            self.points_xyz = points_xyz
            self.points_vel = points_vel
            self.tracks = {t.id: [p.copy() for p in t.history[-80:]] for t in tracks}
            self.track_positions = {t.id: t.position for t in tracks}
            if gesture_label != self.gesture_label:
                self.gesture_label = gesture_label
                self.gesture_confidence = gesture_confidence
                self.gesture_ts = time.time()
            self.fps = fps
            self.frame_count = frame_count

    def snapshot(self) -> "VisualisationState":
        with self._lock:
            snap = VisualisationState.__new__(VisualisationState)
            snap._lock = threading.Lock()
            snap.points_xyz = self.points_xyz.copy() if self.points_xyz is not None else None
            snap.points_vel = self.points_vel.copy() if self.points_vel is not None else None
            snap.tracks = {tid: [p.copy() for p in hist] for tid, hist in self.tracks.items()}
            snap.track_positions = {tid: pos.copy() for tid, pos in self.track_positions.items()}
            snap.gesture_label = self.gesture_label
            snap.gesture_confidence = self.gesture_confidence
            snap.gesture_ts = self.gesture_ts
            snap.fps = self.fps
            snap.frame_count = self.frame_count
            return snap


class RadarVisualizer:
    def __init__(self, settings: VisualizationSettings):
        self._s = settings
        self._state = VisualisationState()
        self._anim: Optional[animation.FuncAnimation] = None

    @property
    def state(self) -> VisualisationState:
        return self._state

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------

    def _build_figure(self):
        self._fig = plt.figure(figsize=(14, 9), facecolor="#1a1a2e")
        self._fig.canvas.manager.set_window_title("IWR1443 Radar – Gesture & Tracking")

        gs = GridSpec(2, 2, figure=self._fig, hspace=0.35, wspace=0.3)

        # 3-D scatter
        self._ax3d = self._fig.add_subplot(gs[0, 0], projection="3d")
        self._setup_3d(self._ax3d)

        # Top-down X–Y
        self._ax_top = self._fig.add_subplot(gs[0, 1])
        self._setup_2d(self._ax_top, "Top view  (X – Y)", "X  lateral (m)", "Y  depth (m)",
                       (-self._s.x_range, self._s.x_range), (0, self._s.y_range))

        # Side Y–Z
        self._ax_side = self._fig.add_subplot(gs[1, 0])
        self._setup_2d(self._ax_side, "Side view  (Y – Z)", "Y  depth (m)", "Z  elevation (m)",
                       (0, self._s.y_range), (-self._s.z_range, self._s.z_range))

        # Gesture panel
        self._ax_gest = self._fig.add_subplot(gs[1, 1])
        self._setup_gesture_panel(self._ax_gest)

    def _setup_3d(self, ax):
        ax.set_facecolor("#16213e")
        ax.set_xlim(-self._s.x_range, self._s.x_range)
        ax.set_ylim(0, self._s.y_range)
        ax.set_zlim(-self._s.z_range, self._s.z_range)
        ax.set_xlabel("X (m)", color="white", fontsize=7)
        ax.set_ylabel("Y (m)", color="white", fontsize=7)
        ax.set_zlabel("Z (m)", color="white", fontsize=7)
        ax.set_title("Point Cloud", color="white", fontsize=9, pad=4)
        ax.tick_params(colors="gray", labelsize=6)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
        ax.grid(True, color="gray", alpha=0.2, linewidth=0.5)

    def _setup_2d(self, ax, title, xlabel, ylabel, xlim, ylim):
        ax.set_facecolor("#16213e")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, color="white", fontsize=8)
        ax.set_ylabel(ylabel, color="white", fontsize=8)
        ax.set_title(title, color="white", fontsize=9)
        ax.tick_params(colors="gray", labelsize=7)
        ax.grid(True, color="gray", alpha=0.2, linewidth=0.5)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)

    def _setup_gesture_panel(self, ax):
        ax.set_facecolor("#16213e")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Gesture", color="white", fontsize=9)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._build_figure()
        self._anim = animation.FuncAnimation(
            self._fig,
            self._update,
            interval=self._s.update_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()

    def _update(self, _frame):
        snap = self._state.snapshot()
        self._draw_3d(snap)
        self._draw_top(snap)
        self._draw_side(snap)
        self._draw_gesture(snap)

    # ------------------------------------------------------------------
    # Per-panel drawing helpers
    # ------------------------------------------------------------------

    def _draw_3d(self, s: VisualisationState):
        ax = self._ax3d
        ax.cla()
        self._setup_3d(ax)

        if s.points_xyz is not None and len(s.points_xyz):
            vel = s.points_vel if s.points_vel is not None else np.zeros(len(s.points_xyz))
            v_norm = np.clip((vel + 3) / 6, 0, 1)
            colors = cm.RdBu(v_norm)
            ax.scatter(
                s.points_xyz[:, 0],
                s.points_xyz[:, 1],
                s.points_xyz[:, 2],
                c=colors, s=10, alpha=0.9, depthshade=True,
            )

        # Track centroids in 3D
        for tid, hist in s.tracks.items():
            if not hist:
                continue
            col = TRACK_PALETTE[tid % len(TRACK_PALETTE)]
            if len(hist) > 1:
                h = np.array(hist)
                ax.plot(h[:, 0], h[:, 1], h[:, 2], "-", color=col,
                        alpha=0.5, linewidth=1)
            pos = hist[-1]
            ax.scatter([pos[0]], [pos[1]], [pos[2]], color=col, s=60,
                       marker="o", edgecolors="white", linewidths=0.5, zorder=5)

        ax.text2D(0.02, 0.97, f"Frame {s.frame_count}  |  {s.fps:.1f} Hz",
                  transform=ax.transAxes, color="white", fontsize=7, va="top")

    def _draw_top(self, s: VisualisationState):
        ax = self._ax_top
        ax.cla()
        self._setup_2d(ax, "Top view  (X – Y)", "X  lateral (m)", "Y  depth (m)",
                       (-self._s.x_range, self._s.x_range), (0, self._s.y_range))

        if s.points_xyz is not None and len(s.points_xyz):
            ax.scatter(s.points_xyz[:, 0], s.points_xyz[:, 1],
                       c="#4fc3f7", s=6, alpha=0.6)

        for tid, hist in s.tracks.items():
            if not hist:
                continue
            col = TRACK_PALETTE[tid % len(TRACK_PALETTE)]
            h = np.array(hist)
            if len(h) > 1:
                ax.plot(h[:, 0], h[:, 1], "-", color=col, alpha=0.6, linewidth=1.5)
            pos = hist[-1]
            ax.scatter([pos[0]], [pos[1]], color=col, s=80, marker="o",
                       edgecolors="white", linewidths=0.8, zorder=5)
            ax.text(pos[0] + 0.05, pos[1] + 0.05, f"T{tid}", color=col,
                    fontsize=7, fontweight="bold")

    def _draw_side(self, s: VisualisationState):
        ax = self._ax_side
        ax.cla()
        self._setup_2d(ax, "Side view  (Y – Z)", "Y  depth (m)", "Z  elevation (m)",
                       (0, self._s.y_range), (-self._s.z_range, self._s.z_range))

        if s.points_xyz is not None and len(s.points_xyz):
            ax.scatter(s.points_xyz[:, 1], s.points_xyz[:, 2],
                       c="#80cbc4", s=6, alpha=0.6)

        for tid, hist in s.tracks.items():
            if not hist:
                continue
            col = TRACK_PALETTE[tid % len(TRACK_PALETTE)]
            h = np.array(hist)
            if len(h) > 1:
                ax.plot(h[:, 1], h[:, 2], "-", color=col, alpha=0.6, linewidth=1.5)
            pos = hist[-1]
            ax.scatter([pos[1]], [pos[2]], color=col, s=80, marker="o",
                       edgecolors="white", linewidths=0.8, zorder=5)

    def _draw_gesture(self, s: VisualisationState):
        ax = self._ax_gest
        ax.cla()
        self._setup_gesture_panel(ax)

        label = s.gesture_label
        conf = s.gesture_confidence
        color = GESTURE_COLORS.get(label, "#bdc3c7")

        # Large gesture label
        ax.text(0.5, 0.62, label.upper().replace("_", " "),
                ha="center", va="center", fontsize=18, fontweight="bold",
                color=color, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f3460",
                          edgecolor=color, linewidth=2))

        # Confidence bar
        bar_w = 0.7
        bar_x = (1 - bar_w) / 2
        ax.add_patch(plt.Rectangle(
            (bar_x, 0.30), bar_w, 0.06,
            facecolor="#2c3e50", edgecolor="gray", linewidth=0.5,
            transform=ax.transAxes, clip_on=False
        ))
        ax.add_patch(plt.Rectangle(
            (bar_x, 0.30), bar_w * conf, 0.06,
            facecolor=color, alpha=0.85,
            transform=ax.transAxes, clip_on=False
        ))
        ax.text(0.5, 0.22, f"Confidence: {conf*100:.0f}%",
                ha="center", va="top", color="white", fontsize=9,
                transform=ax.transAxes)

        # Track count
        n_tracks = len(s.tracks)
        ax.text(0.5, 0.10, f"Active tracks: {n_tracks}   |   {s.fps:.1f} fps",
                ha="center", va="top", color="#95a5a6", fontsize=8,
                transform=ax.transAxes)