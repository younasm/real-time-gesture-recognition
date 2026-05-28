# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
Feature extraction from a sliding window of point-cloud frames.

Used by both the rule-based and the ML classifier.

Window → FeatureSet  (one per tracked target, or global if no tracker)
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class WindowFeatures:
    """
    All features derived from one gesture window.
    Also used as an intermediate for the rule-based classifier.
    """
    # Displacement of centroid from window start → end
    dx: float = 0.0     # lateral
    dy: float = 0.0     # depth
    dz: float = 0.0     # elevation

    # Magnitude and dominant axis
    displacement: float = 0.0
    dominant_axis: str = "none"   # x | y | z

    # Velocity statistics (Doppler)
    mean_velocity: float = 0.0
    max_abs_velocity: float = 0.0
    velocity_sign: float = 0.0    # +1 away, -1 toward sensor

    # Path curvature (detect circular motion)
    path_curvature: float = 0.0   # higher = more curved
    direction_changes: int = 0    # sign changes in dx over window

    # Point-cloud density
    mean_points: float = 0.0
    mean_spread: float = 0.0      # mean std of point positions per frame

    # Temporal centroid trajectory flattened for ML
    traj_x: np.ndarray = None    # shape (window_frames,)
    traj_y: np.ndarray = None
    traj_z: np.ndarray = None
    traj_v: np.ndarray = None    # mean Doppler per frame

    def to_vector(self) -> np.ndarray:
        """Flat feature vector for ML models (≈ 60-D for window=20)."""
        scalars = np.array([
            self.dx, self.dy, self.dz,
            self.displacement,
            self.mean_velocity, self.max_abs_velocity, self.velocity_sign,
            self.path_curvature, self.direction_changes,
            self.mean_points, self.mean_spread,
        ], dtype=np.float32)

        traj = np.concatenate([
            self._safe(self.traj_x),
            self._safe(self.traj_y),
            self._safe(self.traj_z),
            self._safe(self.traj_v),
        ])
        return np.concatenate([scalars, traj])

    @staticmethod
    def _safe(arr) -> np.ndarray:
        if arr is None:
            return np.array([], dtype=np.float32)
        return np.asarray(arr, dtype=np.float32)


def extract_features(
    centroids: List[Optional[np.ndarray]],   # list of [x,y,z] or None
    velocities: List[Optional[float]],        # mean Doppler per frame
    point_counts: List[int],
    spreads: List[float],
) -> WindowFeatures:
    """
    Build WindowFeatures from parallel lists (one entry per frame in window).
    Frames where no cluster was detected should have centroid=None.
    """
    # Filter to frames with valid detections
    valid_c = [(i, c) for i, c in enumerate(centroids) if c is not None]
    f = WindowFeatures()

    f.mean_points = float(np.mean([p for p in point_counts if p > 0]) if point_counts else 0)
    f.mean_spread = float(np.mean([s for s in spreads if s > 0]) if spreads else 0)

    # Velocity stats
    valid_v = [v for v in velocities if v is not None]
    if valid_v:
        f.mean_velocity = float(np.mean(valid_v))
        f.max_abs_velocity = float(np.max(np.abs(valid_v)))
        f.velocity_sign = float(np.sign(f.mean_velocity))

    if len(valid_c) < 2:
        return f

    positions = np.array([c for _, c in valid_c])  # (M,3)

    # Centroid trajectory (interpolated to full window length for ML)
    n = len(centroids)
    indices = np.array([i for i, _ in valid_c])
    traj_x = np.interp(np.arange(n), indices, positions[:, 0])
    traj_y = np.interp(np.arange(n), indices, positions[:, 1])
    traj_z = np.interp(np.arange(n), indices, positions[:, 2])
    # fp must have the same length as xp (indices = valid frame positions)
    valid_v = [velocities[i] if velocities[i] is not None else 0.0
               for i, _ in valid_c]
    traj_v = np.interp(np.arange(n), indices, valid_v)

    f.traj_x = traj_x
    f.traj_y = traj_y
    f.traj_z = traj_z
    f.traj_v = traj_v

    # Displacement
    start, end = positions[0], positions[-1]
    delta = end - start
    f.dx, f.dy, f.dz = float(delta[0]), float(delta[1]), float(delta[2])
    f.displacement = float(np.linalg.norm(delta))

    ax_magnitudes = np.abs([f.dx, f.dy, f.dz])
    f.dominant_axis = ["x", "y", "z"][int(np.argmax(ax_magnitudes))]

    # Path curvature: ratio of arc-length to chord-length
    diffs = np.diff(positions, axis=0)
    arc_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    chord = f.displacement
    f.path_curvature = (arc_len / chord) if chord > 0.01 else 1.0

    # Direction changes in lateral axis
    dx_series = np.diff(positions[:, 0])
    sign_changes = int(np.sum(np.diff(np.sign(dx_series)) != 0))
    f.direction_changes = sign_changes

    return f