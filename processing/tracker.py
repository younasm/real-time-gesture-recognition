# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
SORT-style multi-target Kalman tracker (no external dependencies).

State vector per track:  [x, y, z, vx, vy, vz]  (constant-velocity model)
Observation:             [x, y, z]

Tracks are confirmed after `min_hits` consecutive updates and deleted after
`max_age` frames without a match.  Assignment uses the Hungarian algorithm
(scipy.optimize.linear_sum_assignment) on a Euclidean distance cost matrix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from config.settings import TrackerSettings
from processing.point_cloud import Cluster


# ── Track state ───────────────────────────────────────────────────────────────

_NEXT_ID = 1


def _next_id() -> int:
    global _NEXT_ID
    tid = _NEXT_ID
    _NEXT_ID += 1
    return tid


@dataclass
class Track:
    id: int
    confirmed: bool = False
    age: int = 0                    # total frames this track has lived
    hits: int = 1                   # consecutive matched frames
    misses: int = 0                 # consecutive missed frames
    created_at: float = field(default_factory=time.time)

    # Kalman state / covariance
    x: np.ndarray = field(default_factory=lambda: np.zeros(6))
    P: np.ndarray = field(default_factory=lambda: np.eye(6))

    # History for gesture / visualisation
    history: List[np.ndarray] = field(default_factory=list)  # list of [x,y,z]
    velocity_history: List[float] = field(default_factory=list)  # Doppler

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity_vec(self) -> np.ndarray:
        return self.x[3:].copy()


# ── Tracker ────────────────────────────────────────────────────────────────────

class MultiTargetTracker:
    """
    Maintains a set of Kalman tracks and matches them to incoming clusters.
    """

    def __init__(self, settings: TrackerSettings):
        s = self._s = settings
        self._tracks: Dict[int, Track] = {}

        # Build Kalman matrices once
        dt = s.dt
        self._F = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ],
        ], dtype=float)

        self._H = np.zeros((3, 6))
        self._H[0, 0] = self._H[1, 1] = self._H[2, 2] = 1.0

        q_p = s.process_noise_pos
        q_v = s.process_noise_vel
        self._Q = np.diag([q_p, q_p, q_p, q_v, q_v, q_v])

        r = s.measurement_noise
        self._R = np.eye(3) * r

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(self, clusters: List[Cluster]) -> List[Track]:
        """
        Feed current-frame clusters, return confirmed+active tracks.
        """
        # 1. Predict all existing tracks forward one step
        for t in self._tracks.values():
            self._predict(t)

        # 2. Build cost matrix (Euclidean distance)
        track_ids = list(self._tracks.keys())
        measurements = [c.centroid for c in clusters]
        matched, unmatched_tracks, unmatched_dets = self._assign(
            track_ids, measurements
        )

        # 3. Update matched tracks
        for tid, det_idx in matched:
            t = self._tracks[tid]
            self._kalman_update(t, measurements[det_idx])
            t.hits += 1
            t.misses = 0
            t.age += 1
            t.history.append(t.position)
            t.velocity_history.append(clusters[det_idx].mean_velocity)
            if t.hits >= self._s.min_hits:
                t.confirmed = True

        # 4. Mark unmatched tracks as missed
        for tid in unmatched_tracks:
            t = self._tracks[tid]
            t.misses += 1
            t.age += 1
            t.hits = 0

        # 5. Spawn new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(measurements[det_idx], clusters[det_idx])

        # 6. Prune dead tracks
        dead = [tid for tid, t in self._tracks.items()
                if t.misses > self._s.max_age]
        for tid in dead:
            del self._tracks[tid]

        # Trim history to avoid unbounded growth
        max_hist = 200
        for t in self._tracks.values():
            if len(t.history) > max_hist:
                t.history = t.history[-max_hist:]
            if len(t.velocity_history) > max_hist:
                t.velocity_history = t.velocity_history[-max_hist:]

        return [t for t in self._tracks.values() if t.confirmed]

    def get_track(self, tid: int) -> Optional[Track]:
        return self._tracks.get(tid)

    def all_tracks(self) -> List[Track]:
        return list(self._tracks.values())

    # ------------------------------------------------------------------
    # Internal – Kalman
    # ------------------------------------------------------------------

    def _predict(self, t: Track) -> None:
        t.x = self._F @ t.x
        t.P = self._F @ t.P @ self._F.T + self._Q

    def _kalman_update(self, t: Track, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=float)
        S = self._H @ t.P @ self._H.T + self._R
        K = t.P @ self._H.T @ np.linalg.inv(S)
        y = z - self._H @ t.x
        t.x = t.x + K @ y
        t.P = (np.eye(6) - K @ self._H) @ t.P

    def _create_track(self, pos: np.ndarray, cluster: Cluster) -> None:
        t = Track(id=_next_id())
        t.x = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0])
        t.P = np.eye(6) * 1.0
        t.history.append(pos.copy())
        t.velocity_history.append(cluster.mean_velocity)
        self._tracks[t.id] = t

    # ------------------------------------------------------------------
    # Internal – Hungarian assignment
    # ------------------------------------------------------------------

    def _assign(
        self,
        track_ids: List[int],
        measurements: List[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not track_ids or not measurements:
            return [], track_ids[:], list(range(len(measurements)))

        cost = np.zeros((len(track_ids), len(measurements)))
        for i, tid in enumerate(track_ids):
            pred_pos = self._tracks[tid].x[:3]
            for j, m in enumerate(measurements):
                cost[i, j] = np.linalg.norm(pred_pos - m)

        row_ind, col_ind = linear_sum_assignment(cost)

        matched, unmatched_trks, unmatched_dets = [], [], []
        threshold = self._s.max_dist

        assigned_rows, assigned_cols = set(), set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= threshold:
                matched.append((track_ids[r], c))
                assigned_rows.add(r)
                assigned_cols.add(c)

        unmatched_trks = [track_ids[i] for i in range(len(track_ids))
                          if i not in assigned_rows]
        unmatched_dets = [j for j in range(len(measurements))
                          if j not in assigned_cols]

        return matched, unmatched_trks, unmatched_dets