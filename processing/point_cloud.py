# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
Point-cloud pre-processing: spatial gating, SNR filter, DBSCAN clustering.

Each call to `process(frame)` returns a list of Cluster objects – one per
detected spatial cluster – ready for the tracker.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config.settings import ProcessingSettings
from radar.parser import DetectedPoint, RadarFrame


def _dbscan(xyz: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Pure-numpy DBSCAN — avoids sklearn/threadpoolctl macOS crash."""
    n = len(xyz)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        dists = np.linalg.norm(xyz - xyz[i], axis=1)
        neighbors = np.where(dists <= eps)[0]
        if len(neighbors) < min_samples:
            continue
        labels[neighbors] = cluster_id
        seed = set(neighbors.tolist()) - {i}
        while seed:
            j = seed.pop()
            dists_j = np.linalg.norm(xyz - xyz[j], axis=1)
            nb_j = np.where(dists_j <= eps)[0]
            if len(nb_j) >= min_samples:
                for k in nb_j:
                    if labels[k] == -1:
                        labels[k] = cluster_id
                        seed.add(int(k))
        cluster_id += 1
    return labels


@dataclass
class Cluster:
    """Aggregated summary of a DBSCAN cluster."""
    centroid: np.ndarray        # [x, y, z]  metres
    points: np.ndarray          # (N,3) raw point positions
    mean_velocity: float        # mean Doppler across cluster (m/s)
    point_count: int
    mean_snr: float = 0.0


class PointCloudProcessor:
    def __init__(self, settings: ProcessingSettings):
        self._s = settings

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, frame: RadarFrame) -> List[Cluster]:
        pts = self._filter(frame.points)
        if len(pts) == 0:
            return []
        return self._cluster(pts)

    def points_to_array(self, frame: RadarFrame) -> Optional[np.ndarray]:
        """Return (N,4) array [x,y,z,v] after spatial gating, or None."""
        pts = self._filter(frame.points)
        if not pts:
            return None
        return np.array([[p.x, p.y, p.z, p.velocity] for p in pts], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _filter(self, points: List[DetectedPoint]) -> List[DetectedPoint]:
        s = self._s
        out = []
        for p in points:
            rng = np.sqrt(p.x**2 + p.y**2 + p.z**2)
            if not (s.min_range <= rng <= s.max_range):
                continue
            if abs(p.x) > s.x_limit:
                continue
            if abs(p.z) > s.z_limit:
                continue
            if p.y < 0:          # behind the sensor
                continue
            out.append(p)
        return out

    def _cluster(self, points: List[DetectedPoint]) -> List[Cluster]:
        xyz = np.array([[p.x, p.y, p.z] for p in points])
        vel = np.array([p.velocity for p in points])
        snr = np.array([p.snr for p in points])

        labels = _dbscan(xyz, self._s.dbscan_eps, self._s.dbscan_min_samples)
        clusters: List[Cluster] = []

        for label in set(labels):
            if label == -1:     # noise
                continue
            mask = labels == label
            cluster_pts = xyz[mask]
            cluster_vel = vel[mask]
            cluster_snr = snr[mask]

            centroid = cluster_pts.mean(axis=0)
            mean_vel = float(np.mean(cluster_vel))
            mean_snr = float(np.mean(cluster_snr)) if cluster_snr.any() else 0.0

            clusters.append(Cluster(
                centroid=centroid,
                points=cluster_pts,
                mean_velocity=mean_vel,
                point_count=int(mask.sum()),
                mean_snr=mean_snr,
            ))

        return clusters