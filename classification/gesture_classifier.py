# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
Two-mode gesture classifier:

  Rule-based  (default, zero training needed)
  ──────────────────────────────────────────
  Interprets WindowFeatures using hand-crafted thresholds.
  Works immediately out-of-the-box.

  ML (Random Forest, optional)
  ────────────────────────────
  Loads a model from disk trained by model_trainer.py.
  Activate with  FrameworkSettings.use_ml_classifier = True
  after you have collected and labelled data.

Both modes emit a GestureResult.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional
from collections import deque

import numpy as np

from config.settings import GestureSettings
from processing.features import WindowFeatures, extract_features
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class GestureResult:
    label: str
    confidence: float         # 0–1  (rule-based uses heuristic confidence)
    timestamp: float
    features: Optional[WindowFeatures] = None


# ── Rule-based classifier ──────────────────────────────────────────────────────

class RuleBasedClassifier:
    """
    Classifies a gesture window purely from geometric / velocity features.

    Gesture map
    ───────────
    swipe_left / right  : dominant axis = X, |dx| large, low curvature
    swipe_up   / down   : dominant axis = Z, |dz| large, low curvature
    push / pull         : dominant axis = Y, velocity sign indicates direction
    wave                : dominant axis = X, many lateral direction changes
    circle_cw / ccw     : high curvature, rotational analysis of XZ plane
    static              : total displacement < threshold
    """

    def __init__(self, settings: GestureSettings):
        self._s = settings

    def classify(self, f: WindowFeatures) -> GestureResult:
        s = self._s

        # Insufficient movement
        if f.displacement < s.min_displacement_m:
            return GestureResult("static", 0.9, time.time(), f)

        ax = f.dominant_axis
        conf = min(f.displacement / (s.min_displacement_m * 3), 1.0)  # heuristic

        # ── Wave detection: lateral oscillation ──────────────────────────────
        if ax == "x" and f.direction_changes >= 3:
            return GestureResult("wave", conf * 0.85, time.time(), f)

        # ── Circular motion: high path curvature + non-dominant axes active ──
        if f.path_curvature > 2.5 and f.displacement > s.min_displacement_m:
            if f.traj_x is not None and len(f.traj_x) >= 4:
                label = self._detect_circle(f)
                if label:
                    return GestureResult(label, conf * 0.75, time.time(), f)

        ax_vals = np.abs([f.dx, f.dy, f.dz])
        ratio = float(np.max(ax_vals)) / (float(np.sum(ax_vals)) - float(np.max(ax_vals)) + 1e-6)

        # ── Swipe (clear dominant lateral axis) ──────────────────────────────
        if ax == "x" and ratio >= s.swipe_ratio:
            label = "swipe_right" if f.dx > 0 else "swipe_left"
            return GestureResult(label, conf, time.time(), f)

        # ── Vertical swipe ────────────────────────────────────────────────────
        if ax == "z" and ratio >= s.swipe_ratio:
            label = "swipe_up" if f.dz > 0 else "swipe_down"
            return GestureResult(label, conf, time.time(), f)

        # ── Push / pull (depth axis) ──────────────────────────────────────────
        if ax == "y":
            # Doppler also confirms direction:
            #   push = hand moves toward sensor → dy < 0, Doppler negative
            #   pull = hand moves away         → dy > 0, Doppler positive
            if f.dy < 0 or f.velocity_sign < 0:
                return GestureResult("push", conf, time.time(), f)
            else:
                return GestureResult("pull", conf, time.time(), f)

        # ── Fallback: label by dominant axis ─────────────────────────────────
        if ax == "x":
            label = "swipe_right" if f.dx > 0 else "swipe_left"
        elif ax == "z":
            label = "swipe_up" if f.dz > 0 else "swipe_down"
        else:
            label = "push" if f.dy < 0 else "pull"

        return GestureResult(label, conf * 0.6, time.time(), f)

    @staticmethod
    def _detect_circle(f: WindowFeatures) -> Optional[str]:
        """
        Fit an ellipse to XZ trajectory; check winding direction.
        Returns 'circle_cw', 'circle_ccw', or None.
        """
        x = f.traj_x
        z = f.traj_z
        if x is None or len(x) < 6:
            return None

        # Signed area via the shoelace formula (x-z plane)
        n = len(x)
        signed_area = 0.0
        for i in range(n):
            j = (i + 1) % n
            signed_area += x[i] * z[j] - x[j] * z[i]
        signed_area *= 0.5

        # Spread check – must be roughly circular, not just drifting
        x_std = float(np.std(x))
        z_std = float(np.std(z))
        if x_std < 0.05 or z_std < 0.05:
            return None

        aspect = max(x_std, z_std) / (min(x_std, z_std) + 1e-6)
        if aspect > 4.0:
            return None

        # In standard right-hand coords: CW when viewed from front → area < 0
        if signed_area < -0.01:
            return "circle_cw"
        elif signed_area > 0.01:
            return "circle_ccw"
        return None


# ── ML classifier wrapper ──────────────────────────────────────────────────────

class MLClassifier:
    def __init__(self, model_path: str, label_names: List[str]):
        import joblib
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"ML model not found: {p}\n"
                "Run  python -m classification.model_trainer  first."
            )
        saved = joblib.load(p)
        # model_trainer.py saves {"model": pipeline, "label_encoder": le}
        if isinstance(saved, dict):
            self._model = saved["model"]
            self._le    = saved.get("label_encoder", None)
        else:
            self._model = saved   # legacy: raw model
            self._le    = None
        self._labels = label_names
        log.info("ML model loaded from %s", p)

    def classify(self, f: WindowFeatures) -> GestureResult:
        vec = f.to_vector().reshape(1, -1)

        # Sanitise NaN/Inf that arise from empty frames (e.g. arc_length=0)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad/trim to match feature count the model was trained on
        if hasattr(self._model, "n_features_in_"):
            n = self._model.n_features_in_
            if vec.shape[1] < n:
                vec = np.pad(vec, ((0, 0), (0, n - vec.shape[1])))
            elif vec.shape[1] > n:
                vec = vec[:, :n]

        proba = self._model.predict_proba(vec)[0]
        idx   = int(np.argmax(proba))

        # Use label encoder classes if available (more reliable than gesture list)
        if self._le is not None:
            label = str(self._le.classes_[idx])
        else:
            label = self._labels[idx] if idx < len(self._labels) else "unknown"

        return GestureResult(label, float(proba[idx]), time.time(), f)


# ── Deep Learning classifier wrapper ──────────────────────────────────────────

class DLClassifier:
    """
    Wraps the CNN-BiLSTM model trained by dl_trainer.py.
    Works on raw (centroid, velocity, count, spread) sequences — no hand-crafted
    features needed. The GestureEngine passes the raw window directly.
    """

    def __init__(self, model_path: str, label_names: List[str]):
        try:
            import torch
            self._torch = torch
        except ImportError:
            raise ImportError("PyTorch required for DL model. Run: pip install torch")

        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"DL model not found: {p}\n"
                "Run  python -m classification.dl_trainer  first."
            )

        checkpoint = torch.load(str(p), map_location="cpu", weights_only=False)
        from classification.dl_trainer import GestureNet

        self._le      = checkpoint["label_encoder"]
        self._scaler  = checkpoint["scaler"]
        self._n_feat  = checkpoint["n_features"]
        self._seq_len = checkpoint["seq_len"]

        self._model = GestureNet(
            checkpoint["n_features"],
            checkpoint["n_classes"],
            checkpoint["seq_len"],
        )
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()
        self._labels = label_names
        log.info("DL model loaded from %s", p)

    def classify_raw(self,
                     centroids:  list,
                     velocities: list,
                     counts:     list,
                     spreads:    list) -> GestureResult:
        """Classify directly from raw per-frame lists (no feature extraction)."""
        rows = []
        for i in range(len(centroids)):
            c = centroids[i]
            x, y, z = (float(c[0]), float(c[1]), float(c[2])) if c is not None else (0., 0., 0.)
            v = float(velocities[i]) if velocities[i] is not None else 0.
            n = float(counts[i]) if i < len(counts) else 0.
            s = float(spreads[i]) if i < len(spreads) else 0.
            rows.append([x, y, z, v, n, s])

        # Pad or trim to seq_len
        seq = np.array(rows, dtype=np.float32)
        if len(seq) < self._seq_len:
            pad = np.zeros((self._seq_len - len(seq), self._n_feat), np.float32)
            seq = np.vstack([pad, seq])
        else:
            seq = seq[-self._seq_len:]

        # Normalise
        seq_flat = self._scaler.transform(seq.reshape(-1, self._n_feat))
        seq_norm = np.nan_to_num(seq_flat.reshape(1, self._seq_len, self._n_feat))

        tensor = self._torch.tensor(seq_norm, dtype=self._torch.float32)
        with self._torch.no_grad():
            logits = self._model(tensor)[0]
            proba  = self._torch.softmax(logits, dim=0).numpy()

        idx   = int(np.argmax(proba))
        label = str(self._le.classes_[idx])
        return GestureResult(label, float(proba[idx]), time.time())

    def classify(self, f) -> GestureResult:
        """Fallback: classify from WindowFeatures (less accurate for DL)."""
        return GestureResult("unknown", 0.0, time.time())


# ── Attention classifier wrapper ───────────────────────────────────────────────

class AttnClassifier:
    """
    Wraps the CNN + Multi-Head Self-Attention + BiLSTM model
    trained by dl_attention_trainer.py.
    Uses the same raw-sequence interface as DLClassifier.
    """

    def __init__(self, model_path: str, label_names: List[str]):
        try:
            import torch
            self._torch = torch
        except ImportError:
            raise ImportError("PyTorch required. Run: pip install torch")

        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Attention model not found: {p}\n"
                "Run  python -m classification.dl_attention_trainer  first."
            )

        checkpoint = torch.load(str(p), map_location="cpu", weights_only=False)
        from classification.dl_attention_trainer import GestureNetAttention

        self._le      = checkpoint["label_encoder"]
        self._scaler  = checkpoint["scaler"]
        self._n_feat  = checkpoint["n_features"]
        self._seq_len = checkpoint["seq_len"]

        self._model = GestureNetAttention(
            checkpoint["n_features"],
            checkpoint["n_classes"],
            checkpoint["seq_len"],
        )
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()
        self._labels = label_names
        log.info("Attention model loaded from %s", p)

    @staticmethod
    def _normalize_trajectory(seq: np.ndarray) -> np.ndarray:
        """
        Translate trajectory so the first active centroid = (0, 0, 0).
        Must exactly match _normalize_trajectory() in dl_attention_trainer.py.
        Only active frames (count > 0, col 4) are shifted.
        """
        seq = seq.copy()
        active = seq[:, 4] > 0
        if not active.any():
            return seq
        ref = seq[np.argmax(active), :3].copy()
        seq[active, 0] -= ref[0]
        seq[active, 1] -= ref[1]
        seq[active, 2] -= ref[2]
        seq[active, 6]  = np.sqrt(seq[active, 0]**2 + seq[active, 1]**2)
        return seq

    @staticmethod
    def _add_motion_features(seq: np.ndarray) -> np.ndarray:
        """
        Add 6 motion features (cols 9-14) to a single (W, 9) window.
        Output: (W, 15)
        Must exactly match _add_motion_features_batch() in dl_attention_trainer.py.

        [9]  vx          lateral velocity
        [10] vy          depth velocity
        [11] vz          vertical velocity
        [12] dist_xz     XZ-plane distance from gesture start
        [13] angular_xz  rotation direction (CCW→+, CW→-)
        [14] z_fraction  fraction of motion that is vertical
        """
        W = len(seq)
        result = np.zeros((W, 15), dtype=np.float32)
        result[:, :9] = seq
        active = seq[:, 4] > 0

        for t in range(1, W):
            if active[t] and active[t - 1]:
                vx = seq[t, 0] - seq[t - 1, 0]
                vy = seq[t, 1] - seq[t - 1, 1]
                vz = seq[t, 2] - seq[t - 1, 2]
                result[t, 9]  = vx
                result[t, 10] = vy
                result[t, 11] = vz
                # angular velocity in XZ plane: x[t-1]*vz - z[t-1]*vx
                result[t, 13] = seq[t - 1, 0] * vz - seq[t - 1, 2] * vx

        # dist_xz: XZ-plane distance from normalized start (x,z = 0 at start)
        for t in range(W):
            if active[t]:
                result[t, 12] = float(np.sqrt(seq[t, 0]**2 + seq[t, 2]**2))

        # z_fraction: fraction of instantaneous motion that is vertical
        for t in range(W):
            if active[t]:
                total = abs(result[t, 9]) + abs(result[t, 10]) + abs(result[t, 11]) + 1e-6
                result[t, 14] = result[t, 11] / total

        return result

    def classify_raw(self, centroids, velocities, counts, spreads) -> GestureResult:
        """
        Build a 9-feature sequence with carry-forward on empty frames,
        then apply trajectory normalization to match training format.

        Feature vector per frame (9):
          [x, y, z, velocity, count, spread, radial_dist, norm_count, abs_velocity]
        After normalization: x, y, z are relative to the first active centroid.
        """
        rows = []
        last_x, last_y, last_z = 0., 0., 0.

        for i in range(len(centroids)):
            c   = centroids[i]
            cnt = float(counts[i])   if i < len(counts)   else 0.
            spr = float(spreads[i])  if i < len(spreads)  else 0.

            if c is not None and cnt > 0:
                x, y, z = float(c[0]), float(c[1]), float(c[2])
                v       = float(velocities[i]) if velocities[i] is not None else 0.
                last_x, last_y, last_z = x, y, z
            else:
                x, y, z = last_x, last_y, last_z
                v       = 0.
                cnt     = 0.
                spr     = 0.

            radial   = float(np.sqrt(x**2 + y**2))
            norm_cnt = cnt / 15.0
            abs_v    = abs(v)
            rows.append([x, y, z, v, cnt, spr, radial, norm_cnt, abs_v])

        seq = np.array(rows, dtype=np.float32)
        if len(seq) < self._seq_len:
            pad = np.zeros((self._seq_len - len(seq), self._n_feat), np.float32)
            seq = np.vstack([pad, seq])
        else:
            seq = seq[-self._seq_len:]

        # Apply trajectory normalization — must match training exactly
        seq = self._normalize_trajectory(seq)

        # Add motion features (vx, vy, vz) — must match training exactly
        seq = self._add_motion_features(seq)   # (W, 9) → (W, 12)

        seq_flat = self._scaler.transform(seq.reshape(-1, self._n_feat))
        seq_norm = np.nan_to_num(seq_flat.reshape(1, self._seq_len, self._n_feat))

        tensor = self._torch.tensor(seq_norm, dtype=self._torch.float32)
        with self._torch.no_grad():
            logits = self._model(tensor)[0]
            proba  = self._torch.softmax(logits, dim=0).numpy()

        idx   = int(np.argmax(proba))
        label = str(self._le.classes_[idx])
        return GestureResult(label, float(proba[idx]), time.time())

    def classify(self, f) -> GestureResult:
        return GestureResult("unknown", 0.0, time.time())


# ── Sliding-window gesture engine ─────────────────────────────────────────────

class GestureEngine:
    """
    Rolling-buffer onset/offset gesture detector.

    Key design: a circular buffer always holds the last W frames (matching
    the training window size exactly). When onset/offset triggers, we
    classify the rolling buffer — the model receives the same dense
    20-frame format it was trained on, with the gesture naturally
    occupying the recent portion of the window.

    State machine:
      IDLE   → motion detected              → ACTIVE
      ACTIVE → IDLE_TIMEOUT silent frames   → classify rolling buffer → IDLE
    """

    _IDLE   = 0
    _ACTIVE = 1

    _IDLE_TIMEOUT = 5    # 5 × 50 ms = 250 ms silence → gesture ended
    _MIN_ACTIVE   = 3    # minimum active frames in rolling buffer to classify

    def __init__(
        self,
        settings: GestureSettings,
        classifier: "RuleBasedClassifier | MLClassifier",
    ):
        self._s   = settings
        self._clf = classifier
        W = settings.window_frames

        # Rolling buffer — always exactly W frames (pre-filled with empty)
        self._roll_c: Deque = deque([None] * W, maxlen=W)
        self._roll_v: Deque = deque([0.0]  * W, maxlen=W)
        self._roll_n: Deque = deque([0]    * W, maxlen=W)
        self._roll_s: Deque = deque([0.0]  * W, maxlen=W)

        self._state          = self._IDLE
        self._idle_count     = 0
        self._frame_counter  = 0
        self._last_emit_time: float = 0.0
        self._last_result: Optional[GestureResult] = None

    def push_frame(
        self,
        centroid: Optional[np.ndarray],
        mean_velocity: Optional[float],
        point_count: int = 0,
        spread: float = 0.0,
    ) -> Optional[GestureResult]:
        # Always update rolling buffer
        self._roll_c.append(centroid)
        self._roll_v.append(mean_velocity if mean_velocity is not None else 0.0)
        self._roll_n.append(point_count)
        self._roll_s.append(spread)
        self._frame_counter += 1

        active = centroid is not None and point_count > 0

        if self._state == self._IDLE:
            if active:
                self._state      = self._ACTIVE
                self._idle_count = 0
            return None

        # --- ACTIVE state ---
        if active:
            self._idle_count = 0
        else:
            self._idle_count += 1

        # Trigger 1: idle timeout (gesture motion ended)
        if self._idle_count >= self._IDLE_TIMEOUT:
            result = self._classify_gesture()
            self._state      = self._IDLE
            self._idle_count = 0
            return result

        # Trigger 2: sliding window — classify every hop_frames while active
        # Only fires after the buffer has seen enough ACTIVE gesture frames
        # (not warmup/background frames) to avoid false positives
        active_in_buffer = sum(1 for x in self._roll_c if x is not None)
        active_since_start = self._idle_count == 0 and self._frame_counter >= self._s.hop_frames
        if (active and
                active_in_buffer >= 10 and
                active_since_start and
                self._frame_counter % self._s.hop_frames == 0):
            return self._classify_gesture()

        return None

    def _classify_gesture(self) -> Optional[GestureResult]:
        c = list(self._roll_c)
        v = list(self._roll_v)
        n = list(self._roll_n)
        s = list(self._roll_s)

        active_frames = sum(1 for x in c if x is not None)
        if active_frames < self._MIN_ACTIVE:
            return None

        # Paper improvement #4 (m-Activity): require meaningful total point mass
        total_points = sum(x for x in n if x > 0)
        if total_points < 6:
            return None

        # Cooldown
        now = time.time()
        if now - self._last_emit_time < self._s.gesture_cooldown_s:
            return None

        if isinstance(self._clf, (DLClassifier, AttnClassifier)):
            result = self._clf.classify_raw(c, v, n, s)
        else:
            f = extract_features(c, v, n, s)
            result = self._clf.classify(f)

        if result.label == "static" or result.confidence < self._s.min_confidence:
            return None

        self._last_emit_time = now
        self._last_result    = result
        return result

    @property
    def last_result(self) -> Optional[GestureResult]:
        return self._last_result