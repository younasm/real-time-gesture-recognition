# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
classification/dl_attention_trainer.py
CNN + Multi-Head Self-Attention + BiLSTM gesture classifier.

Architecture improvements over the baseline CNN-BiLSTM:
  ┌─────────────────────────────────────────────────────────┐
  │  Input  (B, 20, 6)                                       │
  │    ↓  Linear projection + LayerNorm                      │
  │  (B, 20, 64)                                             │
  │    ↓  Learnable positional encoding                      │
  │  (B, 20, 64)                                             │
  │    ↓  Transformer encoder (2 layers, 4 heads)            │
  │       → self-attention learns which frames matter most   │
  │  (B, 20, 64)                                             │
  │    ↓  Bidirectional LSTM (2 layers)                      │
  │  (B, 20, 256)                                            │
  │    ↓  Attention pooling (soft-weights over time steps)   │
  │  (B, 256)                                                │
  │    ↓  FC head with LayerNorm + Dropout                   │
  │  (B, n_classes)                                          │
  └─────────────────────────────────────────────────────────┘

Key improvements:
  • Multi-head self-attention focuses on the most informative gesture frames
  • Learnable positional encoding preserves temporal order
  • Attention pooling weights time steps adaptively instead of taking last state
  • Label smoothing prevents overconfident wrong predictions
  • Richer augmentation with explicit mirror symmetry for circle_cw/ccw balance

Usage
─────
  python -m classification.dl_attention_trainer
  python -m classification.dl_attention_trainer --epochs 80 --augment 10
"""

import argparse
import csv as _csv
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_curve, auc)
    from sklearn.preprocessing import label_binarize
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

from config.settings import GestureSettings

GESTURE_SETTINGS = GestureSettings()
WINDOW       = GESTURE_SETTINGS.window_frames   # 20
N_FEAT       = 9    # base features used by data loading and augmentation
N_FEAT_FULL  = 15   # final features fed to model (base + vx,vy,vz + dist_xz,angular_xz,z_frac)
OUT_DIR      = "classification/models/attention"


# ── Feature helpers ───────────────────────────────────────────────────────────

def _expand_frames(data: np.ndarray) -> np.ndarray:
    """
    Expand (N, 6) recordings to (N, 9) by appending 3 derived features:
      [6] radial_dist  = sqrt(x²+y²)   — XY-plane distance from sensor
      [7] norm_count   = count / 15.0   — normalised point density
      [8] abs_velocity = |velocity|     — velocity magnitude (direction-agnostic)
    All three can be computed from existing recordings without re-collection.
    """
    x      = data[:, 0];  y   = data[:, 1]
    vel    = data[:, 3];  cnt = data[:, 4]
    radial   = np.sqrt(x**2 + y**2)
    norm_cnt = cnt / 15.0
    abs_vel  = np.abs(vel)
    return np.column_stack(
        [data, radial, norm_cnt, abs_vel]
    ).astype(np.float32)


def _recompute_derived(w: np.ndarray) -> np.ndarray:
    """Recompute derived feature columns (6-8) after augmentation changes base cols."""
    w[:, 6] = np.sqrt(w[:, 0]**2 + w[:, 1]**2)   # radial_dist
    w[:, 7] = w[:, 4] / 15.0                        # norm_count
    w[:, 8] = np.abs(w[:, 3])                       # abs_velocity
    return w


def _normalize_trajectory(w: np.ndarray) -> np.ndarray:
    """
    Trajectory normalization — translate so the first active centroid = (0, 0, 0).

    Why this helps:
      After normalization, the model directly sees DISPLACEMENT from gesture start:
        swipe_right → x increases from 0 to +Δx
        swipe_left  → x decreases from 0 to -Δx
        swipe_up    → z increases from 0 to +Δz
        swipe_down  → z decreases from 0 to -Δz
        push        → y decreases from 0 to -Δy  (hand moves toward sensor)
        pull        → y increases from 0 to +Δy

    Only active frames (count > 0) are shifted — inactive frames stay at (0,0,0)
    so the model can still detect gesture onset/offset.
    Radial distance (col 6) is recomputed from normalized x, y.
    """
    w = w.copy()
    active = w[:, 4] > 0          # frames where point_count > 0
    if not active.any():
        return w

    # Reference: first active centroid
    ref = w[np.argmax(active), :3].copy()

    # Shift only active frames
    w[active, 0] -= ref[0]        # x relative to start
    w[active, 1] -= ref[1]        # y relative to start
    w[active, 2] -= ref[2]        # z relative to start

    # Recompute radial from normalised x, y
    w[active, 6] = np.sqrt(w[active, 0]**2 + w[active, 1]**2)
    return w


def _add_motion_features_batch(X: np.ndarray) -> np.ndarray:
    """
    Add 6 motion-derived features (cols 9-14) to base 9-feature windows.
    Input : (N, WINDOW, 9)   — after trajectory normalization
    Output: (N, WINDOW, 15)

    Features added
    ──────────────
    [9]  vx  = x[t]-x[t-1]   lateral velocity       swipe_right→+, swipe_left→-
    [10] vy  = y[t]-y[t-1]   depth velocity          pull→+, push→-
    [11] vz  = z[t]-z[t-1]   vertical velocity       swipe_up→+, swipe_down→-
    [12] dist_xz              XZ-plane distance from  circles: peaks then returns
                              gesture start           swipes: monotonically grows
    [13] angular_vel_xz       rotation direction in   CCW→+, CW→-
                              XZ plane                (directly encodes circle direction)
    [14] z_fraction           vz/(|vx|+|vy|+|vz|+ε) swipe_up→≈+1, swipe_down→≈-1
                                                      push/pull→≈0 (vy dominates)

    vx, vy, vz only computed where both current AND previous frame are active.
    angular_vel_xz = x[t-1]*vz[t] - z[t-1]*vx[t]  (cross product, signed area)
    """
    N, W, _ = X.shape
    result = np.zeros((N, W, N_FEAT_FULL), dtype=np.float32)
    result[:, :, :N_FEAT] = X

    active      = X[:, :, 4] > 0
    both_active = active[:, 1:] & active[:, :-1]   # (N, W-1)

    dx = X[:, 1:, 0] - X[:, :-1, 0]
    dy = X[:, 1:, 1] - X[:, :-1, 1]
    dz = X[:, 1:, 2] - X[:, :-1, 2]

    # vx, vy, vz
    result[:, 1:, 9][both_active]  = dx[both_active]
    result[:, 1:, 10][both_active] = dy[both_active]
    result[:, 1:, 11][both_active] = dz[both_active]

    # dist_xz: XZ-plane distance from gesture start (x and z are 0-normalised)
    result[:, :, 12] = np.sqrt(X[:, :, 0]**2 + X[:, :, 2]**2) * (active)

    # angular_vel_xz: x[t-1]*vz[t] - z[t-1]*vx[t]  → CCW=+, CW=-
    angular = (X[:, :-1, 0] * dz - X[:, :-1, 2] * dx)
    result[:, 1:, 13][both_active] = angular[both_active]

    # z_fraction: fraction of instantaneous motion that is vertical
    total_vel = (np.abs(result[:, :, 9]) +
                 np.abs(result[:, :, 10]) +
                 np.abs(result[:, :, 11]) + 1e-6)
    result[:, :, 14] = result[:, :, 11] / total_vel * (active)

    return result


# ── Model ─────────────────────────────────────────────────────────────────────

class GestureNetAttention(nn.Module):
    """
    1D CNN + Multi-Head Self-Attention + BiLSTM gesture classifier.

    Architecture
    ────────────
    Input (B, T, F)
      ↓  1D CNN  — extracts LOCAL motion patterns across adjacent frames
         kernel_size=3: each output frame sees 3 consecutive input frames
         This is the key improvement over the plain Linear projection
    (B, T, d_model)
      ↓  Learnable positional encoding
      ↓  Transformer encoder (2 layers, 4 heads)
         — captures GLOBAL dependencies across the full gesture window
    (B, T, d_model)
      ↓  BiLSTM (2 layers)
         — models sequential order and direction
    (B, T, n_lstm*2)
      ↓  Attention pooling
         — soft-weights each frame by importance
    (B, n_lstm*2)
      ↓  FC head
    (B, n_classes)
    """

    def __init__(self, n_features=N_FEAT, n_classes=10, seq_len=WINDOW,
                 d_model=64, n_heads=4, n_lstm=128, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len

        # ── 1D CNN feature extractor (replaces plain Linear projection) ────
        # Conv1d operates on (B, channels, T) — we transpose in forward()
        # kernel_size=3 with padding=1 preserves sequence length
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

        # ── Learnable positional encoding ─────────────────────────────────
        self.pos_enc = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_enc, std=0.02)

        # ── Transformer encoder (2 layers, 4 attention heads) ─────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # ── Bidirectional LSTM ────────────────────────────────────────────
        self.lstm = nn.LSTM(
            d_model, n_lstm, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout,
        )

        # ── LayerNorm after LSTM (improvement 5 — from Pantomime paper) ──────
        # Normalises LSTM outputs per time-step so sparse frames don't
        # push activations out-of-distribution at inference time
        self.lstm_norm = nn.LayerNorm(n_lstm * 2)

        # ── Attention pooling — learns which frames to focus on ────────────
        self.attn_pool = nn.Sequential(
            nn.Linear(n_lstm * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # ── Classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(n_lstm * 2, 256),
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        B, T, _ = x.shape

        # 1D CNN: needs (B, F, T) → outputs (B, d_model, T) → back to (B, T, d_model)
        x = x.transpose(1, 2)                          # (B, F, T)
        x = self.cnn(x)                                # (B, d_model, T)
        x = x.transpose(1, 2)                          # (B, T, d_model)

        # Add positional encoding
        x = x + self.pos_enc[:, :T, :]

        # Self-attention over all time steps
        x = self.transformer(x)                        # (B, T, d_model)

        # BiLSTM temporal modelling
        lstm_out, _ = self.lstm(x)                     # (B, T, n_lstm*2)
        lstm_out = self.lstm_norm(lstm_out)            # LayerNorm per time-step

        # Soft attention pooling
        attn_w = torch.softmax(self.attn_pool(lstm_out), dim=1)  # (B, T, 1)
        ctx = (lstm_out * attn_w).sum(dim=1)           # (B, n_lstm*2)

        return self.head(ctx)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-frame attention weights for visualisation."""
        B, T, _ = x.shape
        x = x.transpose(1, 2)
        x = self.cnn(x).transpose(1, 2)
        x = x + self.pos_enc[:, :T, :]
        x = self.transformer(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        return torch.softmax(self.attn_pool(lstm_out), dim=1).squeeze(-1)  # (B, T)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_windows(data_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load training windows with zero-frame filtering.

    For gesture classes (non-static):
      1. Standard sliding window — only windows with ≥ 60% active frames
      2. Dense-only extraction  — strip all zero rows, slide over active frames
         This ensures the model trains on clean gesture data without zero padding.

    For static class: zero frames are intentional, use as-is.
    """
    X, y = [], []
    hop = GESTURE_SETTINGS.hop_frames

    for fp in sorted(Path(data_dir).glob("*.npy")):
        label = fp.stem.rsplit("_", 1)[0]
        data  = np.load(fp).astype(np.float32)
        if len(data) < WINDOW:
            continue

        if label == "static":
            for start in range(0, len(data) - WINDOW + 1, hop):
                win9 = _expand_frames(data[start: start + WINDOW])
                # No trajectory normalization for static — no meaningful start point
                X.append(win9)
                y.append(label)
        else:
            # ── Pass 1: standard sliding window, require ≥ 60% active ─────
            for start in range(0, len(data) - WINDOW + 1, hop):
                win    = data[start: start + WINDOW]
                active = np.any(win[:, :3] != 0, axis=1).mean()
                if active >= 0.60:
                    X.append(_normalize_trajectory(_expand_frames(win)))
                    y.append(label)

            # ── Pass 2: dense-only extraction — strip zero rows entirely ───
            active_mask   = np.any(data[:, :3] != 0, axis=1)
            active_frames = data[active_mask]
            if len(active_frames) >= WINDOW:
                for start in range(0, len(active_frames) - WINDOW + 1,
                                   max(1, hop // 2)):
                    X.append(_normalize_trajectory(
                        _expand_frames(active_frames[start: start + WINDOW])
                    ))
                    y.append(label)
            elif len(active_frames) >= 4:
                pad  = WINDOW - len(active_frames)
                pre  = pad // 2
                post = pad - pre
                dense9 = _expand_frames(active_frames)
                win9   = np.vstack([
                    np.zeros((pre,  N_FEAT), dtype=np.float32),
                    dense9,
                    np.zeros((post, N_FEAT), dtype=np.float32),
                ])
                X.append(_normalize_trajectory(win9))
                y.append(label)
    return X, y


def augment(X_raw: List[np.ndarray], y_raw: List[str],
            n_copies: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Richer augmentation with explicit mirror symmetry to balance circle_cw/ccw.
    Each original window produces n_copies augmented variants.
    """
    rng = np.random.default_rng(42)
    X_aug, y_aug = list(X_raw), list(y_raw)

    mirror_map = {
        "circle_cw":   "circle_ccw",
        "circle_ccw":  "circle_cw",
        "swipe_left":  "swipe_right",
        "swipe_right": "swipe_left",
    }

    for win, lbl in zip(X_raw, y_raw):
        # Always add one explicit mirror for symmetric gestures
        if lbl in mirror_map:
            m = win.copy()
            m[:, 0] = -m[:, 0]   # flip x
            m[:, 3] = -m[:, 3]   # flip doppler
            _recompute_derived(m)
            X_aug.append(m)
            y_aug.append(mirror_map[lbl])

        for _ in range(n_copies):
            w = win.copy()
            aug_type = rng.integers(8)  # 8 augmentation types

            if aug_type == 0:
                # Gaussian noise on xyz and velocity
                w[:, :3] += rng.normal(0, 0.015, (WINDOW, 3))
                w[:,  3] += rng.normal(0, 0.04,  WINDOW)
                _recompute_derived(w)

            elif aug_type == 1:
                # Scale amplitude (distance variation)
                w[:, :3] *= rng.uniform(0.80, 1.20)
                _recompute_derived(w)

            elif aug_type == 2:
                # Temporal shift (pad with zeros)
                s   = rng.integers(1, 5)
                pad = np.zeros((s, N_FEAT), dtype=np.float32)
                w   = (np.vstack([w[s:], pad]) if rng.random() > 0.5
                       else np.vstack([pad, w[:-s]]))

            elif aug_type == 3:
                # Speed perturbation (resample temporal axis)
                factor = rng.uniform(0.8, 1.2)
                old_t  = np.linspace(0, 1, WINDOW)
                new_t  = np.clip(np.linspace(0, 1, WINDOW) * factor, 0, 1)
                for col in range(N_FEAT):
                    w[:, col] = np.interp(new_t, old_t, w[:, col])
                _recompute_derived(w)

            elif aug_type == 4:
                # Random frame dropout — simulate sparse detection
                n_drop = rng.integers(2, 8)
                idx    = rng.choice(WINDOW, n_drop, replace=False)
                w[idx] = 0.0

            elif aug_type == 5:
                # Rolling-buffer simulation: gesture in last N frames only
                # Directly matches live inference rolling-buffer pattern
                n_pre = rng.integers(6, 14)
                w[:n_pre] = 0.0

            elif aug_type == 6:
                # Aggressive sparse simulation (paper improvement #1)
                # Keep only 4-8 random frames, zero rest
                n_keep = rng.integers(4, 9)
                keep   = rng.choice(WINDOW, n_keep, replace=False)
                mask   = np.zeros(WINDOW, dtype=bool)
                mask[keep] = True
                w[~mask] = 0.0

            else:  # aug_type == 7
                # Slight rotation in XZ plane
                theta = rng.uniform(-0.2, 0.2)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                x_orig = w[:, 0].copy()
                z_orig = w[:, 2].copy()
                w[:, 0] = cos_t * x_orig - sin_t * z_orig
                w[:, 2] = sin_t * x_orig + cos_t * z_orig
                _recompute_derived(w)

            X_aug.append(w)
            y_aug.append(lbl)

    return np.array(X_aug, np.float32), np.array(y_aug)


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_dir: str = "data/recordings",
          output_dir: str = OUT_DIR,
          epochs: int = 80,
          lr: float = 3e-4,
          batch_size: int = 64,
          n_augment: int = 8,
          label_smoothing: float = 0.1) -> None:

    if not TORCH_OK:
        print("PyTorch not installed. Run:  pip install torch"); return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device  : {device}")
    print(f"  Out dir : {out.resolve()}\n")

    # ── Load recordings and split by FILE before windowing ───────────────
    # Fix: split by recording file (group), not by individual window.
    # Splitting after windowing leaks overlapping windows from the same
    # recording across train/val, inflating reported accuracy.
    print(f"  Loading recordings from {data_dir}…")
    all_files = sorted(Path(data_dir).glob("*.npy"))
    if not all_files:
        print("No recordings found."); return

    # Stratified split by recording file — 85% train / 15% val per class.
    # Each class contributes ~15% of its recordings to val, so no class is
    # over/under-represented. Recordings are kept whole (no window leakage).
    from collections import defaultdict
    rng = np.random.default_rng(42)
    by_class = defaultdict(list)
    for fp in all_files:
        by_class[fp.stem.rsplit("_", 1)[0]].append(fp)

    train_files, val_files = [], []
    for cls, files in sorted(by_class.items()):
        files = list(rng.permutation(files))
        n_val = max(1, round(len(files) * 0.15))
        val_files.extend(files[:n_val])
        train_files.extend(files[n_val:])

    print(f"  Recordings    : {len(train_files)} train / {len(val_files)} val "
          f"(stratified by class, split before windowing — no leakage)")

    def _files_to_windows(files):
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        for f in files:
            shutil.copy(f, tmp / f.name)
        X_w, y_w = load_windows(str(tmp))
        shutil.rmtree(tmp)
        return X_w, y_w

    X_tr_raw, y_tr_raw = _files_to_windows(train_files)
    X_val_raw, y_val_raw = _files_to_windows(val_files)
    print(f"  Raw windows   : {len(X_tr_raw)} train / {len(X_val_raw)} val")

    if n_augment == 0:
        print("  No augmentation — training on real data only.")
        X_tr_aug  = np.array(X_tr_raw,  dtype=np.float32)
        y_tr_aug  = np.array(y_tr_raw)
    else:
        print(f"  Augmenting train set ×{n_augment} + mirror symmetry…")
        X_tr_aug, y_tr_aug = augment(X_tr_raw, y_tr_raw, n_augment)
    # Validation: no augmentation (evaluate on real data only)
    X_val_aug = np.array(X_val_raw, dtype=np.float32)
    y_val_aug = np.array(y_val_raw)
    print(f"  After augment : {len(X_tr_aug)} train / {len(X_val_aug)} val windows")

    # Add motion features
    X_tr_aug  = _add_motion_features_batch(X_tr_aug)
    X_val_aug = _add_motion_features_batch(X_val_aug)
    print(f"  Features      : {N_FEAT} base + 6 motion = {N_FEAT_FULL}")

    le = LabelEncoder()
    le.fit(np.concatenate([y_tr_aug, y_val_aug]))
    y_tr  = le.transform(y_tr_aug)
    y_val = le.transform(y_val_aug)
    n_classes = len(le.classes_)
    print(f"  Classes ({n_classes}) : {list(le.classes_)}")

    # Fit scaler on TRAIN only — no val data leaks into normalisation
    scaler = StandardScaler()
    X_tr_flat  = scaler.fit_transform(X_tr_aug.reshape(-1, N_FEAT_FULL))
    X_val_flat = scaler.transform(X_val_aug.reshape(-1, N_FEAT_FULL))
    X_tr_norm  = np.nan_to_num(X_tr_flat.reshape(-1, WINDOW, N_FEAT_FULL))
    X_val_norm = np.nan_to_num(X_val_flat.reshape(-1, WINDOW, N_FEAT_FULL))

    counts  = np.bincount(y_tr)
    weights = 1.0 / counts[y_tr]
    sampler = WeightedRandomSampler(weights, len(weights))

    tr_ds  = TensorDataset(torch.tensor(X_tr_norm,  dtype=torch.float32),
                            torch.tensor(y_tr,       dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val_norm, dtype=torch.float32),
                            torch.tensor(y_val,      dtype=torch.long))
    tr_loader  = DataLoader(tr_ds,  batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    model = GestureNetAttention(N_FEAT_FULL, n_classes, WINDOW).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters : {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    crit  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_acc, best_state, patience_count = 0.0, None, 0
    patience = 15
    history  = {"epoch": [], "loss": [], "val_acc": []}

    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'Val acc':>8}")
    print("─" * 28)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * len(xb)
        sched.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                correct += (model(xb.to(device)).argmax(1) == yb.to(device)).sum().item()
        val_acc  = correct / len(val_ds)
        avg_loss = total_loss / len(tr_ds)

        history["epoch"].append(ep)
        history["loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        marker = " ← best" if val_acc > best_acc else ""
        if ep % 5 == 0 or val_acc > best_acc:
            print(f"{ep:>6}  {avg_loss:>8.4f}  {val_acc:>7.1%}{marker}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {ep}"); break

    # ── Save model ────────────────────────────────────────────────────────
    pt_path = out / "gesture_attn.pt"
    torch.save({
        "model_state":   best_state,
        "n_features":    N_FEAT_FULL,   # 12 (includes vx, vy, vz)
        "n_classes":     n_classes,
        "seq_len":       WINDOW,
        "label_encoder": le,
        "scaler":        scaler,
        "architecture":  "GestureNetAttention",
    }, str(pt_path))
    joblib.dump(scaler, str(out / "gesture_attn_scaler.joblib"))
    print(f"\n  Best val accuracy : {best_acc:.1%}")
    print(f"  Model saved       → {pt_path}")

    # ── Collect predictions ───────────────────────────────────────────────
    model.load_state_dict(best_state); model.to(device).eval()
    preds, trues, probs_all = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            probs_all.extend(torch.softmax(logits, dim=1).cpu().numpy())
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(yb.numpy())
    preds     = np.array(preds)
    trues     = np.array(trues)
    probs_all = np.array(probs_all)

    # ── Save matplotlib artifacts ─────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    report_str  = classification_report(trues, preds,
                                        target_names=le.classes_, zero_division=0)
    report_dict = classification_report(trues, preds, target_names=le.classes_,
                                        zero_division=0, output_dict=True)

    print("\n── Validation Report ─────────────────────────────────────")
    print(report_str)

    # 1. Text report
    (out / "gesture_attn_report.txt").write_text(
        "CNN + Multi-Head Self-Attention + BiLSTM — Validation Report\n"
        f"Best val accuracy: {best_acc:.4f}\n\n" + report_str
    )

    # 2. CSV report
    with open(out / "gesture_attn_report.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1-score", "support"])
        for cls in le.classes_:
            r = report_dict[cls]
            w.writerow([cls, f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                        f"{r['f1-score']:.4f}", int(r['support'])])
        w.writerow([])
        w.writerow(["accuracy", "", "", f"{report_dict['accuracy']:.4f}", len(trues)])
    print(f"  Reports saved     → {out}/gesture_attn_report.txt / .csv")

    # 3. Confusion matrices
    cm      = confusion_matrix(trues, preds)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        [".0f", ".2f"],
    ):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set(xticks=range(n_classes), yticks=range(n_classes),
               xticklabels=le.classes_, yticklabels=le.classes_,
               xlabel="Predicted", ylabel="True")  # title=title
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        thresh = data.max() / 2
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        fontsize=8,
                        color="white" if data[i, j] > thresh else "black")
    # fig.suptitle("Confusion Matrix — CNN + Self-Attention + BiLSTM",
    #              fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out / "gesture_attn_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["epoch"], history["loss"], color="tab:blue", linewidth=1.5)
    # title="Training Loss" removed
    ax1.set(xlabel="Epoch", ylabel="Training Loss (with label smoothing)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["epoch"], [v * 100 for v in history["val_acc"]],
             color="tab:green", linewidth=1.5)
    ax2.axhline(best_acc * 100, color="red", linestyle="--", linewidth=1,
                label=f"Best {best_acc:.1%}")
    # title="Validation Accuracy" removed
    ax2.set(xlabel="Epoch", ylabel="Validation Accuracy (%)")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax2.legend(); ax2.grid(True, alpha=0.3)
    # fig.suptitle("Training History — CNN + Self-Attention + BiLSTM",
    #              fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out / "gesture_attn_training_history.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 5. ROC curves
    y_bin = label_binarize(trues, classes=range(n_classes))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, (cls, ax) in enumerate(zip(le.classes_, axes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs_all[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="tab:blue", lw=1.5, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8, linestyle="--")
        ax.set(xlim=[0, 1], ylim=[0, 1.02],
               xlabel="FPR", ylabel="TPR", title=cls)   # gesture name as subplot title
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    # fig.suptitle("Per-Class ROC Curves — CNN + Self-Attention + BiLSTM",
    #              fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out / "gesture_attn_roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  All artifacts saved to {out}/")
    print("  Files:")
    print("    gesture_attn.pt               ← model weights")
    print("    gesture_attn_report.txt / .csv← classification report")
    print("    gesture_attn_confusion_matrix.png")
    print("    gesture_attn_training_history.png")
    print("    gesture_attn_roc_curves.png")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train CNN + Multi-Head Self-Attention + BiLSTM gesture classifier"
    )
    ap.add_argument("--data_dir",        default="data/recordings")
    ap.add_argument("--output_dir",      default=OUT_DIR)
    ap.add_argument("--epochs",          type=int,   default=80)
    ap.add_argument("--lr",              type=float, default=3e-4)
    ap.add_argument("--batch_size",      type=int,   default=64)
    ap.add_argument("--augment",         type=int,   default=8)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    args = ap.parse_args()
    train(args.data_dir, args.output_dir, args.epochs, args.lr,
          args.batch_size, args.augment, args.label_smoothing)
