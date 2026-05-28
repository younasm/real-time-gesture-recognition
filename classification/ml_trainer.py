# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
Advanced gesture classifier trainer with:
  • Data augmentation (noise, scaling, time-shift, mirror)
  • Normalised trajectory features (position-independent)
  • Multi-model comparison (RF, SVM, GradientBoosting, MLP)
  • Feature standardisation + pipeline
  • Stratified k-fold with confusion matrix

Usage
─────
  python -m classification.ml_trainer
  python -m classification.ml_trainer --data_dir data/recordings --augment 8
"""

import argparse
import warnings
from pathlib import Path
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from config.settings import GestureSettings
from processing.features import extract_features

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

GESTURE_SETTINGS = GestureSettings()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_recordings(data_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    X_raw, y_labels = [], []
    for fp in sorted(Path(data_dir).glob("*.npy")):
        label = fp.stem.rsplit("_", 1)[0]
        data  = np.load(fp)
        if len(data) < 4:          # skip empty / very short recordings
            continue
        X_raw.append(data)
        y_labels.append(label)
    print(f"  Loaded {len(X_raw)} recordings from {data_dir}")
    return X_raw, y_labels


# ── Data augmentation ─────────────────────────────────────────────────────────

def augment_recording(frames: np.ndarray, n_copies: int) -> List[np.ndarray]:
    """
    Generate n_copies augmented variants of one recording.

    Techniques
    ──────────
    noise      – Gaussian noise on xyz (σ = 1 cm) and velocity (σ = 0.03 m/s)
    scale      – uniform scale of xyz trajectory ×[0.85 – 1.15]
    time_shift – drop/pad 1-3 frames at start or end
    mirror_x   – negate the x axis (left↔right mirror)
    """
    rng = np.random.default_rng()
    augmented = []
    for _ in range(n_copies):
        f = frames.copy().astype(np.float32)
        choice = rng.integers(4)

        if choice == 0:   # noise
            f[:, :3] += rng.normal(0, 0.01, f[:, :3].shape)
            f[:,  3] += rng.normal(0, 0.03, f[:,  3].shape)

        elif choice == 1: # spatial scale
            scale = rng.uniform(0.85, 1.15)
            f[:, :3] *= scale

        elif choice == 2: # time shift (trim + pad with zeros)
            shift = rng.integers(1, 4)
            direction = rng.choice([-1, 1])
            if direction == 1:          # trim start, pad end
                f = np.vstack([f[shift:], np.zeros((shift, f.shape[1]), dtype=np.float32)])
            else:                       # pad start, trim end
                f = np.vstack([np.zeros((shift, f.shape[1]), dtype=np.float32), f[:-shift]])

        elif choice == 3: # mirror x axis (swipe_left ↔ swipe_right, etc.)
            f[:, 0] = -f[:, 0]
            f[:,  3] = -f[:,  3]   # also flip Doppler sign

        augmented.append(f)
    return augmented


# ── Feature extraction ────────────────────────────────────────────────────────

def _frames_to_feature_vector(window: np.ndarray) -> np.ndarray:
    """
    Extract one feature vector from a fixed-length window.
    Trajectories are CENTRED so position is gesture-shape-only, not absolute.
    """
    centroids  = [w[:3] for w in window]
    velocities = [float(w[3]) for w in window]
    counts     = [int(w[4]) for w in window]
    spreads    = [float(w[5]) if window.shape[1] > 5 else 0.0 for w in window]

    f = extract_features(centroids, velocities, counts, spreads)
    base_vec = f.to_vector()   # existing features (scalar + traj)

    # ── Additional features ──────────────────────────────────────
    pos = np.array(centroids, dtype=np.float32)
    vel = np.array(velocities, dtype=np.float32)

    extra = []

    # 0. Active frame ratio (how much of the window has detections)
    extra.append(_active_ratio(window))

    # 1. Normalised trajectory (centred at first valid point)
    valid = np.any(pos != 0, axis=1)
    if valid.any():
        origin = pos[np.argmax(valid)]
        norm_pos = pos - origin
    else:
        norm_pos = pos

    # statistics of normalised trajectory
    extra += list(np.std(norm_pos, axis=0))          # xyz spread
    extra += list(np.ptp(norm_pos, axis=0))          # xyz range (peak-to-peak)

    # 2. Velocity statistics
    extra.append(float(np.std(vel)))
    extra.append(float(np.ptp(vel)))
    # velocity sign changes (oscillation count)
    sign_changes = int(np.sum(np.diff(np.sign(vel)) != 0))
    extra.append(float(sign_changes))

    # 3. Dominant frequency of velocity signal (simple FFT peak)
    fft_mag = np.abs(np.fft.rfft(vel - vel.mean()))
    dom_freq_idx = int(np.argmax(fft_mag[1:]) + 1) if len(fft_mag) > 1 else 0
    extra.append(float(dom_freq_idx))
    extra.append(float(fft_mag[dom_freq_idx] if dom_freq_idx < len(fft_mag) else 0))

    # 4. Trajectory arc-length in each axis separately
    if len(norm_pos) > 1:
        extra += list(np.sum(np.abs(np.diff(norm_pos, axis=0)), axis=0))
    else:
        extra += [0.0, 0.0, 0.0]

    # 5. Straightness ratio per axis (displacement / arc-length)
    end_disp   = np.abs(norm_pos[-1] - norm_pos[0]) if len(norm_pos) > 1 else np.zeros(3)
    arc_per_ax = extra[-3:]  # just appended above
    for d, a in zip(end_disp, arc_per_ax):
        extra.append(float(d / (a + 1e-6)))

    # 6. Circular score: variance of distance from centroid
    centroid_center = norm_pos.mean(axis=0)
    dists = np.linalg.norm(norm_pos - centroid_center, axis=1)
    extra.append(float(np.std(dists)))

    return np.concatenate([base_vec, np.array(extra, dtype=np.float32)])


def _active_ratio(window: np.ndarray) -> float:
    """Fraction of frames in window that have actual detections (non-zero centroid)."""
    return float(np.any(window[:, :3] != 0, axis=1).mean())


def recording_to_features(frames: np.ndarray,
                           label: str = "",
                           min_active: float = 0.4) -> np.ndarray:
    """
    Slide window over recording and return feature matrix.
    For motion gestures, skip windows where < min_active frames have detections
    (these windows look like 'static' and confuse the model).
    """
    W   = GESTURE_SETTINGS.window_frames
    hop = GESTURE_SETTINGS.hop_frames
    n   = len(frames)
    is_motion = label not in ("static", "")
    vecs = []
    for start in range(0, n - W + 1, hop):
        window = frames[start: start + W]
        # For motion gestures: skip nearly-empty windows
        if is_motion and _active_ratio(window) < min_active:
            continue
        vecs.append(_frames_to_feature_vector(window))
    return np.array(vecs) if vecs else np.zeros((0, 1))


# ── Build models ──────────────────────────────────────────────────────────────

def _make_models():
    """Return dict of name → sklearn Pipeline."""
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=400, max_depth=12,
            min_samples_split=3, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1)),
    ])

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                       class_weight="balanced", probability=True,
                       random_state=42)),
    ])

    gbt = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=300, max_depth=5,
            learning_rate=0.08, subsample=0.8,
            random_state=42)),
    ])

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu", solver="adam",
            max_iter=500, early_stopping=True,
            random_state=42)),
    ])

    # Soft-voting ensemble of the best individual models
    ensemble = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    VotingClassifier(
            estimators=[
                ("rf",  RandomForestClassifier(n_estimators=300, max_depth=10,
                                               class_weight="balanced",
                                               random_state=42, n_jobs=-1)),
                ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                            class_weight="balanced", probability=True,
                            random_state=42)),
                ("gbt", GradientBoostingClassifier(n_estimators=200,
                                                    max_depth=5,
                                                    random_state=42)),
            ],
            voting="soft",
        )),
    ])

    return {"RandomForest": rf, "SVM": svm,
            "GradientBoosting": gbt, "MLP": mlp,
            "Ensemble (RF+SVM+GBT)": ensemble}


# ── Main training function ────────────────────────────────────────────────────

def train(data_dir: str, output_path: str, n_augment: int = 8) -> None:
    # 1. Load
    recordings, labels = load_recordings(data_dir)
    if not recordings:
        print("No recordings found. Collect data first.")
        return

    # 2. Augment
    print(f"\n  Augmenting data ×{n_augment} per recording…")
    aug_recs, aug_lbls = list(recordings), list(labels)
    for rec, lbl in zip(recordings, labels):
        for arec in augment_recording(rec, n_augment):
            aug_recs.append(arec)
            aug_lbls.append(lbl)
    print(f"  Original: {len(recordings)} → Augmented: {len(aug_recs)} recordings")

    # 3. Feature extraction
    print("  Extracting features…")
    X_all, y_all = [], []
    for rec, lbl in zip(aug_recs, aug_lbls):
        feats = recording_to_features(rec, label=lbl)
        if len(feats) == 0:
            continue
        X_all.append(feats)
        y_all.extend([lbl] * len(feats))

    X = np.vstack(X_all)
    y = np.array(y_all)
    # Replace NaN / Inf that may arise from empty frames
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_classes = len(le.classes_)
    print(f"\n  Dataset after augmentation: {X.shape[0]} windows, "
          f"{X.shape[1]} features, {n_classes} classes")
    print(f"  Classes: {list(le.classes_)}")

    # 4. Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    # 5. Cross-validate all models and pick the best
    models = _make_models()
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n── Cross-validation (5-fold) ─────────────────────────────────")
    cv_results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_tr, y_tr, cv=skf,
                                 scoring="accuracy", n_jobs=-1)
        cv_results[name] = scores
        print(f"  {name:<28}  {scores.mean():.3f} ± {scores.std():.3f}")

    best_name = max(cv_results, key=lambda k: cv_results[k].mean())
    print(f"\n  Best model: {best_name}  "
          f"(CV={cv_results[best_name].mean():.3f})")

    # 6. Final fit on full training set with best model
    best_pipe = models[best_name]
    best_pipe.fit(X_tr, y_tr)

    # 7. Test set report
    y_pred = best_pipe.predict(X_te)
    print("\n── Test-set Classification Report ────────────────────────────")
    print(classification_report(y_te, y_pred,
                                target_names=le.classes_,
                                zero_division=0))

    acc = (y_pred == y_te).mean()
    print(f"  Test accuracy: {acc:.3f}")

    # 8. Save best model AND a fast RandomForest-only model for real-time use
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_pipe, "label_encoder": le}, str(out))
    print(f"\nModel saved → {out}")

    # Also save the RF model separately (faster for real-time inference)
    rf_pipe = models["RandomForest"]
    rf_pipe.fit(X_tr, y_tr)
    rt_path = out.parent / "gesture_rf_realtime.joblib"
    joblib.dump({"model": rf_pipe, "label_encoder": le}, str(rt_path))
    print(f"Fast real-time model saved → {rt_path}  (use this if Ensemble is too slow)")

    # 9. Confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_te, y_pred, display_labels=le.classes_, ax=ax,
        cmap="Blues", colorbar=False,
    )
    ax.set_title(f"Gesture Classifier — {best_name}  (test acc={acc:.2f})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    cm_path = str(out.with_suffix(".png"))
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved → {cm_path}")
    plt.show()

    # 10. All-models summary
    print("\n── All models summary ────────────────────────────────────────")
    for name, scores in sorted(cv_results.items(),
                                key=lambda kv: -kv[1].mean()):
        bar_len = int(scores.mean() * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {name:<28}  {bar}  {scores.mean():.3f}")

    print("\n  Tips to further improve accuracy:")
    print("  1. Collect MORE data — aim for 50+ samples per gesture")
    print("  2. Ensure consistent distance (0.5-1.5m) during collection")
    print("  3. Perform gestures clearly with a definite start/stop")
    print("  4. Enable ML classifier in settings.py:  use_ml_classifier=True")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Advanced gesture ML trainer")
    ap.add_argument("--data_dir", default="data/recordings")
    ap.add_argument("--output",   default="classification/models/gesture_rf.joblib")
    ap.add_argument("--augment",  type=int, default=8,
                    help="Augmented copies per original recording (default: 8)")
    args = ap.parse_args()
    train(args.data_dir, args.output, args.augment)