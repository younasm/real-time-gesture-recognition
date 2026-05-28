# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
Deep learning gesture classifier — CNN + Bidirectional LSTM.

Architecture
────────────
  Input  : (batch, 20 frames, 6 features)  raw centroid/velocity sequence
  1D CNN : extracts local motion patterns per timestep
  BiLSTM : captures temporal dependencies in both directions
  Head   : FC → 10 gesture classes

Usage
─────
  pip install torch
  python -m classification.dl_trainer
  python -m classification.dl_trainer --epochs 80 --lr 1e-3
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_curve, auc)
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

from config.settings import GestureSettings

GESTURE_SETTINGS = GestureSettings()
WINDOW = GESTURE_SETTINGS.window_frames   # 20
N_FEAT = 6   # x, y, z, velocity, point_count, spread


# ── Model ─────────────────────────────────────────────────────────────────────

class GestureNet(nn.Module):
    """Lightweight CNN-BiLSTM for real-time gesture recognition."""

    def __init__(self, n_features=N_FEAT, n_classes=10, seq_len=WINDOW):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)           # (B, features, seq)
        x = self.cnn(x)                  # (B, 64, seq)
        x = x.transpose(1, 2)           # (B, seq, 64)
        _, (h, _) = self.lstm(x)        # h: (4, B, 128)
        h = torch.cat([h[-2], h[-1]], 1) # (B, 256)
        return self.head(h)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_windows(data_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    X, y = [], []
    hop = GESTURE_SETTINGS.hop_frames
    for fp in sorted(Path(data_dir).glob("*.npy")):
        label = fp.stem.rsplit("_", 1)[0]
        data  = np.load(fp).astype(np.float32)
        if len(data) < WINDOW:
            continue
        for start in range(0, len(data) - WINDOW + 1, hop):
            win = data[start: start + WINDOW]
            active = np.any(win[:, :3] != 0, axis=1).mean()
            if label != "static" and active < 0.30:
                continue
            X.append(win)
            y.append(label)
    return X, y


def augment(X_raw, y_raw, n_copies=6):
    rng = np.random.default_rng(42)
    X_aug, y_aug = list(X_raw), list(y_raw)
    for win, lbl in zip(X_raw, y_raw):
        for _ in range(n_copies):
            w = win.copy()
            c = rng.integers(4)
            if c == 0:
                w[:, :3] += rng.normal(0, 0.01, (WINDOW, 3))
                w[:,  3] += rng.normal(0, 0.03, WINDOW)
            elif c == 1:
                w[:, :3] *= rng.uniform(0.85, 1.15)
            elif c == 2:
                s = rng.integers(1, 4)
                pad = np.zeros((s, N_FEAT), dtype=np.float32)
                w = np.vstack([w[s:], pad]) if rng.random() > 0.5 else np.vstack([pad, w[:-s]])
            else:
                w[:, 0] = -w[:, 0]; w[:, 3] = -w[:, 3]
            X_aug.append(w); y_aug.append(lbl)
    return np.array(X_aug, np.float32), np.array(y_aug)


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_dir="data/recordings", output_dir="classification/models",
          epochs=60, lr=5e-4, batch_size=64, n_augment=6):

    if not TORCH_OK:
        print("PyTorch not installed.  Run:  pip install torch"); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    print(f"\n  Loading recordings from {data_dir}…")
    X_raw, y_raw = load_windows(data_dir)
    if not X_raw:
        print("No recordings found. Collect data first."); return
    print(f"  Raw windows : {len(X_raw)}")

    print(f"  Augmenting ×{n_augment}…")
    X, y_str = augment(X_raw, y_raw, n_augment)
    print(f"  Total windows : {len(X)}")

    le = LabelEncoder()
    y  = le.fit_transform(y_str)
    n_classes = len(le.classes_)
    print(f"  Classes ({n_classes}) : {list(le.classes_)}")

    # Normalise
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X.reshape(-1, N_FEAT))
    X_norm = np.nan_to_num(X_flat.reshape(-1, WINDOW, N_FEAT))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_norm, y, test_size=0.15, random_state=42, stratify=y)

    # Weighted sampler for class balance
    counts  = np.bincount(y_tr)
    weights = 1.0 / counts[y_tr]
    sampler = WeightedRandomSampler(weights, len(weights))

    tr_ds  = TensorDataset(torch.tensor(X_tr,  dtype=torch.float32),
                            torch.tensor(y_tr,  dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.long))
    tr_loader  = DataLoader(tr_ds,  batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model     = GestureNet(N_FEAT, n_classes, WINDOW).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters : {n_params:,}")

    optim  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    crit   = nn.CrossEntropyLoss()

    best_acc, best_state, patience_count = 0.0, None, 0
    patience = 12
    history = {"epoch": [], "loss": [], "val_acc": []}

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
        val_acc = correct / len(val_ds)
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

    # ── Save model ────────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pt_path = out / "gesture_dl.pt"
    torch.save({
        "model_state":   best_state,
        "n_features":    N_FEAT,
        "n_classes":     n_classes,
        "seq_len":       WINDOW,
        "label_encoder": le,
        "scaler":        scaler,
    }, str(pt_path))
    joblib.dump(scaler, str(out / "gesture_dl_scaler.joblib"))
    print(f"\n  Best val accuracy : {best_acc:.1%}")
    print(f"  Model saved       → {pt_path}")

    # ── Collect predictions & probabilities ───────────────────────────────────
    model.load_state_dict(best_state); model.to(device).eval()
    preds, trues, probs_all = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            prob   = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.extend(prob)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(yb.numpy())
    preds     = np.array(preds)
    trues     = np.array(trues)
    probs_all = np.array(probs_all)

    # ── 1. Classification report (txt + csv) ──────────────────────────────────
    report_str = classification_report(trues, preds,
                                       target_names=le.classes_, zero_division=0)
    print("\n── Validation Report ─────────────────────────────────────")
    print(report_str)
    (out / "gesture_dl_report.txt").write_text(
        f"CNN-BiLSTM Gesture Classifier — Validation Report\n"
        f"Copyright (c) 2026 Muhammad Younas\n"
        f"Best val accuracy: {best_acc:.4f}\n\n"
        + report_str
    )
    import csv as _csv
    report_dict = classification_report(trues, preds, target_names=le.classes_,
                                        zero_division=0, output_dict=True)
    with open(out / "gesture_dl_report.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1-score", "support"])
        for cls in le.classes_:
            r = report_dict[cls]
            w.writerow([cls, f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                        f"{r['f1-score']:.4f}", int(r['support'])])
        w.writerow([])
        w.writerow(["accuracy", "", "", f"{report_dict['accuracy']:.4f}", len(trues)])
    print(f"  Report saved      → {out}/gesture_dl_report.txt / .csv")

    # Import matplotlib here (not at module level) to avoid overriding
    # the GUI backend set by visualizer.py when dl_trainer is imported
    import matplotlib
    matplotlib.use("Agg")   # non-interactive — safe for saving without display
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # ── 2. Confusion matrix (counts + normalised) ─────────────────────────────
    cm  = confusion_matrix(trues, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

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
               xlabel="Predicted label", ylabel="True label", title=title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        thresh = data.max() / 2
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black",
                        fontsize=8)
    fig.suptitle("Confusion Matrix — CNN-BiLSTM Gesture Classifier",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = out / "gesture_dl_confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix  → {cm_path}")

    # ── 3. Training history (loss + accuracy) ────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["epoch"], history["loss"], color="tab:blue", linewidth=1.5)
    ax1.set(xlabel="Epoch", ylabel="Training Loss", title="Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["epoch"], [v * 100 for v in history["val_acc"]],
             color="tab:green", linewidth=1.5)
    ax2.axhline(best_acc * 100, color="red", linestyle="--", linewidth=1,
                label=f"Best {best_acc:.1%}")
    ax2.set(xlabel="Epoch", ylabel="Validation Accuracy (%)", title="Validation Accuracy")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("Training History — CNN-BiLSTM Gesture Classifier",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    hist_path = out / "gesture_dl_training_history.png"
    plt.savefig(str(hist_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training history  → {hist_path}")

    # ── 4. Per-class ROC curves (one-vs-rest) ────────────────────────────────
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(trues, classes=range(n_classes))
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, (cls, ax) in enumerate(zip(le.classes_, axes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs_all[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="tab:blue", lw=1.5,
                label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8, linestyle="--")
        ax.set(xlim=[0, 1], ylim=[0, 1.02],
               xlabel="False Positive Rate", ylabel="True Positive Rate",
               title=cls)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-Class ROC Curves (One-vs-Rest) — CNN-BiLSTM Gesture Classifier",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    roc_path = out / "gesture_dl_roc_curves.png"
    plt.savefig(str(roc_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves        → {roc_path}")

    print(f"\n  All research artifacts saved to {out}/")
    print("  Files: gesture_dl.pt | _report.txt | _report.csv |")
    print("         _confusion_matrix.png | _training_history.png | _roc_curves.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   default="data/recordings")
    ap.add_argument("--output_dir", default="classification/models")
    ap.add_argument("--epochs",     type=int,   default=60)
    ap.add_argument("--lr",         type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int,   default=64)
    ap.add_argument("--augment",    type=int,   default=6)
    args = ap.parse_args()
    train(args.data_dir, args.output_dir, args.epochs, args.lr,
          args.batch_size, args.augment)