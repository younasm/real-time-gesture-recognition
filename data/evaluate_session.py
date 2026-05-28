# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
data/evaluate_session.py — Real-time gesture recognition evaluation.

Prompts the user to perform specific gestures one at a time, records what
the DL/ML model actually detected, then generates a full accuracy report
with confusion matrix, per-class metrics, and CSV log — ready for research papers.

Usage
─────
  python -m data.evaluate_session
  python -m data.evaluate_session --trials 5   # 5 trials per gesture
  python -m data.evaluate_session --gestures swipe_left,swipe_right,push,pull
"""

import argparse
import csv
import queue
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from config.settings import FrameworkSettings
from radar.interface import RadarInterface
from radar.parser import RadarParser
from processing.point_cloud import PointCloudProcessor
from classification.gesture_classifier import GestureEngine, GestureResult
from utils.logger import get_logger

log = get_logger(__name__)

ALL_GESTURES = [
    "swipe_left", "swipe_right", "swipe_up", "swipe_down",
    "push", "pull", "wave", "circle_cw", "circle_ccw",
]


# ── Terminal helpers ──────────────────────────────────────────────────────────

def _hr(char="─", width=56):
    return char * width

def _countdown(n):
    for i in range(n, 0, -1):
        print(f"  {i}...", end=" ", flush=True)
        time.sleep(1)
    print("GO!", flush=True)


# ── Build classifier (mirrors Pipeline._build_classifier) ────────────────────

def _build_classifier(settings: FrameworkSettings):
    from classification.gesture_classifier import (
        AttnClassifier, DLClassifier, MLClassifier, RuleBasedClassifier
    )
    if settings.use_attn_classifier:
        try:
            return AttnClassifier(settings.attn_model_path, settings.gesture.gestures)
        except (FileNotFoundError, ImportError) as e:
            print(f"  [WARN] Attention model unavailable ({e}) — trying DL.")
    if settings.use_dl_classifier:
        try:
            return DLClassifier(settings.dl_model_path, settings.gesture.gestures)
        except (FileNotFoundError, ImportError) as e:
            print(f"  [WARN] DL model unavailable ({e}) — trying ML.")
    if settings.use_ml_classifier:
        try:
            return MLClassifier(settings.ml_model_path, settings.gesture.gestures)
        except FileNotFoundError as e:
            print(f"  [WARN] ML model unavailable ({e}) — using rule-based.")
    return RuleBasedClassifier(settings.gesture)


# ── Core evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(settings: FrameworkSettings,
                   gestures: List[str],
                   trials_per_gesture: int,
                   detection_timeout: float = 8.0,
                   out_dir: str = "classification/models") -> None:

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build trial list (randomised order)
    trial_list = gestures * trials_per_gesture
    random.shuffle(trial_list)
    n_total = len(trial_list)

    print(f"\n{_hr('═')}")
    print(f"  Real-Time Gesture Evaluation")
    print(f"  Gestures : {', '.join(gestures)}")
    print(f"  Trials   : {trials_per_gesture} per gesture  ({n_total} total)")
    print(f"  Timeout  : {detection_timeout}s per trial")
    print(_hr("═"))
    input("\n  Press Enter to connect to the radar...")

    # ── Connect radar ──────────────────────────────────────────────────────
    raw_q   = queue.Queue(maxsize=500)
    frame_q = queue.Queue(maxsize=200)
    iface   = RadarInterface(settings.radar, raw_q)
    parser  = RadarParser(raw_q, frame_q)
    proc    = PointCloudProcessor(settings.processing)
    clf     = _build_classifier(settings)
    engine  = GestureEngine(settings.gesture, clf)

    # Lower cooldown so evaluation can run quickly
    settings.gesture.gesture_cooldown_s = 1.0

    try:
        iface.open()
        iface.configure()
        parser.start()
        iface.start_streaming()
        print("  Radar connected. Warming up 2 s...\n")
        time.sleep(2.0)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    # ── Evaluation ────────────────────────────────────────────────────────
    results = []   # list of (prompted, detected, correct, confidence)

    def _check_sensor_presence(timeout=5.0) -> bool:
        """
        Check for hand presence AND fill the rolling buffer with real frames.
        Feeding real background/hand frames here means the model gets genuine
        context (not empty zeros) when the gesture arrives after countdown.
        The engine state machine is suppressed during this warmup.
        """
        deadline = time.time() + timeout
        found = False
        while time.time() < deadline:
            try:
                frame = frame_q.get(timeout=0.2)
            except queue.Empty:
                continue
            clusters = proc.process(frame)
            if clusters:
                best   = max(clusters, key=lambda c: c.point_count)
                spread = float(np.std(best.points)) if len(best.points) > 1 else 0.0
                # Feed directly into rolling buffer (bypass state machine)
                engine._roll_c.append(best.centroid)
                engine._roll_v.append(best.mean_velocity)
                engine._roll_n.append(best.point_count)
                engine._roll_s.append(spread)
                found = True
            else:
                engine._roll_c.append(None)
                engine._roll_v.append(0.0)
                engine._roll_n.append(0)
                engine._roll_s.append(0.0)
            if found:
                return True  # hand found + buffer has real frames
        return False

    def _reset_engine():
        """
        Full reset: clear buffer + state machine + drain queue.
        The buffer will be refilled with real background frames during
        the sensor-presence check that follows immediately after.
        """
        engine._state          = engine._IDLE
        engine._idle_count     = 0
        engine._frame_counter  = 0
        engine._last_emit_time = 0.0
        W = settings.gesture.window_frames
        from collections import deque as _deque
        engine._roll_c = _deque([None] * W, maxlen=W)
        engine._roll_v = _deque([0.0]  * W, maxlen=W)
        engine._roll_n = _deque([0]    * W, maxlen=W)
        engine._roll_s = _deque([0.0]  * W, maxlen=W)
        while not frame_q.empty():
            try: frame_q.get_nowait()
            except queue.Empty: break

    try:
        for trial_idx, prompted in enumerate(trial_list, 1):
            print(f"\n  Trial {trial_idx}/{n_total}")
            print(f"  {'─'*40}")
            print(f"  ► Perform gesture:  {prompted.upper().replace('_', ' ')}")

            # ── Sensor presence check + buffer warmup ─────────────────────
            # Reset first, THEN check presence — the check fills the buffer
            # with real background frames so the model has genuine context.
            _reset_engine()
            print(f"  Checking sensor... hold your hand in front of the radar", end="", flush=True)
            visible = _check_sensor_presence(timeout=4.0)
            if visible:
                print("  ✓ Hand detected")
            else:
                print("  ⚠ No detection — move closer (30–80 cm) and press Enter")
                input()

            # Only reset the state machine — keep the warmed-up buffer
            engine._state          = engine._IDLE
            engine._idle_count     = 0
            engine._frame_counter  = 0
            engine._last_emit_time = 0.0
            # Drain frames accumulated during countdown (queue only)
            _countdown(3)
            while not frame_q.empty():
                try: frame_q.get_nowait()
                except queue.Empty: break

            detected: Optional[str] = None
            confidence: float       = 0.0
            deadline = time.time() + detection_timeout

            while time.time() < deadline:
                try:
                    frame = frame_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                clusters = proc.process(frame)
                if clusters:
                    best   = max(clusters, key=lambda c: c.point_count)
                    spread = float(np.std(best.points)) if len(best.points) > 1 else 0.0
                    result: Optional[GestureResult] = engine.push_frame(
                        best.centroid, best.mean_velocity, best.point_count, spread
                    )
                else:
                    result = engine.push_frame(None, None, 0, 0.0)

                if result and result.label != "static":
                    detected   = result.label
                    confidence = result.confidence
                    break

            correct = detected == prompted
            status  = "✓ CORRECT" if correct else f"✗ WRONG  (got: {detected or 'none'})"
            conf_str = f"{confidence:.2f}" if detected else "—"
            print(f"  Result: {status}   conf={conf_str}")

            results.append({
                "trial":      trial_idx,
                "prompted":   prompted,
                "detected":   detected or "none",
                "correct":    correct,
                "confidence": round(confidence, 4),
            })

            # Short pause between trials
            time.sleep(1.5)

    except KeyboardInterrupt:
        print("\n\n  Evaluation interrupted.")
    finally:
        iface.stop()
        parser.stop()

    if not results:
        print("  No results recorded."); return

    # ── Compute metrics ───────────────────────────────────────────────────
    _save_reports(results, gestures, out)


# ── Report generation ─────────────────────────────────────────────────────────

def _save_reports(results, gestures, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (confusion_matrix, classification_report,
                                 accuracy_score)

    prompted_list = [r["prompted"]  for r in results]
    detected_list = [r["detected"]  for r in results]
    correct_list  = [r["correct"]   for r in results]

    # Replace "none" with a special label so it shows in the confusion matrix
    detected_clean = [d if d != "none" else "no_detection" for d in detected_list]

    accuracy = sum(correct_list) / len(correct_list)
    n_total  = len(results)
    n_correct = sum(correct_list)

    print(f"\n{_hr('═')}")
    print(f"  EVALUATION RESULTS")
    print(_hr("─"))
    print(f"  Total trials   : {n_total}")
    print(f"  Correct        : {n_correct}")
    print(f"  Incorrect      : {n_total - n_correct}")
    print(f"  Overall accuracy : {accuracy:.1%}")
    print(_hr("═"))

    # ── Per-gesture accuracy ───────────────────────────────────────────────
    print(f"\n  {'Gesture':<22} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"  {_hr('-', 50)}")
    per_gesture = {}
    for g in sorted(set(prompted_list)):
        g_results = [r for r in results if r["prompted"] == g]
        g_correct = sum(r["correct"] for r in g_results)
        g_total   = len(g_results)
        g_acc     = g_correct / g_total if g_total else 0
        per_gesture[g] = (g_correct, g_total, g_acc)
        print(f"  {g:<22} {g_correct:>8} {g_total:>6} {g_acc:>9.1%}")
    print()

    # ── 1. Save CSV log ───────────────────────────────────────────────────
    csv_path = out / "evaluation_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial","prompted","detected",
                                          "correct","confidence"])
        w.writeheader()
        w.writerows(results)
    print(f"  Trial log        → {csv_path}")

    # ── 2. Save text summary ──────────────────────────────────────────────
    txt_path = out / "evaluation_summary.txt"
    all_labels = sorted(set(prompted_list) | set(detected_clean))
    report_str = classification_report(
        prompted_list, detected_clean,
        labels=sorted(set(prompted_list)),
        zero_division=0
    )
    with open(txt_path, "w") as f:
        f.write(f"Real-Time Gesture Evaluation Summary\n")
        f.write(f"{'='*56}\n")
        f.write(f"Total trials   : {n_total}\n")
        f.write(f"Correct        : {n_correct}\n")
        f.write(f"Overall accuracy: {accuracy:.4f} ({accuracy:.1%})\n\n")
        f.write(f"Per-gesture accuracy\n{'-'*40}\n")
        for g, (gc, gt, ga) in sorted(per_gesture.items()):
            f.write(f"  {g:<22} {gc}/{gt}  ({ga:.1%})\n")
        f.write(f"\nClassification Report\n{'-'*40}\n")
        f.write(report_str)
    print(f"  Text summary     → {txt_path}")

    # ── 3. Confusion matrix ───────────────────────────────────────────────
    labels = sorted(set(prompted_list))
    all_det_labels = sorted(set(detected_clean))
    plot_labels = sorted(set(labels) | set(all_det_labels))

    cm = confusion_matrix(prompted_list, detected_clean, labels=plot_labels)
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
        ax.set(xticks=range(len(plot_labels)), yticks=range(len(plot_labels)),
               xticklabels=plot_labels, yticklabels=plot_labels,
               xlabel="Detected", ylabel="Prompted", title=title)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        thresh = data.max() / 2
        for i in range(len(plot_labels)):
            for j in range(len(plot_labels)):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center", fontsize=8,
                        color="white" if data[i, j] > thresh else "black")

    # fig.suptitle(
    #     f"Real-Time Detection Evaluation  |  Accuracy: {accuracy:.1%}  ({n_correct}/{n_total})",
    #     fontsize=11, fontweight="bold"
    # )
    plt.tight_layout()
    cm_path = out / "evaluation_confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {cm_path}")

    # ── 4. Per-gesture accuracy bar chart ─────────────────────────────────
    g_names = sorted(per_gesture.keys())
    g_accs  = [per_gesture[g][2] * 100 for g in g_names]
    colors  = ["#2ecc71" if a >= 80 else "#f39c12" if a >= 60 else "#e74c3c"
               for a in g_accs]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(g_names, g_accs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(accuracy * 100, color="navy", linestyle="--", linewidth=1.5,
               label=f"Overall {accuracy:.1%}")
    ax.axhline(80, color="gray", linestyle=":", linewidth=1, label="80% threshold")
    for bar, acc in zip(bars, g_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set(xlabel="Gesture", ylabel="Accuracy (%)", ylim=[0, 110])
           # title="Per-Gesture Detection Accuracy — Real-Time Evaluation"
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    bar_path = out / "evaluation_per_gesture_accuracy.png"
    plt.savefig(str(bar_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Accuracy bar chart → {bar_path}")

    print(f"\n{_hr('═')}")
    print(f"  All evaluation artifacts saved to {out}/")
    print(_hr("═"))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Real-time gesture evaluation session")
    ap.add_argument("--trials",    type=int, default=5,
                    help="Trials per gesture (default: 5)")
    ap.add_argument("--timeout",   type=float, default=8.0,
                    help="Seconds to wait for detection per trial (default: 8)")
    ap.add_argument("--gestures",  default="",
                    help="Comma-separated gestures to test (default: all)")
    ap.add_argument("--out_dir",   default="classification/models/evaluation")
    args = ap.parse_args()

    settings = FrameworkSettings()

    gestures = ([g.strip() for g in args.gestures.split(",") if g.strip()]
                if args.gestures else ALL_GESTURES)

    invalid = [g for g in gestures if g not in ALL_GESTURES]
    if invalid:
        print(f"  Unknown gestures: {invalid}")
        print(f"  Valid: {ALL_GESTURES}")
        sys.exit(1)

    run_evaluation(settings, gestures, args.trials, args.timeout, args.out_dir)


if __name__ == "__main__":
    main()
