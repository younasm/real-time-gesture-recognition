# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
data/inspect_recordings.py — Quality check for all recordings.
Flags recordings that are likely too noisy to train on.

Usage:  python -m data.inspect_recordings
        python -m data.inspect_recordings --clean   (moves bad files to data/bad/)
"""

import argparse
import shutil
from pathlib import Path

import numpy as np


MIN_ACTIVE_RATIO  = 0.35   # at least 35% of frames must have detections
MIN_DISPLACEMENT  = 0.05   # hand must move at least 5cm (not for 'static')
MIN_FRAMES        = 20     # recording must have at least 20 frames

# Gestures that move LATERALLY — Doppler (radial velocity) is naturally
# near-zero for these because the hand moves perpendicular to the radar beam.
# Do NOT warn about low velocity for these classes.
LATERAL_GESTURES = {
    "swipe_left", "swipe_right", "swipe_up", "swipe_down",
    "wave", "circle_cw", "circle_ccw",
}


def grade_recording(data: np.ndarray, label: str) -> tuple[str, str]:
    """Return (grade, reason). Grade: 'OK' | 'WARN' | 'BAD'"""
    n = len(data)
    if n < MIN_FRAMES:
        return "BAD", f"too short ({n} frames < {MIN_FRAMES})"

    active = np.any(data[:, :3] != 0, axis=1).mean()
    if active < MIN_ACTIVE_RATIO:
        return "BAD", f"low detection ({active:.0%} frames have points)"

    if label != "static":
        xyz  = data[np.any(data[:, :3] != 0, axis=1)]   # active frames only
        if len(xyz) < 4:
            return "WARN", "almost no detections"

        # Check spatial displacement using active frames only
        disp = np.max(np.linalg.norm(xyz[:, :3] - xyz[0, :3], axis=1))
        if disp < MIN_DISPLACEMENT:
            return "BAD", f"almost no movement (max disp {disp:.3f}m)"

        # Only check velocity for depth gestures (push/pull).
        # Lateral gestures (swipes, circles, wave) naturally have ~0 Doppler
        # because the hand moves perpendicular to the radar beam — this is
        # physically correct, NOT a data quality problem.
        if label not in LATERAL_GESTURES:
            vel = data[:, 3]
            if np.max(np.abs(vel)) < 0.03:
                return "WARN", "very low velocity — may look like 'static'"

    return "OK", ""


def inspect(data_dir: str, clean: bool = False) -> None:
    bad_dir = Path(data_dir) / "bad"
    counts = {"OK": 0, "WARN": 0, "BAD": 0}
    bad_files = []

    label_stats: dict = {}

    print(f"\n{'File':<45}  {'Grade':<5}  Reason")
    print("─" * 90)

    for fp in sorted(Path(data_dir).glob("*.npy")):
        label = fp.stem.rsplit("_", 1)[0]
        data  = np.load(fp)
        grade, reason = grade_recording(data, label)

        counts[grade] += 1
        if label not in label_stats:
            label_stats[label] = {"OK": 0, "WARN": 0, "BAD": 0}
        label_stats[label][grade] += 1

        if grade != "OK":
            bad_files.append(fp)
            marker = "⚠" if grade == "WARN" else "✗"
            print(f"  {marker} {fp.name:<42}  {grade:<5}  {reason}")

    print("\n── Per-gesture quality ──────────────────────────────────")
    print(f"  {'Gesture':<20}  {'OK':>4}  {'WARN':>5}  {'BAD':>4}  {'Total':>6}")
    print(f"  {'─'*20}  {'─'*4}  {'─'*5}  {'─'*4}  {'─'*6}")
    for lbl in sorted(label_stats):
        s = label_stats[lbl]
        total = sum(s.values())
        print(f"  {lbl:<20}  {s['OK']:>4}  {s['WARN']:>5}  {s['BAD']:>4}  {total:>6}")

    print(f"\n── Summary ──────────────────────────────────────────────")
    total = sum(counts.values())
    print(f"  OK:   {counts['OK']:>4} / {total}")
    print(f"  WARN: {counts['WARN']:>4} / {total}")
    print(f"  BAD:  {counts['BAD']:>4} / {total}")
    print(f"  Bad rate: {(counts['WARN']+counts['BAD'])/total:.0%}")

    # Only move BAD files — WARN files are valid recordings
    truly_bad = [fp for fp in Path(data_dir).glob("*.npy")
                 if grade_recording(np.load(fp),
                                    fp.stem.rsplit("_", 1)[0])[0] == "BAD"]
    if clean and truly_bad:
        bad_dir.mkdir(exist_ok=True)
        print(f"\n  Moving {len(truly_bad)} BAD files to {bad_dir}/")
        for fp in truly_bad:
            shutil.move(str(fp), str(bad_dir / fp.name))
        print("  Done. Re-run model_trainer after cleaning.")
    elif truly_bad:
        print(f"\n  Run with --clean to move {len(truly_bad)} BAD files out.")
        print(f"  NOTE: WARN files are kept — low velocity is normal for lateral gestures.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/recordings")
    ap.add_argument("--clean",  action="store_true",
                    help="Move bad/warn recordings to data/recordings/bad/")
    args = ap.parse_args()
    inspect(args.data_dir, args.clean)