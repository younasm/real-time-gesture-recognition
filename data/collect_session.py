# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
Guided data collection session for building a gesture training dataset.

Usage
─────
  python -m data.collect_session
  python -m data.collect_session --cli_port COM5 --data_port COM6

Prompts you for:
  - Person / subject ID
  - Which gestures to collect (or all)
  - Repetitions per gesture
  - Sample duration in seconds
  - Rest time between reps

For each rep it counts down 3-2-1-GO, records, shows a live progress bar,
then saves a .npy file compatible with classification/model_trainer.py.

All session metadata (person ID, gesture, rep number) is appended to
data/recordings/session_log.csv so you can filter by person later.
"""

import argparse
import csv
import queue
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from config.settings import FrameworkSettings
from radar.interface import RadarInterface
from radar.parser import RadarParser, RadarFrame
from processing.point_cloud import PointCloudProcessor, Cluster
from utils.logger import get_logger

log = get_logger(__name__)

# ── Gesture catalogue (must match GestureSettings.gestures) ──────────────────

ALL_GESTURES: List[str] = [
    "swipe_left",
    "swipe_right",
    "swipe_up",
    "swipe_down",
    "push",
    "pull",
    "wave",
    "circle_cw",
    "circle_ccw",
    "static",
]

# Friendly one-line instructions shown before each gesture
GESTURE_HINTS = {
    "swipe_left":  "Move your hand horizontally to the LEFT",
    "swipe_right": "Move your hand horizontally to the RIGHT",
    "swipe_up":    "Move your hand UPWARD in front of the sensor",
    "swipe_down":  "Move your hand DOWNWARD in front of the sensor",
    "push":        "Push your hand AWAY from you toward the sensor",
    "pull":        "Pull your hand TOWARD you away from the sensor",
    "wave":        "Wave your hand side-to-side (2–3 oscillations)",
    "circle_cw":   "Trace a CLOCKWISE circle in the air",
    "circle_ccw":  "Trace a COUNTER-CLOCKWISE circle in the air",
    "static":      "Hold your hand STILL in front of the sensor",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hr(char: str = "-", width: int = 54) -> str:
    return char * width


def _countdown(seconds: int) -> None:
    print("  Get ready", end="", flush=True)
    for i in range(seconds, 0, -1):
        print(f"  {i}", end="", flush=True)
        time.sleep(1)
    print("  GO!", flush=True)


def _progress_bar(done: int, total: int, width: int = 24) -> str:
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total} frames"


def _drain_queue(q: queue.Queue) -> None:
    """Discard any frames that accumulated while we were idle."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


# ── Core collection ───────────────────────────────────────────────────────────

def collect_sample(
    frame_queue: queue.Queue,
    processor: PointCloudProcessor,
    num_frames: int,
    timeout: float = 1.0,
) -> np.ndarray:
    """
    Block until num_frames radar frames have been processed and return a
    (num_frames, 6) float32 array:  [x, y, z, doppler, point_count, spread]

    Frames where the radar detects no points are stored as all-zeros so the
    sample always has exactly num_frames rows (required by the feature extractor).
    """
    _drain_queue(frame_queue)

    rows: List[np.ndarray] = []
    while len(rows) < num_frames:
        try:
            frame: RadarFrame = frame_queue.get(timeout=timeout)
        except queue.Empty:
            # Sensor may have missed a frame – insert a zero row and keep going
            rows.append(np.zeros(6, dtype=np.float32))
            remaining = num_frames - len(rows)
            print(f"\r  {_progress_bar(len(rows), num_frames)}  (no detection)", end="", flush=True)
            continue

        clusters: List[Cluster] = processor.process(frame)
        if clusters:
            best = max(clusters, key=lambda c: c.point_count)
            spread = float(np.std(best.points)) if len(best.points) > 1 else 0.0
            row = np.array([
                best.centroid[0], best.centroid[1], best.centroid[2],
                best.mean_velocity, float(best.point_count), spread,
            ], dtype=np.float32)
        else:
            row = np.zeros(6, dtype=np.float32)

        rows.append(row)
        print(f"\r  {_progress_bar(len(rows), num_frames)}", end="", flush=True)

    print()  # newline after the progress bar
    return np.array(rows, dtype=np.float32)


# ── Interactive prompts ───────────────────────────────────────────────────────

def _prompt_str(prompt: str, default: str) -> str:
    val = input(f"  {prompt} [{default}]: ").strip()
    return val if val else default


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("  Please enter a whole number.")


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number (e.g. 2.0).")


def _select_gestures() -> List[str]:
    print(f"\n  {'#':>3}  {'Gesture':<20}")
    print(f"  {_hr('-', 26)}")
    for i, g in enumerate(ALL_GESTURES, 1):
        print(f"  {i:>3}. {g}")
    raw = input("\n  Select gesture numbers (e.g. 1,2,3) or press Enter for all: ").strip()
    if not raw:
        return list(ALL_GESTURES)
    try:
        indices = [int(x.strip()) - 1 for x in raw.split(",")]
        chosen = [ALL_GESTURES[i] for i in indices if 0 <= i < len(ALL_GESTURES)]
        if not chosen:
            raise ValueError
        return chosen
    except (ValueError, IndexError):
        print("  Invalid selection – collecting all gestures.")
        return list(ALL_GESTURES)


# ── Session ───────────────────────────────────────────────────────────────────

def run_session(settings: FrameworkSettings) -> None:
    print(f"\n{_hr('=')}")
    print(f"  {settings.radar.radar_model.upper()}  –  Guided Data Collection Session")
    print(_hr("="))

    # ── Collect session parameters ─────────────────────────────────────────
    print("\n  Session setup\n" + _hr())

    person_id  = _prompt_str("Person / subject ID", "person1")
    gestures   = _select_gestures()
    reps       = _prompt_int("Repetitions per gesture", 10)
    duration_s = _prompt_float("Sample duration (seconds)", 2.0)
    rest_s     = _prompt_float("Rest between reps (seconds)", 2.0)
    countdown_s = _prompt_int("Countdown before each rep (seconds)", 3)

    dt         = settings.tracker.dt          # seconds per frame
    num_frames = max(1, round(duration_s / dt))
    fps        = 1.0 / dt

    print(f"\n{_hr()}")
    print(f"  Person:    {person_id}")
    print(f"  Gestures:  {', '.join(gestures)}")
    print(f"  Reps:      {reps} x {duration_s}s  ({num_frames} frames @ {fps:.0f} fps)")
    print(f"  Rest:      {rest_s}s between reps")
    print(f"  Save dir:  {settings.data.recording_dir}")
    print(_hr())

    input("\n  Press Enter to connect to the radar sensor...")

    # ── Connect radar ──────────────────────────────────────────────────────
    raw_q: queue.Queue   = queue.Queue(maxsize=500)
    frame_q: queue.Queue = queue.Queue(maxsize=200)

    iface     = RadarInterface(settings.radar, raw_q)
    parser    = RadarParser(raw_q, frame_q)

    # For data collection use permissive processing settings:
    #   • static_vel_threshold = 0.0  → keep static hands (for 'static' gesture)
    #   • dbscan_min_samples = 1      → allow single-point detections
    #   • dbscan_eps = 0.6            → larger neighbourhood catches sparse reflections
    # The default settings are optimised for real-time tracking, not recording.
    from copy import deepcopy
    collect_proc_settings = deepcopy(settings.processing)
    collect_proc_settings.static_vel_threshold = 0.0
    collect_proc_settings.dbscan_min_samples   = 1
    collect_proc_settings.dbscan_eps           = 0.6
    processor = PointCloudProcessor(collect_proc_settings)

    try:
        iface.open()
        iface.configure()
        parser.start()
        iface.start_streaming()
        print("  Radar connected. Warming up...")
        time.sleep(1.5)
    except Exception as exc:
        print(f"\n  ERROR: Could not connect to radar – {exc}")
        print("  Check COM ports in config/settings.py and try again.")
        sys.exit(1)

    # ── Prepare output ─────────────────────────────────────────────────────
    out_dir = Path(settings.data.recording_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path   = out_dir / "session_log.csv"
    write_header = not log_path.exists()
    log_file   = open(log_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)
    if write_header:
        csv_writer.writerow(["filename", "person_id", "gesture", "rep", "timestamp_ms", "num_frames"])

    total_saved = 0

    # ── Main collection loop ───────────────────────────────────────────────
    try:
        for g_idx, gesture in enumerate(gestures, 1):
            print(f"\n{_hr('=')}")
            print(f"  Gesture {g_idx}/{len(gestures)}: {gesture.upper()}")
            print(f"  Instruction: {GESTURE_HINTS.get(gesture, '')}")
            print(_hr("="))
            input(f"\n  Press Enter when you are ready to start '{gesture}'...")

            for rep in range(1, reps + 1):
                print(f"\n  Rep {rep}/{reps}:")
                _countdown(countdown_s)

                data = collect_sample(frame_q, processor, num_frames)

                ts_ms = int(time.time() * 1000)
                fname = f"{gesture}_{ts_ms}.npy"
                fpath = out_dir / fname
                np.save(str(fpath), data)
                total_saved += 1

                csv_writer.writerow([fname, person_id, gesture, rep, ts_ms, len(data)])
                log_file.flush()

                non_zero = int(np.any(data != 0, axis=1).sum())
                print(f"  Saved: {fname}  ({non_zero}/{num_frames} frames with detections)")

                if rep < reps:
                    print(f"  Resting {rest_s}s...")
                    time.sleep(rest_s)

            print(f"\n  Done: {gesture}  ({reps} samples saved)")

    except KeyboardInterrupt:
        print("\n\n  Collection interrupted by user.")

    finally:
        log_file.close()
        iface.stop()
        parser.stop()

        print(f"\n{_hr('=')}")
        print("  Session complete")
        print(_hr())
        print(f"  Total samples saved : {total_saved}")
        print(f"  Output directory    : {out_dir.resolve()}")
        print(f"  Session log         : {log_path.resolve()}")
        print(_hr())
        print("\n  To train the ML model run:")
        print("    python -m classification.model_trainer")
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Guided gesture data collection session")
    ap.add_argument("--radar",     default=None, choices=["iwr1443", "iwr1843"],
                    help="Radar hardware model (default: iwr1443)")
    ap.add_argument("--cli_port",  default=None, help="Override CLI serial port")
    ap.add_argument("--data_port", default=None, help="Override data serial port")
    ap.add_argument("--config",    default=None, help="Override radar config file path")
    args = ap.parse_args()

    settings = FrameworkSettings()
    if args.radar:
        settings.radar.radar_model = args.radar
        settings.radar.config_file = ""
        settings.radar.__post_init__()
    if args.cli_port:
        settings.radar.cli_port = args.cli_port
    if args.data_port:
        settings.radar.data_port = args.data_port
    if args.config:
        settings.radar.config_file = args.config

    run_session(settings)


if __name__ == "__main__":
    main()