# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
IWR1443 / IWR1843 – Real-time Gesture Classification & Movement Tracking
========================================================================

Pipeline (all threads run concurrently):

  [Radar HW]
      │ UART bytes
      ▼
  RadarInterface._read_loop  ──►  raw_queue
      │
      ▼
  RadarParser._parse_loop    ──►  frame_queue
      │
      ▼
  Pipeline._process_loop     ──►  VisualisationState
      │
      └── PointCloudProcessor
      └── MultiTargetTracker
      └── GestureEngine (per track)
      └── DataRecorder (optional)
      │
      ▼
  RadarVisualizer (main thread, matplotlib animation)

Quick start
───────────
  1. Flash the IWR1443 with TI mmWave SDK OOB demo firmware.
  2. Connect USB.  Note the two COM ports (Device Manager → Ports).
  3. Edit config/settings.py  →  cli_port / data_port.
  4. Run:  python main.py
  5. To train ML model after collecting data:
       python -m classification.model_trainer
     Then set  use_ml_classifier = True  in settings.py.

Optional flags
──────────────
  --radar     iwr1443     Select radar hardware model (iwr1443 or iwr1843)
  --cli_port  COM3        Override CLI port
  --data_port COM4        Override data port
  --config    config/iwr1443_gesture.cfg  Override .cfg file explicitly
  --record                Enable data recording
  --ml                    Use ML classifier (requires trained model)
  --no_viz                Run headless (useful for recording)
  --demo                  Replay a recorded .npy file without hardware
"""

import argparse
import queue
import sys
import threading
import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from config.settings import FrameworkSettings
from radar.interface import RadarInterface
from radar.parser import RadarParser, RadarFrame
from processing.point_cloud import PointCloudProcessor, Cluster
from processing.tracker import MultiTargetTracker, Track
from classification.gesture_classifier import (
    GestureEngine,
    GestureResult,
    RuleBasedClassifier,
    MLClassifier,
    DLClassifier,
    AttnClassifier,
)
from data.collector import DataRecorder
from utils.logger import get_logger

log = get_logger(__name__)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(self, settings: FrameworkSettings, visualizer=None):
        self._settings = settings
        self._vis = visualizer

        self._raw_q: queue.Queue = queue.Queue(maxsize=200)
        self._frame_q: queue.Queue = queue.Queue(maxsize=100)

        self._iface = RadarInterface(settings.radar, self._raw_q)
        self._parser = RadarParser(self._raw_q, self._frame_q)
        self._proc = PointCloudProcessor(settings.processing)
        self._tracker = MultiTargetTracker(settings.tracker)

        # One GestureEngine per tracked target
        clf = self._build_classifier()
        self._clf_factory = lambda: GestureEngine(settings.gesture, clf)
        self._engines: Dict[int, GestureEngine] = {}

        # Global engine for when there are no confirmed tracks
        self._global_engine = self._clf_factory()

        self._recorder: Optional[DataRecorder] = None
        if settings.enable_recording:
            self._recorder = DataRecorder(settings.data, settings.gesture)

        self._stop_event = threading.Event()
        self._frame_count = 0
        self._fps_counter = 0
        self._fps_ts = time.time()
        self._fps = 0.0
        self._last_gesture = GestureResult("static", 1.0, time.time())
        # Global cooldown — shared across all per-track engines to prevent rapid-fire
        self._last_gesture_emit_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._iface.open()
        self._iface.configure()
        self._parser.start()
        self._iface.start_streaming()

        process_thread = threading.Thread(
            target=self._process_loop, name="pipeline", daemon=True
        )
        process_thread.start()
        log.info("Pipeline running.  Press Ctrl+C to stop.")

    def stop(self) -> None:
        self._stop_event.set()
        self._iface.stop()
        self._parser.stop()
        log.info("Pipeline stopped.  Frames parsed: %d  Errors: %d",
                 self._parser.frames_parsed, self._parser.parse_errors)

    # ------------------------------------------------------------------
    # Processing loop (background thread)
    # ------------------------------------------------------------------

    def _process_loop(self) -> None:
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
        while not self._stop_event.is_set():
            try:
                frame: RadarFrame = self._frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            self._frame_count += 1
            self._update_fps()

            # 1. Cluster point cloud
            clusters = self._proc.process(frame)
            raw_pts = self._proc.points_to_array(frame)

            # 2. Track clusters
            tracks: list[Track] = self._tracker.update(clusters)

            # 3. Gesture classification per track
            gesture_result = self._classify(tracks, clusters)
            if gesture_result and gesture_result.label != "static":
                now = time.time()
                cooldown = self._settings.gesture.gesture_cooldown_s
                if now - self._last_gesture_emit_time >= cooldown:
                    self._last_gesture_emit_time = now
                    self._last_gesture = gesture_result
                    log.info("GESTURE %-18s  conf=%.2f", gesture_result.label, gesture_result.confidence)

            # 4. Data recording
            if self._recorder and clusters:
                best = max(clusters, key=lambda c: c.point_count)
                spread = float(np.std(best.points)) if len(best.points) > 1 else 0.0
                self._recorder.record_frame(
                    best.centroid, best.mean_velocity, best.point_count, spread
                )

            # 5. Update visualisation state
            if self._vis:
                pts_xyz = raw_pts[:, :3] if raw_pts is not None else None
                pts_vel = raw_pts[:, 3] if raw_pts is not None else None
                self._vis.state.update(
                    pts_xyz, pts_vel, tracks,
                    self._last_gesture.label,
                    self._last_gesture.confidence,
                    self._fps,
                    self._frame_count,
                )

    # ------------------------------------------------------------------
    # Gesture dispatch
    # ------------------------------------------------------------------

    def _classify(self, tracks: list[Track], clusters: list[Cluster]) -> Optional[GestureResult]:
        # Always use the best cluster's raw data — matches training format exactly:
        # [x, y, z, mean_velocity, point_count, spread] as recorded by collect_session.py
        if not clusters:
            return self._global_engine.push_frame(None, None, 0, 0.0)

        best = max(clusters, key=lambda c: c.point_count)
        spread = float(np.std(best.points)) if len(best.points) > 1 else 0.0
        return self._global_engine.push_frame(
            best.centroid, best.mean_velocity, best.point_count, spread
        )

    # ------------------------------------------------------------------
    # FPS counter
    # ------------------------------------------------------------------

    def _update_fps(self) -> None:
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_ts
        if elapsed >= 1.0:
            self._fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_ts = now

    # ------------------------------------------------------------------
    # Classifier factory
    # ------------------------------------------------------------------

    def _build_classifier(self):
        s = self._settings
        if s.use_attn_classifier:
            try:
                return AttnClassifier(s.attn_model_path, s.gesture.gestures)
            except (FileNotFoundError, ImportError) as e:
                log.warning("%s  →  Falling back to CNN-BiLSTM.", e)
        if s.use_dl_classifier:
            try:
                return DLClassifier(s.dl_model_path, s.gesture.gestures)
            except (FileNotFoundError, ImportError) as e:
                log.warning("%s  →  Falling back to ML/rule-based classifier.", e)
        if s.use_ml_classifier:
            try:
                return MLClassifier(s.ml_model_path, s.gesture.gestures)
            except FileNotFoundError as e:
                log.warning("%s  →  Falling back to rule-based classifier.", e)
        return RuleBasedClassifier(s.gesture)


# ── Demo replay (no hardware) ─────────────────────────────────────────────────

def run_demo_replay(npy_path: str, settings: FrameworkSettings, visualizer=None) -> None:
    """Replay a recorded .npy file using the same classifier as live mode."""
    data = np.load(npy_path)   # (N, 6)
    log.info("Demo replay: %s  (%d frames)", npy_path, len(data))

    # Use the configured classifier, not hardcoded rule-based
    clf    = Pipeline(settings)._build_classifier()
    engine = GestureEngine(settings.gesture, clf)
    dt = settings.tracker.dt

    for row in data:
        x, y, z, vel, cnt, spread = row
        centroid = np.array([x, y, z])
        result = engine.push_frame(centroid, float(vel), int(cnt), float(spread))
        if result and result.label != "static":
            print(f"  GESTURE: {result.label:<20} conf={result.confidence:.2f}")
        time.sleep(dt)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Radar Gesture & Tracking Framework")
    ap.add_argument("--radar",     default=None, choices=["iwr1443", "iwr1843"],
                    help="Radar hardware model (default: iwr1443)")
    ap.add_argument("--cli_port",  default=None)
    ap.add_argument("--data_port", default=None)
    ap.add_argument("--config",    default=None,
                    help="Override radar .cfg file (overrides --radar config selection)")
    ap.add_argument("--classifier", default=None,
                    choices=["rule", "ml", "dl", "attn"],
                    help="Classifier to use: rule=rule-based, ml=Random Forest, "
                         "dl=CNN-BiLSTM, attn=Attention model (default: from settings.py)")
    ap.add_argument("--no_viz",    action="store_true", help="Headless mode")
    ap.add_argument("--demo",      default=None, metavar="NPY_FILE",
                    help="Replay a .npy recording (no hardware required)")
    return ap.parse_args()


def main():
    args = parse_args()
    settings = FrameworkSettings()

    # Apply log level from settings to all loggers
    import logging
    logging.getLogger().setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    for name in ("__main__", "radar.interface", "radar.parser",
                 "processing.point_cloud", "classification.gesture_classifier"):
        logging.getLogger(name).setLevel(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        )

    # Apply CLI overrides
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
    # Apply explicit --classifier override (all others disabled)
    if args.classifier:
        settings.use_ml_classifier   = (args.classifier == "ml")
        settings.use_dl_classifier   = (args.classifier == "dl")
        settings.use_attn_classifier = (args.classifier == "attn")
        # "rule" leaves all False → falls through to RuleBasedClassifier

    # ── Demo mode ──────────────────────────────────────────────────────
    if args.demo:
        run_demo_replay(args.demo, settings)
        return

    # ── Live mode ──────────────────────────────────────────────────────
    visualizer = None
    if not args.no_viz:
        from visualization.visualizer import RadarVisualizer
        visualizer = RadarVisualizer(settings.visualization)

    pipeline = Pipeline(settings, visualizer)

    try:
        pipeline.run()

        if visualizer:
            # Blocking – runs matplotlib event loop in main thread
            visualizer.start()
        else:
            log.info("Headless mode.  Press Ctrl+C to stop.")
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down…")
    except Exception as exc:
        log.error("Fatal error: %s", exc, exc_info=True)
    finally:
        pipeline.stop()
        print("Done.")


if __name__ == "__main__":
    main()