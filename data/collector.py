# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
Data recorder for building an ML training dataset.

When  FrameworkSettings.enable_recording = True  the pipeline calls
`recorder.record_frame(...)` every frame.  Press a gesture key to assign a
label for the next N frames (label window = gesture_settings.window_frames).

Labels are flush-saved as  <label>_<timestamp>.npy  under  data/recordings/.
Each file is a (N_frames, 6) float32 array:
    columns: x, y, z, doppler, point_count, point_spread
"""

import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import DataSettings, GestureSettings
from utils.logger import get_logger

log = get_logger(__name__)


class DataRecorder:
    """
    Call  record_frame  each radar frame.
    Call  start_label(label)  to begin capturing a labelled window.
    The window auto-completes after window_frames frames.
    """

    KEY_MAP = {
        "l": "swipe_left",
        "r": "swipe_right",
        "u": "swipe_up",
        "d": "swipe_down",
        "p": "push",
        "b": "pull",       # b = back i used f in the .bin dataset collected using mmWave studio.
        "w": "wave",
        "c": "circle_cw",
        "v": "circle_ccw",
        "s": "static",
    }

    def __init__(self, ds: DataSettings, gs: GestureSettings):
        self._ds = ds
        self._gs = gs
        self._buf: deque = deque(maxlen=ds.max_buffer_frames)
        self._label_buf: list = []
        self._current_label: Optional[str] = None
        self._label_remaining: int = 0
        self._total_saved = 0

        Path(ds.recording_dir).mkdir(parents=True, exist_ok=True)
        log.info("Recorder ready.  dir=%s", ds.recording_dir)
        self._print_keybindings()

    # ------------------------------------------------------------------

    def record_frame(
        self,
        centroid: Optional[np.ndarray],
        mean_velocity: float,
        point_count: int,
        spread: float,
    ) -> None:
        x, y, z = (centroid[0], centroid[1], centroid[2]) if centroid is not None else (0, 0, 0)
        row = np.array([x, y, z, mean_velocity, point_count, spread], dtype=np.float32)
        self._buf.append(row)

        if self._label_remaining > 0:
            self._label_buf.append(row)
            self._label_remaining -= 1
            if self._label_remaining == 0:
                self._save_window()

    def start_label(self, label: str) -> None:
        if label not in self._gs.gestures:
            log.warning("Unknown label: %s", label)
            return
        self._current_label = label
        self._label_buf = []
        self._label_remaining = self._gs.window_frames
        log.info("Recording label '%s' (%d frames)", label, self._gs.window_frames)

    def handle_key(self, key: str) -> None:
        label = self.KEY_MAP.get(key.lower())
        if label:
            self.start_label(label)

    # ------------------------------------------------------------------

    def _save_window(self) -> None:
        if not self._label_buf:
            return
        arr = np.array(self._label_buf, dtype=np.float32)
        ts = int(time.time() * 1000)
        fname = f"{self._current_label}_{ts}.npy"
        fpath = Path(self._ds.recording_dir) / fname
        np.save(str(fpath), arr)
        self._total_saved += 1
        log.info("Saved %s  (%d total)", fname, self._total_saved)

    @staticmethod
    def _print_keybindings():
        print("\n─── Recorder key bindings ───────────────────────────────")
        bindings = {
            "L": "swipe_left",    "R": "swipe_right",
            "U": "swipe_up",      "D": "swipe_down",
            "P": "push",          "B": "pull (back)",
            "W": "wave",          "C": "circle_cw",
            "V": "circle_ccw",    "S": "static",
        }
        for k, v in bindings.items():
            print(f"  {k}  →  {v}")
        print("─────────────────────────────────────────────────────────\n")