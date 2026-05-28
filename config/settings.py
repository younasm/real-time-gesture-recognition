# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
Centralised configuration for the radar gesture framework.
Edit the values below to match your setup – no other file needs to change.
"""

from dataclasses import dataclass, field
from typing import List


# Supported radar models and their default .cfg files
RADAR_CONFIG_MAP: dict = {
    "iwr1443": "config/iwr1443_gesture.cfg",
    "iwr1843": "config/iwr1843_gesture.cfg",
}


# ---------------------------------------------------------------------------
# Radar / serial
# ---------------------------------------------------------------------------
@dataclass
class RadarSettings:
    # Radar hardware model – determines the default config file.
    # Supported values: "iwr1443", "iwr1843"
    radar_model: str = "iwr1843"

    # On Windows: Device Manager → Ports → "XDS110 Class …"
    #   CLI port  → lower COM number  (115 200 baud)
    #   Data port → higher COM number (921 600 baud)
    cli_port: str = "/dev/tty.usbmodemR21010501" #"COM15"    # ← update to your IWR1843 CLI  port after checking Device Manager
    data_port: str = "/dev/tty.usbmodemR21010504" #"COM16"  # ← update to your IWR1843 Data port after checking Device Manager
    cli_baud: int = 115_200
    data_baud: int = 921_600
    # Leave empty to auto-select from radar_model; set explicitly to override.
    config_file: str = ""
    # Milliseconds to wait between CLI command lines
    cmd_delay_ms: int = 50

    def __post_init__(self) -> None:
        if not self.config_file:
            self.config_file = RADAR_CONFIG_MAP.get(
                self.radar_model, "config/iwr1443_gesture.cfg"
            )


# ---------------------------------------------------------------------------
# Point-cloud pre-processing
# ---------------------------------------------------------------------------
@dataclass
class ProcessingSettings:
    # Spatial gate (meters / degrees) – trim points outside the ROI
    min_range: float = 0.3
    max_range: float = 4.0
    x_limit: float = 2.5        # ± lateral limit
    z_limit: float = 2.0        # ± elevation limit

    # DBSCAN clustering — must match collect_session.py collection settings
    dbscan_eps: float = 0.6          # metres (0.6 matches single-hand reflections)
    dbscan_min_samples: int = 1      # 1 = any single detected point counts

    # Doppler filter — 0.0 keeps all points including slow/stationary hands
    static_vel_threshold: float = 0.0   # m/s


# ---------------------------------------------------------------------------
# Multi-object tracker (SORT-style Kalman)
# ---------------------------------------------------------------------------
@dataclass
class TrackerSettings:
    dt: float = 0.05            # 0.05 = 20 fps (50 ms frame period) – change to 0.04 for 25 fps
    max_age: int = 8            # delete track after this many missed frames
    min_hits: int = 2           # confirm track after this many consecutive hits
    max_dist: float = 0.8       # metres – max centroid-distance for assignment
    process_noise_pos: float = 0.05
    process_noise_vel: float = 0.5
    measurement_noise: float = 0.15


# ---------------------------------------------------------------------------
# Gesture recognition
# ---------------------------------------------------------------------------
@dataclass
class GestureSettings:
    window_frames: int = 20     # sliding window length
    hop_frames: int = 5         # window hop
    min_active_frames: int = 5  # frames with ≥ min_points to attempt classify
    min_points_per_frame: int = 2

    # Rule-based thresholds
    min_displacement_m: float = 0.20   # minimum total movement to classify
    swipe_ratio: float = 1.8           # dominant/secondary axis ratio for swipe

    # Accuracy / noise-rejection (research-backed)
    min_confidence: float = 0.45       # lowered — sparse point clouds produce lower confidence
    gesture_cooldown_s: float = 2.0    # seconds locked out after emitting a gesture
    vote_buffer_size: int = 1          # DL model accurate enough — single window sufficient
    min_doppler_ms: float = 0.0        # disabled — IWR1843 reports 0 velocity for most points

    gestures: List[str] = field(default_factory=lambda: [
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
    ])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
@dataclass
class VisualizationSettings:
    update_ms: int = 100        # matplotlib animation interval (lower = faster but more CPU)
    x_range: float = 2.5
    y_range: float = 4.0
    z_range: float = 2.0
    max_track_history: int = 40   # points kept for track tail


# ---------------------------------------------------------------------------
# Data recording
# ---------------------------------------------------------------------------
@dataclass
class DataSettings:
    recording_dir: str = "data/recordings"
    max_buffer_frames: int = 2000


# ---------------------------------------------------------------------------
# Top-level bundle
# ---------------------------------------------------------------------------
@dataclass
class FrameworkSettings:
    radar: RadarSettings = field(default_factory=RadarSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    tracker: TrackerSettings = field(default_factory=TrackerSettings)
    gesture: GestureSettings = field(default_factory=GestureSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)
    data: DataSettings = field(default_factory=DataSettings)

    # Classifier mode:
    #   use_ml_classifier = False → rule-based (no training needed)
    #   use_ml_classifier = True  → sklearn ML model (gesture_rf_realtime.joblib)
    #   use_dl_classifier = True  → CNN-BiLSTM deep learning (gesture_dl.pt)
    use_ml_classifier: bool  = False
    use_dl_classifier: bool  = False   # original CNN-BiLSTM
    use_attn_classifier: bool = True   # CNN + Self-Attention + BiLSTM (recommended)
    ml_model_path:   str = "classification/models/gesture_rf_realtime.joblib" # ML model
    dl_model_path:   str = "classification/models/gesture_dl.pt" # original CNN-BiLSTM
    attn_model_path: str = "classification/models/attention/gesture_attn.pt" #  CNN + Self-Attention + BiLSTM 

    # Enable to collect labelled data for ML training
    enable_recording: bool = False

    log_level: str = "INFO"