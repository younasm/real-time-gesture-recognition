# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
Serial interface to the IWR1443BOOST and IWR1843BOOST.

Two ports are used:
  CLI  port (115 200 baud) – send configuration commands one-by-one
  Data port (921 600 baud) – receive binary TLV frames (read thread)

The raw byte stream from the data port is placed into `raw_queue` as chunks;
the parser thread reads from that queue and emits parsed RadarFrame objects.
"""

import queue
import threading
import time
from pathlib import Path
from typing import Optional

import serial

from config.settings import RadarSettings
from utils.logger import get_logger

log = get_logger(__name__)


class RadarInterface:
    def __init__(self, settings: RadarSettings, raw_queue: queue.Queue):
        self._s = settings
        self._raw_queue: queue.Queue = raw_queue
        self._cli: Optional[serial.Serial] = None
        self._data: Optional[serial.Serial] = None
        self._read_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> None:
        log.info("Opening CLI  port %s @ %d", self._s.cli_port, self._s.cli_baud)
        self._cli = serial.Serial(
            self._s.cli_port,
            baudrate=self._s.cli_baud,
            timeout=1,
        )
        self._cli.rts = True
        self._cli.dtr = True

        log.info("Opening Data port %s @ %d", self._s.data_port, self._s.data_baud)
        self._data = serial.Serial(
            self._s.data_port,
            baudrate=self._s.data_baud,
            timeout=1,
        )
        self._data.rts = True
        self._data.dtr = True

    def configure(self) -> None:
        cfg_path = Path(self._s.config_file)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        log.info("Sending config: %s", cfg_path)
        with open(cfg_path, "r") as f:
            lines = f.readlines()

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            log.debug("CMD > %s", line)
            self._cli.write((line + "\n").encode())
            time.sleep(self._s.cmd_delay_ms / 1000.0)

            # Read response and verify sensorStart succeeded
            resp = b""
            deadline = time.time() + 0.5
            while time.time() < deadline:
                waiting = self._cli.in_waiting
                if waiting:
                    resp += self._cli.read(waiting)
                    if b"Done" in resp or b"Error" in resp:
                        break
                time.sleep(0.01)

            resp_text = resp.decode(errors="replace")
            if line == "sensorStart":
                if "Done" in resp_text:
                    log.info("Configuration complete – sensor started successfully")
                else:
                    error_msg = resp_text.strip().replace("\n", " ")[:120]
                    raise RuntimeError(
                        f"sensorStart FAILED: {error_msg}\n"
                        f"Check config/{self._s.config_file} and radar hardware."
                    )
            else:
                # For all other commands, just drain the buffer
                self._cli.flushInput()

    def start_streaming(self) -> None:
        self._stop_event.clear()
        self._read_thread = threading.Thread(
            target=self._read_loop, name="radar-read", daemon=True
        )
        self._read_thread.start()
        log.info("Data read thread started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._read_thread:
            self._read_thread.join(timeout=3)

        # Send sensorStop before closing ports
        if self._cli and self._cli.is_open:
            try:
                self._cli.write(b"sensorStop\n")
                time.sleep(0.1)
            except Exception:
                pass
            self._cli.close()

        if self._data and self._data.is_open:
            self._data.close()

        log.info("Radar interface closed")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        READ_CHUNK = 4096
        total_bytes = 0
        import time
        t_next_log = time.time() + 3.0
        while not self._stop_event.is_set():
            try:
                waiting = self._data.in_waiting
                n = max(waiting, 1)
                chunk = self._data.read(min(n, READ_CHUNK))
                if chunk:
                    total_bytes += len(chunk)
                    self._raw_queue.put(chunk)
                now = time.time()
                if now >= t_next_log:
                    log.info("Data port bytes received so far: %d", total_bytes)
                    t_next_log = now + 3.0
            except serial.SerialException as exc:
                log.error("Serial read error: %s", exc)
                break
            except Exception as exc:
                log.error("Unexpected read error: %s", exc)
                break