# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
Binary TLV parser for the TI mmWave OOB Demo.

Packet layout
─────────────
  [Magic Word 8 B]
  [Header      32 B]   version · totalLen · platform · frameNum ·
                        timeCycles · numDetObj · numTLVs · subFrameNum
  [TLV₀ hdr    8 B]   type(u32) · length(u32)
  [TLV₀ data   N B]
  ...

TLV types used here
  1  – detected point cloud  (x,y,z,velocity) × numDetObj  float32×4
  7  – side info per point   (snr,noise) × numDetObj        int16×2
  6  – statistics frame

Reference: mmWave SDK User Guide / DPC / DPIF datatypes
"""

import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

# ── magic word (little-endian bytes) ──────────────────────────────────────────
MAGIC = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

# header sizes (bytes)
MAGIC_SIZE = 8
HEADER_STRUCT_SIZE = 32          # 8 × uint32 after magic
HEADER_TOTAL = MAGIC_SIZE + HEADER_STRUCT_SIZE
TLV_HDR_SIZE = 8                 # type(u32) + length(u32)

# TLV type IDs
TLV_DETECTED_POINTS = 1
TLV_SIDE_INFO = 7
TLV_STATS = 6

# SDK 3.x point format: x,y,z,velocity as float32 (16 bytes/point)
POINT_BYTES = 16
SIDE_INFO_BYTES = 4

# SDK 2.x point format: 4-byte header + int16 Q-format values (12 bytes/point)
SDK2_POINT_HDR_BYTES = 4   # numDetectedObj(u16) + xyzQFormat(u16)
SDK2_POINT_BYTES = 12      # rangeIdx(u16) dopplerIdx(i16) peakVal(u16) x(i16) y(i16) z(i16)
SDK2_DOPPLER_VEL_RES = 0.13  # m/s per Doppler bin for standard IWR1443 config


@dataclass
class DetectedPoint:
    x: float        # lateral  (m)
    y: float        # depth    (m)
    z: float        # elevation(m)
    velocity: float # Doppler  (m/s)
    snr: float = 0.0
    noise: float = 0.0


@dataclass
class RadarFrame:
    frame_number: int
    timestamp: float                            # host time (s)
    num_detected: int
    points: List[DetectedPoint] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class RadarParser:
    """
    Runs in a background thread.
    Consumes raw bytes from `raw_queue`, emits RadarFrame into `frame_queue`.
    """

    def __init__(self, raw_queue: queue.Queue, frame_queue: queue.Queue):
        self._raw_q = raw_queue
        self._frame_q = frame_queue
        self._buf = bytearray()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.frames_parsed = 0
        self.parse_errors = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._parse_loop, name="radar-parser", daemon=True
        )
        self._thread.start()
        log.info("Parser thread started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _parse_loop(self) -> None:
        while not self._stop_event.is_set():
            # Drain raw queue into local buffer
            try:
                chunk = self._raw_q.get(timeout=0.05)
                self._buf.extend(chunk)
            except queue.Empty:
                continue

            # Process as many complete frames as possible
            while True:
                frame = self._try_parse_frame()
                if frame is None:
                    break
                try:
                    self._frame_q.put_nowait(frame)
                except queue.Full:
                    pass  # drop oldest if consumer is slow

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _try_parse_frame(self) -> Optional[RadarFrame]:
        # 1. Find magic word
        idx = self._find_magic()
        if idx < 0:
            # Keep last 7 bytes in case magic spans chunks
            if len(self._buf) > 7:
                self._buf = self._buf[-7:]
            return None

        if idx > 0:
            # Discard bytes before magic
            del self._buf[:idx]

        # 2. Need at least full header
        if len(self._buf) < HEADER_TOTAL:
            return None

        # 3. Parse header (skip past magic)
        hdr_data = self._buf[MAGIC_SIZE: HEADER_TOTAL]
        (
            version,
            total_len,
            platform,
            frame_num,
            time_cycles,
            num_obj,
            num_tlvs,
            sub_frame,
        ) = struct.unpack_from("<8I", hdr_data)

        # 4. Wait until full packet is buffered
        if len(self._buf) < total_len:
            return None

        # 5. Extract packet bytes, advance buffer
        packet = bytes(self._buf[:total_len])
        del self._buf[:total_len]

        # 6. Parse TLVs
        try:
            frame = self._parse_packet(packet, frame_num, num_obj, num_tlvs)
            self.frames_parsed += 1
            return frame
        except Exception as exc:
            self.parse_errors += 1
            log.debug("Parse error frame %d: %s", frame_num, exc)
            return None

    def _find_magic(self) -> int:
        """Return index of MAGIC in buffer, or -1."""
        buf = bytes(self._buf)
        pos = buf.find(MAGIC)
        return pos

    # ------------------------------------------------------------------
    # Packet → RadarFrame
    # ------------------------------------------------------------------

    def _parse_packet(
        self, packet: bytes, frame_num: int, num_obj: int, num_tlvs: int
    ) -> RadarFrame:
        offset = HEADER_TOTAL
        points: List[DetectedPoint] = []
        side_infos: List[Tuple[float, float]] = []
        stats: dict = {}

        for _ in range(num_tlvs):
            if offset + TLV_HDR_SIZE > len(packet):
                break
            tlv_type, tlv_len = struct.unpack_from("<II", packet, offset)
            offset += TLV_HDR_SIZE
            tlv_data = packet[offset: offset + tlv_len]
            offset += tlv_len

            if tlv_type == TLV_DETECTED_POINTS:
                points = self._parse_point_cloud(tlv_data, num_obj)

            elif tlv_type == TLV_SIDE_INFO:
                side_infos = self._parse_side_info(tlv_data, num_obj)

            elif tlv_type == TLV_STATS:
                stats = self._parse_stats(tlv_data)

        # Merge side info into points
        for i, si in enumerate(side_infos):
            if i < len(points):
                points[i].snr = si[0]
                points[i].noise = si[1]

        return RadarFrame(
            frame_number=frame_num,
            timestamp=time.time(),
            num_detected=num_obj,
            points=points,
            stats=stats,
        )

    @staticmethod
    def _parse_point_cloud(data: bytes, num_obj: int) -> List[DetectedPoint]:
        if not data or num_obj == 0:
            return []
        # Auto-detect SDK version by data length:
        #   SDK 2.x: 4-byte header + 12 bytes/point
        #   SDK 3.x: 16 bytes/point (float32 x4), no header
        sdk2_len = SDK2_POINT_HDR_BYTES + num_obj * SDK2_POINT_BYTES
        sdk3_len = num_obj * POINT_BYTES
        if len(data) == sdk2_len:
            return RadarParser._parse_point_cloud_sdk2(data, num_obj)
        elif len(data) >= sdk3_len:
            return RadarParser._parse_point_cloud_sdk3(data, num_obj)
        # Ambiguous length — try SDK 2.x first then SDK 3.x
        if len(data) >= SDK2_POINT_HDR_BYTES:
            return RadarParser._parse_point_cloud_sdk2(data, num_obj)
        return []

    @staticmethod
    def _parse_point_cloud_sdk2(data: bytes, num_obj: int) -> List[DetectedPoint]:
        if len(data) < SDK2_POINT_HDR_BYTES:
            return []
        num_detected, xyz_q = struct.unpack_from("<HH", data, 0)
        q = float(1 << xyz_q) if xyz_q < 16 else 128.0
        pts = []
        n = min(num_detected, (len(data) - SDK2_POINT_HDR_BYTES) // SDK2_POINT_BYTES)
        for i in range(n):
            off = SDK2_POINT_HDR_BYTES + i * SDK2_POINT_BYTES
            _, doppler_idx, _, x_q, y_q, z_q = struct.unpack_from("<HhHhhh", data, off)
            pts.append(DetectedPoint(
                x=x_q / q, y=y_q / q, z=z_q / q,
                velocity=doppler_idx * SDK2_DOPPLER_VEL_RES,
            ))
        return pts

    @staticmethod
    def _parse_point_cloud_sdk3(data: bytes, num_obj: int) -> List[DetectedPoint]:
        pts = []
        if len(data) < num_obj * POINT_BYTES:
            num_obj = len(data) // POINT_BYTES
        for i in range(num_obj):
            x, y, z, v = struct.unpack_from("<4f", data, i * POINT_BYTES)
            pts.append(DetectedPoint(x=x, y=y, z=z, velocity=v))
        return pts

    @staticmethod
    def _parse_side_info(data: bytes, num_obj: int) -> List[Tuple[float, float]]:
        sis = []
        expected = num_obj * SIDE_INFO_BYTES
        if len(data) < expected:
            num_obj = len(data) // SIDE_INFO_BYTES
        for i in range(num_obj):
            snr, noise = struct.unpack_from("<2h", data, i * SIDE_INFO_BYTES)
            sis.append((snr * 0.1, noise * 0.1))  # convert to dB
        return sis

    @staticmethod
    def _parse_stats(data: bytes) -> dict:
        if len(data) < 24:
            return {}
        fields = struct.unpack_from("<6I", data)
        keys = [
            "interframe_proc_time_us",
            "transmit_output_time_us",
            "interframe_margin_us",
            "interchirp_margin_us",
            "active_frame_cpu_load",
            "interframe_cpu_load",
        ]
        return dict(zip(keys, fields))