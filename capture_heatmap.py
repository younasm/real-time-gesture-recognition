# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
capture_heatmap.py — Capture and visualise Range-Doppler heatmaps from IWR1843.

The IWR1843 firmware outputs TLV type 5 (Range-Doppler heatmap) when
guiMonitor has the 5th parameter set to 1.  Each heatmap frame is a
(numDopplerBins × numRangeBins) array of uint16 log-magnitude values —
the 2D FFT output before CFAR.  This is the closest to raw IQ available
over the standard UART data port without a DCA1000EVM capture board.

Usage
─────
  python capture_heatmap.py                    # live display
  python capture_heatmap.py --save             # save frames to .npy
  python capture_heatmap.py --frames 100       # capture 100 frames then exit
  python capture_heatmap.py --gesture swipe_up # tag saved file with gesture name

Radar parameters (matching iwr1843_heatmap.cfg)
─────────────────────────────────────────────────
  numRangeBins   = 256   range res ≈ 0.058 m/bin  →  max range ≈ 14.8 m
  numDopplerBins =  16   velocity res ≈ 0.13 m/s/bin
  Frame rate     =  20 Hz
"""

import argparse
import queue
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ── Radar parameters (must match iwr1843_heatmap.cfg) ────────────────────────
CLI_PORT   = "/dev/tty.usbmodemR21010501"
DATA_PORT  = "/dev/tty.usbmodemR21010504"
CLI_BAUD   = 115_200
DATA_BAUD  = 921_600
CFG_FILE   = "config/iwr1843_heatmap.cfg"

NUM_RANGE_BINS   = 256
NUM_DOPPLER_BINS = 16
RANGE_RES_M      = 0.058        # metres per bin
VEL_RES_MPS      = 0.13         # m/s per bin
MAX_RANGE_M      = NUM_RANGE_BINS  * RANGE_RES_M
MAX_VEL_MPS      = NUM_DOPPLER_BINS * VEL_RES_MPS / 2

# TLV constants (mmWave SDK)
MAGIC_WORD       = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])
TLV_TYPE_HEATMAP = 5
HEATMAP_BYTES    = NUM_RANGE_BINS * NUM_DOPPLER_BINS * 2  # uint16 per cell


# ── Serial helpers ────────────────────────────────────────────────────────────

def send_config(cli_port_name: str) -> None:
    import serial, time as _time
    print(f"  Opening CLI port {cli_port_name}…")
    cli = serial.Serial(cli_port_name, baudrate=CLI_BAUD, timeout=1)
    cli.rts = True; cli.dtr = True

    with open(CFG_FILE) as f:
        lines = f.readlines()

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        cli.write((line + "\n").encode())
        _time.sleep(0.05)
        resp = b""
        deadline = _time.time() + 0.5
        while _time.time() < deadline:
            if cli.in_waiting:
                resp += cli.read(cli.in_waiting)
                if b"Done" in resp or b"Error" in resp:
                    break
            _time.sleep(0.01)
        status = "✓" if b"Done" in resp else "✗"
        print(f"    {status}  {line[:60]}")
    cli.close()


def read_data(data_port_name: str, raw_q: queue.Queue,
              stop: threading.Event) -> None:
    import serial
    port = serial.Serial(data_port_name, baudrate=DATA_BAUD, timeout=1)
    port.rts = True; port.dtr = True
    buf = bytearray()
    while not stop.is_set():
        chunk = port.read(4096)
        if chunk:
            buf.extend(chunk)
            raw_q.put(bytes(chunk))
    port.close()


# ── TLV parser ────────────────────────────────────────────────────────────────

def parse_frames(raw_q: queue.Queue, heatmap_q: queue.Queue,
                 stop: threading.Event) -> None:
    """Extract TLV type-5 heatmap from raw byte stream."""
    buf = bytearray()
    while not stop.is_set():
        try:
            chunk = raw_q.get(timeout=0.1)
        except queue.Empty:
            continue
        buf.extend(chunk)

        # Search for magic word
        while True:
            idx = buf.find(MAGIC_WORD)
            if idx == -1:
                buf = buf[-8:] if len(buf) > 8 else buf
                break
            if idx > 0:
                buf = buf[idx:]

            # Need at least 40 bytes for header
            if len(buf) < 40:
                break

            # Parse frame header (40 bytes)
            # magic(8) version(4) totalLen(4) platform(4) frameNum(4)
            # cpuCycles(4) numDetObj(4) numTLVs(4) subFrameNum(4)
            total_len = struct.unpack_from("<I", buf, 12)[0]
            num_tlvs  = struct.unpack_from("<I", buf, 32)[0]

            if len(buf) < total_len:
                break   # wait for more data

            frame_data = buf[:total_len]
            buf = buf[total_len:]

            offset = 40  # skip header
            for _ in range(num_tlvs):
                if offset + 8 > len(frame_data):
                    break
                tlv_type   = struct.unpack_from("<I", frame_data, offset)[0]
                tlv_length = struct.unpack_from("<I", frame_data, offset + 4)[0]
                offset += 8

                if tlv_type == TLV_TYPE_HEATMAP:
                    if tlv_length >= HEATMAP_BYTES:
                        raw_vals = struct.unpack_from(
                            f"<{NUM_RANGE_BINS * NUM_DOPPLER_BINS}H",
                            frame_data, offset
                        )
                        heatmap = np.array(raw_vals, dtype=np.float32).reshape(
                            NUM_DOPPLER_BINS, NUM_RANGE_BINS
                        )
                        heatmap_q.put(heatmap)

                offset += tlv_length


# ── Live display ──────────────────────────────────────────────────────────────

def run_live(heatmap_q: queue.Queue, save: bool,
             gesture: str, max_frames: int) -> None:
    import matplotlib
    matplotlib.use("MacOSX")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    range_axis = np.linspace(0, MAX_RANGE_M, NUM_RANGE_BINS)
    vel_axis   = np.linspace(-MAX_VEL_MPS, MAX_VEL_MPS, NUM_DOPPLER_BINS)

    fig, ax = plt.subplots(figsize=(10, 5))
    dummy   = np.zeros((NUM_DOPPLER_BINS, NUM_RANGE_BINS))
    im      = ax.imshow(
        dummy, aspect="auto", origin="lower", cmap="jet",
        extent=[0, MAX_RANGE_M, -MAX_VEL_MPS, MAX_VEL_MPS],
        vmin=0, vmax=500,
    )
    plt.colorbar(im, ax=ax, label="Log magnitude")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Doppler Velocity (m/s)")
    ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
    plt.tight_layout()

    saved_frames = []
    frame_count  = [0]
    title_text   = ax.set_title("")

    def update(_):
        updated = False
        while not heatmap_q.empty():
            try:
                hm = heatmap_q.get_nowait()
                # fftshift doppler axis so 0 velocity is in the centre
                hm = np.fft.fftshift(hm, axes=0)
                im.set_data(hm)
                im.set_clim(0, np.percentile(hm, 99))
                frame_count[0] += 1
                title_text.set_text(
                    f"Frame {frame_count[0]}  |  "
                    f"gesture: {gesture if gesture else '—'}"
                )
                if save:
                    saved_frames.append(hm.copy())
                updated = True
                if max_frames and frame_count[0] >= max_frames:
                    _save_and_exit(saved_frames, gesture, frame_count[0])
                    plt.close()
                    return
            except queue.Empty:
                break
        if updated:
            fig.canvas.draw_idle()

    import matplotlib.animation as animation
    ani = animation.FuncAnimation(fig, update, interval=50,
                                  cache_frame_data=False)
    plt.show()

    if save and saved_frames:
        _save_and_exit(saved_frames, gesture, frame_count[0])


def _save_and_exit(frames, gesture, count):
    if not frames:
        return
    out = Path("data/heatmaps")
    out.mkdir(parents=True, exist_ok=True)
    tag  = gesture if gesture else "capture"
    ts   = int(time.time() * 1000)
    path = out / f"{tag}_{ts}.npy"
    np.save(str(path), np.array(frames, dtype=np.float32))
    print(f"\n  Saved {count} frames → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Capture Range-Doppler heatmap from IWR1843"
    )
    ap.add_argument("--save",    action="store_true",
                    help="Save captured heatmap frames to data/heatmaps/")
    ap.add_argument("--frames",  type=int, default=0,
                    help="Stop after N frames (0 = run until Ctrl+C)")
    ap.add_argument("--gesture", default="",
                    help="Label for saved file (e.g. swipe_up)")
    ap.add_argument("--no_config", action="store_true",
                    help="Skip sending config (sensor already running)")
    args = ap.parse_args()

    print("\n" + "═" * 56)
    print("  IWR1843 Range-Doppler Heatmap Capture")
    print(f"  Range bins   : {NUM_RANGE_BINS}  ({RANGE_RES_M:.3f} m/bin)")
    print(f"  Doppler bins : {NUM_DOPPLER_BINS}  ({VEL_RES_MPS:.3f} m/s/bin)")
    print(f"  Max range    : {MAX_RANGE_M:.1f} m")
    print(f"  Max velocity : ±{MAX_VEL_MPS:.2f} m/s")
    print("═" * 56 + "\n")

    if not args.no_config:
        print("  Configuring radar…")
        send_config(CLI_PORT)
        print("  Configuration sent. Warming up 2 s…")
        time.sleep(2.0)

    raw_q     = queue.Queue(maxsize=500)
    heatmap_q = queue.Queue(maxsize=100)
    stop      = threading.Event()

    read_t  = threading.Thread(target=read_data,
                                args=(DATA_PORT, raw_q, stop), daemon=True)
    parse_t = threading.Thread(target=parse_frames,
                                args=(raw_q, heatmap_q, stop), daemon=True)
    read_t.start()
    parse_t.start()

    print("  Streaming…  (close the window or press Ctrl+C to stop)\n")
    try:
        run_live(heatmap_q, args.save, args.gesture, args.frames)
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        stop.set()


if __name__ == "__main__":
    main()
