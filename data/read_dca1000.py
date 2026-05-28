# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
data/read_dca1000.py — Python equivalent of TI's readDCA1000.m + full processing.

Reads raw IQ .bin files captured by DCA1000EVM + mmWave Studio,
organises them into chirp/loop/antenna arrays, computes Range-Doppler
and Range-Angle maps, and saves publication-quality plots.

Matches the MATLAB function provided by TI:
  readDCA1000('<ADC capture bin file>')

Workflow
────────
  1. Capture .bin file using mmWave Studio + DCA1000EVM
  2. Copy .bin file to  data/raw/
  3. Run this script to produce Range-Doppler and Range-Angle maps

Usage
─────
  # Single file
  python -m data.read_dca1000 --file data/raw/swipe_up.bin --gesture swipe_up

  # All .bin files in a folder
  python -m data.read_dca1000 --folder data/raw/

IWR1843 parameters — update to match your mmWave Studio config
───────────────────────────────────────────────────────────────
  NUM_ADC_SAMPLES  = 256    (samples per chirp per RX antenna)
  NUM_RX           = 4      (receive antennas)
  NUM_TX           = 3      (transmit antennas, MIMO)
  NUM_LOOPS        = 16     (chirp loops per frame)
  SAMPLE_RATE_KSPS = 5000
  FREQ_SLOPE       = 29.982 (MHz/us)
  START_FREQ_GHZ   = 77
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Radar parameters — update these to match your mmWave Studio config ────────
NUM_ADC_SAMPLES   = 256
NUM_RX            = 4
NUM_TX            = 3
NUM_LOOPS         = 16
NUM_ADC_BITS      = 16
NUM_LANES         = 4      # always 4 for DCA1000
IS_REAL           = False  # False = complex IQ (standard)

SAMPLE_RATE_KSPS  = 5000.0
FREQ_SLOPE_MHZ_US = 29.982
START_FREQ_GHZ    = 77.0
IDLE_TIME_US      = 7.0
RAMP_END_TIME_US  = 60.0
LIGHT_SPEED       = 3e8


# ── Derived constants ──────────────────────────────────────────────────────────

def compute_radar_params() -> dict:
    c   = LIGHT_SPEED
    B   = FREQ_SLOPE_MHZ_US * 1e6 * (RAMP_END_TIME_US - IDLE_TIME_US) * 1e-6
    range_res  = c / (2 * B)
    max_range  = range_res * NUM_ADC_SAMPLES / 2
    chirp_time = RAMP_END_TIME_US * 1e-6
    lambda_m   = c / (START_FREQ_GHZ * 1e9)
    vel_res    = lambda_m / (2 * NUM_TX * NUM_LOOPS * chirp_time)
    max_vel    = vel_res * NUM_LOOPS / 2
    return {
        "range_res_m": range_res,
        "max_range_m": max_range,
        "vel_res_mps": vel_res,
        "max_vel_mps": max_vel,
    }


# ── 1. Read .bin file (Python equivalent of TI's readDCA1000.m) ───────────────

def read_dca1000_bin(filepath: str) -> np.ndarray:
    """
    Direct Python translation of TI's readDCA1000.m MATLAB function.

    MATLAB:
        adcData = fread(fid, 'int16');
        adcData = reshape(adcData, numLanes*2, []);
        adcData = adcData(1:4,:) + sqrt(-1)*adcData(5:8,:);

    Returns complex ndarray of shape (NUM_RX, total_samples).
    """
    raw = np.fromfile(filepath, dtype=np.int16)

    # Sign-extend if not 16-bit ADC
    if NUM_ADC_BITS != 16:
        l_max = 2 ** (NUM_ADC_BITS - 1) - 1
        raw[raw > l_max] = raw[raw > l_max] - 2 ** NUM_ADC_BITS

    if IS_REAL:
        adc = raw.reshape(NUM_LANES, -1)
    else:
        # Lanes 0-3 = I parts, lanes 4-7 = Q parts
        adc = raw.reshape(NUM_LANES * 2, -1)
        adc = adc[:4, :] + 1j * adc[4:8, :]

    return adc.astype(np.complex64)   # (NUM_RX, total_samples)


# ── 2. Organise into 4D frame array ───────────────────────────────────────────

def organise_frame(adc: np.ndarray) -> np.ndarray:
    """
    Reshape (NUM_RX, total_samples) → (NUM_TX, NUM_LOOPS, NUM_RX, NUM_ADC_SAMPLES).
    Chirps are interleaved in TX order: TX0_loop0, TX1_loop0, TX2_loop0, TX0_loop1…
    """
    num_rx, _ = adc.shape
    total_chirps = NUM_TX * NUM_LOOPS
    data = adc.reshape(num_rx, total_chirps, NUM_ADC_SAMPLES)

    frame = np.zeros((NUM_TX, NUM_LOOPS, num_rx, NUM_ADC_SAMPLES),
                     dtype=np.complex64)
    for tx in range(NUM_TX):
        frame[tx] = data[:, tx::NUM_TX, :].transpose(1, 0, 2)

    return frame   # (NUM_TX, NUM_LOOPS, NUM_RX, NUM_ADC_SAMPLES)


# ── 3. Range-Doppler map ───────────────────────────────────────────────────────

def range_doppler_map(frame: np.ndarray) -> np.ndarray:
    """
    2D FFT → Range-Doppler magnitude (dB).
    Output: (NUM_LOOPS, NUM_ADC_SAMPLES//2) — positive ranges, centred Doppler.
    """
    win_r = np.hanning(NUM_ADC_SAMPLES)
    win_d = np.hanning(NUM_LOOPS)

    data = frame * win_r
    data = data * win_d[np.newaxis, :, np.newaxis, np.newaxis]

    rfft = np.fft.fft(data, axis=-1)[:, :, :, :NUM_ADC_SAMPLES // 2]
    dfft = np.fft.fftshift(np.fft.fft(rfft, axis=1), axes=1)

    rd = np.mean(np.abs(dfft), axis=(0, 2))
    return rd.astype(np.float32)


# ── 4. Range-Angle map ────────────────────────────────────────────────────────

def range_angle_map(frame: np.ndarray, n_fft_angle: int = 64) -> np.ndarray:
    """
    Spatial FFT over 12 virtual antennas (3 TX × 4 RX) → Range-Azimuth map.
    Output: (n_fft_angle, NUM_ADC_SAMPLES//2).
    """
    win_r = np.hanning(NUM_ADC_SAMPLES)
    data  = frame * win_r
    rfft  = np.fft.fft(data, axis=-1)[:, :, :, :NUM_ADC_SAMPLES // 2]

    # Build virtual array (NUM_TX*NUM_RX, loops, range_bins)
    va     = np.concatenate([rfft[tx] for tx in range(NUM_TX)], axis=1)
    va     = va.transpose(1, 0, 2)
    va_mean = np.mean(va, axis=1)   # average over loops

    angle_fft = np.fft.fftshift(
        np.fft.fft(va_mean, n=n_fft_angle, axis=0), axes=0
    )
    return np.abs(angle_fft).astype(np.float32)


# ── 5. Save plots ──────────────────────────────────────────────────────────────

def save_plots(rd: np.ndarray, ra: np.ndarray,
               params: dict, gesture: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = gesture if gesture else "capture"

    range_axis = np.linspace(0,                   params["max_range_m"], rd.shape[1])
    vel_axis   = np.linspace(-params["max_vel_mps"], params["max_vel_mps"], rd.shape[0])
    angle_axis = np.linspace(-90, 90, ra.shape[0])

    # ── Range-Doppler ─────────────────────────────────────────────────────────
    rd_db = 20 * np.log10(rd + 1e-6)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.pcolormesh(range_axis, vel_axis, rd_db, cmap="jet", shading="auto",
                       vmin=rd_db.max() - 50, vmax=rd_db.max())
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Doppler Velocity (m/s)")
    ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.6)
    plt.tight_layout()
    p = out_dir / f"rd_map_{tag}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Range-Doppler      → {p}")

    # ── Range-Angle ───────────────────────────────────────────────────────────
    ra_db = 20 * np.log10(ra + 1e-6)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.pcolormesh(range_axis, angle_axis, ra_db, cmap="jet", shading="auto",
                       vmin=ra_db.max() - 50, vmax=ra_db.max())
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Azimuth Angle (°)")
    ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.6)
    plt.tight_layout()
    p = out_dir / f"ra_map_{tag}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Range-Angle        → {p}")

    # ── Range profile ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range_axis, 20 * np.log10(rd.mean(axis=0) + 1e-6),
            color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / f"range_profile_{tag}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Range profile      → {p}")

    # Save numpy arrays
    np.save(str(out_dir / f"rd_{tag}.npy"), rd)
    np.save(str(out_dir / f"ra_{tag}.npy"), ra)
    print(f"  Numpy arrays       → rd_{tag}.npy / ra_{tag}.npy")


# ── 6. Process a single .bin file ─────────────────────────────────────────────

def process_file(filepath: str, gesture: str, out_dir: Path) -> None:
    print(f"\n  Reading  {filepath}")
    adc   = read_dca1000_bin(filepath)
    print(f"  ADC shape  : {adc.shape}  (NUM_RX × total_samples)")

    frame = organise_frame(adc)
    print(f"  Frame shape: {frame.shape}  (TX × loops × RX × samples)")

    params = compute_radar_params()
    print(f"  Range res  : {params['range_res_m']*100:.1f} cm/bin  |  "
          f"Max range : {params['max_range_m']:.2f} m")
    print(f"  Vel res    : {params['vel_res_mps']*100:.1f} cm/s/bin  |  "
          f"Max vel   : ±{params['max_vel_mps']:.2f} m/s")

    rd = range_doppler_map(frame)
    ra = range_angle_map(frame)
    save_plots(rd, ra, params, gesture, out_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Read DCA1000 .bin and compute Range-Doppler / Range-Angle maps"
    )
    ap.add_argument("--file",    default="",
                    help="Path to .bin file captured by mmWave Studio")
    ap.add_argument("--folder",  default="",
                    help="Process all .bin files in this folder")
    ap.add_argument("--gesture", default="",
                    help="Gesture label used in output file names")
    ap.add_argument("--out_dir", default="data/iq_maps")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if args.file:
        process_file(args.file, args.gesture, out_dir)

    elif args.folder:
        files = sorted(Path(args.folder).glob("*.bin"))
        if not files:
            print("No .bin files found in", args.folder); return
        for f in files:
            gesture = f.stem.split("_")[0] if "_" in f.stem else f.stem
            process_file(str(f), gesture, out_dir)

    else:
        ap.print_help()
        print("\n  Examples:")
        print("    python -m data.read_dca1000 --file data/raw/swipe_up.bin --gesture swipe_up")
        print("    python -m data.read_dca1000 --folder data/raw/")


if __name__ == "__main__":
    main()
