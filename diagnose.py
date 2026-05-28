# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
diagnose.py – IWR1443 hardware connectivity checker
====================================================
Run BEFORE main.py to confirm serial ports and data flow.

Usage:
    python diagnose.py
    python diagnose.py --cli COM15 --data COM16   # explicit ports (example)
    python diagnose.py --swap                    # swap CLI/data to test
"""

import argparse
import time
import struct
import serial

MAGIC = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

# Auto-read config file path from settings.py
try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from config.settings import RadarSettings as _RS
    CFG_FILE = _RS().config_file
except Exception:
    CFG_FILE = "config/iwr1443_gesture.cfg"


def load_cfg_commands(path: str) -> list:
    commands = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("%"):
                    commands.append(line)
    except FileNotFoundError:
        print(f"  ERROR: config file not found: {path}")
    return commands


def hr(title=""):
    print("\n" + ("─" * 60))
    if title:
        print(f"  {title}")
        print("─" * 60)


def check_port(port: str, baud: int, label: str) -> bool:
    print(f"  Opening {label} port  {port} @ {baud} baud … ", end="", flush=True)
    try:
        s = serial.Serial(port, baudrate=baud, timeout=1)
        s.close()
        print("OK")
        return True
    except serial.SerialException as e:
        print(f"FAILED  ({e})")
        return False


def send_config_and_read(cli_port: str, data_port: str, duration_s: float = 5.0):
    """Open CLI and data ports simultaneously, send config, then read data.
    Both ports stay open together — required for firmware to output data."""
    commands = load_cfg_commands(CFG_FILE)
    if not commands:
        return

    # Open data port FIRST (before sending sensorStart)
    try:
        data = serial.Serial(data_port, baudrate=921_600, timeout=0.2)
        data.rts = True
        data.dtr = True
    except serial.SerialException as e:
        print(f"  Could not open data port: {e}")
        return

    hr(f"Sending configuration from {CFG_FILE}  (raw responses)")
    done_count = 0
    try:
        cli = serial.Serial(cli_port, baudrate=115_200, timeout=1)
        cli.rts = True
        cli.dtr = True
        for line in commands:
            cmd = (line + "\n").encode()
            cli.write(cmd)
            time.sleep(0.25)
            resp = cli.read(max(cli.in_waiting, 1))
            resp_text = resp.decode(errors="replace").strip()
            if "Done" in resp_text:
                done_count += 1
                status = "Done ✓"
            elif resp_text == "":
                status = "(no response)"
            else:
                status = repr(resp_text[:80])
            print(f"    > {line:<50}  {status}")
        # Keep CLI open while reading data!
        print()
        if done_count == 0:
            print("  [WARN] No 'Done' responses received from the radar.")
        else:
            print(f"  Config sent  ({done_count}/{len(commands)} commands acknowledged).")
    except serial.SerialException as e:
        print(f"  CLI FAILED: {e}")
        data.close()
        return

    MAGIC = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

    # Read BOTH ports simultaneously while CLI stays open
    hr(f"Reading {data_port} AND {cli_port} for {duration_s:.0f} s  (sensor running, hand wave helps)")
    buf10 = bytearray()   # data port frames
    buf9  = bytearray()   # cli  port frames (check if data arrives there instead)
    t_end = time.time() + duration_s

    print("  Reading… (wave your hand in front of the sensor)")
    while time.time() < t_end:
        chunk10 = data.read(min(data.in_waiting or 1, 4096))
        if chunk10:
            buf10.extend(chunk10)
        chunk9 = cli.read(min(cli.in_waiting or 0, 4096)) if cli.in_waiting else b""
        if chunk9:
            buf9.extend(chunk9)
        elapsed = duration_s - (t_end - time.time())
        m10 = bytes(buf10).count(MAGIC)
        m9  = bytes(buf9).count(MAGIC)
        print(f"\r  {elapsed:4.1f}s  {data_port}={len(buf10):6d}B/{m10}frm  {cli_port}={len(buf9):6d}B/{m9}frm   ", end="", flush=True)

    data.close()
    cli.close()
    print()

    m10 = bytes(buf10).count(MAGIC)
    m9  = bytes(buf9).count(MAGIC)

    hr("Results")
    print(f"  {data_port} bytes: {len(buf10):6d}  frames: {m10}")
    print(f"  {cli_port}  bytes: {len(buf9):6d}  frames: {m9}")

    if m10 > 0:
        print()
        print(f"  [OK] Data is on {data_port}!  ~{m10/duration_s:.1f} frames/sec")
        print("       Run:  python main.py")
        idx = bytes(buf10).find(MAGIC)
        if idx >= 0 and len(buf10)-idx >= 40:
            try:
                ver,tot,plat,fnum,cyc,nobj,ntlv,sub = __import__('struct').unpack_from("<8I", buf10[idx+8:idx+40])
                print(f"  Frame #{fnum}  objects={nobj}  tlvs={ntlv}")
            except: pass
    elif m9 > 0:
        print()
        print(f"  [!!] Data is on {cli_port} (CLI port), NOT {data_port}!")
        print(f"       The data_port setting is wrong — swap ports in config/settings.py")
        print(f"       Update config/settings.py:  data_port = '{cli_port}'  cli_port = '{data_port}'")
    elif len(buf10) > 0 or len(buf9) > 0:
        total = len(buf10) + len(buf9)
        sample = bytes(buf10[:64]) if buf10 else bytes(buf9[:64])
        print(f"\n  [WARN] {total} bytes received but no frames. First bytes:")
        print("  " + " ".join(f"{b:02x}" for b in sample))
    else:
        print()
        print("  [FAIL] No bytes on either port.")
        print("         The sensor may not be transmitting. Check guiMonitor setting.")
        print("         Ensure something is within 0.5-2m of the sensor.")


def read_data_port(data_port: str, duration_s: float = 5.0):
    """Legacy stub — not used when CLI port must stay open."""
    hr(f"Reading data port {data_port} for {duration_s:.0f} s")
    try:
        data = serial.Serial(data_port, baudrate=921_600, timeout=0.2)
        data.rts = True
        data.dtr = True
    except serial.SerialException as e:
        print(f"  Could not open data port: {e}")
        return

    buf = bytearray()
    total_bytes = 0
    magic_count = 0
    t_end = time.time() + duration_s

    print("  Reading… (wave your hand in front of the sensor)")
    while time.time() < t_end:
        chunk = data.read(4096)
        if chunk:
            buf.extend(chunk)
            total_bytes += len(chunk)

        # Count magic words found so far
        tmp = bytes(buf)
        pos = 0
        while True:
            idx = tmp.find(MAGIC, pos)
            if idx < 0:
                break
            magic_count += 1
            pos = idx + len(MAGIC)

        elapsed = duration_s - (t_end - time.time())
        print(f"\r  {elapsed:4.1f}s  bytes={total_bytes:7d}  frames_found={magic_count}   ", end="", flush=True)

    data.close()
    print()

    hr("Results")
    print(f"  Total bytes received : {total_bytes}")
    print(f"  Magic words (frames) : {magic_count}")

    if total_bytes == 0:
        print()
        print("  [FAIL] No bytes received on the data port.")
        print()
        print("  *** FIRMWARE IS MOST LIKELY WRONG ***")
        print()
        print("  The IWR1443BOOST must be flashed with the TI mmWave SDK")
        print("  'Out-of-Box Demo' firmware.  Steps:")
        print()
        print("  1. Download  mmWave SDK  from  ti.com/tool/MMWAVE-SDK")
        print("     (version 3.x for IWR1443)")
        print()
        print("  2. Locate the pre-built binary:")
        print("     <SDK>\\packages\\ti\\demo\\xwr14xx\\mmw\\")
        print("         xwr14xx_mmw_demo.bin  (or .xer4f + .xdsplink)")
        print()
        print("  3. Flash using TI UniFlash:")
        print("     ti.com/tool/UNIFLASH")
        print("     - Select device: IWR1443")
        print("     - Program xwr14xx_mmw_demo.bin")
        print()
        print("  4. After flashing, re-run this script.")
    elif magic_count == 0:
        print()
        print("  [WARN] Bytes received but NO valid frames detected.")
        print("         Possible causes:")
        print("           1. Wrong firmware – not TI mmWave SDK OOB demo binary format")
        print("           2. Config was sent to the wrong port (try --swap)")
        print("           3. Sensor is running but 0 objects detected (move hand closer)")

        hr("First 64 raw bytes (hex)")
        print("  " + " ".join(f"{b:02x}" for b in buf[:64]))
        print()
        print("  Expected frame start: 02 01 04 03 06 05 08 07")
    else:
        print()
        print("  [OK] Hardware is working correctly!")
        print(f"       Detected ~{magic_count / duration_s:.1f} frames/second")
        print("       You can now run:  python main.py")

        # Decode first frame header
        tmp = bytes(buf)
        idx = tmp.find(MAGIC)
        if idx >= 0 and len(tmp) - idx >= 40:
            hdr = tmp[idx + 8: idx + 40]
            try:
                ver, tot_len, platform, frame_num, cycles, num_obj, num_tlvs, sub = \
                    struct.unpack_from("<8I", hdr)
                hr("First frame header")
                print(f"  version      : 0x{ver:08x}")
                print(f"  total_len    : {tot_len} bytes")
                print(f"  platform     : 0x{platform:08x}")
                print(f"  frame_num    : {frame_num}")
                print(f"  num_detected : {num_obj}")
                print(f"  num_tlvs     : {num_tlvs}")
            except Exception:
                pass


def main():
    # Read defaults from settings.py so diagnose.py always matches main.py
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from config.settings import RadarSettings
        _s = RadarSettings()
        default_cli  = _s.cli_port
        default_data = _s.data_port
    except Exception:
        default_cli  = "COM9"   # fallback if settings.py can't be read
        default_data = "COM10"  # fallback if settings.py can't be read

    ap = argparse.ArgumentParser()
    ap.add_argument("--cli",  default=default_cli,  help=f"CLI port  (default {default_cli})")
    ap.add_argument("--data", default=default_data, help=f"Data port (default {default_data})")
    ap.add_argument("--swap", action="store_true",
                    help="Swap CLI and data ports (test if they are reversed)")
    ap.add_argument("--duration", type=float, default=5.0,
                    help="Seconds to read from data port (default 5)")
    args = ap.parse_args()

    cli_port  = args.data if args.swap else args.cli
    data_port = args.cli  if args.swap else args.data

    if args.swap:
        print("  *** SWAP MODE: using CLI={data_port}  DATA={cli_port} ***")

    hr("Port connectivity check")
    cli_ok  = check_port(cli_port,  115_200, "CLI ")
    data_ok = check_port(data_port, 921_600, "Data")

    if not cli_ok or not data_ok:
        print("\n  Fix port issues above before continuing.")
        return

    send_config_and_read(cli_port, data_port, args.duration)


if __name__ == "__main__":
    main()