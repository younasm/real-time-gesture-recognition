# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
serial_test.py - Raw serial communication test for the configured radar.
Reads port settings from config/settings.py automatically.
"""
import serial
import time
import sys
import os

# Load ports from settings.py so this script always matches main.py
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config.settings import RadarSettings
    _s = RadarSettings()
    CLI_PORT  = _s.cli_port
    DATA_PORT = _s.data_port
except Exception:
    CLI_PORT  = "/dev/tty.usbmodemR21010501"  # for windows:"COM9"    |   for Mac:  "/dev/tty.usbmodemR21010501"  ← update to your IWR1843 CLI  port after checking Device Manager
    DATA_PORT = "/dev/tty.usbmodemR21010504"  # for windows:"COM10"   |   for Mac:  "/dev/tty.usbmodemR21010504" ← update to your IWR1843 Data port after checking Device Manager

BAUD = 115200

print(f"Opening {CLI_PORT} @ {BAUD}...")
try:
    s = serial.Serial(
        CLI_PORT,
        baudrate=BAUD,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
        timeout=2,
    )
    # Assert RTS and DTR — PuTTY does this by default; some firmware
    # requires these lines active before it enables its UART transmitter.
    s.rts = True
    s.dtr = True
except serial.SerialException as e:
    print(f"Failed to open port: {e}")
    raise SystemExit(1)

print("Port open (RTS+DTR asserted). Listening for 3 seconds first (catch boot banner)...")
time.sleep(3)
boot_data = s.read(s.in_waiting or 1)
if boot_data:
    print(f"Boot output received ({len(boot_data)} bytes):")
    print(boot_data.decode(errors="replace"))
else:
    print("No boot output received.")

print()
print(f"=== Checking {CLI_PORT} for binary data (sensor may be outputting on CLI port) ===")
try:
    d9 = serial.Serial(CLI_PORT, baudrate=115_200, timeout=0.2)
    d9.rts = True; d9.dtr = True
    buf9 = bytearray()
    for _ in range(15): buf9.extend(d9.read(4096))
    d9.close()
    MAGIC = bytes([0x02,0x01,0x04,0x03,0x06,0x05,0x08,0x07])
    frames9 = bytes(buf9).count(MAGIC)
    print(f"  {CLI_PORT} @ 115200: {len(buf9)} bytes, {frames9} magic frames")
except Exception as e: print(f"  {CLI_PORT} error: {e}")
print()

print("=== Testing bpmCfgAdvanced (showing actual error) ===")
for params in [
    "", "0", "1",
    "0 0", "0 1", "0 2",
    "0 0 0", "0 0 2", "0 0 1",
    "0 0 0 0", "0 0 2 0",
    "0 0 2 0 0", "0 0 0 1 0 2",
    "0 0 2 0 0 0 0", "0 0 0 0 0 0 0",
]:
    cmd = f"bpmCfgAdvanced {params}\n".strip() + "\n"
    s.write(cmd.encode())
    time.sleep(0.4)
    resp = s.read(s.in_waiting or 1).decode(errors="replace").strip()
    resp_short = resp.replace("\n"," ").replace("\r","")[:80]
    status = "Done ✓" if "Done" in resp else f"→ {resp_short}"
    print(f"  bpmCfgAdvanced {params:<22} {status}")
print()

print(f"=== sensorStart 0 then check {DATA_PORT} ===")
s.write(b"sensorStop\n"); time.sleep(0.5); s.read(s.in_waiting or 1)
s.write(b"sensorStart 0\n"); time.sleep(1)
resp = s.read(s.in_waiting or 1).decode(errors="replace").strip()
print(f"  sensorStart 0 → {resp[:60]}")
print(f"  Reading {DATA_PORT} for 5 seconds...")
try:
    d = serial.Serial(DATA_PORT, baudrate=921_600, timeout=0.2)
    d.rts = True; d.dtr = True
    buf = bytearray()
    for _ in range(25):
        buf.extend(d.read(4096))
    d.close()
    MAGIC = bytes([0x02,0x01,0x04,0x03,0x06,0x05,0x08,0x07])
    frames = 0; pos = 0
    while True:
        idx = bytes(buf).find(MAGIC, pos)
        if idx < 0: break
        frames += 1; pos = idx + 8
    print(f"  {DATA_PORT}: {len(buf)} bytes, {frames} frames")
except Exception as e:
    print(f"  {DATA_PORT} error: {e}")
print()

print("=== Testing sensorStart 0 (restart without reconfig) ===")
s.write(b"sensorStop\n"); time.sleep(1)
s.read(s.in_waiting or 1)
s.write(b"sensorStart 0\n"); time.sleep(2)
resp = s.read(s.in_waiting or 1).decode(errors="replace").strip()
print(f"  sensorStart 0 → {resp[:120]}")
print()

print("=== Testing peakGrouping parameter combinations ===")
for params in [
    "1",
    "0",
    "1 1",
    "0 0",
    "1 1 1",
    "1 4.0 0.3",
    "1 1.5 0.5",
    "1 4.0 0.3 1",
    "1 1 1 1",
    "1 4.0 0.3 1 1",
    "1 1 1 1 1",
]:
    cmd = f"peakGrouping {params}\n".encode()
    s.write(cmd)
    time.sleep(0.5)
    resp = s.read(s.in_waiting or 1).decode(errors="replace").strip()
    result = "Done ✓" if "Done" in resp else ("FAIL: " + resp.replace("\n", " ").replace("\r", "")[:60])
    print(f"  peakGrouping {params:<25} → {result}")
print()

print("Sending 'sensorStop' first to reset state...")
s.write(b"sensorStop\n"); time.sleep(0.5); s.read(s.in_waiting or 1)
print("Sending 'help' command (listing all supported CLI commands)...")
s.write(b"help\n")
time.sleep(2)
resp = s.read(s.in_waiting or 1)
if resp:
    print(resp.decode(errors="replace"))
else:
    print("No response to 'help'.")

print("Sending 'version' command...")
s.write(b"version\n")
time.sleep(1)
resp = s.read(s.in_waiting or 1)
if resp:
    print(f"Response ({len(resp)} bytes):")
    print(resp.decode(errors="replace"))
else:
    print("No response to 'version'.")

print()
print("Sending 'sensorStop' command...")
s.write(b"sensorStop\n")
time.sleep(1)
resp = s.read(s.in_waiting or 1)
if resp:
    print(f"Response ({len(resp)} bytes):")
    print(resp.decode(errors="replace"))
else:
    print("No response to 'sensorStop'.")

print()
print("Sending empty Enter...")
s.write(b"\n")
time.sleep(1)
resp = s.read(s.in_waiting or 1)
if resp:
    print(f"Response ({len(resp)} bytes):")
    print(resp.decode(errors="replace"))
else:
    print("No response to empty Enter.")

s.close()
print("\nDone.")