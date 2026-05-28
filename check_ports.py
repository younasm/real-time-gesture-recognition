# Copyright (c) 2026 Muhammad Younas
# Licensed under CC BY-NC 4.0 — free for scientific use, commercial use prohibited.
# https://creativecommons.org/licenses/by-nc/4.0/

"""
check_ports.py - Read IWR1843 ports directly for 10 seconds (macOS).
Run this IMMEDIATELY after diagnose.py confirms sensorStart Done.
The sensor should still be running.
"""
import serial
import time
import threading

MAGIC = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

CLI_PORT  = "/dev/tty.usbmodemR21010501"   # XDS110 CLI  port (macOS)
DATA_PORT = "/dev/tty.usbmodemR21010504"   # XDS110 Data port (macOS)

results = {}

def read_port(port, baud, duration=10):
    try:
        s = serial.Serial(port, baudrate=baud, timeout=0.1)
        s.rts = True
        s.dtr = True
        buf = bytearray()
        t_end = time.time() + duration
        while time.time() < t_end:
            chunk = s.read(4096)
            if chunk:
                buf.extend(chunk)
        s.close()
        frames = bytes(buf).count(MAGIC)
        results[port] = (len(buf), frames, bytes(buf[:64]))
    except Exception as e:
        results[port] = (0, 0, b"", str(e))

print(f"Reading {CLI_PORT} @ 115200 and {DATA_PORT} @ 921600 simultaneously for 10 seconds...")
print("(Sensor should already be running from previous diagnose.py run)")
print("Wave your hand in front of the sensor!\n")

t_cli  = threading.Thread(target=read_port, args=(CLI_PORT,  115200, 10))
t_data = threading.Thread(target=read_port, args=(DATA_PORT, 921600, 10))

t_cli.start()
t_data.start()

for i in range(10):
    time.sleep(1)
    print(f"  {i+1}s elapsed...", end="\r")

t_cli.join()
t_data.join()

print("\n")
print("=" * 50)
for port, data in results.items():
    if len(data) == 3:
        nbytes, frames, sample = data
    else:
        nbytes, frames, sample, err = data
        print(f"{port}: ERROR - {err}")
        continue
    print(f"{port}: {nbytes} bytes,  {frames} magic frames")
    if nbytes > 0:
        print(f"       First bytes: {sample.hex()}")

print("=" * 50)

# Conclusion
r_cli  = results.get(CLI_PORT,  (0, 0, b""))
r_data = results.get(DATA_PORT, (0, 0, b""))

if r_data[1] > 0:
    print(f"\n[OK] Data is on {DATA_PORT}!  {r_data[1]} frames  -> python main.py")
elif r_cli[1] > 0:
    print(f"\n[!!] Data is on {CLI_PORT}!  {r_cli[1]} frames")
    print("     Update config/settings.py: swap cli_port and data_port")
elif r_data[0] > 0:
    print(f"\n[?]  {r_data[0]} bytes on {DATA_PORT} but no magic word found")
    print("     Data may be there but frame parsing is off")
elif r_cli[0] > 0:
    print(f"\n[?]  {r_cli[0]} bytes on {CLI_PORT} but no magic word found")
else:
    print("\n[FAIL] No data on either port.")
    print("       The sensor may not be outputting data.")
    print("       Try putting your hand 50cm in front of the radar antenna.")