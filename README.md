# IWR1843BOOST Radar Gesture Framework

Real-time gesture classification and movement tracking using the TI IWR1843BOOST mmWave radar sensor.

---

## Citation

If you use this work in your research, please cite it as:

```bibtex
@software{younas_2026,
  author  = {Younas, Muhammad},
  title   = {Real-Time Gesture Recognition using mmWave Radar},
  year    = {2026},
  url     = {https://github.com/younasm/real-time-gesture-recognition},
  orcid   = {https://orcid.org/0009-0009-7893-1191},
  license = {CC-BY-NC-4.0}
}
```

---

## Requirements

- Python 3.8+
- TI IWR1843BOOST radar (for live mode)

Install dependencies:

```
pip install -r requirements.txt
```

---

## Configuration

Before running in live mode, open `config/settings.py` and set the correct serial ports:

**macOS:**

```python
cli_port: str = "/dev/tty.usbmodemXXXXXX01"   # XDS110 CLI  port (lower suffix)
data_port: str = "/dev/tty.usbmodemXXXXXX04"  # XDS110 Data port (higher suffix)
```

**Windows:**

```python
cli_port: str = "COM3"   # lower COM number  (115,200 baud)
data_port: str = "COM4"  # higher COM number (921,600 baud)
```

See the **Verifying the Sensor** section below for how to find your exact port names.

---

## Verifying the Sensor (macOS / Linux)

After setting the correct ports in `config/settings.py`, run these two scripts
to confirm the sensor is connected and streaming data before launching the main pipeline.

### Step 1 — Full diagnostic (recommended first time)

Sends the full radar configuration and reads data for 5 seconds:

```bash
python diagnose.py
```

A successful run looks like this:

```
Opening CLI  port  /dev/tty.usbmodemR21010501 @ 115200 baud … OK
Opening Data port  /dev/tty.usbmodemR21010504 @ 921600 baud … OK
...
> sensorStart    Done ✓

[OK] Data is on /dev/tty.usbmodemR21010504!  ~20.4 frames/sec
     Run:  python main.py
```

If it fails, check that:

- The USB cable is connected
- The port names in `config/settings.py` match your system (use `python check_ports.py` to find them)
- The radar is powered on (green LED)

### Step 2 — Quick port check (after sensor is already running)

Run this immediately after `diagnose.py` while the sensor is still streaming:

```bash
python check_ports.py
```

Expected output:

```
[OK] Data is on /dev/tty.usbmodemR21010504!  199 frames  -> python main.py
```

### Finding your port names

**macOS / Linux** — run this in terminal to list connected radar ports:

```bash
python -c "import serial.tools.list_ports; [print(p.device, p.description) for p in serial.tools.list_ports.comports()]"
```

Look for entries containing `XDS110` — the lower-numbered port is CLI, the higher is Data.

**Windows** — open Device Manager → Ports (COM & LPT) → look for `XDS110 Class Application/User UART`.

---

## Running

### Live mode (hardware required)

1. Flash the IWR1443 with TI mmWave SDK OOB demo firmware
2. Connect the board via USB
3. Set the correct COM ports in `config/settings.py`
4. Run:

```
python main.py
```

### Demo / replay mode (no hardware)

Replay a previously recorded `.npy` file:

```
python main.py --demo path\to\recording.npy
```

### Headless mode (no visualization window)

```
python main.py --no_viz
```

---

## Command-line Flags

| Flag                        | Description                                     |
| --------------------------- | ----------------------------------------------- |
| `--cli_port COM5`           | Override the CLI serial port                    |
| `--data_port COM6`          | Override the data serial port                   |
| `--config path/to/file.cfg` | Use a custom radar config file                  |
| `--record`                  | Enable data recording to `data/recordings/`     |
| `--ml`                      | Use trained ML classifier instead of rule-based |
| `--no_viz`                  | Run headless (no matplotlib window)             |
| `--demo file.npy`           | Replay a recorded file without hardware         |

---

## Guided Data Collection (recommended)

Use the guided session script to collect training data in a structured way.
It handles countdowns, progress bars, rest periods, and logs which person
performed each sample.

```
python -m data.collect_session
```

It will prompt you for:

| Prompt                      | Example                  | Description                      |
| --------------------------- | ------------------------ | -------------------------------- |
| Person / subject ID         | `person1`                | Labels who performed the gesture |
| Gestures to collect         | `1,2,3` or Enter for all | Pick from the numbered list      |
| Repetitions per gesture     | `10`                     | How many samples per gesture     |
| Sample duration (seconds)   | `2.0`                    | How long each recording lasts    |
| Rest between reps (seconds) | `2.0`                    | Recovery time between samples    |
| Countdown before each rep   | `3`                      | Seconds of 3-2-1 before GO       |

**Collecting from multiple people:** run the script once per person and enter
a different subject ID each time (e.g. `person1`, `person2`). All files land
in `data/recordings/` and are indexed in `session_log.csv`.

**To override COM ports:**

```
python -m data.collect_session --cli_port COM5 --data_port COM6
```

---

## Model Training

### Option 1 — ML Model (Random Forest, fast, no GPU needed)

1. Collect data using the guided session script above
2. Train the model:

```bash
python -m classification.ml_trainer
```

3. Enable in `config/settings.py`:

```python
use_ml_classifier: bool = True
use_dl_classifier: bool = False
```

The trained model is saved to `classification/models/gesture_rf_realtime.joblib`.

---

### Option 2 — Deep Learning Model (CNN-BiLSTM, higher accuracy)

Requires PyTorch (`pip install torch`).

1. Collect data using the guided session script above
2. Train the model:

```bash
python -m classification.dl_trainer
```

With custom parameters (more augmentation / longer training):

```bash
python -m classification.dl_trainer --epochs 60 --augment 8
```

without augmentation

```bash
python -m classification.dl_trainer --epochs 60 --augment 0
```

| Flag           | Default | Description                           |
| -------------- | ------- | ------------------------------------- |
| `--epochs`     | `60`    | Number of training epochs             |
| `--augment`    | `6`     | Augmentation multiplier per recording |
| `--lr`         | `5e-4`  | Learning rate                         |
| `--batch_size` | `64`    | Batch size                            |

3. Enable in `config/settings.py`:

```python
use_ml_classifier: bool = False
use_dl_classifier: bool = True
```

The trained model is saved to `classification/models/gesture_dl.pt`.

> **Tip:** Collect data from multiple people at different distances for best accuracy.
> Run `python -m data.collect_session` once per person with a different subject ID.

---

## Supported Gestures

- `swipe_left` / `swipe_right`
- `swipe_up` / `swipe_down`
- `push` / `pull`
- `wave`
- `circle_cw` / `circle_ccw`
- `static`

---

## Project Structure

```
radar_gesture_framework/
├── main.py                  # Entry point
├── requirements.txt
├── config/
│   ├── settings.py          # All configuration (edit this)
│   └── iwr1443_gesture.cfg  # Radar hardware config
├── radar/                   # Serial interface and frame parser
├── processing/              # Point cloud, clustering, tracking
├── classification/          # Gesture engine and ML trainer
├── visualization/           # Matplotlib live display
├── data/                    # Data recorder
└── utils/                   # Logging helpers
```
