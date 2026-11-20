# Autonomous Robot: Vision & Match Control

Control software for an autonomous robot built for a robotics competition (Eurobot-style: start cord, 100 s match limit).

A **Raspberry Pi** is the brain of the robot. After the start cord (the *tirette*) is pulled, it runs the whole match sequence on its own:

1. **Reads the number (1 or 2)** printed on the wall.
2. **Identifies three shapes** (cube / sphere / pyramid) with an open-vocabulary YOLO model.
3. **Finds the odd one out** among the three shapes.
4. **Reads three wall letters** (E / W / S / N) with OCR.
5. **Drives the robot** through the matching movements, using an **STM32** for motion and an **ESP32** for the actuators.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │              Raspberry Pi                │
                    │                                          │
 Start cord ──GPIO17──►  main_etirette.py  (match sequence)     │
                    │      │        │         │                │
                    │      │        │         └─► subprocess: wall_number_detector.py   (camera 0)
                    │      │        └───────────► subprocess: letter_detector.py        (camera 0)
                    │      └────── TCP :5000 ───► detector_server.py  (YOLO-World)      (camera 1)
                    │                                          │
                    └───────┬──────────────────────┬───────────┘
                       UART │                 USB  │ serial
                            ▼                      ▼
                         STM32                   ESP32
                    (wheels / motion)     (actuators, LEDs)
```

| Component | Link | Role |
|---|---|---|
| STM32 | UART `/dev/ttyAMA0`, 9600 baud | Movement. Receives `cmd x y speed` and echoes it back as confirmation. |
| ESP32 | USB serial `/dev/ttyUSB0`, 9600 baud | Actuators. Receives a single digit `0`–`6` and answers `DONE`. |
| Start cord | GPIO 17 (physical pin 11) + GND (pin 9) | Arms the robot when inserted and starts the match when pulled. |

---

## Repository contents

| File | Purpose |
|---|---|
| `main_etirette.py` | **Main program.** Runs the full match sequence: start cord, movement, vision calls, actuator commands. |
| `wall_number_detector.py` | Detects whether the wall number is **1** or **2** using classical computer vision (OpenCV) with an optional OCR fallback. |
| `letter_detector.py` | Reads the three wall letters (**E / W / S / N**) using blob detection and Tesseract OCR. |
| `detector_server.py` | Persistent TCP server that keeps **YOLO-World** loaded and the camera open, and returns the shape seen in front of the camera. |
| `detector_client.py` | Small client used by the main program to talk to `detector_server.py`. |

---

## How the match unfolds

### 1. Startup (before the match)
- The STM32 and ESP32 serial ports are opened **early**, because the ESP32 resets when its port opens (~2 s).
- `wall_number_detector.py` is **pre-launched in the background** in `--wait-for-go` mode: Python, imports and the camera are already warm when the robot needs them.
- The program blocks on the **tirette**. It must be inserted to arm the robot, then pulled to start.

### 2. Wall number (1 / 2)
- After the first moves, the main program sends `GO` to the pre-launched detector and parses `Detected number: X` from its output.
- The number is forwarded to the ESP32 (`1` or `2`), and a different STM32 movement sequence is executed depending on it.

### 3. Shape detection (cube / sphere / pyramid)
- `detector_server.py` is started while the robot is still moving, so the model is loaded by the time it is needed.
- The robot scans **three positions** by rotating between them (rotation steps `43` then `37`), with a settle delay before each detection so images are not blurry.
- The server returns `c`, `s`, `t` or `n` (nothing found). If nothing is found at a position, `T` (pyramid) is assumed.
- After the third shape, the **odd letter** and its **index** are determined, and the robot moves to the matching position. The ESP32 receives `4`, `5` or `6` depending on whether the odd shape is a cube, sphere or pyramid.

### 4. Wall letters (E / W / S / N)
- `letter_detector.py` is also pre-launched, then triggered with `GO` once the robot is in front of the wall.
- The robot waits for a full stop before reading, with up to two fallback attempts. A debug image (`letter_debug.jpg`) is saved to help diagnose misreads.
- It returns a 3-letter string such as `WES`.

### 5. Final moves
- A final move is chosen from a table indexed by **(wall number, odd shape)**.
- Then, for each of the three wall letters, the robot executes one move defined in `WALL_LETTER_MOVES`:

| Letter | Move `(x, y)` |
|---|---|
| N | `(2000, 0)` |
| W | `(1075, 400)` |
| E | `(1075, -400)` |
| S | `(0, 0)` |

---

## The vision pipeline

### Wall number: `wall_number_detector.py`
Since the wall carries a high-contrast digit and only two answers are possible, a lightweight **geometric classifier** is used instead of running OCR on every frame:
- Contours are filtered by size, then classified using **aspect ratio** and **solidity** (a "1" is tall and thin, a "2" is wider and more complex).
- Detections must be **stable over several frames** before being trusted.
- An optional **Tesseract fallback** only runs on borderline frames, which saves CPU on the Pi. Common "1" misreads (`7`, `l`, `I`, `|`) are mapped to `1`, which is safe because the only possible answers are 1 and 2.
- It can optionally send the result over serial to the ESP32.

### Shapes: `detector_server.py`
- Uses **YOLO-World** (`yolov8s-worldv2.pt`), an open-vocabulary detector driven by text prompts such as `cube`, `box`, `sphere`, `ball`, `pyramid`, `triangle`, `cone`.
- Prompts are mapped to three classes: Cube, Sphere, Triangle (pyramid).
- Only a **region of interest** at the bottom-centre of the frame is analysed.
- Per-class confidence thresholds and **IoU-based duplicate merging** are applied. The detection closest to the centre of the ROI wins.
- **3 frames** are analysed per request and the result is decided by **majority vote**.
- A background thread continuously reads the camera so every request gets a **fresh frame**, not a stale buffered one.
- The server is persistent, so the model loads once and each detection is fast.

**Protocol (TCP, port 5000, one line per message):**

| Command | Reply |
|---|---|
| `DETECT` | `c`, `s`, `t` or `n` |
| `PING` | `pong` |
| `QUIT` | `bye` (stops the server) |

### Wall letters: `letter_detector.py`
- Finds the three letter-shaped blobs and reads each with **Tesseract**, restricted to the letters `E, W, S, N`.
- A **confusion map** folds common OCR mistakes into the right letter (for example `F`/`3` → `E`, `M` → `W`, `5`/`8` → `S`, `H`/`Z` → `N`).
- **Per-position temporal memory** and a stability check keep the result from flickering.
- Checks that the three letters are distinct (can be disabled).
- Supports a configurable ROI and a right-to-left mode if the string comes out reversed.

---

## Robustness features

- **Pre-launching** of detectors and early opening of serial ports, so slow start-up work overlaps with robot movement.
- **STM32 command confirmation**: every command is checked against its echoed reply, with a retry function when the reply is corrupted.
- **Graceful fallbacks**: one-shot detection if the pre-launched process fails, a default shape if nothing is detected, and the program keeps running if the ESP32 is unavailable.
- **Cleanup**: background processes are killed at exit, and the YOLO server is shut down in a background thread so the robot never waits on it.
- **Bench-test options**: numbers and letters can be typed in, and the start cord can be skipped.

---

## Hardware and software requirements

**Hardware**
- Raspberry Pi with two USB cameras (one for the number and letters, one for the shapes)
- STM32 motion controller
- ESP32 actuator controller
- Start cord wired to GPIO 17 (via a 330 Ω resistor) and GND

**Software** (Python 3)
```bash
pip install opencv-python numpy pyserial pytesseract ultralytics gpiozero
sudo apt install tesseract-ocr
```
The YOLO-World weights (`yolov8s-worldv2.pt`) are not stored in the repo (`*.pt` is git-ignored). Ultralytics downloads them on first use, or you can place the file next to `detector_server.py`.

---

## Usage

```bash
# Full run: detects the number and the shapes
python3 main_etirette.py

# Number typed in (no wall number detection)
python3 main_etirette.py 1

# Letters typed in (no shape detection)
python3 main_etirette.py CTC

# Both typed in: no vision at all
python3 main_etirette.py 1 CTC

# Bench test without the start cord
python3 main_etirette.py --no-tirette
```
The number (`1`/`2`) and the shape letters (three among `C`, `T`, `S`) are recognised by their content, so the order of arguments does not matter.

**Run the vision modules on their own**

```bash
# Shape server, then a test client
python3 detector_server.py 1 5000      # camera 1, port 5000
python3 detector_client.py

# Wall number
python3 wall_number_detector.py --source 0 --no-display --serial-port ""

# Wall letters (saves an annotated debug image)
python3 letter_detector.py --source 0 --no-display --save-debug
```

---

## Configuration

Most settings are constants at the top of `main_etirette.py`:

- Serial ports and baud rates (`ESP32_PORT`, `STM32_PORT`)
- Camera indices (`NUMBER_CAMERA`, `DETECTOR_CAMERA`, `LETTER_CAMERA`)
- Timeouts and settle delays (`ROTATE_SETTLE`, `FIRST_POSITION_SETTLE`, `LETTER_SETTLE_DELAY`)
- Default shape when nothing is detected (`DEFAULT_SHAPE`)
- Letter ROI (`LETTER_ROI`) and wall letter moves (`WALL_LETTER_MOVES`)
- Final move table (`FINAL_MOVE_BY_NUMBER_AND_SHAPE`)

Shape detection thresholds (ROI, confidence, number of frames) are at the top of `detector_server.py`.

---

## Notes

- `main_etirette.py` imports a `letter_handler` module (`LetterHandler`) that is **not included in this repository**. It computes the odd letter and its index and communicates with the STM32 (and LCD). Add it to the repo so the project runs from a fresh clone.
- Paths such as `PROJECT_DIR = "/home/omar/main"` and the virtualenv location are specific to the robot's Raspberry Pi and should be adapted on other machines.
