import atexit
import os
import serial
import threading
import time
import subprocess
import re
import argparse
import sys


# ============================================================
# COMMAND-LINE ARGUMENTS (all optional)
#
#   python3 main.py             -> number detection + shape detection
#   python3 main.py 1 CTC       -> number 1 and letters CTC typed:
#                                         no detection at all
#   python3 main.py 1           -> number typed, shape detection runs
#   python3 main.py CTC         -> letters typed, number detection runs
#   python3 main.py --no-tirette ...  -> skip the tirette (bench tests)
#
# The number (1 or 2) and the letters (3 letters among C, T, S)
# are recognised by their content, so the order does not matter.
#
# At the very end, the wall letters (E / W / S / N) are read with
# letter_detector.py (see WALL LETTER DETECTION below).
# ============================================================

NUM_SHAPES = 3

parser = argparse.ArgumentParser(description="Robot main sequence")
parser.add_argument(
    "values",
    nargs="*",
    help="Optional: the number (1 or 2) and/or the letter string (e.g. CTC). "
         "Anything not given is detected normally.",
)
parser.add_argument(
    "--no-tirette",
    action="store_true",
    help="Skip the tirette (start cord) and start immediately (bench tests).",
)
args = parser.parse_args()

MANUAL_NUMBER = None
MANUAL_LETTERS = None

for value in args.values:
    if value in ("1", "2"):
        if MANUAL_NUMBER is not None:
            parser.error("le numéro est donné deux fois")
        MANUAL_NUMBER = int(value)
    elif len(value) == NUM_SHAPES and all(c in "CTS" for c in value.upper()):
        if MANUAL_LETTERS is not None:
            parser.error("la chaîne de lettres est donnée deux fois")
        MANUAL_LETTERS = value.upper()
    else:
        parser.error(
            f"argument invalide : {value!r} (attendu : un numéro 1 ou 2, "
            f"et/ou {NUM_SHAPES} lettres parmi C, T, S)"
        )


# ============================================================
# ESP32 CONFIGURATION
# ============================================================

ESP32_PORT = "/dev/ttyUSB0"
ESP32_BAUDRATE = 9600


# ============================================================
# STM32 CONFIGURATION
# ============================================================

STM32_PORT = "/dev/ttyAMA0"
STM32_BAUD = 9600

# Pause between two commands sent to the STM32 (seconds)
STM_GAP = 0.1


# ============================================================
# DETECTION SERVER CONFIGURATION
# ============================================================

PROJECT_DIR = "/home/omar/main"

# Python used for the number / letter detectors (needs cv2, numpy...).
# If it is missing, fall back to the Python that is running this script.
VENV_PYTHON = "/home/omar/detector-env/bin/python3"
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

# The shape server (YOLO-World / ultralytics). Same environment by default;
# it must contain ultralytics:
#   /home/omar/detector-env/bin/pip install ultralytics
YOLO_PYTHON = "/home/omar/detector-env/bin/python3"
if not os.path.exists(YOLO_PYTHON):
    YOLO_PYTHON = sys.executable

DETECTOR_CAMERA = 1          # camera index used by the shape detector
DETECTOR_HOST = "127.0.0.1"  # server runs on the same Raspberry Pi
DETECTOR_PORT = 5000
DETECTOR_STARTUP_TIMEOUT = 90   # seconds to wait for the model to load
DETECTOR_REPLY_TIMEOUT = 15     # seconds to wait for one DETECT answer

# Settle times before asking the camera for a shape (seconds).
# Detection used to start while the robot was still moving / the image was
# still blurry, which caused wrong shapes. Raise these if mistakes remain.
FIRST_POSITION_SETTLE = 1.5   # after the final positioning move, before position 0
ROTATE_SETTLE = 1.0           # after each rotation to position 1 and 2

# Shape that is assumed when the detector finds nothing at a position
# (there is no second attempt any more).
DEFAULT_SHAPE = "T"

sys.path.insert(0, PROJECT_DIR)


# ============================================================
# WALL NUMBER DETECTION (1 / 2) CONFIGURATION
# ============================================================

NUMBER_CAMERA = 0        # camera index for wall_number_detector.py
NUMBER_TIMEOUT = 15      # seconds allowed after "GO" for the detection


# ============================================================
# WALL LETTER DETECTION (E / W / S / N) CONFIGURATION
# ============================================================

LETTER_CAMERA = 0              # camera index for letter_detector.py
LETTER_ROI = "0.2,0.4,0.6,0.4" # "X,Y,W,H" (pixels) or fractions of the frame
                               # (same as: --roi 0.2,0.4,0.6,0.4). None = whole frame
LETTER_RIGHT_TO_LEFT = False   # True if the string comes out reversed
LETTER_TIMEOUT = 20            # seconds allowed for one attempt
LETTER_ATTEMPTS = 2            # tries before giving up
LETTER_SETTLE_DELAY = 1.0      # seconds to let the robot stop moving before looking

# Move for each wall letter: (x, y) sent with cmd=4, speed=1.
# A letter set to None is skipped (with a message).
WALL_LETTER_MOVES = {
    "N": (2000, 0),
    "W": (1075, 400),
    "E": (1075, -400),
    "S": (0, 0),
}


# ============================================================
# TIRETTE (start cord) CONFIGURATION
# ============================================================

TIRETTE_GPIO = 17          # BCM number (physical pin 11); GND on physical pin 9
TIRETTE_BOUNCE = 0.05      # seconds of debounce


# ============================================================
# ESP32 CLIENT
# ============================================================

class ESP32Client:

    def __init__(self):

        # Open ESP32
        self.ser = serial.Serial(
            ESP32_PORT,
            ESP32_BAUDRATE,
            timeout=1
        )

        # ESP32 resets when the port opens
        time.sleep(2)

        # Remove boot messages / ESP32 READY
        self.ser.reset_input_buffer()

        print("ESP32 connected")
    def send_command(self, value):

        value = str(value)

        # Valid commands:
        # 0 = LEDs OFF
        # 1 = LED1 ON
        # 2 = LED2 ON
        if value not in ("0", "1", "2", "3","4","5","6"):
            print("Invalid ESP32 command")
            return False

        print(f"[RPi -> ESP32] Sending: {value}")

        self.ser.write(f"{value}\n".encode())

        # Make sure the data is transmitted
        self.ser.flush()

        start = time.time()

        while time.time() - start < 10:

            response = self.ser.readline().decode(
                errors="ignore"
            ).strip()

            if response:

                print("[ESP32 -> RPi]", response)

                if response == "DONE":

                    print("ESP32 DONE received")

                    return True

        print("Timeout: no DONE received")

        return False
    def close(self):

        if self.ser.is_open:
            self.ser.close()

        print("ESP32 connection closed")


# ============================================================
# STM32 SERIAL CONNECTION
# ============================================================

def open_stm32_serial():

    s = serial.Serial()

    s.port = STM32_PORT
    s.baudrate = STM32_BAUD
    s.timeout = 1.0

    s.dtr = False
    s.rts = False

    s.open()

    return s


# ============================================================
# SEND COMMAND TO STM32
# (returns True if the STM32 echoed the expected command,
#  False otherwise - existing calls simply ignore the value)
# ============================================================

def send_stm32_command(
    ser,
    cmd,
    x,
    y,
    speed,
    wait_seconds=10
):

    # Remove stale STM32 messages
    ser.reset_input_buffer()

    command = f"{cmd} {x} {y} {speed}\r\n"

    ser.write(command.encode("utf-8"))
    ser.flush()

    print(f"[RPi -> STM32] {command.strip()}")

    start = time.time()

    while time.time() - start < wait_seconds:

        if ser.in_waiting:

            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if line:

                print("[STM32 -> RPi]", line)

                if line.startswith("cmd="):

                    expected = (
                        f"cmd={cmd} "
                        f"x={x} "
                        f"y={y} "
                        f"speed={speed}"
                    )

                    if line != expected:

                        print(
                            f"ATTENTION: réponse inattendue "
                            f"(attendu: {expected})"
                        )

                        return False

                    return True

    print("STM32: Aucune réponse reçue.")

    return False


def send_stm32_command_retry(
    ser,
    cmd,
    x,
    y,
    speed,
    retries=3,
    wait_seconds=10
):
    """Same as send_stm32_command but re-sends the command if the STM32
    answered something different (e.g. a command corrupted by stray bytes
    on the shared port)."""

    for attempt in range(1, retries + 1):

        if send_stm32_command(ser, cmd, x, y, speed, wait_seconds):
            return True

        print(f"Réponse incorrecte, nouvel essai {attempt}/{retries}...")

        time.sleep(0.5)   # was 1 s: enough for the STM32 to recover

    return False


# ============================================================
# WALL NUMBER DETECTOR (pre-launched, then triggered with "GO")
#
# Python start-up + imports + opening/warming the camera take ~2 s.
# So wall_number_detector.py is started at the very beginning (while the
# robot is still doing its first moves) in --wait-for-go mode: it opens the
# camera, keeps it streaming, and waits. When the robot is in position we
# send "GO" and it only has to detect.
# ============================================================

number_proc = None


def _kill_number_detector():
    global number_proc
    if number_proc is not None and number_proc.poll() is None:
        number_proc.kill()
    number_proc = None


atexit.register(_kill_number_detector)


def start_number_detector():
    """Start wall_number_detector.py in the background (non-blocking)."""
    global number_proc

    if number_proc is not None:
        return

    print("Pré-lancement du détecteur de numéro (caméra prête à l'avance)...")

    try:
        number_proc = subprocess.Popen(
            [
                VENV_PYTHON,
                "wall_number_detector.py",
                "--source", str(NUMBER_CAMERA),
                "--no-display",
                "--serial-port", "",
                "--wait-for-go",
            ],
            cwd=PROJECT_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as e:
        print(f"Impossible de pré-lancer le détecteur de numéro : {e}")
        number_proc = None


def run_number_detector_oneshot():
    """Old behaviour: start the detector and wait for it (slower)."""
    try:
        return subprocess.run(
            [
                VENV_PYTHON,
                "wall_number_detector.py",
                "--source", str(NUMBER_CAMERA),
                "--no-display",
                "--serial-port", "",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=NUMBER_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        print(f"La détection a dépassé le délai imparti "
              f"({NUMBER_TIMEOUT}s), aucun résultat.")
        return None
    except OSError as e:
        print(f"Impossible de lancer le détecteur de numéro : {e}")
        return None


def get_number_detector_result():
    """Trigger the pre-launched detector and return its result
    (an object with .stdout / .stderr), or None on timeout."""
    global number_proc

    t = time.time()

    if number_proc is None:
        # Pre-launch did not happen / failed: use the old slower path.
        result = run_number_detector_oneshot()
    else:
        proc = number_proc
        number_proc = None
        try:
            out, err = proc.communicate(input="GO\n", timeout=NUMBER_TIMEOUT)
            result = subprocess.CompletedProcess(
                proc.args, proc.returncode, out, err
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            print(f"La détection a dépassé le délai imparti "
                  f"({NUMBER_TIMEOUT}s), aucun résultat.")
            result = None

        # The pre-launched process died without a result (e.g. camera could
        # not be opened): try once the normal way.
        if (result is not None
                and result.returncode != 0
                and "Detected number" not in (result.stdout or "")):
            print("Le détecteur pré-lancé a échoué, nouvel essai classique...")
            result = run_number_detector_oneshot()

    print(f"DETECTION TIME (après GO) = {time.time() - t:.2f} seconds")

    return result


# ============================================================
# SHAPE DETECTION SERVER (launched in the background)
#
# detector_server.py loads YOLO-World + opens the camera ONCE.
# We start it while the robot is still moving, so the model is
# already loaded when we need the first detection.
# ============================================================

detector_proc = None      # the server subprocess
detector_log = None       # its log file
detector_client = None    # the TCP client (created on first use)


def start_detector_server():
    """Start detector_server.py in the background (non-blocking).
    Safe to call several times: it only starts once."""
    global detector_proc, detector_log

    if detector_proc is not None:
        return

    print("Lancement du serveur de détection de forme en arrière-plan...")

    # Log to a file (a PIPE that nobody reads would eventually block the server)
    detector_log = open(f"{PROJECT_DIR}/detector_server.log", "w")

    detector_proc = subprocess.Popen(
        [
            YOLO_PYTHON,
            "detector_server.py",
            str(DETECTOR_CAMERA),
            str(DETECTOR_PORT),
        ],
        cwd=PROJECT_DIR,
        stdout=detector_log,
        stderr=subprocess.STDOUT,
    )


def connect_detector():
    """Connect to the detection server, retrying until the model is loaded.
    Returns the Detector client, or None if the server never came up."""
    global detector_client

    if detector_client is not None:
        return detector_client

    from detector_client import Detector

    start_detector_server()   # no-op if already started

    start = time.time()

    while time.time() - start < DETECTOR_STARTUP_TIMEOUT:

        if detector_proc.poll() is not None:
            print("Le serveur de détection s'est arrêté "
                  "(voir detector_server.log).")
            return None

        try:
            detector_client = Detector(
                DETECTOR_HOST,
                DETECTOR_PORT,
                timeout=DETECTOR_REPLY_TIMEOUT,
            )
            print("Connecté au serveur de détection.")
            return detector_client
        except OSError:
            # Server not listening yet (model still loading)
            time.sleep(0.5)

    print(f"Serveur de détection indisponible après "
          f"{DETECTOR_STARTUP_TIMEOUT}s.")
    return None


def stop_detector_server():
    """Ask the server to quit, then make sure the process is gone."""
    global detector_client, detector_proc, detector_log

    if detector_client is not None:
        try:
            detector_client._ask("QUIT")
        except Exception:
            pass
        try:
            detector_client.close()
        except Exception:
            pass
        detector_client = None

    if detector_proc is not None:
        try:
            detector_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            detector_proc.terminate()
            try:
                detector_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                detector_proc.kill()
        detector_proc = None

    if detector_log is not None:
        detector_log.close()
        detector_log = None

    print("Serveur de détection arrêté.")


# ============================================================
# WALL LETTER DETECTION (E / W / S / N)
#
# letter_detector.py is PRE-LAUNCHED (same idea as the number detector):
# start_letter_detector() is called when the odd letter is sent to the
# ESP32. The script starts, opens and warms up the camera in
# --wait-for-go mode, and waits. When the robot is in front of the wall,
# run_letter_detector() sends "GO" and it only has to detect.
#
# If the pre-launched process fails (or does not support --wait-for-go),
# run_letter_detector() falls back to the normal one-shot run.
# ============================================================

letter_proc = None


def _kill_letter_detector():
    global letter_proc
    if letter_proc is not None and letter_proc.poll() is None:
        letter_proc.kill()
    letter_proc = None


atexit.register(_kill_letter_detector)


def _letter_cmd(wait_for_go=False):
    """Command line of letter_detector.py."""
    cmd = [
        VENV_PYTHON,
        "letter_detector.py",
        "--source", str(LETTER_CAMERA),
        "--no-display",
        "--quiet",
        "--save-debug",   # saves letter_debug.jpg in PROJECT_DIR
    ]
    if LETTER_ROI:
        cmd += ["--roi", LETTER_ROI]
    if LETTER_RIGHT_TO_LEFT:
        cmd.append("--right-to-left")
    if wait_for_go:
        cmd.append("--wait-for-go")
    return cmd


def start_letter_detector():
    """Start letter_detector.py in the background, waiting for "GO"
    (non-blocking). Safe to call several times: it only starts once."""
    global letter_proc

    if letter_proc is not None:
        return

    print("Pré-lancement du détecteur de lettres (caméra prête à l'avance)...")

    try:
        letter_proc = subprocess.Popen(
            _letter_cmd(wait_for_go=True),
            cwd=PROJECT_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as e:
        print(f"Impossible de pré-lancer le détecteur de lettres : {e}")
        letter_proc = None


def _parse_letters(stdout):
    match = re.search(r"Detected letters:\s*([EWSN]{3})", stdout or "")
    return match.group(1) if match else None


def run_letter_detector():
    """Return the 3-letter string (e.g. 'WES'), or None if nothing was
    detected. Uses the pre-launched detector first, then the one-shot run."""
    global letter_proc

    # --- Attempt with the pre-launched detector (send "GO") ---
    if letter_proc is not None:

        proc = letter_proc
        letter_proc = None

        print("Détection des lettres du mur (détecteur pré-lancé, GO)...")

        t = time.time()

        try:
            out, err = proc.communicate(input="GO\n", timeout=LETTER_TIMEOUT)

            print(f"LETTER DETECTION TIME (après GO) = {time.time() - t:.2f} seconds")

            if err:
                print(err)

            letters_found = _parse_letters(out)

            if letters_found:
                return letters_found

            print("Le détecteur pré-lancé n'a rien trouvé, essai classique...")

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            print(f"Délai dépassé ({LETTER_TIMEOUT}s), essai classique...")

    # --- Normal one-shot attempts ---
    cmd = _letter_cmd()

    for attempt in range(1, LETTER_ATTEMPTS + 1):

        print(f"Détection des lettres du mur (essai {attempt}/{LETTER_ATTEMPTS})...")

        t = time.time()

        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_DIR,
                timeout=LETTER_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            print(f"Délai dépassé ({LETTER_TIMEOUT}s), aucun résultat.")
            continue
        except OSError as e:
            print(f"Impossible de lancer le détecteur de lettres : {e}")
            return None

        print(f"LETTER DETECTION TIME = {time.time() - t:.2f} seconds")

        if r.stderr:
            print(r.stderr)

        letters_found = _parse_letters(r.stdout)

        if letters_found:
            return letters_found

        print("Aucune chaîne de lettres détectée.")

    return None


# ============================================================
# TIRETTE (start cord)
# Inserted = pin LOW (button "pressed"), pulled = pin HIGH.
# Wiring: GPIO17 (physical pin 11) -> [330 ohm] -> contact A
#         GND    (physical pin 9)  -> contact B
# ============================================================

def wait_for_tirette():
    # Imported here so the script still runs without gpiozero
    # when --no-tirette is used.
    from gpiozero import Button

    tirette = Button(
        TIRETTE_GPIO,
        pull_up=True,
        bounce_time=TIRETTE_BOUNCE,
    )

    try:
        # The tirette must be inserted before the match.
        if not tirette.is_pressed:
            print("TIRETTE absente ! Insère la tirette pour armer le robot...")
            tirette.wait_for_press()

        print("Robot ARMÉ. Tire la tirette pour démarrer.")

        tirette.wait_for_release()      # blocks until the tirette is pulled

        print("TIRETTE retirée -> DÉPART !")
    finally:
        tirette.close()


# ============================================================
# MAIN PROGRAM
# ============================================================

# ============================================================
# OPEN STM32
# ============================================================

try:

    ser = open_stm32_serial()

    print("Connexion STM32 ouverte, attente...")

    time.sleep(2)

    ser.reset_input_buffer()

except serial.SerialException as e:

    print("ERREUR: impossible de connecter le STM32.")
    print(e)

    exit(1)


# ============================================================
# OPEN ESP32 EARLY
#
# The ESP32 resets when its port is opened (~2 s), so it is opened NOW,
# while the robot is still waiting for the tirette. It is then ready to
# receive commands the moment the match starts.
# If it cannot be opened here, a second attempt is made after the number
# detection (see ESP32 CONNECTION below).
# ============================================================

esp32 = None

try:

    print("Ouverture de l'ESP32 (avant la tirette)...")

    esp32 = ESP32Client()

except serial.SerialException as e:

    print("ATTENTION: ESP32 non disponible pour l'instant.")
    print(e)

    esp32 = None


# Start the number detector NOW: its start-up (imports, camera) then happens
# while the robot waits for the tirette and performs the first moves below.
if MANUAL_NUMBER is None:
    start_number_detector()


# ============================================================
# WAIT FOR THE TIRETTE
# (the number detector camera keeps warming up while we wait)
# ============================================================

if args.no_tirette:
    print("Tirette ignorée (--no-tirette) : départ immédiat.")
else:
    wait_for_tirette()

MATCH_START = time.time()   # reference for the 100 s Eurobot limit


# ============================================================
# INITIAL MOTION SEQUENCE
# ============================================================

send_stm32_command(ser, cmd=5, x=240, y=-270, speed=600)

time.sleep(STM_GAP)


send_stm32_command(
    ser,
    cmd=4,
    x=-270,
    y=-270,
    speed=1
)

time.sleep(STM_GAP)


# ============================================================
# LAUNCH VISION DETECTOR
# ============================================================

if MANUAL_NUMBER is not None:

    # Number typed in the terminal -> no number detection
    print(
        f"Numéro fourni au terminal : {MANUAL_NUMBER} -> "
        "détection du numéro ignorée."
    )

    # The detection normally takes a few seconds, during which the robot
    # finishes its last move. Keep that time so the ESP32 is not opened
    # while the robot is still moving.
    TYPED_NUMBER_DELAY = 5   # seconds - set it close to your usual "DETECTION TIME"

    print(f"Pause de {TYPED_NUMBER_DELAY}s (remplace la durée de la détection)...")

    time.sleep(TYPED_NUMBER_DELAY)

    result = None

else:

    print(
        "Lancement de la détection du numéro sur le mur..."
    )

    print("START DETECTION")

    result = get_number_detector_result()


# ============================================================
# READ DETECTED NUMBER
# ============================================================

detected_number = MANUAL_NUMBER   # None unless typed in the terminal


if result is not None:

    print("---- Sortie du détecteur ----")

    print(result.stdout)

    if result.stderr:

        print("---- Erreurs / logs ----")

        print(result.stderr)

    match = re.search(
        r"Detected number:\s*(\d+)",
        result.stdout
    )

    if match:

        detected_number = int(match.group(1))

        print(
            f"Numéro détecté : {detected_number}"
        )

    else:

        print(
            "Aucun numéro détecté dans la sortie."
        )


# ============================================================
# ESP32: SEND THE DETECTED NUMBER
#
# The ESP32 was already opened before the tirette, so the number is
# sent immediately. Only if that early opening failed do we retry here.
# ============================================================

if detected_number in (1, 2):

    print(
        f"Numéro {detected_number} détecté"
    )

    try:

        if esp32 is None:

            print("Nouvelle tentative d'ouverture de l'ESP32...")

            esp32 = ESP32Client()

        # detected 1 -> ESP32 receives "1"
        # detected 2 -> ESP32 receives "2"
        success = esp32.send_command(
            detected_number
        )

        if success:

            print(
                "ESP32: commande terminée avec succès."
            )

        else:

            print(
                "ESP32: pas de DONE reçu."
            )

    except serial.SerialException as e:

        print(
            "ERREUR: impossible de communiquer "
            "avec l'ESP32."
        )

        print(e)


# ============================================================
# STM32 SEQUENCE FOR NUMBER 2
# ============================================================

if detected_number == 2:

    print(
        "Numéro 2 détecté -> "
        "envoi de la séquence STM32 associée."
    )

    ser.reset_input_buffer()


    # ESP32:
    # Number 2 -> LED2 ON


    send_stm32_command(
        ser,
        cmd=2,
        x=305,
        y=0,
        speed=500
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=3,
        x=-90,
        y=0,
        speed=0
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=2,
        x=430,
        y=0,
        speed=500
    )

    # Launch the detection server now (model loads while the robot keeps going).
    # Not needed if the letters were typed in the terminal.
    if MANUAL_LETTERS is None:
        start_detector_server()


    # Keep LED ON for 3 seconds

    time.sleep(3)


    # Turn LEDs OFF

    if esp32 is not None:
        esp32.send_command("0")


    send_stm32_command(
        ser,
        cmd=5,
        x=1095,
        y=-50,
        speed=600
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=5,
        x=1071,
        y=-658,
        speed=600
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=2,
        x=105,
        y=0,
        speed=250
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=5,
        x=1365,
        y=-650,
        speed=500
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=4,
        x=1710,
        y=-670,
        speed=1
    )

    time.sleep(STM_GAP)
    if esp32 is not None:
        esp32.send_command("3")


    send_stm32_command(
        ser,
        cmd=2,
        x=190,
        y=0,
        speed=250
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=5,
        x=1450,
        y=-380,
        speed=600
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=5,
        x=2000,
        y=-420,
        speed=600
    )

    time.sleep(STM_GAP)


# ============================================================
# STM32 SEQUENCE FOR NUMBER 1
# ============================================================

elif detected_number == 1:

    print(
        "Numéro 1 détecté -> "
        "envoi de la séquence STM32 associée."
    )

    ser.reset_input_buffer()


    # ESP32:
    # Number 1 -> LED1 ON


    send_stm32_command(
        ser,
        cmd=1,
        x=55,
        y=0,
        speed=75
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=3,
        x=-90,
        y=0,
        speed=0
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=2,
        x=430,
        y=0,
        speed=500
    )

    # Launch the detection server now (model loads while the robot keeps going).
    # Not needed if the letters were typed in the terminal.
    if MANUAL_LETTERS is None:
        start_detector_server()


    # Keep LED ON for 2 seconds

    time.sleep(3)


    # Turn LEDs OFF

    if esp32 is not None:
        esp32.send_command("0")


    send_stm32_command(
        ser,
        cmd=5,
        x=1095,
        y=-50,
        speed=600
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=5,
        x=1071,
        y=-658,
        speed=600
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=2,
        x=105,
        y=0,
        speed=250
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=5,
        x=1365,
        y=-650,
        speed=500
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=4,
        x=1710,
        y=-670,
        speed=1
    )

    time.sleep(STM_GAP)
    if esp32 is not None:
        esp32.send_command("3")
    time.sleep(STM_GAP)

    send_stm32_command(
        ser,
        cmd=2,
        x=190,
        y=0,
        speed=250
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=5,
        x=1470,
        y=-150,
        speed=600
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=5,
        x=1760,
        y=-20,
        speed=600
    )

    time.sleep(STM_GAP)


    send_stm32_command(
        ser,
        cmd=5,
        x=1700,
        y=-710,
        speed=600
    )

    time.sleep(STM_GAP)


# ============================================================
# NO VALID NUMBER
# ============================================================

else:

    print(
        f"Numéro détecté = {detected_number}, "
        "pas d'action définie."
    )


# ============================================================
# FINAL STM32 POSITIONING
# ============================================================

ser.reset_input_buffer()


send_stm32_command(
    ser,
    cmd=2,
    x=190,
    y=0,
    speed=250
)

time.sleep(STM_GAP)


send_stm32_command(
    ser,
    cmd=5,
    x=1200,
    y=210,
    speed=600
)

time.sleep(STM_GAP)


send_stm32_command(
    ser,
    cmd=4,
    x=1600,
    y=220,
    speed=1
)

time.sleep(STM_GAP)


# ======================================================================
# ---- New, separate task: shape detection (persistent server) ----
# ======================================================================
print("Séquence numéro terminée. Lancement de la détection de forme (nouvelle tâche)...")

from letter_handler import LetterHandler

handler = LetterHandler(
    stm_port=STM32_PORT,
    baudrate=STM32_BAUD,
)

# Make sure the server is running and reachable (it was normally started
# earlier, during the number sequence; this also covers the case where no
# number was detected and the server was never launched).
if MANUAL_LETTERS is None:
    connect_detector()

# The server answers with these letters; the rest of the program uses C/T/S.
SERVER_TO_LETTER = {"c": "C", "s": "S", "t": "T"}   # "n" = nothing found

# Rotation (cmd=3, x value) to go from position i to position i+1.
# Going back from i+1 to i uses the same value with a negative sign.
STEP_TO_NEXT = [43, 37]

position = 0              # the robot starts in front of shape index 0


def run_shape_detector():
    """Ask the persistent detection server for the shape currently in front
    of the camera. Returns 'C'/'T'/'S' or None."""
    global detector_client

    det = connect_detector()
    if det is None:
        return None


    t0 = time.time()

    try:
        answer = det.detect()
    except OSError as e:
        # Timeout or broken connection: drop it, a fresh one is made next time
        # (also avoids reading a stale answer on the next call).
        print(f"Erreur de communication avec le serveur de détection : {e}")
        try:
            det.close()
        except Exception:
            pass
        detector_client = None
        return None

    print(f"[Serveur détection] réponse : {answer!r} ({time.time() - t0:.2f}s)")

    letter = SERVER_TO_LETTER.get(answer)
    if letter:
        print(f"Forme détectée : {letter}")
        return letter

    print("Aucune forme détectée.")
    return None


def detect_at(index):
    """One detection attempt at physical position `index`.
    If letters were typed in the terminal, no detection: return the typed letter."""
    if MANUAL_LETTERS is not None:
        print(f"Lettre fournie au terminal pour la position {index} : {MANUAL_LETTERS[index]}")
        return MANUAL_LETTERS[index]
    return run_shape_detector()


def go_to(target):
    """Rotate the robot to physical position `target`. Forward uses the
    original rotation values, going back uses the same values negated."""
    global position
    while position < target:
        send_stm32_command(ser, cmd=3, x=STEP_TO_NEXT[position], y=0, speed=0)
        time.sleep(ROTATE_SETTLE)
        position += 1
    while position > target:
        send_stm32_command(ser, cmd=3, x=-STEP_TO_NEXT[position - 1], y=0, speed=0)
        time.sleep(ROTATE_SETTLE)
        position -= 1


# NOTE: `ser` is still open from the final positioning move above.
# letters[i] = the letter seen at physical position i.
letters = [None] * NUM_SHAPES
odd_letter, odd_index = None, None

# --- One pass: scan position 0 -> 1 -> 2, ONE attempt each ---
# Nothing detected at a position -> it is considered to be DEFAULT_SHAPE ("T").
# The handler is fed right after each detection (typed or detected), so the
# odd letter + index are known, and sent to LCD/STM32, the instant the 3rd
# shape has been seen.
for i in range(NUM_SHAPES):
    go_to(i)

    # Position 0 has no rotation, so nothing has given the robot time to stop
    # and the camera time to get a sharp image: wait here.
    if i == 0 and MANUAL_LETTERS is None:
        print(f"Pause de {FIRST_POSITION_SETTLE}s avant la première détection...")
        time.sleep(FIRST_POSITION_SETTLE)

    letters[i] = detect_at(i)

    if letters[i] is None:
        letters[i] = DEFAULT_SHAPE
        print(f"Position {i} : aucune forme détectée -> considérée comme {DEFAULT_SHAPE}")
    else:
        print(f"Position {i} : {letters[i]}")

    # Returns (odd_letter, odd_index) on the 3rd letter, (None, None) before.
    odd_letter, odd_index = handler.add_letter(letters[i])

    # No need to wait after the last position.
    if i < NUM_SHAPES - 1:
        time.sleep(0.5)

# Detection is over: free the camera and the CPU/RAM used by the model.
# Done in a background thread so the robot does not wait for the server to
# shut down (it can take up to 6 s). It is joined before the program ends.
stop_thread = None
if MANUAL_LETTERS is None:
    stop_thread = threading.Thread(target=stop_detector_server, daemon=True)
    stop_thread.start()

print(f"Chaîne de formes : {''.join(letters)}")
print(f"Résultat letter_handler -> lettre impaire : {odd_letter}, index : {odd_index}")

# Small pause so the STM32 finishes with the letter/index the handler just wrote
# on the shared port, otherwise they get glued to the next command.
# Lower it only while no "ATTENTION: réponse inattendue" line appears.
DELAY_BEFORE_ODD_MOVE = 0.3   # seconds (lower step by step: 0.2, 0.1)
if DELAY_BEFORE_ODD_MOVE > 0:
    time.sleep(DELAY_BEFORE_ODD_MOVE)

# Pre-launch the wall letter detector NOW: its start-up (imports, camera)
# overlaps with the ESP32 action and the moves below.
start_letter_detector()

if esp32 is not None :
    if odd_letter == 'C':
       esp32.send_command(4)
    elif odd_letter == 'S':
       esp32.send_command(5)
    elif odd_letter =='T':
        esp32.send_command(6)
    else :
        print("ons 3ella")
if odd_index == 0:
    send_stm32_command_retry(ser, cmd=5, x=1490, y=220, speed=500)
elif odd_index == 1:
    send_stm32_command_retry(ser, cmd=5, x=1500, y=430, speed=500)
else:
    send_stm32_command_retry(ser, cmd=1, x=250, y=0, speed=500)

send_stm32_command_retry(ser, cmd=5, x=950, y=530, speed=600)
time.sleep(STM_GAP)
send_stm32_command_retry(ser, cmd=4, x=0, y=580, speed=1)

# Let the robot come to a complete stop before looking at the wall letters.
time.sleep(LETTER_SETTLE_DELAY)


# ============================================================
# WALL LETTER DETECTION (E / W / S / N)
# ============================================================

wall_letters = run_letter_detector()

if wall_letters:
    print(f"Lettres du mur : {wall_letters}")
else:
    print("Lettres du mur : non détectées "
          "(voir letter_debug.jpg dans le dossier du projet).")

# `wall_letters` is a string like "WES" (or None).

# Final move depends on the wall number (1 / 2) AND the odd shape found earlier
#   S = sphere, C = cube, T = pyramid
#   each entry is (x, y, speed)
# Number 2 currently has the same values as number 1: edit them separately.
FINAL_MOVE_BY_NUMBER_AND_SHAPE = {
    1: {
        "S": (820, 690, 250),   # sphere
        "C": (450, 730, 600),   # cube
        "T": (90, 790, 700),    # pyramid
    },
    2: {
        "S": (790, 650, 250),   # sphere
        "C": (420, 690, 600),   # cube
        "T": (60, 695, 700),    # pyramid
    },
}

# If no valid number was detected, fall back on the table of number 1.
if detected_number in FINAL_MOVE_BY_NUMBER_AND_SHAPE:
    FINAL_MOVE_BY_SHAPE = FINAL_MOVE_BY_NUMBER_AND_SHAPE[detected_number]
else:
    print(f"Numéro {detected_number!r} : coordonnées finales du numéro 1 utilisées.")
    FINAL_MOVE_BY_SHAPE = FINAL_MOVE_BY_NUMBER_AND_SHAPE[1]

if odd_letter in FINAL_MOVE_BY_SHAPE:

    fx, fy, fspeed = FINAL_MOVE_BY_SHAPE[odd_letter]

    print(f"Numéro {detected_number} / forme impaire : {odd_letter} -> "
          f"déplacement final ({fx}, {fy}) vitesse {fspeed}")

    send_stm32_command(ser, cmd=5, x=fx, y=fy, speed=fspeed)
    time.sleep(STM_GAP)
    send_stm32_command(ser, cmd=4, x=900, y=0, speed=1)

else:

    print(f"Forme impaire inconnue ({odd_letter!r}) : "
          "déplacement final ignoré.")

send_stm32_command(ser, cmd=5, x=1050, y=0, speed=600)              # emchi ll centre ya ka7louch


# ============================================================
# WALL LETTERS -> ONE VARIABLE PER LETTER, THEN ONE MOVE PER LETTER
# (coordinates are in WALL_LETTER_MOVES at the top of the file)
# ============================================================

if wall_letters and len(wall_letters) == 3:

    first_letter = wall_letters[0]
    second_letter = wall_letters[1]
    third_letter = wall_letters[2]

    print(f"Lettres : {first_letter} {second_letter} {third_letter}")

    for step, letter in enumerate(
        (first_letter, second_letter, third_letter), start=1
    ):

        print(f"Lettre {step}/3 : {letter}")

        move = WALL_LETTER_MOVES.get(letter)

        if move is None:

            print(f"Aucune coordonnée définie pour la lettre {letter} "
                  "(voir WALL_LETTER_MOVES) : déplacement ignoré.")

        else:

            mx, my = move
            send_stm32_command(ser, cmd=4, x=mx, y=my, speed=1)
            time.sleep(STM_GAP)

else:

    print("Lettres du mur non valides : séquence N/W/E/S ignorée.")


# ============================================================
# CLOSE BACKGROUND DETECTOR SHUTDOWN
# ============================================================

# Make sure the detection server is really gone before exiting.
if stop_thread is not None:
    stop_thread.join(timeout=10)


# ============================================================
# CLOSE ESP32
# ============================================================

if esp32 is not None:

    esp32.close()


# ============================================================
# CLOSE STM32
# ============================================================

ser.close()

handler.close()

print("STM32 connection closed.")
print("Programme terminé.")
