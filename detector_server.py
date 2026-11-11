"""Persistent shape-detection server.

Start this ONCE at the beginning of the match. It loads YOLO-World, warms it up,
opens the camera, then waits for commands. Detection is then fast because
nothing is loaded anymore.

Protocol (TCP, one line per command, one line per answer):
  DETECT  -> "c", "s", "t"  or "n" (nothing found)
  PING    -> "pong"
  QUIT    -> stops the server

  python detector_server.py              # camera 0, port 5000
  python detector_server.py 1 5000       # camera 1, port 5000
"""
import socket
import sys
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLOWorld

CAMERA = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
HOST = "0.0.0.0"   # use "127.0.0.1" if the client runs on the same machine

BASE_CONF = 0.01
CUBE_MIN_CONF = 0.1   # higher floor for Cube to suppress weak false positives
ROI_X, ROI_Y, ROI_W, ROI_H = 0.15, 0.50, 0.70, 0.40
N_FRAMES = 3       # frames analysed per DETECT command (majority vote)

PROMPT_TO_CLASS = {
    "cube": "Cube", "box": "Cube",
    "sphere": "Sphere", "ball": "Sphere",
    "pyramid": "Triangle", "triangle": "Triangle", "cone": "Triangle",
    "square": "Square",
    "circle": "Circle",
}
PROMPTS = list(PROMPT_TO_CLASS)
CONF_BY_CLASS = {"Cube": max(BASE_CONF, CUBE_MIN_CONF),
                 "Sphere": BASE_CONF, "Square": BASE_CONF, "Circle": BASE_CONF,
                 "Triangle": BASE_CONF / 2}
CLASS_TO_LETTER = {"Cube": "c", "Sphere": "s", "Triangle": "t", "Square": "c", "Circle": "s"}
MERGE_IOU = 0.5


# ---------------------------------------------------------------- model
print("Loading model...", flush=True)
model = YOLOWorld("yolov8s-worldv2.pt")
model.set_classes(PROMPTS)
model.predict(np.zeros((480, 640, 3), np.uint8), verbose=False)   # warm-up run
print("Model ready.", flush=True)


# ---------------------------------------------------------------- camera
class LatestFrame:
    """Reads the camera continuously so we always get a FRESH frame
    (otherwise OpenCV hands back old buffered frames)."""

    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open camera {index}")
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        while self.frame is None:          # wait for first frame
            time.sleep(0.01)

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f

    def read(self):
        with self.lock:
            return self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()


camera = LatestFrame(CAMERA)


# ---------------------------------------------------------------- detection
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detect_once(frame):
    """Same logic as test.py: returns a letter or None."""
    h, w = frame.shape[:2]
    x1r, y1r = int(w * ROI_X), int(h * ROI_Y)
    x2r, y2r = int(w * (ROI_X + ROI_W)), int(h * (ROI_Y + ROI_H))
    roi = frame[y1r:y2r, x1r:x2r]

    r = model.predict(roi, conf=min(CONF_BY_CLASS.values()), verbose=False)[0]
    dets = []
    for box in r.boxes:
        prompt = PROMPTS[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        dets.append((float(box.conf[0]), PROMPT_TO_CLASS[prompt],
                     (x1 + x1r, y1 + y1r, x2 + x1r, y2 + y1r)))
    dets.sort(key=lambda d: d[0], reverse=True)

    kept = []
    for c, cls, b in dets:
        if c < CONF_BY_CLASS[cls]:
            continue
        if any(iou(b, k[2]) > MERGE_IOU for k in kept):
            continue
        kept.append((c, cls, b))
    if not kept:
        return None

    cx, cy = (x1r + x2r) / 2, (y1r + y2r) / 2

    def priority(d):
        c, _, (x1, y1, x2, y2) = d
        return ((x1 + x2) / 2 - cx) ** 2 + ((y1 + y2) / 2 - cy) ** 2, -c

    return CLASS_TO_LETTER[min(kept, key=priority)[1]]


def detect():
    """Analyse N_FRAMES fresh frames and return the most common letter."""
    votes = {}
    for _ in range(N_FRAMES):
        letter = detect_once(camera.read())
        if letter:
            votes[letter] = votes.get(letter, 0) + 1
    return max(votes, key=votes.get) if votes else "n"


# ---------------------------------------------------------------- server
def serve():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"Waiting for commands on port {PORT}", flush=True)

    while True:
        conn, addr = srv.accept()
        print("Client connected:", addr, flush=True)
        with conn, conn.makefile("rw") as f:
            for line in f:
                cmd = line.strip().upper()
                if cmd == "DETECT":
                    t0 = time.time()
                    answer = detect()
                    print(f"DETECT -> {answer} ({time.time() - t0:.2f}s)", flush=True)
                elif cmd == "PING":
                    answer = "pong"
                elif cmd == "QUIT":
                    f.write("bye\n")
                    f.flush()
                    camera.release()
                    return
                else:
                    answer = "error"
                f.write(answer + "\n")
                f.flush()
        print("Client disconnected, waiting for a new one...", flush=True)


if __name__ == "__main__":
    serve()
