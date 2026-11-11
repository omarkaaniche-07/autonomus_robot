#!/usr/bin/env python3
"""
Standalone (no ROS) version of the wall-number detector.

Detects whether the number printed on a (wooden, high-contrast) wall is a
"1" or a "2" and prints/streams the result. Reads frames directly from a
camera (e.g. a webcam or USB camera) or a video file via OpenCV, instead of
subscribing to a ROS2 image topic.

Since the target is a wooden wall in good/clear conditions with only two
possible digits, this uses a lightweight geometric-shape classifier
(contour aspect ratio + solidity) instead of full OCR. This is faster and
more robust on embedded hardware than running Tesseract every frame, and
doesn't require the tesseract binary to be installed.

An optional Tesseract-based fallback/verification pass can be enabled via
a CLI flag if you want extra robustness and don't mind the dependency.

Dependencies:
    pip install opencv-python numpy
    (optional) sudo apt install tesseract-ocr && pip install pytesseract
    (for serial output to an ESP32) pip install pyserial

Usage:
    python3 wall_number_detector.py --source 0
    python3 wall_number_detector.py --source /path/to/video.mp4
    python3 wall_number_detector.py --source 0 --use-ocr-fallback
    python3 wall_number_detector.py --source 0 --no-display --quiet

    # Raspberry Pi -> ESP32 over serial (e.g. USB or the Pi's GPIO UART):
    python3 wall_number_detector.py --source 0 --no-display \
        --serial-port /dev/ttyUSB0 --baud-rate 115200

Output:
    - By default, the program stops as soon as the first confident
      detection is made: it sends the value over serial (if
      --serial-port is set), writes it to --result-file (if set), prints
      "Detected number: X" to the terminal, and exits. Pass --continuous
      to keep running and re-detecting indefinitely instead (the old
      behavior), with the same serial heartbeat/resend-on-change logic
      as before.
    - "Detected number: X" is always printed to the terminal on a
      confident detection, regardless of --quiet (which only affects the
      more verbose per-frame ar/sol logging).
    - If --result-file is given, writes the latest confidently-detected
      value (just "1" or "2") to that file on every confident frame, so
      another process can poll it the way it might have subscribed to a
      ROS topic.
    - If --serial-port is given, writes the detected value out over a
      serial connection (e.g. USB-serial or the Pi's UART pins) to an
      ESP32 or similar microcontroller. See "Serial output" below for the
      wire format and resend behavior.
    - Unless --no-display is passed, shows a live annotated window (like
      the ROS node's debug image topic). Press 'q' or Esc to quit. On a
      headless Pi (no monitor/X server), pass --no-display or this will
      raise an error trying to open a GUI window.

Serial output (Raspberry Pi -> ESP32):
    - Each value is sent as a single ASCII digit followed by '\\n'
      (e.g. "1\\n"), so the ESP32 side can read it with something as
      simple as Serial.parseInt() or readStringUntil('\\n'). Pass
      --raw-byte to instead send one raw byte (0x01 / 0x02, no newline)
      if you'd rather parse bytes directly.
    - Sent immediately the first time a value becomes confident, and again
      whenever it changes.
    - Also re-sent periodically (every --serial-resend-interval seconds,
      default 1.0) even while the value is steady. Serial links to a
      microcontroller are more failure-prone than an in-process topic —
      a dropped byte, a garbled line, or an ESP32 reset/reconnect can
      silently lose the "last known value" on the receiving end. A small
      heartbeat re-send means the ESP32 self-corrects within one interval
      instead of staying stuck on a stale reading indefinitely. Set
      --serial-resend-interval 0 to disable and only send on change.
    - If the serial port can't be opened (wrong path, permissions, device
      unplugged) this logs an error and continues running without serial
      output (stdout / --result-file still work), rather than crashing —
      useful if you're bringing the camera pipeline up before the ESP32
      is wired in.

CHANGE LOG (calibration fix):
    - aspect_ratio_threshold: 2.2 -> 1.8
    - solidity_threshold: 0.85 -> 0.65
      Real photographed "1"s (perspective, serif foot, blur) topped out around
      ar~2.1 / sol~0.70 in test footage and never cleared the old thresholds,
      so real "1"s were being classified as "2". Real "2"s in the same footage
      sat around ar~1.3-1.5 / sol~0.58-0.59, so there's a comfortable margin
      between the two new thresholds and real "2" data.
    - Added min_box_height_ratio size gate, applied in both candidate
      selection and classification. min_area_ratio alone (0.00005 in the
      original) let ~10x10px noise specks through on a 1920x1080 frame.
      During occlusion/paper-flip transitions these tiny specks produced
      suspiciously "clean" readings (ar=1.000, 2.667, 3.000, 5.000 with
      sol=1.000) that were stable for several frames and got published as
      confident (and false) detections. The height-ratio gate rejects
      anything that isn't plausibly digit-sized before it can become a
      candidate at all.
    - Result is emitted on every confident frame, not only on the first
      frame a new value is reached. A downstream vote-counting consumer
      (require N matching reads before trusting a value) was starved of
      repeat votes for a steady digit, since the old code only emitted a
      value when it changed - meaning every read a consumer saw was,
      almost by construction, a change from the previous one, so votes
      kept landing on alternating keys instead of accumulating on one.
      The console log line is still only printed once per transition, to
      keep the console readable.

CHANGE LOG (OCR fallback hardening, this revision):
    - ocr_confirm now maps a wider set of "1"-like Tesseract misreads
      (7, l, I, |) onto "1" instead of discarding them as unconfirmed.
      This is safe specifically because the competition domain is closed
      to only "1" or "2": any output that isn't literally "2" is itself
      evidence for "1", not evidence for "nothing". "7" in particular is
      the single most common OCR confusion for a single-stroke "1", so
      treating it as a "1" vote meaningfully reduces the miss rate without
      creating a new failure mode ("2" does not get misread as any of
      these characters).
    - Added OCR-side temporal smoothing (_ocr_last / _ocr_streak /
      ocr_confidence_frames), independent from the existing
      geometric-classifier streak. Previously a single noisy OCR frame
      (blur, glare, partial occlusion) could flip `value` for that frame
      *and* reset the downstream confidence streak back to 1, costing
      several frames of delay even though the geometric read never
      changed. Now OCR must agree with itself for `ocr_confidence_frames`
      consecutive frames before it is trusted enough to override the
      geometric classifier. Cost is negligible (a couple of integer
      comparisons per frame); the only trade-off is a small extra frame
      or two of latency before a genuine digit change registers, which is
      a good trade against flicker in a competition setting.
    - OCR now only runs when the geometric read is borderline (within
      `ocr_margin` of either the aspect-ratio or solidity threshold),
      instead of on every single frame. Tesseract is a full OCR pass
      (preprocessing + feature extraction + character matching) and can
      easily take tens to a few hundred milliseconds on embedded hardware,
      which can exceed a camera's per-frame budget (e.g. ~33ms at 30fps)
      and cause processing to fall behind the live feed. Restricting it
      to borderline frames spends that cost only where the cheap
      classifier is least confident, which is exactly where the extra
      accuracy is worth the most.
    - ocr_confirm now logs (debug level) when Tesseract returns a value
      that neither confirms "1" nor "2", so it's possible to tell whether
      OCR is failing often or rarely, instead of silently doing nothing.

CHANGE LOG (this revision - ROS removal):
    - Removed all rclpy / sensor_msgs / cv_bridge / std_msgs dependencies.
      Frames now come from cv2.VideoCapture (camera index or video file)
      instead of a subscribed ROS Image topic, and results are printed to
      stdout / optionally written to a plain text file instead of being
      published to a ROS Int32 topic. The debug "image topic" is replaced
      with an optional live cv2.imshow() window. All detection logic
      (preprocessing, candidate selection, classification, OCR fallback,
      both temporal-smoothing passes) is unchanged from the ROS version.

CHANGE LOG (this revision - serial output to ESP32):
    - Added optional pyserial output (--serial-port / --baud-rate) so the
      Raspberry Pi can push the detected value straight to an ESP32 over
      USB-serial or UART, without needing --result-file plus a separate
      polling script on the ESP32 side.
    - Send-on-change plus a periodic heartbeat re-send
      (--serial-resend-interval), rather than sending on every single
      confident frame (which could be tens of writes per second). A
      microcontroller only needs to know the current value and that the
      link is still alive; flooding it with identical writes wastes UART
      bandwidth and offers no benefit here, unlike the ROS topic case
      where a downstream vote-counter specifically wanted repeats. The
      heartbeat still guards against a dropped byte or an ESP32 reset
      leaving it stuck on a stale value.
"""

import argparse
import logging
import sys
import time

import cv2
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import serial as pyserial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# Tesseract characters that are commonly confused with a single vertical
# stroke ("1") due to similar shape (diagonal top, single dominant stroke).
# Safe to fold into "1" here specifically because the only two valid
# answers in this domain are "1" and "2", and none of these characters is
# a plausible misread of "2".
ONE_LIKE_CHARS = {'1', '7', 'l', 'I', '|'}
TWO_LIKE_CHARS = {'2', 'Z', 'z'}


class WallNumberDetector:
    """
    Same detection logic as the ROS node, minus any ROS dependency.
    Call process_frame(frame) once per frame; it returns the confidently
    detected value (1, 2, or None) for that frame, plus an annotated debug
    frame if requested.
    """

    def __init__(
        self,
        publish_debug_image=True,
        use_ocr_fallback=False,
        min_area_ratio=0.00005,
        max_area_ratio=0.5,
        min_box_height_ratio=0.05,
        aspect_ratio_threshold=1.8,
        solidity_threshold=0.65,
        detection_confidence_frames=3,
        ocr_margin=0.15,
        ocr_confidence_frames=2,
        ocr_borderline_only=False,
        logger=None,
    ):
        self.logger = logger or logging.getLogger('wall_number_detector')
        self._last_area = None

        self.publish_debug_image = publish_debug_image
        self.use_ocr_fallback = use_ocr_fallback and TESSERACT_AVAILABLE
        if use_ocr_fallback and not TESSERACT_AVAILABLE:
            self.logger.warning(
                'use_ocr_fallback requested but pytesseract is not installed; '
                'continuing with geometric classifier only.'
            )

        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_box_height_ratio = min_box_height_ratio
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.solidity_threshold = solidity_threshold
        self.confidence_frames = detection_confidence_frames
        self.ocr_margin = ocr_margin
        self.ocr_confidence_frames = ocr_confidence_frames
        # OCR is authoritative by default (self.ocr_borderline_only == False):
        # it runs on every frame and, once its own smoothing confirms a value,
        # that value wins over the geometric classifier's read. The geometric
        # classifier still runs every frame regardless, both as the frame's
        # debug annotation and as the fallback whenever OCR hasn't confirmed
        # anything yet (e.g. pytesseract not installed, a blurry/unreadable
        # frame, or still mid-way through its own confirmation streak).
        # Set ocr_borderline_only=True to restore the old behavior where OCR
        # only ever runs on frames where the geometric read is borderline.
        self.ocr_borderline_only = ocr_borderline_only

        # ---- State for temporal smoothing (avoid publishing on a single noisy frame) ----
        self._last_value = None
        self._streak = 0
        self._published_value = None

        # ---- State for OCR-side temporal smoothing (independent of the
        # geometric-classifier streak above) ----
        self._ocr_last = None
        self._ocr_streak = 0

        self.logger.info('wall_number_detector initialized.')

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------
    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Wooden wall, clear/high-contrast marking -> Otsu works well and is cheap.
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        return gray, thresh

    def find_best_candidate(self, thresh, image_shape):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h_img, w_img = image_shape[:2]
        img_area = h_img * w_img
        min_box_height_px = self.min_box_height_ratio * h_img

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area_ratio * img_area or area > self.max_area_ratio * img_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # Reject anything too small to plausibly be the digit itself. This is
            # what keeps few-pixel noise specks (e.g. during occlusion, motion
            # blur, or a blacked-out frame) from ever becoming a "candidate",
            # regardless of how their area ratio happens to compare.
            if h < min_box_height_px:
                continue
            aspect = h / float(w + 1e-6)
            if aspect < 0.8:
                continue
            candidates.append((area, c))

        if not candidates:
            return None

        # Prefer the candidate closest in area to last frame's accepted contour,
        # not just the largest — reduces flip-flopping between objects.
        if self._last_area is not None:
            candidates.sort(key=lambda t: abs(t[0] - self._last_area))
        else:
            candidates.sort(key=lambda t: -t[0])

        best_area, best = candidates[0]
        self._last_area = best_area
        return best

    def classify_digit(self, contour, thresh_shape):
        """Geometric-shape classifier: '1' vs '2'."""
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        aspect_ratio = h / float(w + 1e-6)
        solidity = area / float(hull_area + 1e-6)

        is_one = (aspect_ratio >= self.aspect_ratio_threshold and
                    solidity >= self.solidity_threshold)

        value = 1 if is_one else 2
        features = {'aspect_ratio': aspect_ratio, 'solidity': solidity}
        return value, features, (x, y, w, h)

    def is_borderline(self, features):
        """
        True when the geometric read sits close enough to either threshold
        that OCR is worth its cost this frame. Keeps Tesseract off the hot
        path for the (common) clearly-confident frames.
        """
        ar_margin = self.ocr_margin * self.aspect_ratio_threshold
        sol_margin = self.ocr_margin * self.solidity_threshold
        near_ar = abs(features['aspect_ratio'] - self.aspect_ratio_threshold) <= ar_margin
        near_sol = abs(features['solidity'] - self.solidity_threshold) <= sol_margin
        return near_ar or near_sol

    def ocr_confirm(self, gray, box):
        """
        Optional Tesseract pass restricted to '1' and '2' for extra confidence.

        Returns 1, 2, or None. A wider set of "1"-like misreads (7, l, I, |)
        is folded into "1" rather than discarded, since in this closed
        two-class domain any non-"2" read is itself evidence for "1".
        """
        x, y, w, h = box
        pad = int(0.15 * max(w, h))
        roi = gray[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        scale = 200 / max(roi.shape[:2])
        if scale > 1:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        config = '--psm 10 -c tessedit_char_whitelist=12'
        text = pytesseract.image_to_string(roi_bin, config=config).strip()

        if text in TWO_LIKE_CHARS:
            return 2
        if text in ONE_LIKE_CHARS:
            return 1

        # Neither a clean "2" nor a "1"-like read — log at debug level so it's
        # possible to tell whether OCR is failing often or rarely, instead of
        # silently doing nothing every time this happens.
        self.logger.debug(f'ocr_confirm: unconfirmed read {text!r}')
        return None

    def ocr_confirm_smoothed(self, gray, box):
        """
        Wraps ocr_confirm with its own short temporal smoothing, independent
        of the geometric-classifier streak. A single noisy OCR frame no
        longer overrides the geometric read on its own, and no longer resets
        the downstream confidence streak just because it flickered once.
        """
        ocr_value = self.ocr_confirm(gray, box)

        if ocr_value == self._ocr_last:
            self._ocr_streak += 1
        else:
            self._ocr_streak = 1
            self._ocr_last = ocr_value

        if ocr_value is not None and self._ocr_streak >= self.ocr_confidence_frames:
            return ocr_value
        return None

    # ------------------------------------------------------------------
    # Per-frame entry point (replaces the ROS image_callback)
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """
        Runs the full pipeline on one BGR frame.

        Returns (confident_value, debug_frame):
            confident_value: 1 or 2 if this frame reached the required
                              confidence streak, else None.
            debug_frame:      annotated BGR frame if publish_debug_image is
                               True, else None.
        """
        gray, thresh = self.preprocess(frame)
        contour = self.find_best_candidate(thresh, frame.shape)

        detected_value = None
        debug_frame = frame.copy() if self.publish_debug_image else None

        if contour is not None:
            value, features, box = self.classify_digit(contour, thresh.shape)
            self.logger.info(f"ar={features['aspect_ratio']:.3f} sol={features['solidity']:.3f}")

            ocr_value = None
            should_run_ocr = self.use_ocr_fallback and (
                not self.ocr_borderline_only or self.is_borderline(features)
            )
            if should_run_ocr:
                # OCR is authoritative here: it runs on (by default) every
                # frame rather than only borderline ones, and once its own
                # temporal smoothing confirms a value, that value overrides
                # the geometric classifier's read below. If OCR hasn't
                # confirmed anything yet this frame (still building its
                # streak, or pytesseract returned an unreadable result), we
                # fall back to the cheap geometric read rather than stalling.
                ocr_value = self.ocr_confirm_smoothed(gray, box)
                if ocr_value is not None:
                    value = ocr_value  # OCR confirmed -> takes priority

            detected_value = value

            if debug_frame is not None:
                x, y, w, h = box
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = (f'{value}  ar={features["aspect_ratio"]:.2f} '
                         f'sol={features["solidity"]:.2f}'
                         + (' [ocr]' if ocr_value is not None else ''))
                cv2.putText(debug_frame, label, (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # No plausible candidate this frame (e.g. occluded/blacked-out camera).
            # Reset the area-matching memory so we don't try to "match" the next
            # real contour to whatever we were tracking before the gap.
            self._last_area = None
            # Also reset OCR smoothing state so a gap in detection doesn't let
            # a stale OCR streak carry over onto an unrelated future contour.
            self._ocr_last = None
            self._ocr_streak = 0

        # ---- Temporal smoothing: require N consecutive agreeing frames ----
        if detected_value is not None and detected_value == self._last_value:
            self._streak += 1
        else:
            self._streak = 1
            self._last_value = detected_value

        confident_value = None
        if detected_value is not None and self._streak >= self.confidence_frames:
            # Return a value on every confident frame, not just on the first
            # frame a new value is reached. A downstream consumer that
            # vote-counts (e.g. requiring N matching reads before locking in
            # an answer) needs repeated confirmations of a *steady* value to
            # accumulate votes quickly - it should not have to wait for the
            # digit to change and change back. self._published_value is kept
            # only for the log message below, so we're not re-announcing
            # "Detected number: 2" every frame in the log.
            confident_value = detected_value
            if detected_value != self._published_value:
                self.logger.info(f'Detected number: {detected_value}')
            self._published_value = detected_value

        return confident_value, debug_frame


# ----------------------------------------------------------------------
# Standalone runner (replaces the ROS node's spin loop)
# ----------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Detect whether a wall shows "1" or "2" from a camera or video file (no ROS).'
    )
    parser.add_argument('--source', default='1',
                         help='Camera index (e.g. 0) or path/URL to a video file. Default: 0')
    parser.add_argument('--result-file', default=None,
                         help='If set, write the latest confident value ("1" or "2") to this file '
                              'on every confident frame, so another process can poll it.')
    parser.add_argument('--no-display', action='store_true',
                         help="Don't open a live preview window. Use this on a headless Pi.")
    parser.add_argument('--serial-port', default='/dev/ttyUSB0',
                         help='Serial device to send the detected value to (e.g. /dev/ttyUSB0 on '
                              'the Pi for a USB-connected ESP32, or /dev/serial0 for the GPIO UART). '
                              'Requires pyserial. If omitted, no serial output is sent.')
    parser.add_argument('--baud-rate', type=int, default=115200,
                         help='Serial baud rate; must match the ESP32 sketch\'s Serial.begin(). Default: 115200')
    parser.add_argument('--serial-resend-interval', type=float, default=1.0,
                         help='Re-send the current value over serial at least this often (seconds), '
                              'even if it hasn\'t changed, as a heartbeat so the ESP32 self-corrects '
                              'after a dropped byte or reset. Set to 0 to only send on change. Default: 1.0')
    parser.add_argument('--raw-byte', action='store_true',
                         help='Send the value as a single raw byte (0x01/0x02) instead of the default '
                              'ASCII digit + newline (e.g. "1\\n").')
    parser.add_argument('--no-ocr-fallback', action='store_false', dest='use_ocr_fallback',
                         help='Disable the Tesseract OCR verification pass (it is ON by default here; '
                              'requires pytesseract to actually run).')
    parser.add_argument('--ocr-borderline-only', action='store_true',
                         help='Only run OCR on frames where the geometric read is borderline, and let '
                              'the geometric classifier win otherwise (the old, cheaper behavior). '
                              'By default (this flag OFF) OCR runs every frame and, once confirmed, '
                              'takes priority over the geometric read.')
    parser.set_defaults(use_ocr_fallback=True)
    parser.add_argument('--min-area-ratio', type=float, default=0.00005)
    parser.add_argument('--max-area-ratio', type=float, default=0.5)
    parser.add_argument('--min-box-height-ratio', type=float, default=0.05)
    parser.add_argument('--aspect-ratio-threshold', type=float, default=1.8)
    parser.add_argument('--solidity-threshold', type=float, default=0.45)
    parser.add_argument('--detection-confidence-frames', type=int, default=3)
    parser.add_argument('--ocr-margin', type=float, default=0.15)
    parser.add_argument('--ocr-confidence-frames', type=int, default=2)
    parser.add_argument('--continuous', action='store_true',
                         help='Keep running and re-detecting after a confident value is found (the old '
                              'behavior). By default (this flag OFF), the program stops as soon as the '
                              'first confident detection is made, after sending it over serial (if '
                              'configured) and printing it.')
    parser.add_argument('--quiet', action='store_true', help='Only log warnings/errors.')
    return parser.parse_args(argv)


def resolve_source(source_str):
    """Camera indices are passed as ints to cv2.VideoCapture; paths/URLs as strings."""
    try:
        return int(source_str)
    except ValueError:
        return source_str


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format='[%(levelname)s] %(message)s',
    )
    logger = logging.getLogger('wall_number_detector')

    detector = WallNumberDetector(
        publish_debug_image=not args.no_display,
        use_ocr_fallback=args.use_ocr_fallback,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        min_box_height_ratio=args.min_box_height_ratio,
        aspect_ratio_threshold=args.aspect_ratio_threshold,
        solidity_threshold=args.solidity_threshold,
        detection_confidence_frames=args.detection_confidence_frames,
        ocr_margin=args.ocr_margin,
        ocr_confidence_frames=args.ocr_confidence_frames,
        ocr_borderline_only=args.ocr_borderline_only,
        logger=logger,
    )

    cap = cv2.VideoCapture(resolve_source(args.source))
    if not cap.isOpened():
        logger.error(f'Could not open video source: {args.source!r}')
        return 1

    ser = None
    if args.serial_port:
        if not SERIAL_AVAILABLE:
            logger.error(
                '--serial-port was given but pyserial is not installed. '
                'Install it with: pip install pyserial. Continuing without serial output.'
            )
        else:
            try:
                ser = pyserial.Serial(args.serial_port, args.baud_rate, timeout=1)
                # Give the ESP32 a moment to finish resetting after the port opens
                # (common on boards that reset on DTR toggle, e.g. most USB-serial
                # adapters) so the very first send isn't lost while it boots.
                time.sleep(2)
                logger.info(f'Serial output enabled on {args.serial_port} @ {args.baud_rate} baud.')
            except Exception as e:
                logger.error(f'Could not open serial port {args.serial_port!r}: {e}. '
                              'Continuing without serial output.')
                ser = None

    last_sent_value = None
    last_sent_time = 0.0

    window_name = 'wall_number_detector'
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info('End of stream / camera read failed; stopping.')
                break

            confident_value, debug_frame = detector.process_frame(frame)

            if confident_value is not None and args.result_file:
                try:
                    with open(args.result_file, 'w') as f:
                        f.write(str(confident_value))
                except OSError as e:
                    logger.error(f'Failed to write result file: {e}')

            if confident_value is not None and ser is not None:
                now = time.time()
                changed = confident_value != last_sent_value
                heartbeat_due = (
                    args.serial_resend_interval > 0
                    and (now - last_sent_time) >= args.serial_resend_interval
                )
                # When we're about to stop (not --continuous), force the send
                # even if neither condition above happened to be true this
                # exact frame, so the final detection is never dropped right
                # before exit.
                if changed or heartbeat_due or not args.continuous:
                    payload = (bytes([confident_value]) if args.raw_byte
                               else f'{confident_value}\n'.encode('ascii'))
                    try:
                        ser.write(payload)
                    except Exception as e:
                        logger.error(f'Serial write failed: {e}')
                    else:
                        last_sent_value = confident_value
                        last_sent_time = now

            if confident_value is not None and not args.continuous:
                # Print unconditionally (independent of --quiet / logging
                # level) so the result is always visible in the terminal,
                # then stop: this is the "detect once and stop" mode.
                print(f'Detected number: {confident_value}')
                break

            if debug_frame is not None:
                cv2.imshow(window_name, debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # 'q' or Esc
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if ser is not None:
            ser.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())