#!/usr/bin/env python3
"""
Standalone (no ROS) letter detector.

Reads frames from a camera (or video file) with OpenCV, finds the three
letter-shaped blobs, reads each one with Tesseract (restricted to the
letters E, W, S, N), and prints the 3-letter string, e.g. "NEW".

All detection logic is unchanged from the ROS node (blob finding, OCR with
confusion mapping, per-position temporal memory, distinct-letter check,
stability check). Only the ROS plumbing was replaced:
    * Image subscription      -> cv2.VideoCapture
    * /detected_letters topic -> stdout, optional --result-file, optional serial
    * debug image topic       -> optional annotated image saved with --save-debug
    * ROS parameters          -> command-line flags
    * get_logger()            -> the standard logging module

Dependencies:
    pip install opencv-python numpy pytesseract
    sudo apt install tesseract-ocr
    (optional, serial output) pip install pyserial

Usage:
    python3 letter_detector.py --source 0
    python3 letter_detector.py --source 0 --no-display --quiet
    python3 letter_detector.py --source 0 --right-to-left
    python3 letter_detector.py --source 0 --no-display --save-debug
    python3 letter_detector.py --source 0 --no-display --save-debug /tmp/seen.jpg
    python3 letter_detector.py --source 0 --roi 100,80,400,300      # pixels: x,y,w,h
    python3 letter_detector.py --source 0 --roi 0.1,0.2,0.8,0.5    # or fractions of the frame
    python3 letter_detector.py --source 0 --verbose --debug-crops crops   # diagnose misreads
    python3 letter_detector.py --source video.mp4 --continuous
    python3 letter_detector.py --source 0 --serial-port /dev/ttyUSB0

Output:
    * By default the program stops at the first confident detection:
      prints "Detected letters: XYZ" (always, even with --quiet), writes
      --result-file and sends over serial if configured, then exits.
    * --continuous keeps running and re-detecting (every confident frame
      is written/sent, same as the old topic publishing behaviour).
    * There is NO preview window in this version (it is meant for a headless
      Pi). Use --save-debug to get an annotated picture of what the camera
      saw. --no-display is still accepted so old command lines keep working,
      but it does nothing.

Serial output (optional): the 3-letter string followed by '\\n', e.g.
"NEW\\n". Re-sent on change and every --serial-resend-interval seconds.
"""

import argparse
import logging
import sys
import time

import cv2
import numpy as np
import pytesseract

try:
    import serial as pyserial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

VALID_LETTERS = {'E', 'W', 'S', 'N'}

# Characters Tesseract is allowed to emit -> the valid letter each one means.
# (Characters it commonly confuses with each letter are folded into it.)
LETTER_CONFUSIONS = {
    'E': {'E', 'F', '3'},
    'W': {'W', 'M', 'w'},
    'S': {'S', '5', '8'},
    'N': {'N', 'H', 'Z'},
}

# Reverse lookup, built from LETTER_CONFUSIONS so the whitelist and the
# mapping can never drift apart.
CHAR_TO_LETTER = {
    ch: letter
    for letter, confusable_chars in LETTER_CONFUSIONS.items()
    for ch in confusable_chars
}

TESSERACT_CONFIG = (
    '--psm 10 -c tessedit_char_whitelist=' + ''.join(sorted(CHAR_TO_LETTER))
)


class LetterDetector:
    """
    Call process_frame(frame) once per BGR frame. It returns
    (confident_result, debug_frame): confident_result is a 3-letter string
    once the same read has been seen confirm_frames times in a row, else
    None.
    """

    def __init__(
        self,
        publish_debug_image=True,
        min_contour_area=500,
        confirm_frames=2,
        min_confidence=30.0,          # Tesseract confidence, 0-100
        min_aspect_ratio=0.5,
        max_aspect_ratio=3.0,
        min_solidity=0.20,
        max_solidity=0.85,
        max_ocr_candidates=8,         # cap blobs sent to Tesseract per frame
        detection_scale=0.5,          # downscale for blob-finding only
        slot_memory_frames=6,
        slot_position_tolerance=60,   # px, full-res
        slot_confidence_decay_per_frame=10.0,
        require_distinct_letters=True,
        right_to_left=False,          # True = highest x (rightmost) first
        min_ocr_crop_height=80,
        roi=None,                     # (x, y, w, h): pixels, or fractions if all <= 1.0
        max_box_width_fraction=0.4,   # reject blobs wider than this share of the search area
        debug_crop_dir=None,          # if set, save the images sent to Tesseract here
        max_debug_crops=60,
        logger=None,
    ):
        self.logger = logger or logging.getLogger('letter_detector')

        self.publish_debug_image = publish_debug_image
        self.min_area = min_contour_area
        self.confirm_frames = confirm_frames
        self.min_confidence = min_confidence
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_solidity = min_solidity
        self.max_solidity = max_solidity
        self.max_ocr_candidates = max_ocr_candidates
        self.detection_scale = detection_scale
        self.slot_memory_frames = slot_memory_frames
        self.slot_position_tolerance = slot_position_tolerance
        self.slot_confidence_decay_per_frame = slot_confidence_decay_per_frame
        self.require_distinct_letters = require_distinct_letters
        self.right_to_left = right_to_left
        self.min_ocr_crop_height = min_ocr_crop_height
        self.roi = roi
        self.max_box_width_fraction = max_box_width_fraction
        self.debug_crop_dir = debug_crop_dir
        self.max_debug_crops = max_debug_crops
        self._debug_crop_count = 0
        if debug_crop_dir:
            import os
            os.makedirs(debug_crop_dir, exist_ok=True)

        # Temporal fusion memory: list of {'x', 'letter', 'conf', 'age'}.
        self.letter_memory = []

        self.last_result = None
        self.stable_count = 0
        self.published_result = None

        # Replacement for ROS's throttle_duration_sec
        self._last_log_time = {}

        self.logger.info('Letter detector (OCR-based, no ROS) started.')

    # -------------------------------------------------------------------------
    def _log_throttled(self, key, message, period):
        """Log at INFO at most once per `period` seconds for a given key."""
        now = time.time()
        if now - self._last_log_time.get(key, 0.0) >= period:
            self._last_log_time[key] = now
            self.logger.info(message)

    # -------------------------------------------------------------------------
    def resolve_roi(self, frame_shape):
        """
        Turn self.roi into integer pixel (x, y, w, h) clamped to the frame.
        If all four numbers are <= 1.0 they are treated as fractions of the
        frame (x, y, w, h), so the same setting works at any resolution.
        Returns the full frame if no ROI is set or the ROI is invalid.
        """
        h_img, w_img = frame_shape[:2]
        if self.roi is None:
            return 0, 0, w_img, h_img

        rx, ry, rw, rh = self.roi
        if max(rx, ry, rw, rh) <= 1.0:
            rx, ry, rw, rh = rx * w_img, ry * h_img, rw * w_img, rh * h_img

        x0 = max(0, min(int(rx), w_img - 1))
        y0 = max(0, min(int(ry), h_img - 1))
        x1 = max(x0 + 1, min(int(rx + rw), w_img))
        y1 = max(y0 + 1, min(int(ry + rh), h_img))
        if (x1 - x0) < 10 or (y1 - y0) < 10:
            self._log_throttled('bad_roi', f'ROI {self.roi} is too small/outside the '
                                f'{w_img}x{h_img} frame - using the full frame.', 5.0)
            return 0, 0, w_img, h_img
        return x0, y0, x1 - x0, y1 - y0

    # -------------------------------------------------------------------------
    def find_candidates(self, gray):
        """
        Find letter-shaped blobs; return bounding boxes in the ORIGINAL
        (full-resolution) coordinate space. Blob finding runs on a
        downscaled copy for speed; boxes are scaled back up before OCR.
        """
        scale = self.detection_scale
        if scale and scale != 1.0:
            small = cv2.resize(gray, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = gray

        blur = cv2.GaussianBlur(small, (5, 5), 0)
        # Inverted binary only FINDS blobs (letters assumed darker than wall)
        _, blob_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area_scaled = self.min_area * (scale ** 2)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_scaled:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # A real letter is a small part of the search area; a blob that
            # spans most of it is a box edge / shadow / background, and would
            # only waste an OCR call (it is the biggest, so it is read first).
            if w > self.max_box_width_fraction * small.shape[1]:
                self.logger.debug(f'find_candidates: rejected oversized blob '
                                  f'(width {w} of {small.shape[1]} px)')
                continue
            aspect = h / float(w) if w > 0 else 0
            if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
                if area >= min_area_scaled * 2:
                    ux, uy, uw, uh = (x, y, w, h) if scale == 1.0 else (
                        int(x / scale), int(y / scale), int(w / scale), int(h / scale))
                    self.logger.debug(
                        f'find_candidates: rejected blob at ({ux},{uy}) '
                        f'{uw}x{uh} - aspect={aspect:.2f} outside '
                        f'[{self.min_aspect_ratio},{self.max_aspect_ratio}]'
                    )
                continue

            # Solidity = contour area / convex hull area. Letters are concave
            # (mid band); round objects are near-solid, wires near-degenerate.
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            if solidity < self.min_solidity or solidity > self.max_solidity:
                if area >= min_area_scaled * 2:
                    ux, uy, uw, uh = (x, y, w, h) if scale == 1.0 else (
                        int(x / scale), int(y / scale), int(w / scale), int(h / scale))
                    self.logger.debug(
                        f'find_candidates: rejected blob at ({ux},{uy}) '
                        f'{uw}x{uh} - solidity={solidity:.2f} outside '
                        f'[{self.min_solidity},{self.max_solidity}]'
                    )
                continue

            if scale != 1.0:
                x, y, w, h = (int(x / scale), int(y / scale),
                              int(w / scale), int(h / scale))
            candidates.append((x, y, w, h))

        return candidates

    # -------------------------------------------------------------------------
    def process_frame(self, frame):
        """
        Runs the full pipeline on one BGR frame.

        Returns (confident_result, debug_frame):
            confident_result: 3-letter string if stable for confirm_frames
                              frames, else None.
            debug_frame:      annotated BGR frame if publish_debug_image,
                              else None.
        """
        debug_frame = frame.copy() if self.publish_debug_image else None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Only look for letters inside the ROI. Boxes are shifted back to
        # full-frame coordinates so OCR crops, memory and drawing all work
        # unchanged.
        rx, ry, rw, rh = self.resolve_roi(gray.shape)
        roi_gray = gray[ry:ry + rh, rx:rx + rw]
        candidates = [(x + rx, y + ry, w, h)
                      for (x, y, w, h) in self.find_candidates(roi_gray)]

        if debug_frame is not None and self.roi is not None:
            # Blue = the region being searched
            cv2.rectangle(debug_frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)

        # Yellow = every blob that passed the area/aspect/solidity filter
        if debug_frame is not None:
            for (x, y, w, h) in candidates:
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        if len(candidates) < 3:
            self._log_throttled(
                'few_candidates',
                f'Only {len(candidates)} letter-shaped candidate(s) found (need 3) - '
                f'check framing/lighting/min_contour_area',
                1.0
            )
            return None, debug_frame

        # Bound how many blobs reach Tesseract (each call is a subprocess).
        # Biggest blobs kept: real letters are usually the largest clean regions.
        ocr_candidates = sorted(candidates, key=lambda b: b[2] * b[3], reverse=True)
        ocr_candidates = ocr_candidates[:self.max_ocr_candidates]

        confirmed = []
        for (x, y, w, h) in ocr_candidates:
            letter, conf = self.classify(gray, x, y, w, h)
            if letter is not None:
                confirmed.append((x, y, w, h, letter, conf))
                if debug_frame is not None:
                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(debug_frame, f'{letter} {conf:.0f}', (x, max(0, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif debug_frame is not None:
                # Red = this box failed to classify confidently
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        result = self.update_letter_memory(confirmed)

        if result is None:
            self._log_throttled(
                'few_confirmed',
                f'Only {len(confirmed)} confirmed letter(s) this frame, '
                f'{len(self.letter_memory)} fresh slot(s) in memory (need 3) - '
                f'dropping this frame',
                1.0
            )
            return None, debug_frame

        confident = self.check_stability(result)

        if debug_frame is not None:
            cv2.putText(debug_frame, f'{result} ({self.stable_count}/{self.confirm_frames})',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return confident, debug_frame

    # -------------------------------------------------------------------------
    def update_letter_memory(self, confirmed):
        """
        Fold this frame's confident (x, y, w, h, letter, conf) reads into
        per-position memory and return a 3-letter result if 3 distinct,
        still-fresh positions are available (they need not all be confident
        in the SAME frame).

        A matched slot is only overwritten if the new read beats the old
        confidence decayed by slot_confidence_decay_per_frame per frame
        since it was last confirmed, so one noisy misread can't clobber a
        solid recent read, but a genuine change still wins after a few
        frames. Slots untouched for slot_memory_frames frames are dropped.
        """
        for slot in self.letter_memory:
            slot['age'] += 1

        for (x, y, w, h, letter, conf) in confirmed:
            cx = x + w / 2.0
            match = None
            best_dist = self.slot_position_tolerance
            for slot in self.letter_memory:
                dist = abs(slot['x'] - cx)
                if dist <= best_dist:
                    best_dist = dist
                    match = slot
            if match is None:
                self.letter_memory.append({'x': cx, 'letter': letter, 'conf': conf, 'age': 0})
                continue

            decayed_conf = match['conf'] - self.slot_confidence_decay_per_frame * match['age']
            if conf >= decayed_conf:
                match['x'], match['letter'], match['conf'], match['age'] = cx, letter, conf, 0
            # else: old value still trusted more even after decay - ignore

        self.letter_memory = [s for s in self.letter_memory if s['age'] <= self.slot_memory_frames]

        if self.require_distinct_letters:
            # The mission code is 3 DISTINCT letters: keep one slot per
            # letter value (freshest / most confident), so a stray misread
            # can't produce garbage like "NNW" / "EEE".
            best_per_letter = {}
            for slot in self.letter_memory:
                cur = best_per_letter.get(slot['letter'])
                if cur is None or (slot['age'], -slot['conf']) < (cur['age'], -cur['conf']):
                    best_per_letter[slot['letter']] = slot
            candidates = list(best_per_letter.values())
        else:
            candidates = self.letter_memory

        if len(candidates) < 3:
            return None

        chosen = sorted(candidates, key=lambda s: (s['age'], -s['conf']))[:3]
        chosen = sorted(chosen, key=lambda s: s['x'], reverse=self.right_to_left)
        return ''.join(s['letter'] for s in chosen)

    # -------------------------------------------------------------------------
    def _read_letter(self, clean):
        """
        One Tesseract pass on a prepared image. Returns (letter, conf, raw).
        A multi-character read like "Ww" or "EE" still counts if every
        character maps to the SAME letter (Tesseract often repeats a glyph);
        mixed reads like "EW" are rejected as ambiguous.
        """
        clean = cv2.copyMakeBorder(clean, 20, 20, 20, 20,
                                   cv2.BORDER_CONSTANT, value=255)
        data = pytesseract.image_to_data(
            clean, config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT
        )

        best_letter, best_conf, best_raw = None, -1.0, None
        for text, conf in zip(data['text'], data['conf']):
            text = text.strip()
            if not text:
                continue
            conf = float(conf)
            mapped = {CHAR_TO_LETTER.get(ch) or CHAR_TO_LETTER.get(ch.upper())
                      for ch in text}
            if len(mapped) == 1 and None not in mapped:
                letter = next(iter(mapped))
                if conf > best_conf:
                    best_letter, best_conf, best_raw = letter, conf, text
            else:
                self.logger.debug(f'classify: unmapped/ambiguous OCR read {text!r} '
                                  f'(conf={conf:.0f})')
        return best_letter, best_conf, best_raw

    def _save_debug_crop(self, img, name):
        if not self.debug_crop_dir or self._debug_crop_count >= self.max_debug_crops:
            return
        import os
        self._debug_crop_count += 1
        cv2.imwrite(os.path.join(self.debug_crop_dir, name), img)

    def classify(self, gray, x, y, w, h):
        pad = 15
        y0, y1 = max(0, y - pad), y + h + pad
        x0, x1 = max(0, x - pad), x + w + pad
        crop = gray[y0:y1, x0:x1]

        if crop.size == 0:
            return None, -1.0

        # Small crops read poorly in Tesseract: a cheap upscale raises confidence
        if crop.shape[0] < self.min_ocr_crop_height:
            up_scale = self.min_ocr_crop_height / crop.shape[0]
            crop = cv2.resize(crop, None, fx=up_scale, fy=up_scale,
                              interpolation=cv2.INTER_CUBIC)

        # Tesseract wants dark text on a light background. Try the local
        # (adaptive) binarisation first; if it gives no confident letter, try
        # a global Otsu one. Different lighting/stroke thickness favours
        # different methods, and the 2nd pass only costs time on failures.
        attempts = []
        attempts.append(('adaptive', cv2.adaptiveThreshold(
            crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 10)))
        attempts.append(('otsu', cv2.threshold(
            cv2.GaussianBlur(crop, (3, 3), 0), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))

        best_letter, best_conf, best_raw, best_method = None, -1.0, None, None
        for method, clean in attempts:
            letter, conf, raw = self._read_letter(clean)
            self._save_debug_crop(
                clean, f'{x}_{y}_{method}_{raw or "none"}_{conf:.0f}.png')
            if letter is not None and conf > best_conf:
                best_letter, best_conf, best_raw, best_method = letter, conf, raw, method
            if best_letter is not None and best_conf >= self.min_confidence:
                break   # good enough - skip the second pass

        if best_letter is None or best_conf < self.min_confidence:
            self.logger.debug(
                f'classify: box ({x},{y}) {w}x{h}: best guess {best_letter!r} '
                f'(raw {best_raw!r}) conf={best_conf:.1f} below '
                f'min_confidence={self.min_confidence}'
            )
            return None, best_conf

        self.logger.debug(
            f'classify: {best_raw!r} -> {best_letter} '
            f'(conf={best_conf:.1f}, {best_method})'
        )
        self.logger.info(f'Letter detected: {best_letter} (conf={best_conf:.1f})')
        return best_letter, best_conf

    # -------------------------------------------------------------------------
    def check_stability(self, result):
        """
        Returns `result` if it has now been read confirm_frames times in a
        row (on every such frame, not just the first, so a vote-counting
        consumer gets repeated confirmations), else None.
        """
        if result == self.last_result:
            self.stable_count += 1
        else:
            self.last_result = result
            self.stable_count = 1

        self._log_throttled(
            'stability',
            f'Read "{result}" - stability {self.stable_count}/{self.confirm_frames}',
            0.5
        )

        if self.stable_count >= self.confirm_frames:
            if result != self.published_result:
                self.logger.info(f'Detected letters: {result}')
            self.published_result = result
            return result
        return None


# ----------------------------------------------------------------------
# Standalone runner
# ----------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Detect the three wall letters (E/W/S/N) from a camera or video file (no ROS).'
    )
    p.add_argument('--source', default='0',
                   help='Camera index (e.g. 0) or path/URL to a video file. Default: 0')
    p.add_argument('--width', type=int, default=640, help='Camera capture width. Default: 640')
    p.add_argument('--height', type=int, default=480, help='Camera capture height. Default: 480')
    p.add_argument('--warmup-frames', type=int, default=5,
                   help='Frames to discard after opening the camera. Default: 5')
    p.add_argument('--save-debug', nargs='?', const='letter_debug.jpg', default=None,
                   metavar='PATH',
                   help='Save the annotated debug frame to PATH (default letter_debug.jpg) when '
                        'a letter string is detected, and on exit if nothing was detected (shows '
                        'what the camera saw). Works with --no-display.')
    p.add_argument('--result-file', default=None,
                   help='Write the latest confident 3-letter string to this file.')
    p.add_argument('--no-display', action='store_true',
                   help='Accepted for compatibility; this version never opens a window.')
    p.add_argument('--serial-port', default='',
                   help='Serial device to send the result to (e.g. /dev/ttyUSB0). '
                        'Empty (default) = no serial output.')
    p.add_argument('--baud-rate', type=int, default=115200)
    p.add_argument('--serial-boot-delay', type=float, default=2.0,
                   help='Seconds to let the board finish resetting after the port opens '
                        '(overlaps with camera warm-up). Default: 2.0')
    p.add_argument('--serial-resend-interval', type=float, default=1.0,
                   help='Re-send the value at least this often (seconds). 0 = only on change.')
    p.add_argument('--continuous', action='store_true',
                   help='Keep running after a confident detection. Default: stop at the first one.')
    p.add_argument('--quiet', action='store_true', help='Only log warnings/errors.')
    p.add_argument('--verbose', action='store_true',
                   help='Debug logging: shows what Tesseract read for every box and why it '
                        'was rejected.')
    p.add_argument('--debug-crops', default=None, metavar='DIR',
                   help='Save the black/white images sent to Tesseract (first 60) into DIR. '
                        'File names show the box position, method, text read and confidence.')
    p.add_argument('--max-box-width-fraction', type=float, default=0.4,
                   help='Reject blobs wider than this fraction of the search area. Default: 0.4')

    # Detector tuning (these replace the ROS parameters)
    p.add_argument('--min-contour-area', type=int, default=500)
    p.add_argument('--confirm-frames', type=int, default=2)
    p.add_argument('--min-confidence', type=float, default=30.0)
    p.add_argument('--min-aspect-ratio', type=float, default=0.5)
    p.add_argument('--max-aspect-ratio', type=float, default=3.0)
    p.add_argument('--min-solidity', type=float, default=0.20)
    p.add_argument('--max-solidity', type=float, default=0.85)
    p.add_argument('--max-ocr-candidates', type=int, default=8)
    p.add_argument('--detection-scale', type=float, default=0.5)
    p.add_argument('--slot-memory-frames', type=int, default=6)
    p.add_argument('--slot-position-tolerance', type=int, default=60)
    p.add_argument('--slot-confidence-decay-per-frame', type=float, default=10.0)
    p.add_argument('--allow-repeated-letters', action='store_false',
                   dest='require_distinct_letters',
                   help='Allow the same letter more than once in the result.')
    p.set_defaults(require_distinct_letters=True)
    p.add_argument('--right-to-left', action='store_true',
                   help='Read the final string starting from the rightmost letter in the image.')
    p.add_argument('--min-ocr-crop-height', type=int, default=80)
    p.add_argument('--roi', default=None, metavar='X,Y,W,H',
                   help='Only detect inside this rectangle. Pixels (e.g. 100,80,400,300) or, if '
                        'all four numbers are <= 1, fractions of the frame (e.g. 0.1,0.2,0.8,0.5). '
                        'The area is drawn in blue on the debug image.')
    args = p.parse_args(argv)
    if args.roi is not None:
        try:
            vals = [float(v) for v in args.roi.split(',')]
            if len(vals) != 4 or min(vals) < 0 or vals[2] <= 0 or vals[3] <= 0:
                raise ValueError
            args.roi = tuple(vals)
        except ValueError:
            p.error('--roi must be four non-negative numbers X,Y,W,H (W and H > 0), '
                    'e.g. --roi 100,80,400,300')
    return args


def open_capture(source_str, width, height, logger):
    """Camera index -> V4L2 + MJPG + small frames; file/URL -> as-is."""
    try:
        index = int(source_str)
    except ValueError:
        return cv2.VideoCapture(source_str)

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return cap
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info(f'Camera {index} opened at '
                f'{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
                f'{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}.')
    return cap


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()

    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else
               logging.WARNING if args.quiet else logging.INFO),
        format='[%(levelname)s] %(message)s',
    )
    logger = logging.getLogger('letter_detector')

    detector = LetterDetector(
        publish_debug_image=bool(args.save_debug),
        min_contour_area=args.min_contour_area,
        confirm_frames=args.confirm_frames,
        min_confidence=args.min_confidence,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        min_solidity=args.min_solidity,
        max_solidity=args.max_solidity,
        max_ocr_candidates=args.max_ocr_candidates,
        detection_scale=args.detection_scale,
        slot_memory_frames=args.slot_memory_frames,
        slot_position_tolerance=args.slot_position_tolerance,
        slot_confidence_decay_per_frame=args.slot_confidence_decay_per_frame,
        require_distinct_letters=args.require_distinct_letters,
        right_to_left=args.right_to_left,
        min_ocr_crop_height=args.min_ocr_crop_height,
        roi=args.roi,
        max_box_width_fraction=args.max_box_width_fraction,
        debug_crop_dir=args.debug_crops,
        logger=logger,
    )

    # Serial first (no sleep yet) so the board boots while the camera warms up
    ser = None
    serial_opened_at = None
    if args.serial_port:
        if not SERIAL_AVAILABLE:
            logger.error('--serial-port given but pyserial is not installed '
                         '(pip install pyserial). Continuing without serial output.')
        else:
            try:
                ser = pyserial.Serial(args.serial_port, args.baud_rate, timeout=1)
                serial_opened_at = time.time()
            except Exception as e:
                logger.error(f'Could not open serial port {args.serial_port!r}: {e}. '
                             'Continuing without serial output.')
                ser = None

    cap = open_capture(args.source, args.width, args.height, logger)
    if not cap.isOpened():
        logger.error(f'Could not open video source: {args.source!r}. '
                     'Check the index with: v4l2-ctl --list-devices')
        if ser is not None:
            ser.close()
        return 1
    logger.info(f'Camera ready (+{time.time() - t0:.2f}s).')

    for _ in range(max(0, args.warmup_frames)):
        cap.read()

    if ser is not None:
        remaining = args.serial_boot_delay - (time.time() - serial_opened_at)
        if remaining > 0:
            time.sleep(remaining)
        logger.info(f'Serial output enabled on {args.serial_port} @ {args.baud_rate} baud.')

    last_sent_value = None
    last_sent_time = 0.0
    last_debug_frame = None
    saved_for_result = None

    def save_debug_frame(img, why):
        try:
            if cv2.imwrite(args.save_debug, img):
                logger.info(f'Saved debug frame ({why}) to {args.save_debug}')
            else:
                logger.error(f'Could not write debug frame to {args.save_debug}')
        except Exception as e:
            logger.error(f'Failed to save debug frame: {e}')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info('End of stream / camera read failed; stopping.')
                break

            confident, debug_frame = detector.process_frame(frame)

            if args.save_debug and debug_frame is not None:
                last_debug_frame = debug_frame
                if confident is not None and confident != saved_for_result:
                    save_debug_frame(debug_frame, f'detected {confident}')
                    saved_for_result = confident

            if confident is not None and args.result_file:
                try:
                    with open(args.result_file, 'w') as f:
                        f.write(confident)
                except OSError as e:
                    logger.error(f'Failed to write result file: {e}')

            if confident is not None and ser is not None:
                now = time.time()
                changed = confident != last_sent_value
                heartbeat_due = (args.serial_resend_interval > 0
                                 and (now - last_sent_time) >= args.serial_resend_interval)
                if changed or heartbeat_due or not args.continuous:
                    try:
                        ser.write(f'{confident}\n'.encode('ascii'))
                        ser.flush()
                    except Exception as e:
                        logger.error(f'Serial write failed: {e}')
                    else:
                        last_sent_value = confident
                        last_sent_time = now

            if confident is not None and not args.continuous:
                print(f'Detected letters: {confident}  (+{time.time() - t0:.2f}s)')
                break
    except KeyboardInterrupt:
        pass
    finally:
        if args.save_debug and saved_for_result is None and last_debug_frame is not None:
            save_debug_frame(last_debug_frame, 'no detection - last frame')
        cap.release()
        if ser is not None:
            ser.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
