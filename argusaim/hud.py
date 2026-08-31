"""The overlay drawn on top of the video.

Everything here is cosmetic -- nothing in this file affects detection or the
numbers. It is kept cheap on purpose: the translucent panels blend small regions
rather than the whole frame, so the overlay costs well under a millisecond and
does not distort the frame rate it is reporting.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

GREEN = (120, 255, 130)     # on target
AMBER = (60, 190, 255)      # tracking, but off centre
GREY = (140, 140, 140)      # nobody
DIM = (90, 90, 90)          # other people, not the one we follow
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def scale_of(img) -> float:
    """Overlay scale, so it stays legible from 640x480 up to 1080p."""
    return max(1.0, min(2.4, img.shape[1] / 640.0))


def text(img, s, org, size=0.45, color=WHITE, weight=1) -> None:
    """Text with a black halo, so it reads over a bright or busy background."""
    thick = max(weight, int(round(size * 1.8)))
    cv2.putText(img, s, org, FONT, size, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, size, color, thick, cv2.LINE_AA)


def text_width(s, size, weight=1) -> int:
    thick = max(weight, int(round(size * 1.8)))
    return cv2.getTextSize(s, FONT, size, thick + 2)[0][0]


def panel(img, x1, y1, x2, y2, alpha=0.6) -> None:
    """Darken a rectangle so text on top of it stays readable."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, 18), alpha, 0, roi)
    cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), (60, 60, 60), 1, cv2.LINE_AA)


def corner_box(img, box, color, thick=2) -> None:
    """Bracket corners instead of a full rectangle -- easier to see through."""
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    cw, ch = max(6, int((x2 - x1) * 0.22)), max(6, int((y2 - y1) * 0.22))
    for cx, cy, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + sx * cw, cy), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + sy * ch), color, thick, cv2.LINE_AA)


def draw_reticle(img, camera, lock_px, color, active=False) -> None:
    """Centre crosshair, the lock ring, and a scale in degrees."""
    k = scale_of(img)
    cx, cy = int(camera.cx), int(camera.cy)

    # The ring's radius is exactly the configured on-target angle, so "inside the
    # ring" and "locked" mean the same thing on screen and in the numbers.
    r = int(max(10, lock_px))
    cv2.circle(img, (cx, cy), r, color, 1, cv2.LINE_AA)
    if active:
        cv2.circle(img, (cx, cy), r + int(5 * k), color, 1, cv2.LINE_AA)

    gap, arm = r + int(4 * k), r + int(26 * k)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img, (cx + dx * gap, cy + dy * gap),
                 (cx + dx * arm, cy + dy * arm), color, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, int(2 * k)), color, -1, cv2.LINE_AA)

    # Ticks every 5 degrees, labelled every 10.
    for deg in range(5, 46, 5):
        px = camera.pixels_for_angle(deg)
        major = deg % 10 == 0
        length = int((7 if major else 4) * k)
        if px < camera.cx - 8:
            for side in (1, -1):
                x = int(cx + side * px)
                cv2.line(img, (x, cy - length), (x, cy + length), color, 1, cv2.LINE_AA)
                if major:
                    label = str(deg)
                    text(img, label, (x - text_width(label, 0.36 * k) // 2,
                                      cy + length + int(13 * k)), 0.36 * k, color)


def draw_person(img, track, aim, color, followed=True) -> None:
    """One person: box, ID, and (for the one we follow) the aim marker."""
    k = scale_of(img)
    corner_box(img, track.box, color, int(2 * k) if followed else max(1, int(k)))
    x1, y1 = int(track.box[0]), int(track.box[1])

    if not followed:
        text(img, str(track.id), (x1 + 3, y1 - int(4 * k)), 0.38 * k, DIM)
        return

    ax, ay, r = int(aim[0]), int(aim[1]), int(9 * k)
    cv2.circle(img, (ax, ay), r, color, 1, cv2.LINE_AA)
    cv2.drawMarker(img, (ax, ay), color, cv2.MARKER_CROSS, int(8 * k), 1, cv2.LINE_AA)
    text(img, "ID %d  %.2f" % (track.id, track.score),
         (x1, y1 - int(6 * k)), 0.42 * k, color)


def draw_link(img, camera, aim, color) -> None:
    """Line from the centre of the frame to the aim point -- shows the offset."""
    cx, cy = int(camera.cx), int(camera.cy)
    ax, ay = int(aim[0]), int(aim[1])
    cv2.line(img, (cx, cy), (ax, ay), color, 1, cv2.LINE_AA)
    cv2.line(img, (ax, cy - 4), (ax, cy + 4), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx - 4, ay), (cx + 4, ay), color, 1, cv2.LINE_AA)


def draw_panel(img, reading, stats, n_people, cfg, paused=False, note=None,
               idle_label="SEARCHING") -> None:
    """Telemetry panel, top left, plus the status bar along the bottom."""
    h, w = img.shape[:2]
    k = scale_of(img)

    def S(v):
        return int(round(v * k))

    has_target = reading is not None
    on_target = has_target and reading["on_target"]
    color = GREEN if on_target else (AMBER if has_target else GREY)

    pad = S(8)
    pw, ph = S(238), S(has_target and 124 or 66)
    panel(img, pad, pad, pad + pw, pad + ph)

    left, col = pad + S(8), pad + S(104)
    text(img, "ARGUSAIM", (left, pad + S(21)), 0.52 * k, color)
    state = "PAUSED" if paused else ("TRACKING" if has_target else idle_label)
    text(img, state, (col, pad + S(21)), 0.44 * k, color)

    text(img, "%.1f fps" % stats["fps"], (left, pad + S(40)), 0.42 * k, WHITE)
    text(img, "det %.1f ms" % stats["infer_ms"], (col, pad + S(40)), 0.42 * k, DIM)
    text(img, "people %d" % n_people, (left, pad + S(56)), 0.40 * k, DIM)

    if has_target:
        rows = (("YAW  (H)", "%+.2f deg" % reading["yaw"]),
                ("PITCH(V)", "%+.2f deg" % reading["pitch"]),
                ("RANGE", ("%.2f m" % reading["range_m"]
                           if reading["range_m"] else "--")))
        for i, (label, value) in enumerate(rows):
            y = pad + S(76) + i * S(16)
            text(img, label, (left, y), 0.40 * k, DIM)
            text(img, value, (col, y), 0.44 * k,
                 WHITE if i < 2 else (AMBER if reading["truncated"] else WHITE))
        if reading["truncated"]:
            text(img, "(cut off)", (col + S(58), pad + S(108)), 0.34 * k, AMBER)

    # On-target banner, so it is obvious from across a room.
    if on_target:
        label = "ON TARGET"
        tw = text_width(label, 0.6 * k)
        bx = (w - tw) // 2
        panel(img, bx - S(10), S(8), bx + tw + S(10), S(32), 0.5)
        text(img, label, (bx, S(26)), 0.6 * k, GREEN)

    # Status bar. The key hints are dropped rather than allowed to collide with
    # the settings when the frame is too narrow for both -- they are printed at
    # startup as well, so nothing is lost.
    panel(img, 0, h - S(20), w, h, 0.5)
    bar = "model:%s  aim:%s  select:%s  lock:%.1fdeg" % (
        cfg.model.split("/")[-1].replace(".onnx", ""), cfg.aim, cfg.select, cfg.lock_deg)
    text(img, bar, (S(6), h - S(6)), 0.38 * k, DIM)

    keys = "[q]uit [space]pause [tab]next [x]unlock [a]im [s]hot"
    keys_w = text_width(keys, 0.38 * k)
    if S(6) + text_width(bar, 0.38 * k) + S(16) + keys_w < w:
        text(img, keys, (w - keys_w - S(6), h - S(6)), 0.38 * k, DIM)

    if note:
        tw = text_width(note, 0.5 * k)
        bx = (w - tw) // 2
        panel(img, bx - S(10), h - S(56), bx + tw + S(10), h - S(30), 0.6)
        text(img, note, (bx, h - S(38)), 0.5 * k, WHITE)
