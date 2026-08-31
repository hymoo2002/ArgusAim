"""Settings, and the command line that fills them in.

Every setting lives in one dataclass. The parser is generated from it by
reflection, so adding an option means adding one field here and nothing else --
the flag, its type and its default can never drift apart.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, fields


@dataclass
class Config:
    # --- camera -------------------------------------------------------
    source: str = "0"       # webcam index, or a path to a video file
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: str = "auto"   # auto | msmf | dshow | v4l2 | any
    flip: bool = False      # mirror horizontally; nicer when testing on yourself

    # --- detector -----------------------------------------------------
    model: str = "models/yolo11n_256.onnx"
    conf: float = 0.35      # detection confidence threshold
    iou: float = 0.50       # NMS overlap threshold
    threads: int = 4        # ONNX Runtime threads
    every: int = 1          # detect every Nth frame, predict in between

    # --- optics -------------------------------------------------------
    # The angles and the distance are both derived from this number, so it is
    # the single most important thing to get right. See the README.
    hfov: float = 60.0      # camera horizontal field of view, degrees

    # --- targeting ----------------------------------------------------
    aim: str = "head"       # head | center -- where on the person to point
    select: str = "center"  # center | largest | left | right
    smooth: float = 0.8     # aim-point smoothing, 0 = frozen .. 1 = no smoothing
    lock_deg: float = 2.0   # angular radius that counts as "on target"
    person_h: float = 1.70  # assumed standing height, metres, used for distance

    # --- pan/tilt servos (optional) -----------------------------------
    turret: bool = False        # drive the servos from the tracking error
    servo_backend: str = "auto"  # auto | hardware | sim
    scan: bool = True            # sweep for people when none are visible

    # --- output -------------------------------------------------------
    display: bool = True    # show the local OpenCV window
    stream: int = 0         # serve the view over HTTP on this port, 0 = off
    stream_fps: float = 15.0
    stream_quality: int = 70
    print_hz: float = 5.0   # console readout rate, 0 = silent
    record: str = ""        # write the annotated view to this .mp4
    max_frames: int = 0     # stop after N frames, 0 = run until quit


HELP = {
    "source": "webcam index (0, 1, ...) or a path to a video file",
    "width": "capture width", "height": "capture height",
    "fps": "requested capture frame rate",
    "backend": "capture backend: auto, msmf, dshow, v4l2 or any",
    "flip": "mirror the image horizontally",
    "model": "person detector, an ONNX YOLO export (256 is fastest, 416 sees furthest)",
    "conf": "confidence threshold; lower finds more, with more false positives",
    "iou": "NMS overlap threshold",
    "threads": "ONNX Runtime threads",
    "every": "run the detector every Nth frame and predict in between",
    "hfov": "camera horizontal field of view in degrees -- MEASURE THIS",
    "aim": "where on the person to aim: head or center",
    "select": "which person to follow when several are visible",
    "smooth": "aim-point smoothing; lower is steadier but lags more",
    "lock_deg": "angular radius counted as on target",
    "person_h": "assumed standing height in metres, used for the distance estimate",
    "turret": "drive the pan/tilt servos to follow the target",
    "servo_backend": "servo driver: auto, hardware, or sim (no motors, for testing)",
    "scan": "sweep the pan axis when nobody is visible (--no-scan holds still)",
    "display": "show the local window (--no-display for headless)",
    "stream": "serve the annotated view as MJPEG on this port (0 = off)",
    "stream_fps": "cap the preview frame rate; does not slow the tracker",
    "stream_quality": "JPEG quality 1-100 for the preview",
    "print_hz": "console readout rate in Hz (0 = silent)",
    "record": "save the annotated view to this .mp4",
    "max_frames": "stop after this many frames; handy for benchmarking",
}


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(
        prog="argusaim",
        description="ArgusAim -- detect people and report how far each one is "
                    "off the centre of the frame, in degrees.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(Config, f.name)
        if isinstance(default, bool):
            # --flip / --no-flip, both generated from the one field.
            p.add_argument(flag, dest=f.name, default=default,
                           action=argparse.BooleanOptionalAction, help=HELP.get(f.name))
        else:
            p.add_argument(flag, dest=f.name, type=type(default), default=default,
                           help=HELP.get(f.name))

    cfg = Config(**vars(p.parse_args(argv)))

    # A window is the sensible default on a desktop, but over SSH there is no
    # display and OpenCV does not fail gracefully -- it aborts the whole process
    # with "could not connect to display". Detect that and go headless instead,
    # unless --display was asked for explicitly, in which case the error is the
    # useful answer.
    if cfg.display and sys.platform != "win32":
        asked = "--display" in (list(argv) if argv is not None else sys.argv[1:])
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and not asked:
            cfg.display = False
            hint = "" if cfg.stream else "  Add --stream 8080 to watch it in a browser."
            print("  [config] no display detected - running headless." + hint)

    return cfg
