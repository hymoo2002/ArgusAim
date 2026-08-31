"""Camera capture on a background thread, keeping only the newest frame.

Why a thread: detection takes 15-40 ms, and a camera that keeps filling a driver
buffer meanwhile hands you frames that are already old. Grabbing continuously and
throwing away everything but the latest frame means the angle we report belongs
to what the camera saw a moment ago, not half a second ago.
"""
from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np

# api constant, human-readable name
BACKENDS = {
    "msmf": (cv2.CAP_MSMF, "MSMF"),
    "dshow": (cv2.CAP_DSHOW, "DSHOW"),
    "v4l2": (cv2.CAP_V4L2, "V4L2"),
    "any": (cv2.CAP_ANY, "ANY"),
}


def _looks_live(cap, frames: int = 5) -> tuple[bool, str]:
    """Check that a camera returns a moving picture, not a dead one.

    Virtual cameras (OBS, laptop vendor "sharing" devices) sit on low indices,
    open perfectly happily, and then serve a black or frozen image forever. That
    looks exactly like a broken detector, so it is worth half a second at startup
    to rule out. Any real sensor has enough noise to pass, even facing a wall.
    """
    got = [f for _ in range(frames)
           for ok, f in [cap.read()] if ok and f is not None]
    if not got:
        return False, "opens but returns no frames"
    if float(np.mean(got[-1])) < 3.0:
        return False, "returns a black image"
    if len(got) >= 2:
        motion = np.mean(np.abs(got[-1].astype(np.int16) - got[-2].astype(np.int16)))
        if float(motion) < 0.05:
            return False, "returns frozen frames (virtual camera?)"
    return True, "live"


class Camera:
    """Opens a webcam or video file and serves its newest frame."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = "?"
        self.is_file = False
        self.frame = None
        self.stamp = 0.0
        self._seq = -1
        self._seen = -1
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._cap = self._open()

    # -- opening -------------------------------------------------------
    def _backend_order(self):
        pref = (self.cfg.backend or "auto").lower()
        if pref != "auto":
            if pref not in BACKENDS:
                raise RuntimeError("unknown --backend %r; choose from %s or auto"
                                   % (pref, ", ".join(sorted(BACKENDS))))
            return [BACKENDS[pref]]
        # MSMF before DSHOW on Windows: DSHOW tends to grab whichever virtual
        # camera registered itself on the low indices.
        if sys.platform == "win32":
            return [BACKENDS["msmf"], BACKENDS["dshow"], BACKENDS["any"]]
        return [BACKENDS["v4l2"], BACKENDS["any"]]

    def _open(self):
        src = self.cfg.source
        if not src.isdigit():                       # a video file
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError("could not open video file %r" % src)
            self.backend, self.is_file = "FILE", True
            return cap

        index, tried, fallback = int(src), [], None
        for api, name in self._backend_order():
            cap = cv2.VideoCapture(index, api)
            if not cap.isOpened():
                tried.append("%s: will not open" % name)
                cap.release()
                continue
            self._configure(cap)
            live, why = _looks_live(cap)
            if live:
                self.backend = name
                return cap
            tried.append("%s: %s" % (name, why))
            # Keep the first camera that at least opened. A black image is a
            # perfectly legitimate state -- a dark room, a covered lens -- so
            # this warns and carries on rather than refusing to start.
            if fallback is None:
                fallback = (cap, name)
            else:
                cap.release()

        if fallback is not None:
            cap, self.backend = fallback
            print("  [camera] warning: index %d via %s %s."
                  % (index, self.backend, tried[0].split(": ", 1)[-1]))
            print("  [camera] carrying on anyway. If the picture is wrong, run "
                  "'python tools/list_cameras.py'.")
            return cap

        raise RuntimeError(
            "could not open camera index %d.\n    %s\n"
            "  Run 'python tools/list_cameras.py' to see which indices work,\n"
            "  then pass one, e.g. 'python main.py --source 1'.\n"
            "  Also close anything else using the camera (Teams, Zoom, Camera app)."
            % (index, "\n    ".join(tried) or "no backend accepted the index"))

    def _configure(self, cap) -> None:
        # MJPG lets most USB webcams reach their higher frame rates; without it
        # they fall back to raw YUYV and cap out around 5-10 fps at 640x480.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # not honoured by every backend
        except cv2.error:
            pass

    # -- running -------------------------------------------------------
    def start(self) -> "Camera":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        deadline = time.time() + 5.0
        while self.frame is None and time.time() < deadline and not self._stop.is_set():
            time.sleep(0.005)
        if self.frame is None:
            raise RuntimeError("no frames from the camera (timed out after 5 s)")
        return self

    def _loop(self) -> None:
        # Play a video file at roughly its real speed; a webcam sets its own pace.
        native = self._cap.get(cv2.CAP_PROP_FPS) if self.is_file else 0.0
        interval = 1.0 / native if (self.is_file and native > 1) else 0.0
        n = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self.is_file:
                    self._stop.set()                # end of the file
                    break
                time.sleep(0.005)
                continue
            if self.cfg.flip:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self.frame, self.stamp, self._seq = frame, time.perf_counter(), n
            n += 1
            if interval:
                time.sleep(interval)

    def read(self):
        """Newest frame as (frame, capture_time, is_new). Never blocks."""
        with self._lock:
            if self.frame is None:
                return None, 0.0, False
            fresh = self._seq != self._seen
            self._seen = self._seq
            return self.frame, self.stamp, fresh

    @property
    def ended(self) -> bool:
        return self._stop.is_set()

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._cap.release()
