"""Find out which camera index actually works.

Laptops often expose several cameras: the built-in one, plus virtual devices
installed by OBS, Teams or the laptop vendor. Those virtual devices sit on low
indices and open perfectly happily, then serve a black or frozen picture. That
looks exactly like a broken detector, so check here first.

    python tools/list_cameras.py

Then pass whichever index says "live":

    python main.py --source 1
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

BACKENDS = ([(cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW")]
            if sys.platform == "win32" else [(cv2.CAP_V4L2, "V4L2")])


def probe(index: int, api: int) -> tuple[str, str]:
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        cap.release()
        return "-", ""
    frames = [f for _ in range(5) for ok, f in [cap.read()] if ok and f is not None]
    size = "%dx%d" % (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                      cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not frames:
        return "opens, no frames", size
    if float(np.mean(frames[-1])) < 3.0:
        return "black image", size
    if len(frames) >= 2:
        motion = np.mean(np.abs(frames[-1].astype(np.int16)
                                - frames[-2].astype(np.int16)))
        if float(motion) < 0.05:
            return "frozen (virtual camera?)", size
    return "LIVE", size


def main() -> int:
    print("  probing camera indices 0-5 ...\n")
    print("  %-6s %-8s %-26s %s" % ("index", "backend", "result", "size"))
    live = []
    for index in range(6):
        for api, name in BACKENDS:
            result, size = probe(index, api)
            if result == "-":
                continue
            print("  %-6d %-8s %-26s %s" % (index, name, result, size))
            if result == "LIVE":
                live.append(index)

    print()
    if live:
        print("  Use:  python main.py --source %d" % live[0])
    else:
        print("  Nothing live. Close anything else using the camera (Teams, Zoom,")
        print("  the Camera app), and on Windows check Settings > Privacy & security")
        print("  > Camera > Let desktop apps access your camera.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
