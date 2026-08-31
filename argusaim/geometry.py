"""Turning pixels into angles and metres.

A detector gives boxes in pixels. Pixels mean nothing to anything downstream --
a servo, a report, a person reading the screen -- so everything here converts to
physical units using a pinhole camera model.
"""
from __future__ import annotations

import math


class Camera:
    """Pinhole model: image coordinates in, angles off the lens axis out.

    The focal length in pixels comes from the horizontal field of view:

        fx = (width / 2) / tan(hfov / 2)

    Angles then use the true relation ``angle = atan((u - cx) / fx)``, not the
    tempting linear shortcut ``(u - cx) / width * hfov``. The linear version is
    fine in the middle of the frame and wrong by several degrees at the edges,
    because a camera projects onto a flat sensor, not onto a sphere. At 60 deg
    HFOV the error at the frame edge is about 2.5 deg -- larger than the lock
    radius we are trying to measure.
    """

    def __init__(self, width: int, height: int, hfov_deg: float = 60.0):
        self.width = width
        self.height = height
        self.cx = width / 2.0
        self.cy = height / 2.0
        # Square pixels, so the vertical focal length is the same number.
        self.fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.fy = self.fx

    @property
    def hfov(self) -> float:
        return math.degrees(2 * math.atan2(self.width / 2.0, self.fx))

    @property
    def vfov(self) -> float:
        return math.degrees(2 * math.atan2(self.height / 2.0, self.fy))

    def angles(self, u: float, v: float) -> tuple[float, float]:
        """Angle from the centre of the frame to a pixel, in degrees.

        Positive yaw means the point is to the right, positive pitch means up.
        The minus sign on pitch is because image rows count downwards.
        """
        yaw = math.degrees(math.atan2(u - self.cx, self.fx))
        pitch = math.degrees(math.atan2(-(v - self.cy), self.fy))
        return yaw, pitch

    def pixels_for_angle(self, deg: float) -> float:
        """How many pixels from the centre correspond to this angle."""
        return self.fx * math.tan(math.radians(deg))

    def deg_per_px(self) -> tuple[float, float]:
        """Degrees per pixel at the centre of the frame, for the readout."""
        return math.degrees(1.0 / self.fx), math.degrees(1.0 / self.fy)


def distance_m(box, camera: Camera, person_h: float = 1.70) -> float | None:
    """Estimate how far away a person is, from how tall they look.

    One camera cannot measure depth, so this assumes a real-world size and works
    backwards from the apparent one:

        distance = real_height * fy / pixel_height

    That makes it an estimate, not a measurement. It is wrong for children, for
    someone sitting, and for anyone unusually tall or short -- and it is only as
    accurate as the field of view the camera was told it has. Treat it as a
    useful ballpark, and see ``truncated`` below for when to distrust it more.
    """
    pixel_h = float(box[3] - box[1])
    if pixel_h < 2:
        return None
    return person_h * camera.fy / pixel_h


def is_truncated(box, width: int, height: int, margin: int = 3) -> bool:
    """True when the box touches a frame edge, so the person is partly cut off.

    This matters for the distance estimate specifically: if the legs are out of
    shot the person looks shorter, so the estimate reads *further away* than
    they really are. The reading is flagged rather than hidden, because it is
    still roughly right and hiding it would be more confusing than qualifying it.
    """
    x1, y1, x2, y2 = box[:4]
    return bool(x1 <= margin or y1 <= margin
                or x2 >= width - 1 - margin or y2 >= height - 1 - margin)
