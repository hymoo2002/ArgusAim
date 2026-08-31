"""Tracking people between frames, and choosing which one to follow.

A detector has no memory: each frame it returns a fresh set of boxes with no
notion that this box and that box are the same person. Three things are added
here, and each one exists to fix a specific visible problem:

1. **Identity** -- boxes are matched to existing tracks by overlap, so a person
   keeps the same ID. Without it the target we follow would swap between people
   whenever the detector happened to list them in a different order.
2. **Velocity** -- each track carries a constant-velocity estimate, so it can
   coast through a frame where the detector missed, or where we deliberately
   skipped detection with ``--every N``.
3. **Smoothing** -- the aim point is passed through an exponential moving
   average. Raw detections jitter by a few pixels every frame; smoothing turns
   that into steady motion.
"""
from __future__ import annotations

import numpy as np

# A head is roughly one seventh of standing height. Aiming at the top of the box
# would point at the top of the skull, so we come down into the middle of the head.
HEAD_FRAC = 0.075


def iou(a, b) -> float:
    """Intersection over union of two boxes -- how much they overlap, 0 to 1."""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    overlap = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if overlap <= 0:
        return 0.0
    union = ((ax2 - ax1) * (ay2 - ay1)) + ((bx2 - bx1) * (by2 - by1)) - overlap
    return overlap / union if union > 0 else 0.0


class Track:
    """One person, followed across frames."""

    __slots__ = ("id", "box", "score", "vel", "aim", "hits", "misses", "truncated")

    def __init__(self, track_id: int, box, score: float):
        self.id = track_id
        self.box = np.asarray(box[:4], dtype=np.float32)
        self.score = score
        self.vel = np.zeros(4, np.float32)
        self.aim = None          # smoothed aim point (x, y)
        self.hits = 1            # frames matched to a detection
        self.misses = 0          # consecutive frames without one
        self.truncated = False   # touching a frame edge

    def predict(self) -> None:
        """Move the box forward one frame along its last known velocity."""
        self.box = self.box + self.vel

    def update(self, box, score: float, alpha: float = 0.5) -> None:
        new = np.asarray(box[:4], dtype=np.float32)
        # predict() has already advanced self.box, so (new - self.box) is what
        # is *left over* after the prediction, not the real frame-to-frame
        # movement. Adding the old velocity back recovers the actual motion.
        # Skipping this step makes the velocity estimate decay towards zero, and
        # coasted frames then lag behind anyone who is moving.
        measured = (new - self.box) + self.vel
        self.vel = alpha * measured + (1 - alpha) * self.vel
        self.box = new
        self.score = score
        self.hits += 1
        self.misses = 0

    def aim_point(self, mode: str, smooth: float) -> tuple[float, float]:
        """Where to point at this person, smoothed."""
        x1, y1, x2, y2 = self.box
        if mode == "center":
            target = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        else:                                     # head
            target = ((x1 + x2) / 2.0, y1 + (y2 - y1) * HEAD_FRAC)

        if self.aim is None or smooth >= 1.0:
            self.aim = (float(target[0]), float(target[1]))
        else:
            ax, ay = self.aim
            self.aim = (ax + smooth * (target[0] - ax),
                        ay + smooth * (target[1] - ay))
        return self.aim


class Tracker:
    """Holds every current track, and which one we are following."""

    # Tuned by watching it run rather than by theory:
    IOU_MATCH = 0.25    # below this, two boxes are not the same person
    MIN_HITS = 2        # seen this many times before we will aim at it
    MAX_COAST = 3       # may go missing this many frames and stay aimable
    MAX_MISSES = 12     # then it is forgotten entirely
    ID_WRAP = 99        # IDs count 1..99 and start again, to stay readable

    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks: list[Track] = []
        self.locked_id: int | None = None
        self._next_id = 1

    # -- matching ------------------------------------------------------
    def _match(self, detections):
        """Greedy overlap matching: best pair first, then the next best.

        With a handful of people in frame this costs microseconds. The optimal
        assignment (Hungarian algorithm) would be more correct in contrived
        cases and is not worth the dependency here.
        """
        pairs, used_t, used_d = [], set(), set()
        if self.tracks and detections:
            scored = [(iou(t.box, d), ti, di)
                      for ti, t in enumerate(self.tracks)
                      for di, d in enumerate(detections)]
            scored.sort(reverse=True)
            for score, ti, di in scored:
                if score < self.IOU_MATCH or ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                pairs.append((ti, di))
        unmatched = [i for i in range(len(detections)) if i not in used_d]
        return pairs, unmatched

    def _new_id(self) -> int:
        """Next ID, wrapping at ID_WRAP but never reusing a live one."""
        live = {t.id for t in self.tracks}
        for _ in range(self.ID_WRAP):
            track_id = self._next_id
            self._next_id = self._next_id % self.ID_WRAP + 1
            if track_id not in live:
                return track_id
        return self._next_id

    # -- per frame -----------------------------------------------------
    def update(self, detections, frame_shape) -> None:
        """Advance one frame that the detector actually ran on."""
        for t in self.tracks:
            t.predict()

        pairs, unmatched = self._match(detections)
        matched = {ti for ti, _ in pairs}
        for ti, di in pairs:
            self.tracks[ti].update(detections[di], detections[di][4])
        for ti, t in enumerate(self.tracks):
            if ti not in matched:
                t.misses += 1

        for di in unmatched:
            self.tracks.append(Track(self._new_id(), detections[di], detections[di][4]))

        self.tracks = [t for t in self.tracks if t.misses <= self.MAX_MISSES]

        h, w = frame_shape[:2]
        for t in self.tracks:
            x1, y1, x2, y2 = t.box
            t.truncated = bool(x1 <= 3 or y1 <= 3 or x2 >= w - 4 or y2 >= h - 4)

    def coast(self) -> None:
        """Advance a frame where detection was skipped (``--every N``)."""
        for t in self.tracks:
            t.predict()

    # -- choosing a target ---------------------------------------------
    def visible(self) -> list[Track]:
        """Tracks solid enough to aim at.

        A short run of misses is tolerated deliberately. Detectors drop the odd
        frame, and if one miss dropped the target the lock would hop to someone
        else and back again -- which looks far worse than coasting for 3 frames.
        """
        return [t for t in self.tracks
                if t.hits >= self.MIN_HITS and t.misses <= self.MAX_COAST]

    def select(self, camera) -> Track | None:
        """The person we are following this frame.

        The lock is sticky: once chosen we keep following that person while they
        remain visible, even if someone else drifts closer to the centre.
        """
        live = self.visible()
        if not live:
            self.locked_id = None
            return None

        for t in live:
            if t.id == self.locked_id:
                return t

        mode = self.cfg.select
        if mode == "largest":
            best = max(live, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]))
        elif mode == "left":
            best = min(live, key=lambda t: t.box[0])
        elif mode == "right":
            best = max(live, key=lambda t: t.box[2])
        else:                                   # nearest the centre of the frame
            best = min(live, key=lambda t:
                       ((t.box[0] + t.box[2]) / 2 - camera.cx) ** 2 +
                       ((t.box[1] + t.box[3]) / 2 - camera.cy) ** 2)
        self.locked_id = best.id
        return best

    def next_target(self) -> None:
        """Follow the next visible person instead (the Tab key)."""
        ids = sorted(t.id for t in self.visible())
        if not ids:
            return
        if self.locked_id in ids:
            self.locked_id = ids[(ids.index(self.locked_id) + 1) % len(ids)]
        else:
            self.locked_id = ids[0]

    def clear_lock(self) -> None:
        """Let select() choose again from scratch."""
        self.locked_id = None
