"""The main loop: capture -> detect -> track -> measure -> display.

Read this file first. Everything else is a component it calls, and each stage is
one short block below.
"""
from __future__ import annotations

import os
import time

import cv2

from . import hud
from .camera import Camera as FrameSource
from .config import Config, parse_args
from .detector import PersonDetector
from .geometry import Camera as CameraModel
from .geometry import distance_m, is_truncated
from .stream import MJPEGServer, local_ip
from .tracker import Tracker

WINDOW = "ArgusAim"
AIM_MODES = ("head", "center")


def banner(cfg: Config, camera, source, detector, streamer, turret=None) -> None:
    deg_x, deg_y = camera.deg_per_px()
    line = "=" * 78
    print(line)
    print("  ARGUSAIM  |  person detection and angular targeting")
    print(line)
    print("  camera   : %s via %s   %dx%d" % (cfg.source, source.backend,
                                              camera.width, camera.height))
    print("  detector : %s  (%dpx input, %d threads, conf>%.2f)"
          % (os.path.basename(detector.model_path), detector.size,
             cfg.threads, cfg.conf))
    print("  optics   : HFOV %.1f  VFOV %.1f  ->  %.4f deg per pixel"
          % (camera.hfov, camera.vfov, deg_x))
    print("  aiming   : aim=%s  select=%s  smooth=%.2f  lock=%.1f deg"
          % (cfg.aim, cfg.select, cfg.smooth, cfg.lock_deg))
    print("  distance : from an assumed standing height of %.2f m" % cfg.person_h)
    if turret is not None:
        print("  servos   : %s   pan GPIO%d %+.0f..%+.0f deg (gain %.1f/s)   "
              "tilt GPIO%d %+.0f..%+.0f deg (gain %.1f/s)"
              % (turret.backend.name, turret.pan.gpio, turret.pan.min_deg,
                 turret.pan.max_deg, turret.pan.gain, turret.tilt.gpio,
                 turret.tilt.min_deg, turret.tilt.max_deg, turret.tilt.gain))
        if turret.scan.enabled:
            span = turret.pan.max_deg - turret.pan.min_deg
            print("  no target: hold %.0fs, then sweep pan %+.0f..%+.0f at %.0f deg/s "
                  "(%.0fs a pass), tilt parked at %+.0f"
                  % (turret.scan.after_s, turret.pan.min_deg, turret.pan.max_deg,
                     turret.scan.rate_deg_s, span / turret.scan.rate_deg_s,
                     turret.scan.tilt_deg))
        else:
            print("  no target: hold position (scanning disabled)")
    if streamer is not None:
        print("  stream   : http://%s:%d/  (open this in a browser)"
              % (local_ip(), cfg.stream))
    if cfg.record:
        print("  recording: %s" % cfg.record)
    print("-" * 78)
    print("  angles are measured from the centre of the frame:")
    print("  +yaw = target is to the RIGHT,  +pitch = target is UP")
    if cfg.display:
        print("  keys     : [q]uit  [space]pause  [tab]next person  [x]unlock")
        print("             [a]im point  [s]creenshot")
    print(line)


def measure(track, camera, cfg, frame_shape):
    """Turn one tracked person into the numbers we actually care about."""
    aim_x, aim_y = track.aim_point(cfg.aim, cfg.smooth)
    yaw, pitch = camera.angles(aim_x, aim_y)

    height, width = frame_shape[:2]
    offset_px = ((aim_x - camera.cx) ** 2 + (aim_y - camera.cy) ** 2) ** 0.5
    lock_px = camera.pixels_for_angle(cfg.lock_deg)

    return {
        "id": track.id,
        "aim": (aim_x, aim_y),
        "yaw": yaw,
        "pitch": pitch,
        "range_m": distance_m(track.box, camera, cfg.person_h),
        "truncated": is_truncated(track.box, width, height),
        "on_target": offset_px <= lock_px,
    }


def run(cfg: Config) -> int:
    # --- set up -------------------------------------------------------
    try:
        source = FrameSource(cfg).start()
    except RuntimeError as exc:
        # Opening a camera fails for mundane, fixable reasons far more often
        # than interesting ones, so show the advice rather than a traceback.
        print("\n  camera error: %s" % exc)
        return 1

    first, _, _ = source.read()
    height, width = first.shape[:2]

    camera = CameraModel(width, height, cfg.hfov)
    detector = PersonDetector(cfg.model, cfg.conf, cfg.iou, cfg.threads)
    tracker = Tracker(cfg)

    turret = None
    if cfg.turret:
        from .turret import Scan, Turret, make_backend
        turret = Turret(make_backend(cfg.servo_backend),
                        scan=Scan(enabled=cfg.scan))

    streamer = MJPEGServer(cfg.stream, cfg.stream_quality, cfg.stream_fps).start() \
        if cfg.stream else None
    writer = cv2.VideoWriter(cfg.record, cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, (width, height)) if cfg.record else None

    banner(cfg, camera, source, detector, streamer, turret)

    lock_px = camera.pixels_for_angle(cfg.lock_deg)
    aim_mode = cfg.aim
    paused = False
    frame_no = 0
    frame_dt = 0.0          # smoothed seconds per frame; fps is its reciprocal
    fps = 0.0
    started = time.perf_counter()
    last_tick = started
    last_print = 0.0
    note, note_until = None, 0.0
    last_frame = first

    try:
        while True:
            # --- get a frame ------------------------------------------
            if paused:
                frame = last_frame
            else:
                frame, _stamp, fresh = source.read()
                if source.ended and not fresh:
                    break
                if not fresh:
                    time.sleep(0.001)
                    if cfg.display and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    continue
                last_frame = frame
                frame_no += 1
                if cfg.max_frames and frame_no > cfg.max_frames:
                    break

            # --- detect, or coast on the prediction -------------------
            if not paused:
                if frame_no % max(1, cfg.every) == 0:
                    tracker.update(detector.detect(frame), frame.shape)
                else:
                    tracker.coast()

            # --- pick a person and measure them -----------------------
            cfg.aim = aim_mode
            target = tracker.select(camera)
            reading = measure(target, camera, cfg, frame.shape) if target else None

            # --- timing -----------------------------------------------
            now = time.perf_counter()
            dt = now - last_tick
            last_tick = now

            # --- move the mount ---------------------------------------
            # dt is measured first, because the control law needs it.
            if turret is not None and not paused:
                turret.update(reading["yaw"] if reading else None,
                              reading["pitch"] if reading else None, dt)

            if dt > 0:
                # Smooth the frame *interval* and invert once, rather than
                # smoothing 1/dt. Webcams deliver in bursts -- here, gaps of
                # 6 ms then 100 ms -- and averaging instantaneous rates weights
                # the short gaps far too heavily, reporting roughly ten times
                # the real frame rate.
                frame_dt = (0.9 * frame_dt + 0.1 * dt) if frame_dt else dt
                fps = 1.0 / frame_dt

            # --- console readout --------------------------------------
            if cfg.print_hz > 0 and now - last_print >= 1.0 / cfg.print_hz:
                last_print = now
                servo = ("  servo %+6.1f,%+6.1f" % (turret.pan_deg, turret.tilt_deg)
                         if turret is not None else "")
                if reading:
                    print("  id %-3d yaw %+7.2f  pitch %+7.2f  range %s%s  %s"
                          % (reading["id"], reading["yaw"], reading["pitch"],
                             ("%5.2f m%s" % (reading["range_m"],
                                             "~" if reading["truncated"] else " "))
                             if reading["range_m"] else "   --  ",
                             servo,
                             "ON TARGET" if reading["on_target"] else ""))
                else:
                    what = ("scanning" if (turret is not None and turret.scanning)
                            else "no target")
                    print("  %-10s%s%s%.1f fps" % (what, servo, " " * 11, fps))

            # --- draw -------------------------------------------------
            view = frame.copy()
            color = (hud.GREEN if reading and reading["on_target"]
                     else hud.AMBER if reading else hud.GREY)

            for t in tracker.visible():
                if target is None or t.id != target.id:
                    hud.draw_person(view, t, None, hud.DIM, followed=False)
            if reading:
                hud.draw_link(view, camera, reading["aim"], color)
                hud.draw_person(view, target, reading["aim"], color)
            hud.draw_reticle(view, camera, lock_px, color, active=target is not None)

            if note and now > note_until:
                note = None
            hud.draw_panel(view, reading,
                           {"fps": fps, "infer_ms": detector.last_ms},
                           len(tracker.visible()), cfg, paused, note,
                           idle_label=("SCANNING" if (turret is not None
                                                      and turret.scanning)
                                       else "SEARCHING"))

            if streamer is not None:
                streamer.publish(view)
            if writer is not None:
                writer.write(view)

            # --- keyboard ---------------------------------------------
            if not cfg.display:
                continue
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == 9:                          # Tab
                tracker.next_target()
            elif key == ord("x"):
                tracker.clear_lock()
                note, note_until = "lock cleared", now + 1.0
            elif key == ord("a"):
                aim_mode = AIM_MODES[(AIM_MODES.index(aim_mode) + 1) % len(AIM_MODES)]
                if target is not None:
                    target.aim = None               # drop the smoothing history
                note, note_until = "aim: " + aim_mode, now + 1.2
            elif key == ord("s"):
                os.makedirs("shots", exist_ok=True)
                path = os.path.join("shots", time.strftime("argus_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(path, view)
                note, note_until = "saved " + path, now + 1.5
                print("  [shot] " + path)

    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        # The servos go first and unconditionally: whatever else fails while
        # shutting down, the motors must end up not being driven.
        if turret is not None:
            turret.close()
        source.release()
        if streamer is not None:
            streamer.stop()
        if writer is not None:
            writer.release()
            print("  recorded -> " + cfg.record)
        if cfg.display:
            cv2.destroyAllWindows()
        elapsed = time.perf_counter() - started
        print("  done. %d frames in %.1f s  (%.1f fps average)."
              % (frame_no, elapsed, frame_no / elapsed if elapsed else 0.0))
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))
