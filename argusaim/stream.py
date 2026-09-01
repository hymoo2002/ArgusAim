"""Serve the annotated video to a browser, over plain HTTP.

Motion JPEG: one JPEG per frame, sent as an endless multipart response. Every
browser plays it natively, so there is nothing to install on the viewing machine
and nothing clever happening -- it is just images in a row.

    python main.py --stream 8080
    # then open http://<this-machine>:8080/ from any browser on the network

Two things keep it from slowing the tracker down:

* frames are only JPEG-encoded when somebody is actually connected, so with no
  viewer ``publish()`` returns immediately;
* the preview has its own frame-rate cap, separate from the detection loop, so
  watching cannot throttle tracking.

There is no authentication. That is fine on your own network and not something
to expose to the internet.
"""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = """<!doctype html><html><head><title>ArgusAim</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{margin:0;background:#111;color:#ccc;font-family:system-ui,sans-serif;
      display:flex;flex-direction:column;align-items:center;
      justify-content:center;min-height:100vh;gap:12px}
 img{max-width:96vw;height:auto;border:1px solid #333;border-radius:4px}
 p{font-size:13px;color:#666;margin:0}
 .bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;
      justify-content:center}
 button{font-family:inherit;font-size:15px;font-weight:600;letter-spacing:1px;
        padding:10px 22px;border-radius:6px;cursor:pointer;
        background:#1d3a1d;border:2px solid #3d7a3d;color:#cfc}
 button.armed{background:#c02020;border-color:#ff6060;color:#fff}
 .pill{padding:4px 11px;border-radius:11px;border:1px solid #333;
       font-size:12px;color:#888;font-variant-numeric:tabular-nums}
 .pill.hot{color:#f77;border-color:#7a3030;background:#2a1010}
 .pill.live{color:#6d6;border-color:#3a5a3a}
</style></head><body>
<img src="/stream.mjpg" alt="ArgusAim live view">
<div class="bar">
  <button id="arm" onclick="toggleArm()">SAFE</button>
  <span class="pill" id="fire">output off</span>
  <span class="pill" id="pos">pan --  tilt --</span>
</div>
<p>ArgusAim &mdash; live view &nbsp;|&nbsp; <b>ACTION</b> fires the laser whenever
the turret is locked on target</p>
<script>
let armed = false;

function paint(s){
  armed = !!s.armed;
  const b = document.getElementById("arm");
  b.textContent = armed ? "ACTION" : "SAFE";
  b.className = armed ? "armed" : "";
  const f = document.getElementById("fire");
  f.textContent = s.firing ? "FIRING" : (armed ? "armed, waiting for lock"
                                               : "output off");
  f.className = "pill" + (s.firing ? " hot" : (s.on_target ? " live" : ""));
  document.getElementById("pos").textContent =
    "pan " + (s.pan ?? 0).toFixed(1) + "  tilt " + (s.tilt ?? 0).toFixed(1);
}

function toggleArm(){
  // Deliberately no confirmation dialog - this is a toggle, and being asked
  // "are you sure?" on every press makes it useless during a live demo.
  fetch("/cmd?a=toggle_arm", {method: "POST"})
    .then(r => r.json()).then(paint).catch(() => {});
}

function refresh(){
  fetch("/state").then(r => r.json()).then(paint).catch(() => {});
}
setInterval(refresh, 600);
refresh();
</script>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "ArgusAim"

    def log_message(self, *args):
        pass                      # keep the console clear for the readout

    def _json(self, obj, code=200):
        import json as _json_mod
        body = _json_mod.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self._command()

    def _command(self):
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        action = (q.get("a") or [""])[0]
        server = self.server.streamer
        if action not in server.ALLOWED:
            self._json({"error": "unknown command"}, 400)
            return
        # Queued, not acted on here: the control loop owns the hardware, so
        # nothing touches the laser or the servos from an HTTP thread.
        server.push_command(action)
        self._json(server.state)

    def do_GET(self):
        from urllib.parse import urlparse
        path = urlparse(self.path).path
        if path == "/cmd":
            self._command()
            return
        if path == "/state":
            self._json(self.server.streamer.state)
            return
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()

        server = self.server.streamer
        server.add_viewer()
        try:
            last = 0.0
            while True:
                jpg, stamp = server.wait_for_frame(last)
                if jpg is None:               # the server is shutting down
                    break
                last = stamp
                self.wfile.write(b"--FRAME\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(jpg))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass                              # the viewer closed the tab
        finally:
            server.remove_viewer()


class MJPEGServer:
    """Publishes annotated frames to any browsers that are watching."""

    ALLOWED = ("toggle_arm",)

    def __init__(self, port: int, quality: int = 70, max_fps: float = 15.0):
        self.port = port
        self.quality = int(quality)
        self.min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self.encode_ms = 0.0
        self._commands = []
        self._cmd_lock = threading.Lock()
        self.state = {"armed": False, "firing": False, "on_target": False,
                      "pan": 0.0, "tilt": 0.0}
        self._jpg = None
        self._stamp = 0.0
        self._viewers = 0
        self._last_encode = 0.0
        self._cond = threading.Condition()
        self._httpd = None

    # -- commands ------------------------------------------------------
    def push_command(self, action: str) -> None:
        with self._cmd_lock:
            self._commands.append(action)

    def take_commands(self) -> list:
        """Drain pending commands. Costs nothing when empty."""
        if not self._commands:
            return []
        with self._cmd_lock:
            out, self._commands = self._commands, []
        return out

    def set_state(self, **fields) -> None:
        self.state = fields

    # -- viewers -------------------------------------------------------
    def add_viewer(self) -> None:
        with self._cond:
            self._viewers += 1

    def remove_viewer(self) -> None:
        with self._cond:
            self._viewers = max(0, self._viewers - 1)

    @property
    def viewers(self) -> int:
        return self._viewers

    def wait_for_frame(self, since: float, timeout: float = 5.0):
        with self._cond:
            if not self._cond.wait_for(lambda: self._stamp > since, timeout=timeout):
                return None, since
            return self._jpg, self._stamp

    # -- producer ------------------------------------------------------
    def publish(self, frame) -> None:
        """Offer a frame. Encodes only if somebody is watching and it is due."""
        if self._viewers <= 0:
            return
        now = time.perf_counter()
        if self.min_interval and (now - self._last_encode) < self.min_interval:
            return

        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        self.encode_ms = (time.perf_counter() - t0) * 1000.0
        self._last_encode = now

        with self._cond:
            self._jpg = buf.tobytes()
            self._stamp = now
            self._cond.notify_all()

    # -- lifecycle -----------------------------------------------------
    def start(self) -> "MJPEGServer":
        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), _Handler)
        self._httpd.daemon_threads = True
        self._httpd.streamer = self
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        with self._cond:                      # release any waiting viewers
            self._jpg = None
            self._stamp = time.perf_counter()
            self._cond.notify_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()


def local_ip() -> str:
    """Best guess at this machine's address on the LAN, for the startup banner."""
    import socket
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))        # no packets are actually sent
        ip = probe.getsockname()[0]
        probe.close()
        return ip
    except OSError:
        return "localhost"
