# ArgusAim

Real-time person detection that reports **how far each person is from the centre
of the frame, in degrees**, plus an estimated distance in metres — with a
targeting overlay on the live video.

The point is the *angles*. A detector gives you a box in pixels, which is only
meaningful for the exact camera and resolution that produced it. Converting to
an angle off the lens axis gives a number that means something physically: it is
what you would tell a motor, a rangefinder, or a person.

```bash
python main.py
```

---

## Quick start

**1. Install** (Python 3.10 or newer)

```bash
pip install -r requirements.txt
```

**2. Run**

```bash
python main.py
```

**3. Watch it in a browser too** (optional — useful for demos)

```bash
python main.py --stream 8080
```

It prints an address like `http://192.168.1.20:8080/`. Open that on any device
on the same network.

### If the camera does not open

Laptops usually expose several cameras — the real one plus virtual devices from
OBS, Teams or the laptop vendor. The virtual ones open happily and then show a
frozen or black picture. Find the real one:

```bash
python tools/list_cameras.py
```

Then pass the index it reports:

```bash
python main.py --source 1
```

---

## What you are looking at

| On screen | Meaning |
|---|---|
| **Orange brackets** | the person currently being followed |
| **Grey brackets** | other people detected, not being followed |
| **Ring in the centre** | the "on target" radius — its size *is* `--lock-deg` |
| **Line from the centre** | the offset from centre to the aim point |
| **Scale bar** | degrees away from the centre, ticks every 5° |
| **Green instead of orange** | the aim point is inside the ring |

The panel reports:

- **YAW (H)** — horizontal angle. Positive means the person is to the **right**.
- **PITCH (V)** — vertical angle. Positive means **up**.
- **RANGE** — estimated distance. A `~` marks a person who is partly out of
  frame, where the estimate reads further away than they really are.

### Keys

| Key | Does |
|---|---|
| `q` or `Esc` | quit |
| `space` | pause |
| `tab` | follow the next person |
| `x` | release the lock, pick again automatically |
| `a` | switch the aim point between head and centre of body |
| `s` | save a screenshot to `shots/` |

(Press `s` while it is tracking you to grab a picture for your report.)

---

## How it works

Five stages, one per module, in the order the loop calls them:

```
camera.py     grab frames on a thread, keep only the newest
    |
detector.py   YOLO11n via ONNX Runtime -> person boxes
    |
tracker.py    match boxes to people across frames, choose one to follow
    |
geometry.py   pixels -> degrees, apparent height -> metres
    |
hud.py        draw it;  stream.py serves the same image to a browser
```

### Detection

Stock **YOLO11n** exported to ONNX, running on the CPU through ONNX Runtime.
It is a general 80-class COCO detector; we keep class 0 (`person`) and drop the
rest. The class filter runs *before* non-maximum suppression, so the expensive
overlap comparison only ever sees the handful of boxes that were people rather
than the full 2100 candidates.

Images are letterboxed — scaled to fit and padded to a square — rather than
squashed to the model's aspect ratio, and the padding is undone afterwards.

### Tracking

A small IoU tracker. Boxes overlapping an existing track by more than 25% are
treated as the same person, so IDs stay stable. Each track also carries a
constant-velocity estimate, which lets it coast through a frame the detector
missed. Without that, a single dropped detection would make the lock jump to
someone else and back, which looks far worse than coasting for three frames.

The lock is **sticky**: once a person is chosen we keep following them while
they are visible, even if someone else moves closer to the centre.

### Pixels to degrees

This is the part worth reading the code for. The obvious conversion is linear:

```python
yaw = (x - centre_x) / width * hfov          # wrong at the edges
```

A camera projects onto a flat sensor, so the correct relation is:

```python
yaw = atan((x - centre_x) / focal_length_px)
```

At 60° field of view the two disagree by about **1.1° a quarter of the way out**
from the centre — larger than the 2° lock radius the program is trying to
measure. `geometry.py` uses the second form.

### Distance

One camera cannot measure depth, so distance is inferred from apparent size:

```
distance = assumed_height * focal_length_px / pixel_height
```

That makes it an **estimate, not a measurement**. It is wrong for children, for
someone sitting down, and for anyone unusually tall — and it is only as accurate
as the field of view you told it the camera has. When someone is cut off by the
frame edge they look shorter, so the estimate reads too far; that case is
detected and flagged with `~` rather than hidden.

---

## Pan/tilt servos (optional)

Off by default -- the vision side runs on its own. Add `--turret` to have the
mount physically follow the person:

```bash
./run --turret --stream 8080
```

With no PWM hardware it falls back to a simulation backend, so the flag is safe
to use on a laptop for testing (`--servo-backend sim` forces that).

### The control law

The camera is bolted to the mount, so it moves with it. That makes the measured
angle to the person *exactly* the correction the mount needs -- there is no
coordinate transform to do:

```python
command += gain * error * dt
```

The `* dt` is the part worth understanding. The obvious version applies the gain
once per **frame**, which makes the real loop gain `gain x frame rate` -- so
swapping to a lighter model, or a faster camera, silently makes the mount more
aggressive until it oscillates. Multiplying by the frame interval instead means
`gain` is "fraction of the error corrected per second" and the behaviour is the
same at 15 fps or 40. That is why the numbers are around 5 rather than 0.2.

The ceiling is **latency, not gain**: detection takes ~30 ms, so the mount always
acts on slightly stale information. Above roughly 8/s it overshoots and hunts.

Tuning lives at the top of `argusaim/turret.py`:

```python
PAN  = Axis(channel=1, gpio=13, min_deg=-90, max_deg=90, gain=6.0, ...)
TILT = Axis(channel=0, gpio=12, min_deg=-30, max_deg=30, gain=4.0, ...)
```

Measured on the real mount at 25 fps: **median yaw error 2.2 deg, median pitch
error 0.6 deg**, with large sign-flips on only 6% and 2% of frames respectively
(frequent sign-flips would mean it was hunting).

### When nobody is visible

It holds still for 2 seconds -- people step behind things and come back -- and
then parks the tilt level and sweeps the pan axis slowly across its full travel,
pausing at each end. A person appearing stops the sweep instantly.

```
no target: hold 2s, then sweep pan -90..+90 at 12 deg/s (15s a pass), tilt parked at +0
```

`--no-scan` makes it hold position instead. The speed and the pause are in the
`Scan` class in `argusaim/turret.py`.

### Wiring

| Signal | BCM pin | Physical pin |
|---|---|---|
| pan servo | GPIO13 | 33 |
| tilt servo | GPIO12 | 32 |
| ground | GND | 6, 9, 14, 20, 25, 30, 34 or 39 |

GPIO12 and GPIO13 are not arbitrary -- they are the two pins the Pi can drive
with **hardware** PWM, so the pulse comes from a peripheral rather than from
Python. Software PWM on any other pin shows up as visible servo jitter, because
the CPU is busy running inference. Enable them once in `/boot/firmware/config.txt`:

```
dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4
```

Give the servos their **own power supply** and share only the ground with the
Pi. They draw far more than the Pi's 3.3 V rail can provide.

---

## Accuracy depends on one number

Everything angular is derived from `--hfov`, the camera's horizontal field of
view. The default of 60° is a reasonable guess for a laptop webcam, not a
measurement of *your* camera. To check it:

1. Put an object at a known distance `d` straight ahead of the camera.
2. Move it sideways until it sits exactly on the left edge of the frame.
   Measure that sideways offset `x`.
3. `hfov = 2 * atan(x / d)` in degrees.

Then run with, say, `--hfov 54`. A 10% error in the field of view is a 10% error
in every angle the program reports.

---

## Options

`python main.py --help` lists everything with its default. The ones that matter:

| Flag | Default | Does |
|---|---|---|
| `--source` | `0` | camera index, or a path to a video file |
| `--hfov` | `60` | camera field of view — **measure this** |
| `--model` | `yolo11n_256` | `_320` and `_416` see further but are slower |
| `--conf` | `0.35` | lower finds more people, and more false positives |
| `--aim` | `head` | `head` or `center` |
| `--select` | `center` | which person to follow: `center`, `largest`, `left`, `right` |
| `--smooth` | `0.6` | aim smoothing; lower is steadier but lags more |
| `--lock-deg` | `2.0` | the on-target radius |
| `--person-h` | `1.70` | assumed standing height for the distance estimate |
| `--every` | `1` | detect every Nth frame and predict in between |
| `--turret` | off | drive the pan/tilt servos to follow the target |
| `--no-scan` | scan on | hold still instead of sweeping when nobody is visible |
| `--servo-backend` | `auto` | `auto`, `hardware`, or `sim` (no motors) |
| `--stream` | `0` | serve the view over HTTP on this port |
| `--no-display` | — | run without the local window (headless) |
| `--record out.mp4` | — | save the annotated video |
| `--max-frames` | `0` | stop after N frames, for benchmarking |

---

## Performance

Two different machines, and interestingly two different bottlenecks.

**Per frame, measured:**

| Stage | Laptop | Pi 5 |
|---|---|---|
| detection (`yolo11n_256`) | 7.6 ms | 30.4 ms |
| tracking + geometry + overlay | 0.8 ms | ~2 ms |
| **end to end** | **29.9 fps** | **25 fps** |

On the **laptop**, detection is cheap enough that the webcam's own 30 fps is the
limit -- the loop is waiting for frames, not for the network.

On the **Pi**, detection is the whole cost, so that is the only thing worth
optimising there.

**Model choice**, measured on its own:

| Model | Input | Laptop | Pi 5 |
|---|---|---|---|
| `yolo11n_256` (default) | 256x256 | 7.4 ms | 30.4 ms |
| `yolo11n_320` | 320x320 | 10.1 ms | 44.0 ms |
| `yolo11n_416` | 416x416 | 16.5 ms | -- |

The 320 model is about 45% more work than the 256 for a modest gain in range,
which is why 256 is the default. `--every 2` runs the detector on every second
frame and predicts in between -- worth using on the Pi, pointless on the laptop
where detection is not the constraint.

Two notes on measuring this yourself:

- Close other programs first. An early version of this table was taken with
  background processes running and reported detection three times slower.
- The frame rate is computed by smoothing the frame *interval* and inverting it
  once, not by averaging instantaneous `1/dt`. Webcams deliver in bursts -- gaps
  of 6 ms then 100 ms here -- and averaging rates weights the short gaps far too
  heavily. Doing it the naive way reported 369 fps from a 30 fps camera.

The browser stream is close to free: frames are only JPEG-encoded when somebody
is actually connected, and the preview has its own frame-rate cap so watching
cannot slow the detector down.

---

## Project layout

```
main.py                 entry point
argusaim/
  config.py             every setting; the CLI is generated from it
  camera.py             threaded capture, newest frame wins
  detector.py           YOLO11n via ONNX Runtime
  tracker.py            IoU tracking, target selection, smoothing
  geometry.py           pinhole model: pixels -> degrees, size -> metres
  hud.py                the overlay
  stream.py             MJPEG server for the browser view
  turret.py             optional: pan/tilt servos and the scan pattern
  app.py                the loop that calls all of the above
models/                 yolo11n_256.onnx (default), _320, _416
run                     launcher that finds the virtualenv (Pi)
tools/list_cameras.py   find which camera index actually works
```

About 900 lines of Python. `app.py` is the place to start reading.

---

## Running it on a Raspberry Pi

Same code, no platform-specific paths. A USB webcam is picked up through V4L2
automatically, and the local window is skipped by itself when there is no
display (so `--stream` is how you watch it).

```bash
cd ~/ArgusAimSimple && ./run --turret --stream 8080
```

Use `./run` rather than `python main.py`: Raspberry Pi OS refuses installs into
the system Python (PEP 668), so onnxruntime lives in a virtualenv and plain
`python main.py` fails with `ModuleNotFoundError`. The `run` script finds it.

**Measured on a Pi 5**, 640x480, 4 threads:

| Model | Per frame | Ceiling |
|---|---|---|
| `yolo11n_256` (default) | 30.4 ms | 32.9 fps |
| `yolo11n_320` | 44.0 ms | 22.7 fps |

End to end with the servos running and a browser watching: **25 fps**.

---

## Notes and limits

- **It detects people, not faces.** A face detector fails the moment somebody
  turns around; a person detector works from any angle. The head aim point is a
  geometric estimate from the box, not a detected face.
- The IoU tracker can swap IDs when two people cross at a similar size. A
  motion-model or appearance-based tracker would fix that at a real cost in
  complexity and speed.
- Distance and angle both inherit any error in `--hfov`.
- The browser stream has no password. Fine on your own network; do not expose
  the port to the internet.

Model: [YOLO11n](https://github.com/ultralytics/ultralytics) by Ultralytics
(AGPL-3.0), exported to ONNX.
