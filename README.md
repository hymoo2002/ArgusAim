# ArgusAim

**A camera that finds a person and turns to keep them centred — by itself.**

It watches a live video feed, works out exactly where a person is, and drives two
motors to follow them. No cloud, no GPU, no subscription.

```bash
python main.py --stream 8080
```

---

## Why this exists

A fixed camera only covers what's in front of it. The moment someone steps
aside, they're gone. Following a person properly needs a human on the controls,
which doesn't scale — and the commercial auto-tracking camera systems that do it
for you are sealed hardware costing thousands.

ArgusAim does the same job on about $80 of open, off-the-shelf parts.

**Where it fits:** security and facilities, live streaming and lecture capture,
interactive attractions, wildlife and field research, and accessibility — any
situation where a camera should follow a person without someone steering it.

---

## What it does, in three steps

| | |
|---|---|
| **1. Sees** | Scans each video frame for people |
| **2. Measures** | Converts the person's position into a real angle in degrees, plus an estimated distance |
| **3. Moves** | Turns two motors to bring them back to the centre |

The measuring step is the interesting part. A detector gives you a box in
pixels, which means nothing outside that one camera. Turning it into an *angle*
gives a number you can hand to a motor — or a rangefinder, or a person.

---

## Results

Measured on both machines, same code:

| | Laptop | Raspberry Pi 5 |
|---|---|---|
| Tracking speed | **29.9 fps** | **25 fps** |
| Detection time per frame | 7.6 ms | 30.4 ms |
| Everything else (tracking, maths, overlay) | 0.8 ms | ~2 ms |

**Aim accuracy on the real mount:** median error **2.2° horizontally**, **0.6°
vertically** — well inside the 2° "on target" ring for the vertical axis.

Two different bottlenecks, which is worth knowing: on the laptop, detection is so
cheap that the webcam's own 30 fps is the ceiling. On the Pi, detection is the
whole cost — so that's the only place worth optimising.

---

## Quick start

**1. Install** (Python 3.10+)

```bash
pip install -r requirements.txt
```

**2. Run**

```bash
python main.py
```

**3. Watch it in a browser** — this is the demo view

```bash
python main.py --stream 8080
```

It prints an address like `http://192.168.1.20:8080/`. Open that on any device
on the same network.

**Camera won't open?** Laptops expose several cameras — the real one plus
virtual devices from Teams or OBS that open fine and show a frozen picture.
Find the real one:

```bash
python tools/list_cameras.py
python main.py --source 1        # whichever index it says is LIVE
```

### Keys

`q` quit · `space` pause · `tab` follow the next person · `x` release the lock ·
`a` aim at head or body centre · `s` save a screenshot

---

## The model, in plain terms

Every frame, a vision model scans the picture and answers one question: **is
there a person here, and exactly where?** It's the same class of technology as a
phone camera locking focus onto a face.

The model is **YOLO11n**, a small pretrained object detector exported to ONNX and
run on the CPU through ONNX Runtime. It was trained by its authors on **COCO**, a
public benchmark of millions of labelled everyday photos. We use it as-is and
keep only the "person" class.

**We did not train or fine-tune it.** The engineering work here is everything
around it: making detection run fast enough on a $80 computer, turning pixels
into physical angles, tracking a person's identity between frames, and closing a
stable control loop onto a pair of motors in real time.

---

## How the pieces fit

```
camera.py     grabs frames on a background thread, keeps only the newest
detector.py   YOLO11n via ONNX Runtime  ->  person boxes
tracker.py    matches boxes to the same person across frames, picks one to follow
geometry.py   pixels -> degrees, apparent height -> metres
turret.py     drives the two servos, and sweeps when nobody is around
hud.py        draws the overlay;  stream.py serves it to a browser
app.py        the loop that calls all of the above
```

Start reading at `app.py`. About 1,600 lines of Python in total.

**Two details worth knowing:**

*Angles use the real pinhole relation* `atan((x - centre) / focal_length)`, not
the tempting linear shortcut. At 60° field of view the two disagree by about
**1.1°** a quarter of the way out from the centre — larger than the 2° lock
radius we're trying to measure.

*The control gain is per second, not per frame.* Multiplying the correction by
the frame interval means the mount behaves the same at 15 fps or 30. The obvious
version — applying the gain once per frame — silently retunes itself whenever the
frame rate moves, which is a real and confusing source of instability.

---

## Pan/tilt motors and the laser (optional)

Off by default. Add `--turret` and the mount physically follows the person:

```bash
./run --turret --stream 8080
```

With no motor hardware attached it falls back to simulation, so the flag is safe
to use on a laptop.

**Wiring:** pan servo on **GPIO 13**, tilt on **GPIO 12** (the two pins the Pi
can drive with hardware PWM, so the pulse is jitter-free), fire output on
**GPIO 17**. Give the servos their own power supply and share only the ground.

**The fire output** has a **SAFE / ACTION** toggle on the stream page. It fires
on exactly one condition — *armed **and** locked on target*. Losing the target
switches it off on the same frame, and so does shutting down.

**When nobody is visible** it holds still for 2 seconds — people step behind
things and come back — then parks the tilt level and sweeps slowly across its
full range until someone appears.

---

## Options

`python main.py --help` lists everything. The ones that matter:

| Flag | Default | Does |
|---|---|---|
| `--source` | `0` | camera index, or a video file |
| `--hfov` | `60` | camera field of view — **every angle depends on this** |
| `--stream PORT` | off | serve the view to a browser |
| `--turret` | off | drive the servos |
| `--model` | `yolo11n_256` | `_320` and `_416` see further but run slower |
| `--conf` | `0.35` | lower finds more people, and more false positives |
| `--no-scan` | scan on | hold still instead of sweeping |
| `--record out.mp4` | — | save the annotated video |

---

## Running on a Raspberry Pi

Same code, no platform-specific paths. A USB webcam is found automatically, and
the local window is skipped by itself when there's no display.

```bash
cd ~/ArgusAimSimple && ./run --turret --stream 8080
```

Use `./run`, not `python main.py` — Raspberry Pi OS blocks installs into the
system Python, so everything lives in a virtual environment and `run` finds it.

---

## Honest limits

- **Distance is an estimate, not a measurement.** It's inferred from how tall a
  person looks, so it's wrong for children, for someone sitting, and for anyone
  unusually tall. A person half out of frame reads as further away than they
  are — that case is detected and flagged with `~` rather than hidden.
- **One camera, one view.** It can't see around a corner.
- **It follows one person at a time.** Choosing between several is a manual step.
- **Every angle inherits any error in `--hfov`.** A 10% error there is a 10%
  error in every number reported.

## Future work

- A depth sensor for true distance instead of an estimate
- Smarter automatic choice of who to follow when several people are in frame
- A weatherproof, higher-torque build for outdoor use
- Fine-tuning the detector on our own footage, so it recognises people in the
  specific conditions it will actually run in

---

## Credits

Detection model: [YOLO11n](https://github.com/ultralytics/ultralytics) by
Ultralytics (AGPL-3.0), trained on [COCO](https://cocodataset.org).
Everything else in this repository was written for this project.
