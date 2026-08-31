#!/usr/bin/env python3
"""
Render the steering headroom arc through a scripted torque sweep, offscreen, to PNGs.

The events this indicator exists for are rare, brief and happen while you are driving: 0.49%
of a measured drive was above 384 counts, in bursts of a few frames, almost all of it below
10 m/s. Iterating on the look by going for a drive and trying to remember what it did is not a
design process. This drives the real widget through every state it has -- rest, the
crossing, a graze at the ceiling, a sustained pin, the peak parked and then decaying, and the
same again mirrored -- and writes a frame from each, on both a dark and a bright background,
so the pastels can be judged where they actually have to work.

It renders the shipping widget, not a mock of it. The only thing substituted is the clock:
offscreen frames run far faster than real time, so on the wall clock the dwell would never
accumulate and the ceiling would never light.

Two steps, because they need different machines.

  render, on the device (needs pyray and a GL context):
      python .elantra/render_headroom_bar.py --out /tmp/headroom
      scp -r comma@<device>:/tmp/headroom .

  contact sheet, on the PC (needs pillow):
      python .elantra/render_headroom_bar.py --sheet headroom

The split is not tidiness. The device's hidden window comes up at 536x240 and cannot be
resized, and its captures come back transposed; pass --window WxH to render into a render
texture of any size instead, which is what makes a higher --scale usable. Neither capture path
is directly viewable, so the sheet and movie steps straighten them, measuring each image rather
than assuming anything about it.

  high resolution, on the device:
      python .elantra/render_headroom_bar.py --out /tmp/hr --window 1500x560 --scale 3
      python .elantra/render_headroom_bar.py --anim --out /tmp/hr --window 1500x560 --scale 3
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# (seconds, counts, label to capture at the end of the segment or None)
SCRIPT = [
    (0.60, 0, "01-rest"),
    (0.80, 200, "02-half"),
    (0.80, 340, "03-nearly-there"),
    (0.15, 392, "04-crossing"),
    (0.50, 392, "05-in-the-headroom"),
    (0.20, 409, "06-ceiling-grazed"),
    (1.50, 409, "07-ceiling-held"),
    (1.00, 120, "08-peak-parked"),
    (8.00, 120, "09-peak-decayed"),
    (1.60, -409, "10-mirrored"),
    (1.00, 0, "11-released"),
]

CAPTION = {
    "01-rest": "0 counts. Cyan, the colour of everything the car could always do.",
    "02-half": "200 counts. Still cyan: well inside the stock envelope.",
    "03-nearly-there": "340 counts. Still cyan, and still short of the line.",
    "04-crossing": "392 counts, 0.15 s in. Cyan giving way to pink, the notch uncovering.",
    "05-in-the-headroom": "392 counts. Pink: this is the part we added.",
    "06-ceiling-grazed": "409 counts, 0.2 s. A graze tints toward purple.",
    "07-ceiling-held": "409 counts, 1.7 s. Full purple: it wants more than 409.",
    "08-peak-parked": "Back to 120. The peak mark holds where it got to.",
    "09-peak-decayed": "9 s later. The mark has walked back down and cooled to cyan.",
    "10-mirrored": "-409 counts. The same, the other way.",
    "11-released": "0 counts. Everything has let go.",
}

BACKGROUNDS = {
    # Not black: the arc is drawn over a camera feed, and pastels that only work on black are
    # not the ones that work at midday.
    "dark": (38, 40, 44),
    "light": (196, 198, 202),
}

FPS = 60
PAD = 14  # px of background kept around the arc in the sheet
ZOOM = 2


class ScriptClock:
    """The clock the widget sees. Advanced one frame at a time, never read from the wall."""

    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def render(out: Path, bg_choice: str, scale: float, windowed: bool, window: str | None) -> int:
    rl, gui_app, SteerHeadroomBar = _open_window(windowed, window)
    out.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(rl, window)
    rect = canvas.rect
    print(f"  canvas {canvas.w}x{canvas.h}")

    backgrounds = list(BACKGROUNDS) if bg_choice == "both" else [bg_choice]
    written = 0
    clock = ScriptClock()
    # The ceiling's pulse reads the frame clock. Point it at the script's clock too, so two
    # runs of the same script produce the same pixels.
    rl.get_time = clock

    for bg_name in backgrounds:
        clock.t = 1000.0
        bar = SteerHeadroomBar(demo=True, scale=scale, clock=clock)
        bg = rl.Color(*BACKGROUNDS[bg_name], 255)

        for seconds, counts, label in SCRIPT:
            for _ in range(max(1, round(seconds * FPS))):
                clock.advance(1.0 / FPS)
                bar.update_counts(float(counts))
                canvas.begin(bg)
                bar.render(rect)
                canvas.end()
            if label is None:
                continue
            path = out / f"{label}-{bg_name}.png"
            image = canvas.grab()
            rl.export_image(image, str(path))
            rl.unload_image(image)
            written += 1
            print(f"  {counts:>5} counts  ->  {path}")

    canvas.close()
    gui_app.close()
    print(f"\n{written} frames in {out}")
    print("now: python .elantra/render_headroom_bar.py --sheet <dir> on a machine with pillow")
    return 0


def orient(arr):
    """Put a captured frame back the way the scene was drawn, measuring rather than assuming.

    Two capture paths, two corrections, and the shape says which one it is:

      * the device's hidden window captures the 536x240 scene as a 240x536 image -- the arc's
        long axis lands on the image's long axis. Transposing puts it back, and the flip is
        what makes positive torque grow to the right, which is the direction upstream's arc
        grows and therefore the direction the car will show.
      * a render texture (--window) captures the right way round but bottom-up, as GL render
        targets do, which puts the arc at the top of the frame instead of the bottom.
    """
    import numpy as np

    if arr.shape[0] > arr.shape[1]:  # portrait capture of a landscape scene: the window
        return np.transpose(arr, (1, 0, 2))[:, ::-1, :]
    return arr[::-1, :, :]           # landscape: a render texture, bottom-up


def content_box(arr, bg):
    """(y0, y1, x0, x1) of everything that is not background, padded."""
    import numpy as np

    lit = np.abs(arr.astype(int) - np.array(bg)).sum(axis=2) > 24
    if not lit.any():
        return None
    ys, xs = np.nonzero(lit)
    return (max(0, ys.min() - PAD), min(arr.shape[0], ys.max() + PAD + 1),
            max(0, xs.min() - PAD), min(arr.shape[1], xs.max() + PAD + 1))


def union_box(frames, bg):
    """One crop for a whole sequence.

    Cropping each frame to its own content instead re-centres the arc whenever the bar grows
    or shrinks, which shows up as the entire arc twitching a pixel or two vertically. That is
    an artefact of the review tool, not of the widget -- on the car the arc never moves -- and
    it measured as the two largest frame-to-frame changes in the whole animation.
    """
    boxes = [b for b in (content_box(f, bg) for f in frames) if b is not None]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), max(b[1] for b in boxes),
            min(b[2] for b in boxes), max(b[3] for b in boxes))


def sheet(src: Path, zoom: int = ZOOM) -> int:
    import base64
    import io

    import numpy as np
    from PIL import Image

    frames = {}
    # One crop box for every frame of a background, so the arc does not jump between rows.
    for bg_name, bg in BACKGROUNDS.items():
        # Only the numbered stills. The directory also holds this tool's own straightened
        # output and, when --anim has run, hundreds of animation frames.
        still = re.compile(r"^\d\d-[a-z-]+-" + bg_name + r"\.png$")
        paths = sorted(q for q in src.glob(f"*-{bg_name}.png") if still.match(q.name))
        if not paths:
            continue
        oriented = [orient(np.array(Image.open(p).convert("RGB"))) for p in paths]
        box = union_box(oriented, bg)
        y0, y1, x0, x1 = box
        h, w = y1 - y0, x1 - x0
        for p, arr in zip(paths, oriented, strict=True):
            img = Image.fromarray(arr[y0:y1, x0:x1]).resize((w * zoom, h * zoom), Image.NEAREST)
            img.save(src / f"sheet-{p.stem}.png")
            # Also inlined, so the sheet is one file that can be sent anywhere and still work.
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            key = p.stem[: -len(bg_name) - 1]
            frames.setdefault(key, {})[bg_name] = (
                "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())

    if not frames:
        print(f"no frames found in {src}", file=sys.stderr)
        return 2

    rows = []
    # The animations first, if --movie has been run in the same directory: the stills say what
    # each state looks like, the movies say what it feels like, and the second is the point.
    for tag, title, blurb in (
        ("story", "A drive",
         "Zero to the ceiling and back, then the same the other way, at the rate the car "
         + "actually moves the command: 3 counts per 100 Hz frame out, 7 back."),
        ("ceiling", "Held at 409",
         "The breathing at full dwell. Five seconds is exactly seven breaths, so it loops."),
    ):
        cells = ""
        for bg_name in sorted(BACKGROUNDS):
            movie = src / f"{tag}-{bg_name}.webp"
            if not movie.is_file():
                continue
            data = base64.b64encode(movie.read_bytes()).decode()
            cells += (f'<figure><img src="data:image/webp;base64,{data}" alt="{tag} {bg_name}">'
                      + f'<figcaption>{bg_name}</figcaption></figure>')
        if cells:
            rows.append(f'<section><h2>{title}</h2><p>{blurb}</p>'
                        + f'<div class="pair">{cells}</div></section>')

    for key in sorted(frames):
        cells = "".join(
            f'<figure><img src="{name}" alt="{key} {bg}"><figcaption>{bg}</figcaption></figure>'
            for bg, name in sorted(frames[key].items()))
        rows.append(f'<section><h2>{key[3:].replace("-", " ").capitalize()}</h2>'
                    + f'<p>{CAPTION.get(key, "")}</p><div class="pair">{cells}</div></section>')

    html = """<!doctype html><meta charset="utf-8"><title>steering headroom arc</title>
<style>
  body{background:#111;color:#ddd;font:15px/1.5 system-ui,sans-serif;margin:0;padding:32px}
  h1{font-size:22px;font-weight:600;margin:0 0 4px}
  .lede{color:#999;max-width:60em;margin:0 0 28px}
  section{border-top:1px solid #2a2a2a;padding:20px 0}
  h2{font-size:16px;font-weight:600;margin:0 0 2px}
  section>p{color:#999;margin:0 0 12px}
  .pair{display:flex;gap:20px;flex-wrap:wrap}
  figure{margin:0}
  img{display:block;max-width:100%;image-rendering:pixelated;border-radius:6px}
  figcaption{color:#777;font-size:12px;margin-top:6px}
</style>
<h1>Steering headroom arc</h1>
<p class="lede">The mici onroad torque arc, driven through every state it has.
<b style="color:#8ce6f5">Cyan</b> up to 384 counts, everything the car could always do.
<b style="color:#fabad4">Pink</b> from 384 to 409, the band this build added.
<b style="color:#ba94f0">Purple</b> at 409, weighted by how long it has been pinned there: a
graze tints, a sustained pin breathes. A dark notch marks 384 and fades in on the approach, and
a mark holds the highest point reached so it is still there when you look down after the corner.
Shown on a dark and a bright background because the arc is drawn over the camera feed. Every
pixel is from the shipping widget rendered on the comma 4""" + (
    "." if zoom == 1 else f", magnified {zoom}x.") + """</p>
""" + "\n".join(rows) + "\n"

    out = src / "contact-sheet.html"
    out.write_text(html, encoding="utf-8")
    print(f"{len(frames)} states -> {out}")
    return 0


# A drive, at the rate the car actually moves the command: openpilot rate limits steering to
# 3 counts per 100 Hz frame going away from centre and 7 coming back, so nothing here snaps.
# (seconds, target counts) -- the target is approached at those rates, then held.
ANIM = [
    (0.5, 0),
    (1.5, 340),
    (0.5, 340),
    (0.6, 400),
    (0.4, 400),
    (0.4, 409),
    (2.1, 409),
    (0.9, 120),
    (2.2, 120),
    (1.7, -409),
    (1.2, -409),
    (1.1, 0),
    (0.6, 0),
]

# 5 s at 1.4 Hz is exactly 7 breaths, so the ceiling loop closes on itself.
CEILING_LOOP_S = 5.0

RATE_UP = 3.0 * 100.0 / FPS    # counts per rendered frame, moving away from centre
RATE_DOWN = 7.0 * 100.0 / FPS  # ...and coming back


def _step(current: float, target: float) -> float:
    rate = RATE_UP if abs(target) > abs(current) else RATE_DOWN
    if target > current:
        return min(target, current + rate)
    return max(target, current - rate)


def _open_window(windowed: bool, window: str | None = None):
    if not windowed:
        os.environ["OFFSCREEN"] = "1"  # must be set before gui_app is imported

    import pyray as rl

    if not windowed:
        rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_HIDDEN)

    from openpilot.selfdrive.ui.sunnypilot.mici.onroad.steer_headroom_bar import SteerHeadroomBar
    from openpilot.system.ui.lib.application import gui_app

    gui_app.init_window("steering headroom", fps=FPS)
    return rl, gui_app, SteerHeadroomBar


class Canvas:
    """Where the frames are drawn, and how they come back.

    The device's hidden window comes up at whatever raylib gives it -- 536x240 -- and
    set_window_size does not move it, so the screen is not a canvas you can choose. The arc's
    geometry is absolute (radius 1200 px at scale 1), so rendering it larger needs both a
    bigger scale and somewhere bigger to put it. A render texture is that somewhere.
    """

    def __init__(self, rl, size: str | None):
        self.rl = rl
        self.target = None
        if size:
            w, h = (int(v) for v in size.lower().split("x"))
        else:
            w, h = rl.get_screen_width(), rl.get_screen_height()
        if size:
            self.target = rl.load_render_texture(w, h)
        self.w, self.h = w, h
        self.rect = rl.Rectangle(0, 0, float(w), float(h))

    def begin(self, bg):
        if self.target is not None:
            self.rl.begin_texture_mode(self.target)
        else:
            self.rl.begin_drawing()
        self.rl.clear_background(bg)

    def end(self):
        if self.target is not None:
            self.rl.end_texture_mode()
        else:
            self.rl.end_drawing()

    def grab(self):
        if self.target is not None:
            return self.rl.load_image_from_texture(self.target.texture)
        return self.rl.load_image_from_screen()

    def close(self):
        if self.target is not None:
            self.rl.unload_render_texture(self.target)


def anim(out: Path, bg_choice: str, scale: float, windowed: bool, every: int,
         window: str | None) -> int:
    """Every frame of the sweep, so the motion can be watched rather than inferred."""
    rl, gui_app, SteerHeadroomBar = _open_window(windowed, window)
    out.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(rl, window)
    rect = canvas.rect
    print(f"  canvas {canvas.w}x{canvas.h}")

    clock = ScriptClock()
    rl.get_time = clock
    written = 0

    for bg_name in (list(BACKGROUNDS) if bg_choice == "both" else [bg_choice]):
        bg = rl.Color(*BACKGROUNDS[bg_name], 255)

        # Both sequences settle before the first captured frame. The arc fades itself in over
        # ~0.1 s at startup, so without this the GIF's first frame is nearly dark and the loop
        # flashes every time it wraps -- measured at 40x the next largest frame-to-frame change.
        for tag, script, settle, settle_at in (("story", ANIM, 0.6, 0.0),
                                               ("ceiling", [(CEILING_LOOP_S, 409)], 2.4, 409.0)):
            clock.t = 1000.0
            bar = SteerHeadroomBar(demo=True, scale=scale, clock=clock)
            counts = 0.0
            # The ceiling loop is meant to show the breathing at full depth, so let the dwell
            # build off camera first rather than filming the ramp into it again.
            for _ in range(int(settle * FPS)):
                clock.advance(1.0 / FPS)
                counts = _step(counts, settle_at)
                bar.update_counts(counts)
                canvas.begin(bg)
                bar.render(rect)
                canvas.end()

            i = 0
            for seconds, target in script:
                for _ in range(max(1, round(seconds * FPS))):
                    clock.advance(1.0 / FPS)
                    counts = _step(counts, float(target))
                    bar.update_counts(counts)
                    canvas.begin(bg)
                    bar.render(rect)
                    canvas.end()
                    if i % every == 0:
                        image = canvas.grab()
                        rl.export_image(image, str(out / f"{tag}-{i // every:04d}-{bg_name}.png"))
                        rl.unload_image(image)
                        written += 1
                    i += 1
            print(f"  {tag}/{bg_name}: {i} frames rendered, {i // every} kept")

    canvas.close()
    gui_app.close()
    print(f"\n{written} frames in {out}")
    print("now: python .elantra/render_headroom_bar.py --movie <dir> on a machine with pillow")
    return 0


def movie(src: Path, fps: int, zoom: int = ZOOM) -> int:
    """Straighten the frames and write one animated WebP per sequence."""
    import numpy as np
    from PIL import Image

    made = []
    for bg_name, bg in BACKGROUNDS.items():
        for tag in ("story", "ceiling"):
            paths = sorted(src.glob(f"{tag}-*-{bg_name}.png"))
            if not paths:
                continue
            arrs = [orient(np.array(Image.open(p).convert("RGB"))) for p in paths]
            y0, y1, x0, x1 = union_box(arrs, bg)
            h, w = y1 - y0, x1 - x0
            frames = [Image.fromarray(a[y0:y1, x0:x1]).resize((w * zoom, h * zoom), Image.NEAREST)
                      for a in arrs]

            # WebP, not GIF. The bar is a long smooth gradient and GIF has 8 bits of palette
            # for the whole frame: quantising per frame drifts the palette and flashes the
            # whole arc, and one shared palette bands the gradient and shifts the bands as it
            # moves. Measured against the source PNGs (max mean-delta 1.7, no isolated frames):
            # per-frame palette 6.8 with 4 isolated flashes, shared palette 8.8 with 3. WebP is
            # 24 bit, so it carries what was rendered.
            path = src / f"{tag}-{bg_name}.webp"
            frames[0].save(path, format="WEBP", save_all=True, append_images=frames[1:],
                           duration=round(1000 / fps), loop=0, lossless=True, method=4)
            made.append(path)
            print(f"  {path.name}: {len(frames)} frames, {path.stat().st_size // 1024} KB")

    if not made:
        print(f"no animation frames found in {src}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/tmp/headroom"),
                    help="directory to write frames into (render mode)")
    ap.add_argument("--sheet", type=Path, default=None, metavar="DIR",
                    help="straighten frames already in DIR and build a contact sheet, then exit")
    ap.add_argument("--anim", action="store_true",
                    help="render every frame of the sweep into --out, for the GIFs")
    ap.add_argument("--every", type=int, default=2,
                    help="keep one frame in N when rendering the animation (default 2, so 30 fps)")
    ap.add_argument("--movie", type=Path, default=None, metavar="DIR",
                    help="build animated WebPs from animation frames already in DIR, then exit")
    ap.add_argument("--bg", choices=(*BACKGROUNDS, "both"), default="both")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="widget scale; 1.0 is what the mici HUD uses")
    ap.add_argument("--windowed", action="store_true", help="show the window while it runs")
    ap.add_argument("--window", default=None, metavar="WxH",
                    help="canvas size to render into, e.g. 1400x520; pair with a larger --scale")
    ap.add_argument("--zoom", type=int, default=ZOOM,
                    help="magnification when building the sheet or the movies (default 2)")
    args = ap.parse_args()

    if args.sheet is not None:
        return sheet(args.sheet, args.zoom)
    if args.movie is not None:
        return movie(args.movie, round(FPS / max(1, args.every)), args.zoom)
    if args.anim:
        return anim(args.out, args.bg, args.scale, args.windowed, max(1, args.every), args.window)
    return render(args.out, args.bg, args.scale, args.windowed, args.window)


if __name__ == "__main__":
    sys.exit(main())
