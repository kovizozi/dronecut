"""Generate synthetic test footage + music with ffmpeg/numpy (nothing here is committed).

Usage: python scripts/make_test_media.py OUT_DIR [--quick]

Clips (good = should survive, junk = must be rejected):
  good_pan_1080     pan over a generated landscape, 1080p, audio
  good_pan_4k       pan over a generated landscape, 4K, no audio
  good_zoomout_1080 slow pull-away zoom over the landscape
  good_orbit_1080   slow rotation + pan (orbit-ish)
  good_testsrc      lavfi testsrc2 (moving counters), 1080p, audio
  good_mandel       lavfi mandelbrot, 1080p
  multi_scene       3 concatenated different shots (hard cuts, for scene detection)
  junk_blur         heavily blurred pan
  junk_shaky        pan with random per-frame jitter
  junk_over         blown-out pan
  junk_under        near-black pan
  junk_sky          featureless blue gradient + faint noise
music: click_120bpm.wav  (kick on each beat, accented downbeat, sparse intro -> full groove)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

FPS = 30
JUNK_PREFIX = "junk_"


def _ff(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode:
        raise RuntimeError(cp.stderr[-1500:])


def make_landscape(path: Path, w: int = 6400, h: int = 3600, seed: int = 7) -> None:
    """Fractal-noise 'terrain' with fields, water, houses and tiny people-like dots."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(2, 9):
        gh, gw = max(2, h >> (9 - o)), max(2, w >> (9 - o))
        g = rng.random((gh // 4 + 2, gw // 4 + 2)).astype(np.float32)
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        acc += amp * np.asarray(im, np.float32) / 255.0
        tot += amp
        amp *= 0.55
    acc /= tot
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    pal = np.array(
        [[40, 90, 160], [70, 120, 200], [60, 130, 60], [90, 160, 70], [150, 130, 80], [120, 100, 70], [230, 230, 235]],
        np.float32,
    )
    idx = np.clip(acc * (len(pal) - 1), 0, len(pal) - 1.001)
    lo = idx.astype(int)
    f = (idx - lo)[..., None]
    img = pal[lo] * (1 - f) + pal[lo + 1] * f
    img += rng.normal(0, 6, img.shape).astype(np.float32)  # grain => sharp texture
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    from PIL import ImageDraw

    d = ImageDraw.Draw(out)
    for _ in range(260):
        x, y = int(rng.integers(0, w - 80)), int(rng.integers(0, h - 80))
        c = tuple(int(v) for v in rng.integers(60, 250, 3))
        d.rectangle([x, y, x + int(rng.integers(12, 70)), y + int(rng.integers(12, 50))], fill=c, outline=(20, 20, 20))
    for _ in range(300):
        x, y = int(rng.integers(0, w - 10)), int(rng.integers(0, h - 10))
        d.ellipse([x, y, x + 8, y + 8], fill=(220, 40, 40))
    out.save(path, quality=92)


def _enc(fps: int = FPS, crf: int = 20) -> list[str]:
    return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps)]


def _img_in(land: Path, dur: float) -> list[str]:
    return ["-i", str(land), "-t", f"{dur}"]


LOOP = f"loop=-1:1:0,setpts=N/{FPS}/TB,"


def pan_clip(land: Path, out: Path, w: int, h: int, dur: float, vf_extra: str = "", audio: bool = False,
             x0: int = 200, y0: int = 300, vx: float = 120, vy: float = 35) -> None:
    # gentle sinusoidal pan so the camera is smooth but not static
    crop = f"crop={w}:{h}:x='{x0}+{vx}*t+40*sin(t*0.9)':y='{y0}+{vy}*t+25*sin(t*0.7)'"
    vf = LOOP + crop + (("," + vf_extra) if vf_extra else "")
    args = _img_in(land, dur)
    if audio:
        args += ["-f", "lavfi", "-t", f"{dur}", "-i", "sine=frequency=220:sample_rate=48000"]
    args += ["-vf", vf, *_enc()]
    if audio:
        args += ["-c:a", "aac", "-shortest"]
    _ff([*args, str(out)])


def make_music(path: Path, bpm: float = 120.0, dur: float = 70.0, sr: int = 44100, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    y = np.zeros(n, np.float32)
    spb = 60.0 / bpm
    t = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 * t + 90 * (1 - np.exp(-t * 30)) / 30)) * np.exp(-t * 14)
    hat = rng.normal(0, 1, int(0.05 * sr)) * np.exp(-np.arange(int(0.05 * sr)) / sr * 90)
    n_beats = int(dur / spb)
    for b in range(n_beats):
        s = int(b * spb * sr)
        t_b = b * spb
        amp = 1.0 if b % 4 == 0 else 0.6
        seg = kick * amp
        y[s : s + len(seg)] += seg[: n - s]
        if t_b >= 16:  # groove kicks in after a sparse intro: off-beat hats + snare-ish clap on 2/4
            h = int((b * spb + spb / 2) * sr)
            y[h : h + len(hat)] += 0.25 * hat[: max(0, n - h)]
            if b % 4 in (1, 3):
                cl = rng.normal(0, 1, int(0.09 * sr)) * np.exp(-np.arange(int(0.09 * sr)) / sr * 45)
                y[s : s + len(cl)] += 0.35 * cl[: max(0, n - s)]
    tt = np.arange(n) / sr
    pad = sum(np.sin(2 * np.pi * f * tt) for f in (130.8, 164.8, 196.0)) * 0.05 * (0.6 + 0.4 * np.sin(2 * np.pi * tt / 8))
    y += pad.astype(np.float32)
    y /= max(1e-6, np.abs(y).max()) * 1.1
    sf.write(path, y, sr)


def write_srt(path: Path, dur: float, fps: int = FPS, east: float = 8.0, north: float = 3.0, climb: float = 1.0) -> None:
    """DJI-style per-frame .SRT: flight east at `east` m/s with a gentle north wiggle, climbing `climb` m/s."""
    import math

    lat0, lon0 = 46.762252, 21.084625
    out = []
    n = int(dur * fps)
    for i in range(n):
        t = i / fps
        x = east * t
        y = north * 3.0 * (1 - math.cos(0.33 * t))
        lat = lat0 + y / 111320.0
        lon = lon0 + x / (111320.0 * math.cos(math.radians(lat0)))
        a, b = i * 1000 // fps, (i + 1) * 1000 // fps

        def ts(ms: int) -> str:
            return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

        out.append(
            chr(10).join([
                str(i + 1),
                f"{ts(a)} --> {ts(b)}",
                f'<font size="28">FrameCnt: {i + 1}, DiffTime: 33ms',
                f"2026-10-04 14:02:{7 + int(t):02d}.{int(t % 1 * 1000):03d}",
                f"[iso: 100] [shutter: 1/1250.0] [fnum: 2.8] [latitude: {lat:.6f}] [longitude: {lon:.6f}] "
                f"[rel_alt: {0.8 + climb * t:.3f} abs_alt: {-38.7 + climb * t:.3f}] </font>",
                "",
            ])
        )
    path.write_text(chr(10).join(out), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--quick", action="store_true", help="shorter clips (CI)")
    a = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    q = 0.6 if a.quick else 1.0
    land = out / "_landscape.jpg"
    make_landscape(land)

    def d(x: float) -> float:
        return round(x * q, 2)

    pan_clip(land, out / "good_pan_1080.mp4", 1920, 1080, d(14), audio=True)
    write_srt(out / "good_pan_1080.SRT", d(14))
    pan_clip(land, out / "good_pan_4k.mp4", 3840, 2160, d(10), x0=100, y0=150, vx=150, vy=40)
    write_srt(out / "good_pan_4k.SRT", d(10), east=12.0, north=-2.0, climb=-0.5)
    # pull-away: crop a shrinking-then-scaled window (zoom out) -> negative divergence
    _ff([*_img_in(land, d(10)), "-vf",
         LOOP + f"crop=3200:1800:1600:800,zoompan=z='max(1.0,2.0-{1.0/(d(10)*FPS):.6f}*on)':"
         f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps={FPS}", *_enc(),
         str(out / "good_zoomout_1080.mp4")])
    _ff([*_img_in(land, d(10)), "-vf",
         LOOP + "crop=2600:1600:x='1500+100*t':y='900+20*t',rotate=a='0.02*t':ow=1920:oh=1080:c=black,scale=1920:1080", *_enc(),
         str(out / "good_orbit_1080.mp4")])
    _ff(["-f", "lavfi", "-i", f"testsrc2=size=1920x1080:rate={FPS}:duration={d(8)}", "-f", "lavfi", "-t", f"{d(8)}",
         "-i", "sine=frequency=440:sample_rate=48000", *_enc(), "-c:a", "aac", "-shortest", str(out / "good_testsrc.mp4")])
    _ff(["-f", "lavfi", "-i", f"mandelbrot=size=1920x1080:rate={FPS}:end_scale=0.001", "-t", f"{d(8)}", *_enc(),
         str(out / "good_mandel.mp4")])
    # multi-scene: three different shots hard-cut together
    parts = []
    for i, (xx, yy) in enumerate([(100, 200), (2800, 1800), (4200, 600)]):
        p = out / f"_ms{i}.mp4"
        pan_clip(land, p, 1920, 1080, d(5), x0=xx, y0=yy, vx=90 + 40 * i, vy=20)
        parts.append(p)
    lst = out / "_ms.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    _ff(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out / "multi_scene.mp4")])
    for p in parts:
        p.unlink()
    lst.unlink()
    # junk
    pan_clip(land, out / "junk_blur.mp4", 1920, 1080, d(8), vf_extra="gblur=sigma=14")
    pan_clip(land, out / "junk_shaky.mp4", 1920, 1080, d(8), x0=0, y0=0)
    _ff([*_img_in(land, d(8)), "-vf",
         LOOP + "crop=w=1700:h=956:x='500+40*t+60*abs(mod(sin(n*12.9898)*43758.5453,1))':"
         "y='500+50*abs(mod(sin(n*78.233)*43758.5453,1))',scale=1920:1080", *_enc(),
         str(out / "junk_shaky.mp4")])
    pan_clip(land, out / "junk_over.mp4", 1920, 1080, d(8), vf_extra="eq=brightness=0.55:contrast=1.3")
    pan_clip(land, out / "junk_under.mp4", 1920, 1080, d(8), vf_extra="eq=brightness=-0.42:contrast=0.7")
    _ff(["-f", "lavfi", "-i", f"gradients=size=1920x1080:rate={FPS}:c0=0x3a78d8:c1=0x6fa8f0:nb_colors=2:speed=0.0001",
         "-t", f"{d(8)}", "-vf", "noise=alls=2:allf=t", *_enc(), str(out / "junk_sky.mp4")])
    make_music(out / "click_120bpm.wav", 120.0, 70.0)
    land.unlink()
    print("generated:", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
