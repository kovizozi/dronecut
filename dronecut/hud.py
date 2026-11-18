"""Telemetry HUD: speed / altitude / climb rate / distance / heading / clock + a live flight-path mini-map.

The overlay is drawn per frame with Pillow (sized for a 1080p frame, scaled by ffmpeg for other resolutions) and piped
as raw RGBA into the render pass, so it is composited in the single final encode. Cuts whose clip has no GPS data
simply get no HUD.
"""
from __future__ import annotations

import bisect
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import Cfg
from .telemetry import Telemetry, compass, parse_srt
from .utils import log

FONT_CANDIDATES = {
    "bold": ["bahnschrift.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"],
    "regular": ["bahnschrift.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"],
}
FONT_DIRS = [os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"), "/usr/share/fonts/truetype/dejavu",
             "/usr/share/fonts/dejavu", "/Library/Fonts", "/System/Library/Fonts", "/System/Library/Fonts/Supplemental"]


@lru_cache(maxsize=64)
def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for name in FONT_CANDIDATES["bold" if bold else "regular"]:
        for d in FONT_DIRS:
            p = Path(d) / name
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except OSError:
                    continue
    return ImageFont.load_default(size)


def _rgb(hexstr: str) -> tuple[int, int, int]:
    h = hexstr.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


WHITE = (255, 255, 255)


def _spaced(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, f, fill, spacing: float = 3.0) -> None:
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=f, fill=fill)
        x += draw.textlength(ch, font=f) + spacing


@dataclass
class _Map:
    base: Image.Image
    pts: np.ndarray  # (N, 2) pixel coordinates on the map, one per telemetry grid sample


class Hud:
    def __init__(self, cfg: Cfg, out_height: int, fps: float, tele: dict | None = None):
        h = cfg.hud
        self.style = h.style if h.style in ("full", "minimal") else "full"
        self.imperial = h.units == "imperial"
        self.accent = _rgb(h.accent)
        self.fade = max(0.0, float(h.fade_s))
        self.show_map = bool(h.show_map) and self.style == "full"
        self.position = h.position
        self.speed_max = float(h.speed_max_kmh)
        self.W, self.PH = (840, 300) if self.style == "full" else (520, 214)  # panel size; the canvas adds a label strip on top
        self.TOP = 56
        self.H = self.PH + self.TOP
        self.mode = h.mode  # "moments": only on the picked highlights | "all": on every cut
        self.scale = float(h.scale) * out_height / 1080.0
        self.fps = float(min(15.0, fps))
        self._tele: dict[str, Telemetry | None] = dict(tele or {})
        self._maps: dict[str, _Map] = {}
        self._static = self._make_static()

    # ------------------------------------------------------------------ geometry for ffmpeg
    def overlay_args(self, out_w: int, out_h: int) -> tuple[float, str, str]:
        m = int(round(48 * self.scale))
        pos = self.position
        x = f"{m}" if "left" in pos else f"main_w-overlay_w-{m}"
        y = f"{m}" if "top" in pos else f"main_h-overlay_h-{m}"
        return self.scale, x, y

    # ------------------------------------------------------------------ telemetry
    def telemetry(self, clip_path: str, srt_hint: str = "") -> Telemetry | None:
        if clip_path not in self._tele:
            tel = None
            cands = [srt_hint] if srt_hint else []
            cands += [str(Path(clip_path).with_suffix(e)) for e in (".SRT", ".srt")]
            for c in cands:
                if c and Path(c).exists():
                    tel = parse_srt(c)
                    break
            if tel is not None and not tel.has_gps:
                log.info("HUD: %s has no GPS fix - no HUD for its cuts", Path(clip_path).name)
                tel = None
            self._tele[clip_path] = tel
        return self._tele[clip_path]

    def _map_for(self, key: str, tel: Telemetry) -> _Map:
        if key in self._maps:
            return self._maps[key]
        S = 230
        x, y = tel.x, tel.y
        ext = max(float(np.ptp(x)), float(np.ptp(y)), 25.0)
        cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
        k = (S - 56) / ext
        px = (x - cx) * k + S / 2
        py = S / 2 - (y - cy) * k
        pts = np.stack([px, py], axis=1)
        base = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(base, "RGBA")
        d.rounded_rectangle((0, 0, S - 1, S - 1), 22, fill=(0, 0, 0, 95), outline=(255, 255, 255, 45))
        step = max(1, len(pts) // 400)
        d.line([tuple(p) for p in pts[::step]], fill=(255, 255, 255, 110), width=3, joint="curve")
        hx, hy = pts[0]
        d.ellipse((hx - 6, hy - 6, hx + 6, hy + 6), outline=(255, 255, 255, 220), width=2)
        d.text((S / 2, 14), "N", font=font(16, True), fill=(255, 255, 255, 200), anchor="mm")
        self._maps[key] = _Map(base, pts)
        return self._maps[key]

    # ------------------------------------------------------------------ drawing
    def _make_static(self) -> Image.Image:
        img = Image.new("RGBA", (self.W, self.PH), (0, 0, 0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        d.rounded_rectangle((0, 0, self.W - 1, self.PH - 1), 30, fill=(8, 10, 14, 150), outline=(255, 255, 255, 50), width=1)
        lab = font(19, True)
        dim = (255, 255, 255, 150)
        full = self.style == "full"
        _spaced(d, (38, 22), "SPEED", lab, dim)
        _spaced(d, (350 if full else 270, 22), "ALTITUDE", lab, dim)
        if full:
            _spaced(d, (38, 204), "DISTANCE", lab, dim)
            _spaced(d, (190, 204), "HEADING", lab, dim)
            _spaced(d, (350, 204), "TIME", lab, dim)
        return img

    def _fmt(self, st: dict, speed_max: float) -> dict:
        sp = st["speed"] * (2.23694 if self.imperial else 3.6)
        alt = st["alt"] * (3.28084 if self.imperial else 1.0)
        vs = st["vspeed"] * (3.28084 if self.imperial else 1.0)
        dist = st["dist"] * (3.28084 if self.imperial else 1.0)
        if self.imperial:
            dist_s = f"{dist:.0f} ft" if dist < 1000 else f"{dist / 5280:.2f} mi"
        else:
            dist_s = f"{dist:.0f} m" if dist < 1000 else f"{dist / 1000:.2f} km"
        clock = st["clock"]
        clock_s = "--:--:--" if not np.isfinite(clock) else f"{int(clock // 3600) % 24:02d}:{int(clock // 60) % 60:02d}:{int(clock) % 60:02d}"
        return {"speed": sp, "speed_unit": "mph" if self.imperial else "km/h", "alt": alt, "alt_unit": "ft" if self.imperial else "m",
                "vs": vs, "vs_unit": "ft/s" if self.imperial else "m/s", "dist": dist_s, "clock": clock_s,
                "frac": min(1.0, st["speed"] * 3.6 / max(speed_max, 1.0))}

    def draw(self, tel: Telemetry, key: str, t: float, alpha: float, speed_max: float, badge: str = "") -> bytes:
        st = tel.at(t)
        v = self._fmt(st, speed_max)
        img = self._static.copy()
        d = ImageDraw.Draw(img, "RGBA")
        full = self.style == "full"
        acc = (*self.accent, 255)
        # speed
        big = font(118 if full else 104, True)
        txt = f"{v['speed']:.0f}"
        d.text((228 if full else 200, 40), txt, font=big, fill=WHITE + (255,), anchor="ra")
        d.text((240 if full else 212, 112 if full else 104), v["speed_unit"], font=font(30), fill=(255, 255, 255, 190))
        by = 182 if full else 168
        bw = 250 if full else 210
        d.rounded_rectangle((38, by, 38 + bw, by + 8), 4, fill=(255, 255, 255, 45))
        if v["frac"] > 0.01:
            d.rounded_rectangle((38, by, 38 + max(8, int(bw * v["frac"])), by + 8), 4, fill=acc)
        # altitude + climb rate
        ax = 350 if full else 270
        d.text((ax, 46), f"{v['alt']:.0f}", font=font(92 if full else 80, True), fill=WHITE + (255,))
        w = d.textlength(f"{v['alt']:.0f}", font=font(92 if full else 80, True))
        d.text((ax + w + 8, 104 if full else 96), v["alt_unit"], font=font(30), fill=(255, 255, 255, 190))
        vy = 158 if full else 140
        vs = v["vs"]
        if abs(vs) >= 0.15:
            up = vs > 0
            tri = [(ax + 12, vy + 4), (ax, vy + 26), (ax + 24, vy + 26)] if up else [(ax, vy + 4), (ax + 24, vy + 4), (ax + 12, vy + 26)]
            d.polygon(tri, fill=acc)
        else:
            d.rounded_rectangle((ax, vy + 13, ax + 24, vy + 17), 2, fill=(255, 255, 255, 150))
        d.text((ax + 36, vy), f"{abs(vs):.1f} {v['vs_unit']}", font=font(30, True), fill=WHITE + (235,))
        if full:
            d.text((38, 226), v["dist"], font=font(38, True), fill=WHITE + (255,))
            d.text((190, 226), f"{st['heading']:03.0f}\u00b0 {compass(st['heading'])}", font=font(38, True), fill=WHITE + (255,))
            d.text((350, 226), v["clock"], font=font(38, True), fill=WHITE + (255,))
            if self.show_map:
                mp = self._map_for(key, tel)
                m = mp.base.copy()
                md = ImageDraw.Draw(m, "RGBA")
                i = min(int(t * 10), len(mp.pts) - 1)
                step = max(1, i // 300)
                trail = [tuple(p) for p in mp.pts[: i + 1 : step]] + [tuple(mp.pts[i])]
                if len(trail) > 1:
                    md.line(trail, fill=acc, width=5, joint="curve")
                px, py = mp.pts[i]
                for r, a in ((16, 40), (11, 90)):
                    md.ellipse((px - r, py - r, px + r, py + r), fill=(*self.accent, a))
                md.ellipse((px - 6, py - 6, px + 6, py + 6), fill=WHITE + (255,), outline=acc, width=2)
                # heading tick on the dot
                a = math.radians(st["heading"])
                md.line([(px, py), (px + 22 * math.sin(a), py - 22 * math.cos(a))], fill=WHITE + (230,), width=3)
                img.alpha_composite(m, (self.W - 230 - 30, 36))
        canvas = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        canvas.alpha_composite(img, (0, self.TOP))
        if badge:
            cd = ImageDraw.Draw(canvas, "RGBA")
            bf = font(24, True)
            tw = cd.textlength(badge, font=bf)
            cd.rounded_rectangle((0, 6, tw + 46, 46), 20, fill=(*self.accent, 255))
            cd.ellipse((17, 21, 27, 31), fill=(20, 20, 20, 255))
            cd.text((38, 26), badge, font=bf, fill=(20, 20, 20, 255), anchor="lm")
        if alpha < 0.999:
            lut = [int(i * alpha) for i in range(256)]
            canvas.putalpha(canvas.getchannel("A").point(lut))
        return canvas.tobytes()

    # ------------------------------------------------------------------ frame stream
    def frames(self, cuts: list, clips_by_path: dict, total_t: float) -> Iterator[bytes]:
        blank = bytes(self.W * self.H * 4)
        starts = [c.out_start for c in cuts]
        n = int(math.ceil(total_t * self.fps))
        smax = {}
        for k in range(n):
            t_out = k / self.fps
            i = max(0, bisect.bisect_right(starts, t_out) - 1)
            c = cuts[i]
            end = c.out_start + c.out_dur
            tel = self.telemetry(c.window["clip_path"], getattr(clips_by_path.get(c.window["clip_path"]), "srt_path", ""))
            if tel is None or t_out >= end or (self.mode == "moments" and not c.window.get("hud")):
                yield blank
                continue
            a = 1.0
            if self.fade > 0:
                a = max(0.0, min(1.0, (t_out - c.out_start) / self.fade, (end - t_out) / self.fade))
            key = c.window["clip_path"]
            if key not in smax:
                smax[key] = max(self.speed_max, float(tel.speed.max()) * 3.6 * 1.1)
            src_t = c.src_start + (t_out - c.out_start) * c.speed
            badge = c.window.get("hud_kind", "") if self.mode == "moments" else ""
            yield self.draw(tel, key, src_t, a, smax[key], badge)

    def summaries(self, cuts: list) -> dict:
        out = {}
        for c in cuts:
            p = c.window["clip_path"]
            tel = self._tele.get(p)
            if tel is not None and p not in out:
                out[Path(p).name] = tel.summary()
        return out
