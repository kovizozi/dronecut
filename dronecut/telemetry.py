"""Flight telemetry from DJI ``.SRT`` sidecars: GPS track, altitude and *derived* speed / heading / distance.

DJI's SRT has latitude, longitude, rel_alt (and abs_alt, iso, shutter...) for every frame but no speed, so speed is
computed from the GPS track (smoothed), vertical speed from the altitude, distance from the first fix (take-off point).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->")
_CLOCK = re.compile(r"\d{4}-\d\d-\d\d[ T](\d\d):(\d\d):(\d\d)(?:[.,](\d+))?")
_KV = re.compile(r"\b(latitude|longitude|rel_alt|abs_alt|altitude)\s*:\s*(-?[\d.]+)")
_GPS_OLD = re.compile(r"GPS\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")
_HEIGHT_OLD = re.compile(r"\bH\s*(-?[\d.]+)\s*m")
R_EARTH = 6371000.0
GRID_HZ = 10.0
COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


@dataclass
class Telemetry:
    t: np.ndarray  # seconds into the clip, uniform GRID_HZ grid
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray  # m above take-off (rel_alt)
    x: np.ndarray  # m east of the first fix
    y: np.ndarray  # m north of the first fix
    speed: np.ndarray  # horizontal m/s
    vspeed: np.ndarray  # m/s, + = climbing
    dist: np.ndarray  # m from the first fix
    heading: np.ndarray  # deg, 0 = north
    clock: np.ndarray  # seconds since midnight (recording time of day), NaN if unknown
    has_gps: bool

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    def at(self, t: float) -> dict:
        t = float(np.clip(t, 0.0, self.duration))
        i = lambda a: float(np.interp(t, self.t, a))  # noqa: E731
        out = {"t": t, "alt": i(self.alt), "speed": i(self.speed), "vspeed": i(self.vspeed), "dist": i(self.dist),
               "heading": float(self.heading[min(int(t * GRID_HZ), len(self.t) - 1)]), "x": i(self.x), "y": i(self.y),
               "lat": i(self.lat), "lon": i(self.lon)}
        out["clock"] = i(self.clock) if np.isfinite(self.clock).any() else float("nan")
        return out

    def summary(self) -> dict:
        return {"max_speed_kmh": float(self.speed.max() * 3.6), "max_alt_m": float(self.alt.max()),
                "max_dist_m": float(self.dist.max()), "path_m": float(np.hypot(np.diff(self.x), np.diff(self.y)).sum()),
                "duration_s": self.duration}


def _smooth(a: np.ndarray, seconds: float) -> np.ndarray:
    k = max(1, int(round(seconds * GRID_HZ)))
    if k <= 1 or len(a) < 3:
        return a
    pad = np.pad(a, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def parse_srt(path: str | Path) -> Telemetry | None:
    """Parse a DJI SRT (new ``[latitude: ..]`` and old ``GPS (lon, lat, alt)`` styles). None if unusable."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    ts, lat, lon, alt, clk = [], [], [], [], []
    for block in re.split(r"\n\s*\n", text):
        m = _TIME.search(block)
        if not m:
            continue
        kv = {k: float(v) for k, v in _KV.findall(block)}
        la, lo = kv.get("latitude"), kv.get("longitude")
        h = kv.get("rel_alt", kv.get("altitude"))
        if la is None:
            g = _GPS_OLD.search(block)
            if g:
                lo, la = float(g.group(1)), float(g.group(2))
                h = h if h is not None else float(g.group(3))
        if h is None:
            hm = _HEIGHT_OLD.search(block)
            h = float(hm.group(1)) if hm else None
        if la is None or lo is None:
            continue
        c = _CLOCK.search(block)
        ts.append(int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / (10 ** len(m[4])))
        lat.append(la)
        lon.append(lo)
        alt.append(h if h is not None else 0.0)
        clk.append(int(c[1]) * 3600 + int(c[2]) * 60 + int(c[3]) + (int(c[4]) / 10 ** len(c[4]) if c[4] else 0.0) if c else np.nan)
    if len(ts) < 5:
        return None
    t = np.asarray(ts)
    order = np.argsort(t)
    t, lat_a, lon_a, alt_a, clk_a = (np.asarray(v, float)[order] for v in (ts, lat, lon, alt, clk))
    grid = np.arange(0.0, t[-1], 1.0 / GRID_HZ)
    if len(grid) < 5:
        return None
    lat_g, lon_g = np.interp(grid, t, lat_a), np.interp(grid, t, lon_a)
    alt_g = _smooth(np.interp(grid, t, alt_a), 0.4)
    clock_g = np.interp(grid, t, clk_a) if np.isfinite(clk_a).all() else np.full(len(grid), np.nan)
    has_gps = bool(np.abs(lat_g).max() > 1e-4 or np.abs(lon_g).max() > 1e-4)  # (0, 0) = no GPS fix
    lat0, lon0 = lat_g[0], lon_g[0]
    x = _smooth(np.radians(lon_g - lon0) * R_EARTH * math.cos(math.radians(lat0)), 0.8)
    y = _smooth(np.radians(lat_g - lat0) * R_EARTH, 0.8)
    vx, vy = np.gradient(x, 1 / GRID_HZ), np.gradient(y, 1 / GRID_HZ)
    speed = _smooth(np.hypot(vx, vy), 0.6)
    speed[speed < 0.35] = 0.0
    vspeed = _smooth(np.gradient(alt_g, 1 / GRID_HZ), 1.0)
    vspeed[np.abs(vspeed) < 0.15] = 0.0
    heading = np.degrees(np.arctan2(vx, vy)) % 360.0
    for i in range(len(heading)):  # hold the last heading while (almost) standing still
        if speed[i] < 0.8:
            heading[i] = heading[i - 1] if i else 0.0
    return Telemetry(grid, lat_g, lon_g, alt_g, x, y, speed, vspeed, np.hypot(x, y), heading, clock_g, has_gps)


def compass(deg: float) -> str:
    return COMPASS[int((deg % 360) / 45.0 + 0.5) % 8]
