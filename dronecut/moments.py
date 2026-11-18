"""Pick the few moments where the telemetry HUD is worth showing: top speed, highest point, fast climb/dive,
farthest from take-off, sustained fast flight.

Each moment becomes a *long* shot (``hud.min_s`` .. ``hud.max_s`` seconds, a whole number of bars) taken from
footage that already passed the quality filter, so the HUD stays up long enough to read and fades only once in and once out.
"""
from __future__ import annotations

import math

import numpy as np

from .config import Cfg
from .telemetry import GRID_HZ, Telemetry

# label, metric key, floor (SI) below which the moment is not interesting enough
KINDS = [
    ("TOP SPEED", "speed_peak"),
    ("HIGHEST POINT", "alt_max"),
    ("FAST CLIMB", "vs_max"),
    ("FARTHEST OUT", "dist_max"),
    ("FAST FLIGHT", "speed_mean"),
]


def _floors(cfg: Cfg) -> dict[str, float]:
    v = float(cfg.hud.min_speed_kmh) / 3.6
    return {"speed_peak": v, "speed_mean": v * 0.8, "alt_max": 15.0, "vs_max": 1.5, "dist_max": 40.0}


def _merge(ranges: list[tuple[float, float]], tol: float = 0.05) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + tol:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def moment_length(bar_dur: float, cfg: Cfg) -> tuple[int, float]:
    bars = max(1, math.ceil(float(cfg.hud.min_s) / bar_dur - 1e-6))
    if bars * bar_dur > float(cfg.hud.max_s) + 1e-6:
        bars = max(1, int(float(cfg.hud.max_s) / bar_dur + 1e-6))
    return bars, bars * bar_dur


def _stats(tel: Telemetry, a: float, b: float) -> dict[str, float]:
    i0, i1 = int(a * GRID_HZ), max(int(b * GRID_HZ), int(a * GRID_HZ) + 2)
    sl = slice(i0, min(i1, len(tel.t)))
    return {"speed_peak": float(tel.speed[sl].max()), "speed_mean": float(tel.speed[sl].mean()), "alt_max": float(tel.alt[sl].max()),
            "vs_max": float(np.abs(tel.vspeed[sl]).max()), "dist_max": float(tel.dist[sl].max())}


def find_moments(ok: list[dict], tels: dict[str, Telemetry], bar_dur: float, cfg: Cfg, avoid: list[dict] | None = None) -> list[dict]:
    """Return pseudo windows (``hud=True``) for up to ``hud.moments`` interesting moments, best first."""
    n_want = int(cfg.hud.moments)
    if n_want <= 0 or not tels:
        return []
    bars, L = moment_length(bar_dur, cfg)
    floors = _floors(cfg)
    by_clip: dict[str, list[dict]] = {}
    for w in ok:
        if w["clip_path"] in tels:
            by_clip.setdefault(w["clip_path"], []).append(w)
    cands: list[dict] = []
    for path, ws in by_clip.items():
        tel = tels[path]
        for a, b in _merge([(w["start"], w["end"]) for w in ws]):
            if b - a < L - 1e-6:
                continue
            for s in np.arange(a, b - L + 1e-6, 0.5):
                if (s + L) * GRID_HZ >= len(tel.t):
                    break
                c = {"path": path, "start": float(s), "end": float(s + L), **_stats(tel, s, s + L)}
                cands.append(c)
    if not cands:
        return []
    chosen: list[dict] = []

    def clash(c: dict) -> bool:
        for o in [*chosen, *[{"path": w["clip_path"], "start": w["start"], "end": w["end"]} for w in avoid or []]]:
            if o["path"] == c["path"] and c["start"] < o["end"] + 3.0 and o["start"] < c["end"] + 3.0:
                return True
        return False

    for strict in (True, False):  # first try for variety (different clips), then allow a second moment per clip
        for _round in range(n_want):
            for label, key in KINDS:
                if len(chosen) >= n_want:
                    break
                if any(c["kind"] == label for c in chosen):
                    continue
                pool = [c for c in cands if c[key] >= floors[key] and not clash(c) and (not strict or all(c["path"] != o["path"] for o in chosen))]
                if pool:
                    best = max(pool, key=lambda c: (c[key], -c["start"]))
                    chosen.append({**best, "kind": label, "metric": key})
        if len(chosen) >= n_want:
            break
    out: list[dict] = []
    for i, c in enumerate(chosen):
        base = max(by_clip[c["path"]], key=lambda w: max(0.0, min(w["end"], c["end"]) - max(w["start"], c["start"])))
        m = dict(base)
        m.update({"id": f"{base['clip_key']}:hud{i}", "start": c["start"], "end": c["end"], "hud": True, "hud_kind": c["kind"],
                  "hud_bars": bars, "reasons": [], "moment": {k: round(c[k], 2) for k in ("speed_peak", "speed_mean", "alt_max", "vs_max", "dist_max")}})
        m.pop("pinned", None)
        out.append(m)
    return sorted(out, key=lambda m: (m["clip_name"], m["start"]))
