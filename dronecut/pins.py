"""User pins: parts of the footage that MUST be in the edit (include) or must never be used (exclude).

A pin is ``clip @ start-end`` where *clip* is any part of the file name (``0010`` or ``DJI_..._0010_D.MP4``) and the
times are seconds or ``m:ss``. ``clip`` alone pins the whole clip. Pins come from ``--include`` / ``--exclude``
and/or a ``must_have.json`` written by the buttons in ``report.html``::

    {"include": [{"clip": "DJI_20261004140316_0010_D.MP4", "start": 45.0, "end": 51.0}],
     "exclude": [{"clip": "0015", "start": 0, "end": 20}]}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .utils import log


@dataclass
class Pin:
    clip: str
    start: float | None = None  # None = from the start of the clip
    end: float | None = None  # None = to the end of the clip

    def label(self) -> str:
        if self.start is None:
            return self.clip
        return f"{self.clip}@{self.start:.1f}-{self.end:.1f}"


@dataclass
class Pins:
    include: list[Pin] = field(default_factory=list)
    exclude: list[Pin] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.include or self.exclude)

    def to_dict(self) -> dict:
        def conv(items: list[Pin]) -> list[dict]:
            return [{"clip": p.clip, "start": p.start, "end": p.end} for p in items]

        return {"include": conv(self.include), "exclude": conv(self.exclude)}


def _t(s: str) -> float:
    s = s.strip()
    if ":" in s:
        m, sec = s.rsplit(":", 1)
        return int(m) * 60 + float(sec)
    return float(s)


def parse_spec(spec: str, default_len: float = 6.0) -> Pin:
    """``name@45-60`` | ``name@1:05-1:20`` | ``name@45`` (a default-length window from 45 s) | ``name``."""
    name, sep, rng = spec.partition("@")
    name = name.strip()
    if not name:
        raise ValueError(f"empty clip name in pin {spec!r}")
    if not sep:
        return Pin(name)
    m = re.fullmatch(r"\s*([\d:.]+)\s*(?:-\s*([\d:.]+))?\s*", rng)
    if not m:
        raise ValueError(f"cannot read time range in {spec!r} (use name@start-end, e.g. 0010@45-60)")
    a = _t(m.group(1))
    b = _t(m.group(2)) if m.group(2) else a + default_len
    if b <= a:
        raise ValueError(f"end must be after start in {spec!r}")
    return Pin(name, a, b)


def load_file(path: str | Path) -> Pins:
    d = json.loads(Path(path).read_text(encoding="utf-8"))

    def conv(items: list[dict]) -> list[Pin]:
        return [Pin(str(i["clip"]), None if i.get("start") is None else float(i["start"]),
                    None if i.get("end") is None else float(i["end"])) for i in items]

    return Pins(conv(d.get("include", [])), conv(d.get("exclude", [])))


def build_pins(file: str | Path | None, include: list[str] | None, exclude: list[str] | None) -> Pins:
    p = load_file(file) if file else Pins()
    p.include += [parse_spec(s) for s in include or []]
    p.exclude += [parse_spec(s) for s in exclude or []]
    return p


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _matches(pin: Pin, w: dict) -> bool:
    return pin.clip.lower() in w["clip_name"].lower()


def _range(pin: Pin, ws: list[dict]) -> tuple[float, float]:
    a = 0.0 if pin.start is None else pin.start
    b = max(w["end"] for w in ws) if pin.end is None else pin.end
    return a, b


def _choose_windows(ws: list[dict], a: float, b: float, max_window: float) -> list[dict]:
    """Windows of one clip that best cover [a, b]: the single best one for short ranges, a tiling for long ones."""
    cand = [w for w in ws if _overlap(w["start"], w["end"], a, b) > 0]
    if not cand:
        return []
    if b - a <= max_window + 1e-6:
        def iou(w: dict) -> float:
            inter = _overlap(w["start"], w["end"], a, b)
            return inter / ((w["end"] - w["start"]) + (b - a) - inter)

        return [max(cand, key=lambda w: (iou(w), w.get("score", 0.0)))]
    out: list[dict] = []
    last_end = -1e9
    for w in sorted(cand, key=lambda w: w["start"]):
        wl = w["end"] - w["start"]
        if _overlap(w["start"], w["end"], a, b) >= 0.6 * wl and w["start"] >= last_end - 0.25 * wl:
            out.append(w)
            last_end = w["end"]
    return out


def apply_pins(windows: list[dict], pins: Pins, max_window: float) -> None:
    """Mark windows: exclude -> reason ``excluded``; include -> ``pinned`` (quality rejections are overridden)."""
    for w in windows:
        w.pop("pinned", None)
    for pin in pins.exclude:
        hit = [w for w in windows if _matches(pin, w)]
        if not hit:
            log.warning("exclude %s: no such clip", pin.label())
            continue
        a, b = _range(pin, hit)
        for w in hit:
            wl = w["end"] - w["start"]
            if _overlap(w["start"], w["end"], a, b) >= 0.3 * wl and "excluded" not in w["reasons"]:
                w["reasons"].append("excluded")
    for pin in pins.include:
        hit = [w for w in windows if _matches(pin, w)]
        if not hit:
            log.warning("include %s: no such clip - check the name", pin.label())
            continue
        clips = {w["clip_key"] for w in hit}
        n = 0
        for key in clips:
            ws = [w for w in hit if w["clip_key"] == key]
            a, b = _range(pin, ws)
            for w in _choose_windows(ws, a, b, max_window):
                if w["reasons"]:
                    w["overridden"] = list(w["reasons"])
                    w["reasons"] = []
                w["pinned"] = True
                w["pin_range"] = [max(a, w["start"]), min(b, w["end"])]
                n += 1
        if n == 0:
            log.warning("include %s: nothing in that time range", pin.label())
        else:
            log.info("must-have %s -> %d window(s)", pin.label(), n)
