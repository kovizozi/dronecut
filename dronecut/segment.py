"""Stage 3: scene detection (Content + Adaptive) and candidate windows of 2-6 s."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Cfg
from .utils import log


@dataclass
class Window:
    id: str
    clip_key: str
    clip_path: str
    start: float
    end: float
    scene_start: float
    scene_end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


def detect_cuts(proxy: Path, cfg: Cfg) -> list[float]:
    """Cut times (s) from ContentDetector + AdaptiveDetector, merged within a tolerance."""
    from scenedetect import AdaptiveDetector, ContentDetector, detect

    s = cfg.segment
    cuts: list[float] = []
    for det in (ContentDetector(threshold=s.content_threshold), AdaptiveDetector(adaptive_threshold=s.adaptive_threshold)):
        try:
            scenes = detect(str(proxy), det, show_progress=False)
        except Exception as e:  # noqa: BLE001
            log.warning("scene detector %s failed on %s: %s", type(det).__name__, proxy.name, e)
            continue
        cuts += [sc[0].get_seconds() for sc in scenes[1:]]
    cuts.sort()
    merged: list[float] = []
    for c in cuts:
        if not merged or c - merged[-1] > s.merge_cut_tol_s:
            merged.append(c)
    return merged


def make_windows(clip_key: str, clip_path: str, duration: float, cuts: list[float], cfg: Cfg) -> list[Window]:
    s = cfg.segment
    bounds = [0.0]
    for c in cuts:
        if c - bounds[-1] >= s.min_scene_len_s and duration - c >= s.min_scene_len_s:
            bounds.append(c)
    bounds.append(duration)
    wins: list[Window] = []
    n = 0
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        length = b - a
        if length < s.min_window_s:
            continue
        if length <= s.max_window_s:
            spans = [(a, b)]
        else:
            w = float(s.window_s)
            step = max(0.5, w * (1.0 - float(s.overlap)))
            spans = []
            t = a
            while t + w <= b + 1e-6:
                spans.append((t, t + w))
                t += step
            if spans and b - spans[-1][1] > 1.0:  # cover the tail
                spans.append((b - w, b))
        for st, en in spans:
            wins.append(Window(f"{clip_key}:{n:03d}", clip_key, clip_path, round(st, 3), round(en, 3), a, b))
            n += 1
    return wins
