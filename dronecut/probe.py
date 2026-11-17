"""Stage 1: ffprobe every input; parse DJI .SRT telemetry sidecars if present."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

from .utils import log, run, which

_TELEM = re.compile(r"\b(rel_alt|abs_alt|altitude|latitude|longitude|iso|shutter|fnum)\b\s*:\s*(-?[\d.]+)")


@dataclass
class ClipInfo:
    path: str
    key: str = ""
    width: int = 0  # display size (after rotation)
    height: int = 0
    fps: float = 0.0
    codec: str = ""
    bitrate: int = 0
    duration: float = 0.0
    has_audio: bool = False
    rotation: int = 0
    pix_fmt: str = ""
    srt: list[dict] = field(default_factory=list)  # sparse telemetry samples
    srt_path: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ClipInfo:
        return cls(**d)


def _fps(s: str) -> float:
    try:
        f = Fraction(s)
        return float(f) if f > 0 else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def parse_srt_telemetry(path: Path, max_samples: int = 400) -> list[dict]:
    """Parse DJI-style .SRT sidecars (``[altitude: 12.3] ...``). Best effort, else empty."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: list[dict] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [x for x in block.strip().splitlines() if x.strip()]
        if len(lines) < 3:
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", lines[1])
        if not m:
            continue
        t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000.0
        rec: dict = {"t": t}
        for k, v in _TELEM.findall(" ".join(lines[2:])):
            try:
                rec[k] = float(v)
            except ValueError:
                pass
        if len(rec) > 1:
            out.append(rec)
    step = max(1, len(out) // max_samples)
    return out[::step]


def probe_file(path: str | Path) -> ClipInfo:
    p = Path(path)
    cp = run([which("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(p)])
    d = json.loads(cp.stdout)
    vs = [
        s
        for s in d.get("streams", [])
        if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) == 0
    ]
    if not vs:
        raise ValueError("no video stream")
    v = vs[0]
    rot = 0
    if "rotate" in v.get("tags", {}):
        rot = int(float(v["tags"]["rotate"]))
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    rot %= 360
    w, h = int(v["width"]), int(v["height"])
    if rot in (90, 270):
        w, h = h, w
    fmt = d.get("format", {})
    dur = float(v.get("duration") or fmt.get("duration") or 0.0)
    fps = _fps(v.get("avg_frame_rate", "0")) or _fps(v.get("r_frame_rate", "0"))
    info = ClipInfo(
        path=str(p.resolve()),
        width=w,
        height=h,
        fps=fps,
        codec=v.get("codec_name", ""),
        bitrate=int(v.get("bit_rate") or fmt.get("bit_rate") or 0),
        duration=dur,
        has_audio=any(s.get("codec_type") == "audio" for s in d["streams"]),
        rotation=rot,
        pix_fmt=v.get("pix_fmt", ""),
    )
    for ext in (".SRT", ".srt"):
        sc = p.with_suffix(ext)
        if sc.exists():
            info.srt_path = str(sc.resolve())
            info.srt = parse_srt_telemetry(sc)
            if info.srt:
                log.debug("telemetry: %d samples from %s", len(info.srt), sc.name)
            break
    if info.duration <= 0 or info.fps <= 0 or info.width <= 0:
        raise ValueError("invalid duration/fps/size")
    return info
