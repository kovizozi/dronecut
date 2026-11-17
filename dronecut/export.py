"""Stage 9a: editable timeline for DaVinci Resolve (FCPXML 1.9) referencing the ORIGINAL media."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from .music import MusicPlan
from .render import out_size, std_fps
from .select import Cut


def _t(frames: int, fd: Fraction) -> str:
    """Rational FCPXML time for a frame count on a grid with frame duration ``fd``."""
    v = Fraction(frames) * fd
    return "0s" if v == 0 else f"{v.numerator}/{v.denominator}s" if v.denominator != 1 else f"{v.numerator}s"


def _frames(sec: float, fps: Fraction) -> int:
    return int(round(sec * float(fps)))


def write_fcpxml(path: Path, cuts: list[Cut], plan: MusicPlan, clips_by_path: dict, fps: Fraction, res: int,
                 music_path: str | None = None, name: str = "dronecut highlight") -> Path:
    W, H = out_size(res)
    fd = 1 / fps
    root = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(root, "resources")
    ET.SubElement(resources, "format", id="r1", name=f"FFVideoFormat{H}p", frameDuration=_t(1, fd), width=str(W), height=str(H),
                  colorSpace="1-1-1 (Rec. 709)")
    fmt_ids: dict[tuple, str] = {}
    asset_ids: dict[str, str] = {}
    n = 2
    for c in cuts:
        p = c.window["clip_path"]
        if p in asset_ids:
            continue
        info = clips_by_path[p]
        cfps = std_fps(info.fps)
        key = (info.width, info.height, cfps)
        if key not in fmt_ids:
            fid = f"r{n}"
            n += 1
            fmt_ids[key] = fid
            ET.SubElement(resources, "format", id=fid, name=f"FFVideoFormat{info.height}p{float(cfps):g}", frameDuration=_t(1, 1 / cfps),
                          width=str(info.width), height=str(info.height), colorSpace="1-1-1 (Rec. 709)")
        aid = f"r{n}"
        n += 1
        asset_ids[p] = aid
        a = ET.SubElement(resources, "asset", id=aid, name=Path(p).stem, start="0s",
                          duration=_t(_frames(info.duration, cfps), 1 / cfps), hasVideo="1", format=fmt_ids[key],
                          hasAudio="1" if info.has_audio else "0", videoSources="1")
        if info.has_audio:
            a.set("audioSources", "1")
            a.set("audioChannels", "2")
            a.set("audioRate", "48000")
        ET.SubElement(a, "media-rep", kind="original-media", src=Path(p).resolve().as_uri())
    music_id = None
    mfile = music_path or plan.track.path
    if mfile:
        music_id = f"r{n}"
        n += 1
        ma = ET.SubElement(resources, "asset", id=music_id, name=Path(mfile).stem, start="0s",
                           duration=_t(_frames(plan.track.duration, fps), fd), hasAudio="1", audioSources="1", audioChannels="2", audioRate="48000")
        ET.SubElement(ma, "media-rep", kind="original-media", src=Path(mfile).resolve().as_uri())

    total = sum(_frames(c.out_dur, fps) for c in cuts)
    lib = ET.SubElement(root, "library")
    ev = ET.SubElement(lib, "event", name="dronecut")
    proj = ET.SubElement(ev, "project", name=name)
    seq = ET.SubElement(proj, "sequence", format="r1", duration=_t(total, fd), tcStart="0s", tcFormat="NDF", audioLayout="stereo", audioRate="48k")
    spine = ET.SubElement(seq, "spine")
    pos = 0
    first_clip = None
    bounds = [0]
    for c in cuts:
        bounds.append(bounds[-1] + _frames(c.out_dur, fps))
    for i, c in enumerate(cuts):
        p = c.window["clip_path"]
        info = clips_by_path[p]
        cfps = std_fps(info.fps)
        dur_f = bounds[i + 1] - bounds[i]
        src_f = _frames(c.src_start, cfps)
        ac = ET.SubElement(spine, "asset-clip", ref=asset_ids[p], name=f"{Path(p).stem} @{c.src_start:.1f}s", offset=_t(pos, fd),
                           start=_t(src_f, 1 / cfps), duration=_t(dur_f, fd), format=fmt_ids[(info.width, info.height, cfps)], tcFormat="NDF")
        if abs(c.speed - 1.0) > 1e-6:
            tm = ET.SubElement(ac, "timeMap")
            ET.SubElement(tm, "timept", time="0s", value="0s", interp="linear")
            ET.SubElement(tm, "timept", time=_t(dur_f, fd), value=_t(_frames(c.src_dur, cfps), 1 / cfps), interp="linear")
        if first_clip is None:
            first_clip = ac
        pos += dur_f
    if music_id and first_clip is not None:
        ET.SubElement(first_clip, "asset-clip", ref=music_id, lane="-1", offset=_t(_frames(plan.start, fps), fd), name=Path(mfile).stem,
                      start=_t(_frames(plan.start, fps), fd), duration=_t(total, fd), audioRole="music")
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = ET.tostring(root, encoding="unicode")
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n\n' + xml, encoding="utf-8")
    return path


def validate_fcpxml(path: Path) -> list[str]:
    """Return a list of problems (empty = OK): well-formed XML, every ref resolves, source files exist."""
    problems: list[str] = []
    tree = ET.parse(path)  # raises on malformed XML
    root = tree.getroot()
    ids = {e.get("id"): e for e in root.iter() if e.get("id")}
    for e in root.iter("asset-clip"):
        r = e.get("ref")
        if r not in ids or ids[r].tag != "asset":
            problems.append(f"unresolved ref {r}")
    for e in root.iter("asset"):
        if e.get("format") and e.get("format") not in ids:
            problems.append(f"asset {e.get('id')} has unknown format")
        rep = e.find("media-rep")
        if rep is None:
            problems.append(f"asset {e.get('id')} has no media-rep")
        else:
            from urllib.parse import unquote, urlparse

            u = urlparse(rep.get("src", ""))
            f = unquote(u.path)
            if len(f) > 2 and f[0] == "/" and f[2] == ":":
                f = f[1:]
            if not Path(f).exists():
                problems.append(f"missing media {f}")
    seq = root.find(".//sequence")
    if seq is None or seq.find("spine") is None:
        problems.append("no sequence/spine")
    return problems
