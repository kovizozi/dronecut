from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronecut import pipeline, telemetry
from dronecut.config import load_config
from dronecut.hud import Hud

ROOT = Path(__file__).resolve().parents[1]


def _make_srt(path: Path, dur: float, **kw) -> None:
    spec = importlib.util.spec_from_file_location("make_test_media", ROOT / "scripts" / "make_test_media.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.write_srt(path, dur, **kw)


def test_telemetry_derives_speed_heading_climb(tmp_path):
    srt = tmp_path / "a.SRT"
    _make_srt(srt, 10.0, east=10.0, north=0.0, climb=1.0)
    tel = telemetry.parse_srt(srt)
    assert tel is not None and tel.has_gps
    mid = tel.at(5.0)
    assert abs(mid["speed"] * 3.6 - 36.0) < 2.0  # 10 m/s eastward
    assert abs(mid["heading"] - 90.0) < 5.0 and telemetry.compass(mid["heading"]) == "E"
    assert abs(mid["vspeed"] - 1.0) < 0.2
    assert tel.at(9.0)["dist"] > tel.at(3.0)["dist"] > 20
    assert mid["clock"] > 14 * 3600  # recording time of day from the SRT timestamp
    s = tel.summary()
    assert 30 < s["max_speed_kmh"] < 45 and s["max_alt_m"] > 9


def test_srt_without_gps_is_ignored(tmp_path):
    f = tmp_path / "b.SRT"
    blocks = [f"{i + 1}\n00:00:0{i // 30},{(i % 30) * 33:03d} --> 00:00:0{i // 30},{(i % 30) * 33 + 33:03d}\n"
              "[latitude: 0.000000] [longitude: 0.000000] [rel_alt: 1.000 abs_alt: 1.000]\n" for i in range(60)]
    f.write_text("\n".join(blocks), encoding="utf-8")
    tel = telemetry.parse_srt(f)
    assert tel is None or not tel.has_gps
    assert telemetry.parse_srt(tmp_path / "missing.SRT") is None


@pytest.mark.parametrize("style,imperial,pos", [("full", False, "bottom-left"), ("minimal", True, "top-right")])
def test_hud_frame_geometry_and_fade(tmp_path, style, imperial, pos):
    srt = tmp_path / "c.SRT"
    _make_srt(srt, 8.0)
    tel = telemetry.parse_srt(srt)
    cfg = load_config(None, {"hud": {"enabled": True, "style": style, "units": "imperial" if imperial else "metric", "position": pos}})
    hud = Hud(cfg, 1080, 30.0)
    raw = hud.draw(tel, "k", 4.0, 1.0, 60.0)
    assert len(raw) == hud.W * hud.H * 4
    arr = np.frombuffer(raw, np.uint8).reshape(hud.H, hud.W, 4)
    assert arr[..., 3].max() > 200 and (arr[..., 3] > 0).mean() > 0.5  # panel + text present
    faded = np.frombuffer(hud.draw(tel, "k", 4.0, 0.3, 60.0), np.uint8).reshape(hud.H, hud.W, 4)
    assert faded[..., 3].max() <= 0.31 * 255 + 1
    s, x, y = hud.overlay_args(1920, 1080)
    assert s == 1.0 and (("main_h-overlay_h" in y) == ("bottom" in pos))
    assert Hud(cfg, 2160, 30.0).overlay_args(3840, 2160)[0] == 2.0  # scales with the output height


def _frame_at(mp4: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    cap.release()
    assert ok
    return f


def test_hud_is_composited_into_render(media, music_file, tmp_path):
    base = {"ingest": {"prefer_nvenc": False, "workers": 2}, "tech": {"workers": 2}, "render": {"nvenc": "off", "crf": 28},
            "paths": {"cache_dir": str(tmp_path / "cache")}}
    cfg_off = load_config(None, base)
    off = pipeline.build(cfg_off, str(music_file), 30.0, 360, False, 5, tmp_path / "off", footage=media, fast=True)
    cfg_on = load_config(None, {**base, "hud": {"enabled": True, "fade_s": 0.0}})
    on = pipeline.build(cfg_on, str(music_file), 30.0, 360, False, 5, tmp_path / "on", footage=media, fast=True)
    assert [c.window["id"] for c in on["selection"].cuts] == [c.window["id"] for c in off["selection"].cuts]  # same edit
    data = json.loads((tmp_path / "on" / "build.json").read_text())
    assert data["telemetry"], "no clip with GPS made it into the edit"
    hit = [c for c in on["selection"].cuts if c.window["clip_name"] in ("good_pan_1080.mp4", "good_pan_4k.mp4")]
    assert hit, "expected a clip with an SRT in the edit"
    c = hit[0]
    t = c.out_start + c.out_dur / 2
    a, b = _frame_at(on["mp4"], t), _frame_at(off["mp4"], t)
    h, w = a.shape[:2]
    region = (slice(int(h * 0.55), h), slice(0, int(w * 0.55)))  # bottom-left panel
    assert np.abs(a[region].astype(int) - b[region].astype(int)).mean() > 6
    other = (slice(0, int(h * 0.3)), slice(int(w * 0.5), w))  # top-right stays untouched
    assert np.abs(a[other].astype(int) - b[other].astype(int)).mean() < 3
