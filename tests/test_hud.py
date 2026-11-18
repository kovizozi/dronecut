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


def _fake_tel(dur=60.0, peak_speed_t=20.0, peak_alt_t=45.0) -> telemetry.Telemetry:
    """60 s flight: a speed bump (12 m/s) around 20 s, climbing to 60 m around 45 s."""
    t = np.arange(0, dur, 1 / telemetry.GRID_HZ)
    speed = 2.0 + 10.0 * np.exp(-((t - peak_speed_t) / 3.0) ** 2)
    alt = 60.0 * np.exp(-((t - peak_alt_t) / 12.0) ** 2) + 3.0
    x = np.cumsum(speed) / telemetry.GRID_HZ
    z = np.zeros_like(t)
    return telemetry.Telemetry(t, z + 46.0, z + 21.0, alt, x, z, speed, np.gradient(alt, 1 / telemetry.GRID_HZ), x.copy(),
                               z + 90.0, z + 50000.0, True)


def _ok_windows(path="/x/c.mp4", dur=60.0, step=3.0):
    return [{"id": f"k:{i}", "clip_key": "k", "clip_name": "c.mp4", "clip_path": path, "start": float(a), "end": float(a + 6.0),
             "scene_start": 0.0, "scene_end": dur, "score": 0.8, "tech": 0.8, "reasons": [], "emb": None, "wide": 0.5,
             "feat": {"zoom": 0.0, "shake": 0.1, "motion": 30.0}} for i, a in enumerate(np.arange(0, dur - 6.0 + 1e-6, step))]


def test_moments_are_long_whole_bars_around_the_peaks():
    from dronecut import moments

    cfg = load_config(None, {"hud": {"enabled": True, "moments": 2, "min_s": 6.0}})
    tel = _fake_tel()
    bar = 2.2857
    ms = moments.find_moments(_ok_windows(), {"/x/c.mp4": tel}, bar, cfg)
    assert {m["hud_kind"] for m in ms} == {"TOP SPEED", "HIGHEST POINT"}
    bars, L = moments.moment_length(bar, cfg)
    assert bars == 3 and abs(L - 3 * bar) < 1e-9
    top = next(m for m in ms if m["hud_kind"] == "TOP SPEED")
    high = next(m for m in ms if m["hud_kind"] == "HIGHEST POINT")
    for m in ms:
        assert abs((m["end"] - m["start"]) - L) < 1e-9 and m["hud"] and not m["reasons"]
    assert top["start"] < 20.0 < top["end"] and high["start"] < 45.0 < high["end"]  # the peak is inside the shot
    assert top["end"] <= high["start"] or high["end"] <= top["start"]  # no overlap
    # boring flight (hovering) -> no moments
    flat = _fake_tel()
    flat.speed[:] = 0.5
    flat.alt[:] = 4.0
    flat.vspeed[:] = 0.0
    flat.dist[:] = 3.0
    assert moments.find_moments(_ok_windows(), {"/x/c.mp4": flat}, bar, cfg) == []


def test_moment_shots_are_laid_out_full_length_and_hud_is_blank_elsewhere():
    from dronecut import select
    from dronecut.music import Track, make_plan

    cfg = load_config(None, {"hud": {"enabled": True, "moments": 2}})
    ws = _ok_windows()
    for w in ws:
        w["feat"] = {"sharp": 100, "luma": 120, "luma_p5": 90, "clip_low": 0, "clip_high": 0, "sky": 0, "occl": 0, "shake": 0.1, "jerk": 0.5,
                     "motion": 30, "chaos": 1, "zoom": 0.0, "zoom_abs": 0.0, "dist_start": 30, "dist_end": 30, "mean_b": 100, "mean_g": 100, "mean_r": 100}
    beats = np.arange(0, 110, 0.5714)  # 105 BPM
    plan = make_plan(Track("m.wav", 110.0, 105.0, beats, beats[::4], np.arange(0, 100, 0.1), np.full(1000, 0.5)), 40.0, cfg)
    sel = select.build_cuts(ws, plan, cfg, seed=1, tels={"/x/c.mp4": _fake_tel()})
    moments_cuts = [c for c in sel.cuts if c.window.get("hud")]
    assert len(moments_cuts) == 2
    bar = float(np.median(np.diff(plan.bar_times)))
    for c in moments_cuts:
        assert c.bars == 3 and abs(c.out_dur - 3 * bar) < 1e-3 and abs(c.src_dur - c.out_dur) < 1e-3  # whole long shot, not squeezed
    # HUD frames: non-blank only inside the moment cuts, one fade in and one fade out each
    hud = Hud(cfg, 1080, 30.0, tele={"/x/c.mp4": _fake_tel()})
    clips = {"/x/c.mp4": type("C", (), {"srt_path": ""})()}
    total = sel.cuts[-1].out_start + sel.cuts[-1].out_dur
    frames = list(hud.frames(sel.cuts, clips, total))
    alpha_max = [np.frombuffer(f, np.uint8).reshape(hud.H, hud.W, 4)[..., 3].max() for f in frames]
    for k, a in enumerate(alpha_max):
        t = k / hud.fps
        inside = [c for c in moments_cuts if c.out_start <= t < c.out_start + c.out_dur]
        if not inside:
            assert a == 0, f"HUD visible outside a moment at t={t:.2f}"
        elif min(t - inside[0].out_start, inside[0].out_start + inside[0].out_dur - t) > 0.15:
            assert a > 0, f"HUD missing inside a moment at t={t:.2f}"  # (the very first/last frame is mid-fade)
    for c in moments_cuts:  # mid-shot fully visible, edges faded
        mid = int((c.out_start + c.out_dur / 2) * hud.fps)
        assert alpha_max[mid] >= 250 and alpha_max[int(c.out_start * hud.fps)] < 120
    # legacy mode: every cut gets the HUD
    cfg_all = load_config(None, {"hud": {"enabled": True, "mode": "all"}})
    hud_all = Hud(cfg_all, 1080, 30.0, tele={"/x/c.mp4": _fake_tel()})
    n_vis = sum(1 for f in hud_all.frames(sel.cuts, clips, total) if np.frombuffer(f, np.uint8)[3::4].max() > 0)
    assert n_vis > 0.9 * len(frames)


def test_hud_only_appears_on_moment_shots_in_the_render(media, music_file, tmp_path):
    from dronecut import render

    base = {"ingest": {"prefer_nvenc": False, "workers": 2}, "tech": {"workers": 2}, "render": {"nvenc": "off", "crf": 28, "hwdecode": "off"},
            "paths": {"cache_dir": str(tmp_path / "cache")}, "hud": {"enabled": True, "fade_s": 0.0, "moments": 2, "min_s": 4.0, "min_speed_kmh": 5.0}}
    cfg = load_config(None, base)
    r = pipeline.build(cfg, str(music_file), 30.0, 360, False, 5, tmp_path / "on", footage=media, fast=True)
    data = json.loads((tmp_path / "on" / "build.json").read_text())
    assert data["telemetry"], "no clip with GPS made it into the edit"
    cuts = r["selection"].cuts
    hud_cuts = [c for c in cuts if c.window.get("hud")]
    assert hud_cuts, "expected at least one HUD moment"
    assert len(hud_cuts) < len(cuts)  # not on every cut
    an = pipeline.analyze(cfg, media, fast=True)
    clips = {c.path: c for c in an.clips}
    from dronecut.hud import Hud

    # same edit rendered once without the HUD: only the moment shots may differ
    off = render.render(cuts, r["plan"], clips, cfg, tmp_path / "nohud.mp4", res=360, music_path=str(music_file), hud=None)
    h = hud_cuts[0]
    a, b = _frame_at(r["mp4"], h.out_start + h.out_dur / 2), _frame_at(off.path, h.out_start + h.out_dur / 2)
    region = (slice(int(a.shape[0] * 0.55), a.shape[0]), slice(0, int(a.shape[1] * 0.55)))
    assert np.abs(a[region].astype(int) - b[region].astype(int)).mean() > 6
    plain = next(c for c in cuts if not c.window.get("hud"))
    t = plain.out_start + plain.out_dur / 2
    assert np.abs(_frame_at(r["mp4"], t).astype(int) - _frame_at(off.path, t).astype(int)).mean() < 2
    _ = Hud
