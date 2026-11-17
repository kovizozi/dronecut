from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronecut import export, music, pipeline, probe, score_tech, select
from dronecut.config import load_config
from dronecut.render import std_fps


@pytest.fixture(scope="module")
def analyzed(media, tmp_path_factory):
    """One shared fast analysis of the synthetic footage."""
    cache = tmp_path_factory.mktemp("cache")
    cfg = load_config(None, {
        "paths": {"cache_dir": str(cache)}, "ingest": {"prefer_nvenc": False, "workers": 2},
        "tech": {"workers": 2}, "render": {"nvenc": "off", "crf": 28},
    })
    an = pipeline.analyze(cfg, media, fast=True)
    return cfg, an


def test_probe_mixed_formats(media):
    a = probe.probe_file(media / "good_pan_1080.mp4")
    b = probe.probe_file(media / "good_pan_4k.mp4")
    assert (a.width, a.height) == (1920, 1080) and a.has_audio
    assert (b.width, b.height) == (3840, 2160) and not b.has_audio
    assert abs(a.fps - 30) < 0.01 and a.codec == "h264" and a.duration > 5


def test_probe_srt_parsing(tmp_path):
    srt = tmp_path / "x.SRT"
    srt.write_text("1\n00:00:00,000 --> 00:00:00,033\n<font size=\"28\">FrameCnt: 1</font>\n[iso: 100] [rel_alt: 12.3 abs_alt: 50.1] [latitude: 47.1]\n\n"
                   "2\n00:00:01,000 --> 00:00:01,033\n[rel_alt: 13.0]\n\n")
    s = probe.parse_srt_telemetry(srt)
    assert len(s) == 2 and s[0]["rel_alt"] == 12.3 and s[1]["t"] == 1.0
    assert probe.parse_srt_telemetry(tmp_path / "missing.srt") == []


def test_bad_file_does_not_kill_run(media, tmp_path, cfg):
    d = tmp_path / "f"
    d.mkdir()
    (d / "broken.mp4").write_bytes(b"not a video")
    import shutil

    shutil.copy(media / "good_pan_1080.mp4", d / "ok.mp4")
    an = pipeline.analyze(cfg, d, fast=True)
    assert len(an.clips) == 1 and any("broken" in k for k in an.failed)


def test_scene_detection_splits_multi_scene(analyzed):
    _, an = analyzed
    ms = [w for w in an.windows if w["clip_name"] == "multi_scene.mp4"]
    assert len({w["scene_start"] for w in ms}) >= 3  # three hard-cut shots found
    assert all(2.0 <= w["end"] - w["start"] <= 6.0 + 1e-6 for w in an.windows)


def test_junk_is_rejected_and_good_survives(analyzed):
    cfg, an = analyzed
    ok, bad = select.score_windows(an.windows, cfg)
    by: dict[str, list[dict]] = {}
    for w in an.windows:
        by.setdefault(w["clip_name"], []).append(w)
    for name, ws in by.items():
        if name.startswith("junk_"):
            assert all(w["reasons"] for w in ws), f"{name} not fully rejected: {[w['reasons'] for w in ws]}"
    reasons = {r for w in bad for r in w["reasons"]}
    assert {"blurry", "overexposed", "underexposed", "sky_only", "shaky"} <= reasons
    for name in ("good_pan_1080.mp4", "good_pan_4k.mp4", "good_orbit_1080.mp4"):
        assert any(not w["reasons"] for w in by[name]), name


def test_cache_is_reused(analyzed, media):
    cfg, an = analyzed
    import time

    t0 = time.time()
    an2 = pipeline.analyze(cfg, media, fast=True)
    assert len(an2.windows) == len(an.windows)
    assert time.time() - t0 < 20  # no proxy/flow recomputation


def test_bpm_and_beat_grid(music_file, cfg):
    tr = music.analyze_track(music_file, cfg)
    assert abs(tr.bpm - 120.0) <= 2.0
    # beats land on the true kick positions (every 0.5 s, 12 ms analysis latency)
    off = np.minimum(tr.beats % 0.5, 0.5 - tr.beats % 0.5)
    assert np.all(off < 0.03)
    # downbeat = accented kick every 4th beat (true bar = 2 s)
    assert np.allclose(np.diff(tr.downbeats), 2.0, atol=0.03)
    assert (tr.downbeats[0] % 2.0 < 0.03) or (2.0 - tr.downbeats[0] % 2.0 < 0.03)


def test_snap_to_beat():
    beats = np.arange(0, 10, 0.5)
    assert music.snap_to_beat(1.74, beats) == 1.5
    assert music.snap_to_beat(1.76, beats) == 2.0


def _fake_windows(n_clips=3, per=8):
    rng = np.random.default_rng(0)
    ws = []
    for c in range(n_clips):
        for i in range(per):
            e = rng.normal(size=16).astype(np.float32)
            e[c] += 6
            e /= np.linalg.norm(e)
            ws.append({"id": f"c{c}:{i}", "clip_key": f"c{c}", "clip_name": f"c{c}.mp4", "clip_path": f"/x/c{c}.mp4",
                       "start": i * 3.0, "end": i * 3.0 + 4.0, "score": 0.9 - 0.01 * i - 0.05 * c, "tech": 0.8, "emb": e,
                       "feat": {"zoom": 0.0, "shake": 0.1, "motion": 20.0}, "wide": 0.5})
    return ws


def test_selection_diversity_and_gap():
    import random

    cfg = load_config(None, {"select": {"min_gap_s": 6.0, "clusters": 3}})
    ws = _fake_windows()
    picks = select._pick_candidates(ws, 15.0, cfg, random.Random(1), 2.0)
    assert picks
    # same-clip picks respect the minimum gap
    for a in picks:
        for b in picks:
            if a is not b and a["clip_key"] == b["clip_key"]:
                assert a["start"] >= b["end"] + 6.0 or b["start"] >= a["end"] + 6.0
    # all three visual clusters represented
    assert len({w["cluster"] for w in picks}) == 3


def test_cuts_land_on_downbeats(analyzed, music_file):
    cfg, an = analyzed
    tr = music.analyze_track(music_file, cfg)
    plan = music.make_plan(tr, 40.0, cfg)
    sel = select.build_cuts(an.windows, plan, cfg, seed=3)
    assert sel.cuts
    for c in sel.cuts:
        t = plan.start + c.out_start
        assert np.min(np.abs(plan.bar_times - t)) < 1e-3
        assert abs(c.out_dur / (plan.bar_times[1] - plan.bar_times[0]) - c.bars) < 1e-3 or c.bars > 1
        w = c.window
        assert w["start"] - 1e-6 <= c.src_start and c.src_start + c.src_dur <= w["end"] + 1e-6
        assert not w["reasons"]
    # deterministic with the same seed
    sel2 = select.build_cuts(an.windows, plan, cfg, seed=3)
    assert [(c.window["id"], c.out_start) for c in sel.cuts] == [(c.window["id"], c.out_start) for c in sel2.cuts]


def test_fcpxml_valid_and_refs_resolve(analyzed, music_file, tmp_path):
    cfg, an = analyzed
    tr = music.analyze_track(music_file, cfg)
    plan = music.make_plan(tr, 30.0, cfg)
    sel = select.build_cuts(an.windows, plan, cfg, seed=1)
    clips = {c.path: c for c in an.clips}
    p = export.write_fcpxml(tmp_path / "t.fcpxml", sel.cuts, plan, clips, std_fps(30.0), 1080, str(music_file))
    assert export.validate_fcpxml(p) == []
    import xml.etree.ElementTree as ET

    root = ET.parse(p).getroot()
    assert root.get("version") == "1.9"
    ids = {e.get("id") for e in root.iter() if e.get("id")}
    refs = [e.get("ref") for e in root.iter("asset-clip")]
    assert refs and all(r in ids for r in refs)
    bad = tmp_path / "bad.fcpxml"
    bad.write_text(p.read_text().replace('ref="r', 'ref="zz', 1))
    assert any("unresolved" in x for x in export.validate_fcpxml(bad))


def test_end_to_end_fast(media, music_file, cfg, tmp_path):
    out = tmp_path / "out"
    r = pipeline.build(cfg, str(music_file), 40.0, 360, False, 7, out, footage=media, fast=True)
    mp4, fcp, rep = r["mp4"], r["fcpxml"], r["report"]
    assert mp4.stat().st_size > 10_000 and fcp.exists() and rep.exists()
    assert "report_thumbs" in rep.read_text(encoding="utf-8")
    assert export.validate_fcpxml(fcp) == []
    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert (int(cap.get(3)), int(cap.get(4))) == (640, 360)
    assert abs(n / fps - r["duration"]) < 0.1
    # picture cuts really sit on the beat/bar grid: frame difference spikes at every boundary between different clips
    prev, d = None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        d.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    d = np.array(d)
    med = np.median(d) + 1e-3
    cuts = r["selection"].cuts
    checked = 0
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        if a.window["clip_key"] == b.window["clip_key"]:
            continue
        idx = int(round(b.out_start * fps))
        lo = max(0, idx - 1)
        assert d[lo : idx + 2].max() > 3 * med, f"no picture cut at {b.out_start}s"
        checked += 1
    assert checked >= 3
    data = json.loads((out / "build.json").read_text())
    assert len(data["cuts"]) == len(cuts)


def test_clean_cache_cli(cfg, tmp_path, monkeypatch):
    from dronecut.cli import main

    (Path(cfg.paths.cache_dir)).mkdir(parents=True, exist_ok=True)
    (Path(cfg.paths.cache_dir) / "x").write_text("1")
    assert main(["clean-cache", "--cache-dir", cfg.paths.cache_dir]) == 0
    assert not Path(cfg.paths.cache_dir).exists()


def test_tech_evaluate_reasons():
    cfg = load_config()
    f = {"sharp": 100, "luma": 120, "luma_p5": 90, "clip_low": 0, "clip_high": 0, "sky": 0, "occl": 0, "shake": 0.1, "jerk": 0.5,
         "motion": 30, "chaos": 1, "zoom": 0.0, "zoom_abs": 0.0, "dist_start": 30, "dist_end": 30}
    s, why = score_tech.evaluate(f, {"sharp_ref": 100.0}, cfg.tech)
    assert not why and s > 0.7
    _, why = score_tech.evaluate({**f, "dist_start": 2.0, "zoom": -0.3}, {"sharp_ref": 100.0}, cfg.tech)
    assert "takeoff" in why
    _, why = score_tech.evaluate({**f, "dist_end": 2.0, "zoom": 0.3}, {"sharp_ref": 100.0}, cfg.tech)
    assert "landing" in why


def test_config_env_expansion_and_friendly_yaml_error(tmp_path, monkeypatch):
    monkeypatch.setenv("DC_TEST_DIR", str(tmp_path))
    good = tmp_path / "c.yaml"
    good.write_text('paths:\n  model_cache_dir: "%DC_TEST_DIR%/models"\n  music_dir: "$DC_TEST_DIR/music"\n', encoding="utf-8")
    c = load_config(good)
    import os

    assert c.paths.model_cache_dir == os.path.expandvars("%DC_TEST_DIR%/models") or str(tmp_path) in c.paths.model_cache_dir
    assert "%" not in c.paths.music_dir
    bad = tmp_path / "b.yaml"
    bad.write_text("paths:\n  model_cache_dir: %LOCALAPPDATA%/x\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        load_config(bad)
    assert "line 2" in str(e.value) and "quote" in str(e.value)


def test_low_altitude_kept_by_default_and_droppable():
    ws = _fake_windows(1, 2)
    for w in ws:
        w["feat"].update({"sharp": 100, "luma": 120, "luma_p5": 90, "clip_low": 0, "clip_high": 0, "sky": 0, "occl": 0, "shake": 0.1,
                          "jerk": 0.5, "motion": 30, "chaos": 1, "zoom": 0.0, "zoom_abs": 0.0, "dist_start": 30, "dist_end": 30, "alt": 0.8})
    ok, bad = select.score_windows(ws, load_config())
    assert len(ok) == 2 and not bad
    ok, bad = select.score_windows(ws, load_config(None, {"tech": {"min_altitude_m": 1.5}}))
    assert len(bad) == 2 and "ground_level" in bad[0]["reasons"]


# ---------------------------------------------------------------- must-have / never-use pins
def _good_feat():
    return {"sharp": 100, "luma": 120, "luma_p5": 90, "clip_low": 0, "clip_high": 0, "sky": 0, "occl": 0, "shake": 0.1,
            "jerk": 0.5, "motion": 30, "chaos": 1, "zoom": 0.0, "zoom_abs": 0.0, "dist_start": 30, "dist_end": 30}


def _plan(n_bars=12):
    from dronecut.music import Track

    beats = np.arange(0, n_bars * 2 + 4, 0.5)
    tr = Track("x.wav", n_bars * 2 + 4.0, 120.0, beats, beats[::4], np.arange(0, 100, 0.1), np.full(1000, 0.5))
    return music.make_plan(tr, n_bars * 2.0, load_config())


def test_parse_pin_specs():
    from dronecut import pins

    p = pins.parse_spec("0010@1:05-1:12.5")
    assert (p.clip, p.start, p.end) == ("0010", 65.0, 72.5)
    assert pins.parse_spec("0010@45").end == 51.0
    assert pins.parse_spec("0010").start is None
    for bad in ("@5-6", "x@9-3", "x@abc"):
        with pytest.raises(ValueError):
            pins.parse_spec(bad)


def test_pinned_part_overrides_quality_filter_and_is_used():
    from dronecut import pins

    cfg = load_config()
    ws = _fake_windows(3, 8)
    for w in ws:
        w["feat"] = _good_feat()
    bad = next(w for w in ws if w["clip_name"] == "c1.mp4" and w["start"] == 9.0)
    bad["feat"] = {**_good_feat(), "shake": 9.0, "sharp": 1.0}  # would be rejected (shaky + blurry)
    bad["score"] = 0.01
    plan = _plan(12)
    sel = select.build_cuts(ws, plan, cfg, seed=1, pins=pins.build_pins(None, ["c1@9.5-12"], []))
    assert any(c.window["id"] == bad["id"] for c in sel.cuts)
    c = next(c for c in sel.cuts if c.window["id"] == bad["id"])
    assert bad["pinned"] and "shaky" in bad["overridden"] and not bad["reasons"]
    assert c.src_start <= 10.75 <= c.src_start + c.src_dur  # the cut is centred on the requested part
    # without the pin the same window is rejected
    sel2 = select.build_cuts(ws, plan, cfg, seed=1)
    assert all(c.window["id"] != bad["id"] for c in sel2.cuts) and "shaky" in bad["reasons"]


def test_excluded_part_never_used_and_pins_win_when_music_is_short():
    from dronecut import pins

    cfg = load_config()
    ws = _fake_windows(3, 8)
    for w in ws:
        w["feat"] = _good_feat()
    plan = _plan(4)  # only 4 bars: far fewer slots than windows
    want = [w for w in ws if w["clip_name"] == "c2.mp4" and w["start"] in (0.0, 21.0)]
    spec = [f"c2@{w['start']}-{w['end']}" for w in want]
    sel = select.build_cuts(ws, plan, cfg, seed=2, pins=pins.build_pins(None, spec, ["c0"]))
    used = {c.window["id"] for c in sel.cuts}
    assert all(w["id"] in used for w in want)  # all must-haves made it despite the short music
    assert not any(c.window["clip_name"] == "c0.mp4" for c in sel.cuts)  # excluded clip is absent
    for c in sel.cuts:  # still on the bar grid
        assert np.min(np.abs(plan.bar_times - (plan.start + c.out_start))) < 1e-3


def test_pins_file_roundtrip_and_end_to_end_include(media, music_file, cfg, tmp_path):
    from dronecut import pins

    f = tmp_path / "must_have.json"
    f.write_text(json.dumps({"include": [{"clip": "junk_blur.mp4", "start": 0.5, "end": 4.5}], "exclude": []}))
    p = pins.build_pins(f, [], [])
    assert p.include[0].clip == "junk_blur.mp4"
    r = pipeline.build(cfg, str(music_file), 30.0, 270, False, 5, tmp_path / "out", footage=media, fast=True, pins=p)
    used = [c.window["clip_name"] for c in r["selection"].cuts]
    assert "junk_blur.mp4" in used  # the blurry clip is in the edit only because the user asked for it
    data = json.loads((tmp_path / "out" / "build.json").read_text())
    assert any(c["pinned"] for c in data["cuts"])
    page = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    assert "Download must_have.json" in page and 'data-state="in"' in page
