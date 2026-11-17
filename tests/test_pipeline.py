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
