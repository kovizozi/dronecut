"""Orchestration of the analysis stages (probe -> proxy -> segment -> technical score -> semantic score).

Every stage caches to ``<cache>/<clip key>/`` so re-runs skip completed work.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import ingest, score_tech
from .config import Cfg
from .probe import ClipInfo
from .segment import detect_cuts, make_windows
from .utils import cached_json, log, params_hash, progress, read_json, stage, store_json, write_json


@dataclass
class Analysis:
    clips: list[ClipInfo] = field(default_factory=list)
    windows: list[dict[str, Any]] = field(default_factory=list)  # one dict per candidate window
    failed: dict[str, str] = field(default_factory=dict)  # clip path -> error


def cache_dir(cfg: Cfg) -> Path:
    p = Path(cfg.paths.cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _clip_worker(args: tuple) -> tuple[str, list[float] | None, dict | None, str]:
    """Worker (runs in a pool process): scene cuts + frame metrics for one proxy."""
    key, proxy, cfg_dict, need_cuts, need_frames = args
    try:
        cfg = Cfg(cfg_dict)
        cuts = detect_cuts(Path(proxy), cfg) if need_cuts else None
        fm = score_tech.frame_metrics(proxy, dict(cfg.tech)) if need_frames else None
        return key, cuts, fm, ""
    except Exception as e:  # noqa: BLE001
        return key, None, None, f"{type(e).__name__}: {e}"


def _srt_alt(info: ClipInfo, a: float, b: float) -> float | None:
    vals = [s.get("rel_alt", s.get("altitude")) for s in info.srt if a <= s["t"] <= b and ("rel_alt" in s or "altitude" in s)]
    return float(np.mean(vals)) if vals else None


def run_clip_analysis(infos: list[ClipInfo], proxies: dict[str, Path], cfg: Cfg, cache: Path) -> tuple[dict, dict, dict]:
    """Returns (cuts per key, frame metrics per key, errors per key), computing only what is missing."""
    cut_ph = params_hash({"seg": {k: cfg.segment[k] for k in ("content_threshold", "adaptive_threshold", "merge_cut_tol_s")}, "fps": cfg.ingest.proxy_fps, "h": cfg.ingest.proxy_height})
    fr_ph = params_hash({"t": {k: cfg.tech[k] for k in ("flow_width", "sharp_width")}, "fps": cfg.ingest.proxy_fps, "h": cfg.ingest.proxy_height, "v": 2})
    cuts: dict[str, list[float]] = {}
    frames: dict[str, dict] = {}
    errors: dict[str, str] = {}
    jobs = []
    for info in infos:
        key = info.key
        if key not in proxies:
            errors[key] = "no proxy"
            continue
        c = cached_json(cache / key / "cuts.json", cut_ph)
        npz = cache / key / "frames.npz"
        fm = None
        if npz.exists() and (cache / key / "frames.ph").exists() and (cache / key / "frames.ph").read_text() == fr_ph:
            try:
                with np.load(npz) as z:
                    fm = {k: z[k] for k in z.files}
            except Exception:  # noqa: BLE001
                fm = None
        if c is not None:
            cuts[key] = c
        if fm is not None:
            frames[key] = fm
        if c is None or fm is None:
            jobs.append((key, str(proxies[key]), dict(cfg), c is None, fm is None))
    if jobs:
        workers = max(1, min(int(cfg.tech.workers), len(jobs)))
        log.info("analysing %d clip(s) with %d worker(s)", len(jobs), workers)
        if workers == 1:
            results = (_clip_worker(j) for j in jobs)
            it = progress(results, "technical analysis", total=len(jobs))
            ex = None
        else:
            ex = ProcessPoolExecutor(max_workers=workers)
            it = progress(ex.map(_clip_worker, jobs), "technical analysis", total=len(jobs))
        for key, c, fm, err in it:
            if err:
                errors[key] = err
                log.warning("analysis failed for %s: %s", key, err.splitlines()[0][:200])
                continue
            if c is not None:
                cuts[key] = c
                store_json(cache / key / "cuts.json", cut_ph, c)
            if fm is not None:
                frames[key] = fm
                np.savez_compressed(cache / key / "frames.npz", **fm)
                (cache / key / "frames.ph").write_text(fr_ph)
        if ex:
            ex.shutdown()
    return cuts, frames, errors


def analyze(cfg: Cfg, footage: str | Path, fast: bool = False, vlm: bool = False, seed: int = 0) -> Analysis:
    cache = cache_dir(cfg)
    an = Analysis()
    with stage("probe"):
        files = ingest.discover(footage, cfg.ingest.video_exts)
        if not files:
            raise SystemExit(f"no video files found in {footage}")
        an.clips = ingest.probe_all(files, cache)
        probed = {c.path for c in an.clips}
        for f in files:
            if str(f.resolve()) not in probed:
                an.failed[str(f)] = "probe failed"
        log.info("%d clip(s) probed, %d failed", len(an.clips), len(an.failed))
    with stage("proxy"):
        proxies = ingest.build_proxies(an.clips, cache, cfg)
    with stage("segment+technical"):
        cuts, frames, errors = run_clip_analysis(an.clips, proxies, cfg, cache)
        for info in an.clips:
            if info.key in errors:
                an.failed[info.path] = errors[info.key]
        good = [c for c in an.clips if c.key in cuts and c.key in frames]
        feats_all: list[dict[str, float]] = []
        for info in good:
            wins = make_windows(info.key, info.path, info.duration, cuts[info.key], cfg)
            for w in wins:
                f = score_tech.window_features(frames[info.key], w.start, w.end, info.duration)
                if f is None:
                    continue
                alt = _srt_alt(info, w.start, w.end)
                if alt is not None:
                    f["alt"] = alt
                d = w.to_dict()
                d["feat"] = f
                d["clip_name"] = Path(info.path).name
                an.windows.append(d)
                feats_all.append(f)
        log.info("%d candidate windows from %d clips", len(an.windows), len(good))
    from . import score_semantic

    with stage("semantic" + (" (fast)" if fast else "")):
        score_semantic.run(an, proxies, cfg, cache, fast=fast)
    if vlm:
        from . import vlm as vlm_mod

        with stage("vlm"):
            vlm_mod.run(an, proxies, cfg, cache, seed=seed)
    write_json(cache / "state.json", {"footage": str(Path(footage).resolve()), "fast": fast, "vlm": vlm,
                                      "clips": [c.key for c in an.clips]})
    return an


def load_state(cfg: Cfg) -> dict | None:
    return read_json(cache_dir(cfg) / "state.json")


# ---------------------------------------------------------------------------------------------
# build


def _clean_window(w: dict) -> dict:
    return {k: v for k, v in w.items() if k != "emb"}


def build(cfg: Cfg, music: str, duration: float, res: int | None, vlm: bool, seed: int, out_dir: str | Path,
          footage: str | Path | None = None, fast: bool = False, cuts_only: bool = False) -> dict:
    """Analyse (cached), pick music, select, render, export. Returns paths + summary."""
    import random

    from . import export, report, select
    from . import music as music_mod
    from .render import render
    from .utils import TIMINGS

    random.seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if footage is None:
        st = load_state(cfg)
        if not st:
            raise SystemExit("no footage known yet: run `dronecut analyze <footage_dir>` first or pass --footage")
        footage = st["footage"]
        fast = fast or bool(st.get("fast"))
    an = analyze(cfg, footage, fast=fast, vlm=vlm, seed=seed)
    cache = cache_dir(cfg)
    with stage("music"):
        if music == "auto":
            if not cfg.paths.music_dir:
                raise SystemExit("--music auto needs paths.music_dir in config.yaml")
            mpath = music_mod.pick_auto(cfg.paths.music_dir, duration, cfg, cache, seed)
            log.info("auto-selected music: %s", mpath.name)
        else:
            mpath = Path(music)
            if not mpath.exists():
                raise SystemExit(f"music file not found: {mpath}")
        track = music_mod.analyze_track(mpath, cfg, cache)
        plan = music_mod.make_plan(track, duration, cfg)
        log.info("music: %.1f BPM, %d bars (%.1fs) from %.1fs", track.bpm, plan.n_bars, plan.duration, plan.start)
    with stage("select"):
        sel = select.build_cuts(an.windows, plan, cfg, seed=seed)
        log.info("%d accepted / %d rejected windows -> %d cuts", len(sel.candidates), len(sel.rejected), len(sel.cuts))
        if not sel.cuts:
            raise SystemExit("no usable windows survived rejection - check report / loosen thresholds in config.yaml")
    clips = {c.path: c for c in an.clips}
    with stage("render"):
        mp4 = out / "dronecut_highlight.mp4"
        info = render(sel.cuts, plan, clips, cfg, mp4, res=res, music_path=str(mpath))
    with stage("export"):
        fcp = export.write_fcpxml(out / "dronecut_highlight.fcpxml", sel.cuts, plan, clips, info.fps, res or cfg.render.resolution, str(mpath))
        problems = export.validate_fcpxml(fcp)
        if problems:
            log.warning("FCPXML issues: %s", problems[:5])
    build_json = out / "build.json"
    write_json(build_json, {
        "output": str(mp4), "fcpxml": str(fcp), "duration": info.duration, "seed": seed, "fast": fast,
        "music": {"path": str(mpath), "name": mpath.name, "bpm": track.bpm, "bars": plan.n_bars, "start": plan.start},
        "cuts": [{"window_id": c.window["id"], "src_start": c.src_start, "src_dur": c.src_dur, "out_start": c.out_start,
                  "out_dur": c.out_dur, "speed": c.speed, "bars": c.bars} for c in sel.cuts],
        "windows": [_clean_window(w) for w in an.windows],
        "timings": dict(TIMINGS),
        "failed": an.failed,
    })
    with stage("report"):
        rep = report.write_report(build_json, cache, out / "report.html")
    return {"mp4": mp4, "fcpxml": fcp, "report": rep, "cuts": len(sel.cuts), "duration": info.duration, "plan": plan, "selection": sel}
