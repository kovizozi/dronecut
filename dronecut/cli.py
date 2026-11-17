"""Command line interface: analyze | build | report | clean-cache."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .config import load_config
from .utils import TIMINGS, console, setup_logging


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="path to config.yaml (default: ./config.yaml if present)")
    p.add_argument("--cache-dir", help="override paths.cache_dir")
    p.add_argument("-v", "--verbose", action="store_true")


def _cfg(a: argparse.Namespace):
    ov: dict = {}
    if getattr(a, "cache_dir", None):
        ov["paths"] = {"cache_dir": a.cache_dir}
    return load_config(a.config, ov)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dronecut", description="Local, offline automatic highlight editor for drone footage.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("analyze", help="probe, proxy, segment and score all footage in a folder (cached)")
    pa.add_argument("footage_dir")
    pa.add_argument("--fast", action="store_true", help="skip the large ML models (CI / quick look)")
    pa.add_argument("--vlm", action="store_true", help="also caption/score the top candidates with a local VLM")
    pa.add_argument("--seed", type=int, default=0)
    _common(pa)
    pb = sub.add_parser("build", help="select, cut to the beat, render MP4 + FCPXML + report")
    pb.add_argument("--music", required=True, help="audio file, or 'auto' to pick from paths.music_dir")
    pb.add_argument("--duration", type=float, default=150.0, help="target length in seconds (default 150)")
    pb.add_argument("--res", type=int, default=None, help="output height: 1080 (default) or 2160")
    pb.add_argument("--vlm", action="store_true")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--fast", action="store_true")
    pb.add_argument("--footage", help="footage folder (default: the one used by the last `analyze`)")
    pb.add_argument("-o", "--output-dir", default=None)
    _common(pb)
    pr = sub.add_parser("report", help="regenerate report.html from the last build")
    pr.add_argument("-o", "--output-dir", default=None)
    _common(pr)
    pc = sub.add_parser("clean-cache", help="delete the analysis cache")
    _common(pc)
    a = ap.parse_args(argv)
    setup_logging(a.verbose)
    cfg = _cfg(a)
    from . import pipeline

    if a.cmd == "analyze":
        an = pipeline.analyze(cfg, a.footage_dir, fast=a.fast, vlm=a.vlm, seed=a.seed)
        ok = sum(1 for w in an.windows)
        console.print(f"[green]analysed {len(an.clips)} clips, {ok} candidate windows[/] (cache: {cfg.paths.cache_dir})")
        for p, e in an.failed.items():
            console.print(f"[yellow]failed:[/] {p}: {e}")
    elif a.cmd == "build":
        out = a.output_dir or cfg.paths.output_dir
        r = pipeline.build(cfg, a.music, a.duration, a.res, a.vlm, a.seed, out, footage=a.footage, fast=a.fast)
        console.print(f"[bold green]done[/]: {r['mp4']}\n  timeline: {r['fcpxml']}\n  report:   {r['report']}")
        console.print("stage timings: " + ", ".join(f"{k} {v:.1f}s" for k, v in TIMINGS.items()))
    elif a.cmd == "report":
        from .pipeline import cache_dir
        from .report import write_report

        out = Path(a.output_dir or cfg.paths.output_dir)
        console.print(f"report: {write_report(out / 'build.json', cache_dir(cfg), out / 'report.html')}")
    elif a.cmd == "clean-cache":
        p = Path(cfg.paths.cache_dir)
        if p.exists():
            shutil.rmtree(p)
        console.print(f"removed {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
