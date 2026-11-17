"""Stage 8: render the final MP4 from the ORIGINAL files in one ffmpeg pass (no concat seams).

All cuts are trimmed to exact frame counts derived from cumulative bar boundaries, so the picture stays
locked to the music grid over the whole edit (no drift, also for 29.97 fps material).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .config import Cfg
from .ingest import encoder_args, nvenc_available
from .music import MusicPlan
from .select import Cut
from .utils import console, log, run, which


@dataclass
class RenderInfo:
    path: Path
    fps: Fraction
    width: int
    height: int
    duration: float
    frames: int
    encoder: str


def std_fps(x: float) -> Fraction:
    """Snap measured fps to the nearest broadcast rate (29.97 -> 30000/1001, 25, 50, 59.94 ...)."""
    for num, den in ((24000, 1001), (24, 1), (25, 1), (30000, 1001), (30, 1), (48, 1), (50, 1), (60000, 1001), (60, 1)):
        if abs(x - num / den) < 0.02:
            return Fraction(num, den)
    return Fraction(x).limit_denominator(1001)


def choose_fps(cuts: list[Cut], clips_by_path: dict, cfg: Cfg) -> Fraction:
    if cfg.render.fps:
        return std_fps(float(cfg.render.fps))
    c = Counter(std_fps(clips_by_path[x.window["clip_path"]].fps) for x in cuts)
    return c.most_common(1)[0][0]


def out_size(res: int) -> tuple[int, int]:
    h = int(res) - int(res) % 2
    w = int(round(h * 16 / 9 / 2)) * 2
    return w, h


def _esc_path(p: str) -> str:
    return p.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def _grade(w: dict, cfg: Cfg) -> str:
    """Per-clip exposure / white-balance normalisation + optional LUT as a filter chain fragment."""
    r = cfg.render
    parts: list[str] = []
    f = w["feat"]
    if r.white_balance:
        mb, mg, mr = f["mean_b"], f["mean_g"], f["mean_r"]
        gray = (mb + mg + mr) / 3.0
        if min(mb, mg, mr) > 5:
            gr, gg, gb = (max(0.88, min(1.12, gray / v)) for v in (mr, mg, mb))
            parts.append(f"colorchannelmixer=rr={gr:.3f}:gg={gg:.3f}:bb={gb:.3f}")
    if r.normalize_exposure and 20 < f["luma"] < 235:
        cur, tgt = f["luma"] / 255.0, float(r.target_luma) / 255.0
        g = math.log(cur) / math.log(tgt)
        g = max(0.8, min(1.3, g))
        if abs(g - 1.0) > 0.03:
            parts.append(f"eq=gamma={g:.3f}")
    if r.lut:
        parts.append(f"lut3d=file='{_esc_path(str(Path(r.lut).resolve()))}'")
    return ",".join(parts)


def build_graph(cuts: list[Cut], plan: MusicPlan, clips_by_path: dict, cfg: Cfg, fps: Fraction, size: tuple[int, int]) -> tuple[str, int, list[str]]:
    """Return (filter_complex script, total frames, per-cut input args)."""
    W, H = size
    F = float(fps)
    cf = float(cfg.render.crossfade_s)
    ext = cf / 2.0
    n = len(cuts)
    bounds = [0.0]
    for c in cuts:
        bounds.append(bounds[-1] + c.out_dur)
    frame_b = [int(round(b * F)) for b in bounds]  # cumulative frame boundaries
    total_frames = frame_b[-1]
    chains: list[str] = []
    seg_frames: list[int] = []
    for i, c in enumerate(cuts):
        info = clips_by_path[c.window["clip_path"]]
        head_want = ext if (cf > 0 and i > 0) else 0.0
        tail_want = ext if (cf > 0 and i < n - 1) else 0.0
        head = min(head_want, c.src_start / c.speed)
        tail = min(tail_want, max(0.0, info.duration - (c.src_start + c.src_dur)) / c.speed)
        nf = frame_b[i + 1] - frame_b[i] + int(round(head_want * F)) + int(round(tail_want * F))
        seg_frames.append(nf)
        grade = _grade(c.window, cfg)
        fl = [
            f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos",
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black",
            "setsar=1",
        ]
        if grade:
            fl.append(grade)
        fl.append(f"setpts=(PTS-STARTPTS)/{c.speed:.4f}")
        miss_h = head_want - head
        miss_t = tail_want - tail
        if miss_h > 1e-3:
            fl.append(f"tpad=start_mode=clone:start_duration={miss_h:.4f}")
        if miss_t > 1e-3:
            fl.append(f"tpad=stop_mode=clone:stop_duration={miss_t:.4f}")
        fl += [f"fps={fps.numerator}/{fps.denominator}", f"trim=end_frame={nf}", "setpts=PTS-STARTPTS", "format=yuv420p"]
        chains.append(f"[{i}:v]" + ",".join(fl) + f"[v{i}]")
    if cf > 0 and n > 1:
        acc = seg_frames[0] / F
        prev = "v0"
        for i in range(1, n):
            off = acc - cf
            out = f"x{i}"
            chains.append(f"[{prev}][v{i}]xfade=transition=fade:duration={cf:.4f}:offset={off:.4f}[{out}]")
            acc += seg_frames[i] / F - cf
            prev = out
        chains.append(f"[{prev}]null[vout]")
    else:
        chains.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vout]")
    return ";\n".join(chains), total_frames, []


def render(cuts: list[Cut], plan: MusicPlan, clips_by_path: dict, cfg: Cfg, out_path: Path, res: int | None = None,
           music_path: str | None = None) -> RenderInfo:
    if not cuts:
        raise SystemExit("nothing to render: no usable cuts were selected")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = choose_fps(cuts, clips_by_path, cfg)
    size = out_size(res or cfg.render.resolution)
    script, total_frames, _ = build_graph(cuts, plan, clips_by_path, cfg, fps, size)
    total_t = total_frames / float(fps)
    args: list[str] = [which("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-stats"]
    for c in cuts:
        info = clips_by_path[c.window["clip_path"]]
        cf = float(cfg.render.crossfade_s)
        ext = cf / 2.0
        head = min(ext, c.src_start / c.speed) if cf > 0 else 0.0
        tail = min(ext, max(0.0, info.duration - c.src_start - c.src_dur) / c.speed) if cf > 0 else 0.0
        s0 = c.src_start - head * c.speed
        dur = c.src_dur + (head + tail) * c.speed + 0.25
        args += ["-ss", f"{s0:.4f}", "-t", f"{dur:.4f}", "-i", info.path]
    music_idx = len(cuts)
    mfile = music_path or plan.track.path
    fade = min(float(cfg.music.fade_out_s), total_t / 3)
    args += ["-ss", f"{plan.start:.4f}", "-t", f"{total_t:.4f}", "-i", mfile]
    script_path = out_path.with_suffix(".filter.txt")
    audio_graph = f"[{music_idx}:a]aresample=48000,afade=t=in:d=0.4,afade=t=out:st={max(0.0, total_t - fade):.3f}:d={fade:.3f}[aout]"
    a_map = "[aout]"
    if cfg.render.mix_source_audio:
        parts, labels = [], []
        for i, c in enumerate(cuts):
            info = clips_by_path[c.window["clip_path"]]
            if info.has_audio:
                parts.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=start=0:duration={c.out_dur:.4f},"
                             f"apad=whole_dur={c.out_dur:.4f},asetpts=PTS-STARTPTS[a{i}]")
            else:
                parts.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={c.out_dur:.4f},asetpts=PTS-STARTPTS[a{i}]")
            labels.append(f"[a{i}]")
        # source audio is cut from c.src_start already (input -ss); gain applied once after concat
        parts.append("".join(labels) + f"concat=n={len(cuts)}:v=0:a=1,volume={cfg.render.source_audio_gain},asplit[srcsc][srcmix]")
        parts.append(f"[{music_idx}:a]aresample=48000,aformat=channel_layouts=stereo,afade=t=in:d=0.4,afade=t=out:st={max(0.0, total_t - fade):.3f}:d={fade:.3f}[mus]")
        parts.append("[mus][srcsc]sidechaincompress=threshold=0.04:ratio=5:attack=30:release=500[duck]")
        parts.append("[duck][srcmix]amix=inputs=2:normalize=0:duration=first[aout]")
        audio_graph = ";\n".join(parts)
    script_path.write_text(script + ";\n" + audio_graph, encoding="utf-8")
    enc = encoder_args(cfg, quality=int(cfg.render.crf))
    tail_args = ["-map", "[vout]", "-map", a_map, *enc, "-r", f"{fps.numerator}/{fps.denominator}", "-c:a", "aac", "-b:a", "192k",
                 "-t", f"{total_t:.4f}", "-movflags", "+faststart", str(out_path)]
    log.info("rendering %d cuts -> %s (%dx%d @ %.3f fps, %.1fs, encoder %s)", len(cuts), out_path.name, *size, float(fps), total_t, enc[1])
    try:
        # ffmpeg >= 7 reads a filter script via "-/filter_complex"; older builds use "-filter_complex_script"
        cp = run([*args, "-/filter_complex", str(script_path), *tail_args], check=False)
        if cp.returncode != 0 and "Unrecognized option" in cp.stderr:
            cp = run([*args, "-filter_complex_script", str(script_path), *tail_args], check=False)
        if cp.returncode != 0:
            raise RuntimeError("ffmpeg render failed:\n" + "\n".join(cp.stderr.strip().splitlines()[-15:]))
    finally:
        script_path.unlink(missing_ok=True)
    console.print()
    return RenderInfo(out_path, fps, size[0], size[1], total_t, total_frames, enc[1])


__all__ = ["RenderInfo", "render", "choose_fps", "out_size", "std_fps", "nvenc_available"]
