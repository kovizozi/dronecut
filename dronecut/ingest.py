"""Stage 2: discover footage, build low-res analysis proxies (NVENC when available)."""
from __future__ import annotations

import functools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Cfg
from .probe import ClipInfo, probe_file
from .utils import cached_json, file_key, log, params_hash, run, store_json, which


@functools.lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """True if ffmpeg can really encode with h264_nvenc on this machine (not just list it)."""
    try:
        cp = run(
            [which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            check=False, timeout=30,
        )
        return cp.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def encoder_args(cfg: Cfg, quality: int = 20, codec: str | None = None, nvenc: str | None = None) -> list[str]:
    """ffmpeg video encoder args honouring render.nvenc (auto/on/off) and codec."""
    codec = codec or cfg.render.get("codec", "h264")
    mode = nvenc or cfg.render.get("nvenc", "auto")
    use = (mode == "on") or (mode == "auto" and nvenc_available())
    if use:
        name = "hevc_nvenc" if codec == "hevc" else "h264_nvenc"
        return ["-c:v", name, "-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0", "-pix_fmt", "yuv420p"]
    name = "libx265" if codec == "hevc" else "libx264"
    extra = ["-preset", "medium"] if codec != "hevc" else ["-preset", "medium", "-tag:v", "hvc1"]
    return ["-c:v", name, "-crf", str(quality), *extra, "-pix_fmt", "yuv420p"]


def discover(footage: str | Path, exts: list[str]) -> list[Path]:
    root = Path(footage)
    if root.is_file():
        return [root]
    exts_l = {e.lower() for e in exts}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_l)


def proxy_path(cache: Path, key: str) -> Path:
    return cache / key / "proxy.mp4"


def probe_all(files: list[Path], cache: Path) -> list[ClipInfo]:
    """Probe every file; failures are recorded per file and never abort the run."""
    infos: list[ClipInfo] = []
    for f in files:
        try:
            key = file_key(f)
            pj = cache / key / "probe.json"
            ph = params_hash("probe-v1")
            d = cached_json(pj, ph)
            if d is None:
                info = probe_file(f)
                info.key = key
                d = info.to_dict()
                store_json(pj, ph, d)
            infos.append(ClipInfo.from_dict(d))
        except Exception as e:  # noqa: BLE001
            log.warning("skipping %s: %s", f.name, str(e).splitlines()[0][:200])
    return infos


def build_proxy(info: ClipInfo, cache: Path, cfg: Cfg) -> Path:
    out = proxy_path(cache, info.key)
    ph = params_hash({"h": cfg.ingest.proxy_height, "fps": cfg.ingest.proxy_fps, "v": 2})
    marker = out.with_suffix(".ok")
    if out.exists() and marker.exists() and marker.read_text() == ph:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    h = int(cfg.ingest.proxy_height)
    h -= h % 2
    vf = f"scale=-2:'min({h},ih)':flags=bilinear,fps={cfg.ingest.proxy_fps}"
    enc = encoder_args(cfg, quality=28, codec="h264", nvenc="auto" if cfg.ingest.prefer_nvenc else "off")
    tmp = out.with_name("proxy.tmp.mp4")
    run([which("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-i", info.path, "-map", "0:v:0", "-an",
         "-vf", vf, *enc, "-g", "20", str(tmp)])
    tmp.replace(out)
    marker.write_text(ph)
    return out


def build_proxies(infos: list[ClipInfo], cache: Path, cfg: Cfg) -> dict[str, Path]:
    """Build proxies in parallel (ffmpeg subprocesses). Returns key -> proxy path for successes."""
    res: dict[str, Path] = {}

    def one(info: ClipInfo) -> tuple[str, Path | None]:
        try:
            return info.key, build_proxy(info, cache, cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("proxy failed for %s: %s", Path(info.path).name, str(e).splitlines()[0][:200])
            return info.key, None

    workers = max(1, int(cfg.ingest.workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for key, p in ex.map(one, infos):
            if p is not None:
                res[key] = p
    return res
