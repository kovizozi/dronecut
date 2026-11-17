"""Configuration: built-in defaults deep-merged with an optional ``config.yaml``."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "paths": {
        "cache_dir": ".dronecut_cache",
        "output_dir": "output",
        "model_cache_dir": None,  # None -> per-user cache dir (never inside the repo)
        "music_dir": None,
    },
    "ingest": {
        "proxy_height": 720,
        "proxy_fps": 10,
        "prefer_nvenc": True,
        "workers": 4,
        "video_exts": [".mp4", ".mov", ".mkv", ".avi", ".m4v"],
    },
    "segment": {
        "min_window_s": 2.0,
        "max_window_s": 6.0,
        "window_s": 4.0,  # sliding window length for long shots
        "overlap": 0.5,  # fraction of overlap between sliding windows
        "content_threshold": 27.0,
        "adaptive_threshold": 3.0,
        "merge_cut_tol_s": 0.5,
        "min_scene_len_s": 1.0,
    },
    "tech": {
        "workers": 4,
        "flow_width": 320,
        "sharp_width": 640,
        # hard rejects (applied at selection time, so they can be tuned without re-analysing)
        "min_sharpness_abs": 12.0,
        "min_sharpness_rel": 0.30,  # fraction of the global median sharpness
        "luma_min": 35.0,
        "luma_max": 215.0,
        "clip_high_max": 0.30,
        "clip_low_max": 0.55,
        "shake_max": 2.2,  # px (at flow_width) of residual camera jitter
        "jerk_max": 14.0,  # px/frame^2 sudden camera acceleration
        "motion_min": 1.5,  # px/s: below = static
        "motion_max": 260.0,  # px/s: above = chaotic
        "zoom_rate_max": 0.45,  # 1/s divergence: rapid zoom-like scale change
        "edge_s": 8.0,  # takeoff / landing zone at clip start/end
        "edge_zoom_rate": 0.18,
        "sky_max": 0.92,
        "occlusion_max": 0.35,
        "motion_target": [12.0, 90.0],  # px/s range scored as 'good motion'
        "weights": {"sharpness": 0.30, "exposure": 0.20, "stability": 0.30, "motion": 0.20},
    },
    "semantic": {
        "backend": "siglip",  # siglip | openclip | mock
        "model": "google/siglip-so400m-patch14-384",
        "sample_fps": 1.0,
        "batch_size": 16,
        "person_backend": "yolo",  # yolo | hog | none
        "person_boost": 0.15,
        "positive_prompts": [
            "people waving at the camera",
            "family outdoors",
            "aerial view of landscape",
            "drone orbiting around a person",
            "children playing",
        ],
        "negative_prompts": ["blurry image", "ground close-up", "sky only", "indoor ceiling"],
        "wide_prompts": ["a wide aerial establishing shot of a landscape"],
    },
    "vlm": {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "fallback_model": "vikhyatk/moondream2",
        "top_k": 40,
        "weight": 0.25,
    },
    "select": {
        "weights": {"technical": 0.5, "semantic": 0.5},
        "min_gap_s": 6.0,  # min time between picks from the same source clip
        "clusters": 8,
        "max_per_cluster_frac": 0.35,
        "open_with_wide": True,
        "close_with_pullaway": True,
    },
    "music": {
        "sr": 22050,
        "bpm_range": [75.0, 165.0],
        "beats_per_bar": 4,
        "bars_high": 1,  # bars per shot in high-energy sections
        "bars_mid": 2,
        "bars_low": 3,
        "fade_out_s": 2.5,
        "auto_prefer_bpm": [95.0, 130.0],
    },
    "render": {
        "resolution": 1080,
        "fps": None,  # None -> most common source fps
        "crossfade_s": 0.0,  # 0 = hard cuts
        "slowmo_factor": 1.0,  # <1 slows smooth orbit-like shots, e.g. 0.8
        "normalize_exposure": True,
        "target_luma": 118.0,
        "white_balance": True,
        "lut": None,  # path to a .cube LUT (applied last)
        "mix_source_audio": False,
        "source_audio_gain": 0.35,
        "nvenc": "auto",  # auto | on | off
        "codec": "h264",  # h264 | hevc
        "crf": 18,
    },
}


class Cfg(dict):
    """dict with attribute access for nested sections."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Cfg(v) if isinstance(v, dict) else v


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Cfg:
    cfg = copy.deepcopy(DEFAULTS)
    if path is None and Path("config.yaml").exists():
        path = "config.yaml"
    if path:
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        _merge(cfg, overrides)
    return Cfg(cfg)


def model_cache_dir(cfg: Cfg) -> Path:
    p = cfg["paths"].get("model_cache_dir")
    if p:
        return Path(p).expanduser()
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "dronecut" / "models"
