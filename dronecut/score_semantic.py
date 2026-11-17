"""Stage 5: semantic scoring - SigLIP embeddings vs. positive/negative prompts + person detection.

Embeddings are cached per clip (npz). Models download on first use into a per-user cache dir, never the repo.
``fast=True`` replaces the big models with a tiny colour-layout "embedding" and no person detector.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Cfg, model_cache_dir
from .utils import log, params_hash, progress

# ---------------------------------------------------------------------------------------------
# frame sampling


def sample_frames(proxy: Path, fps_out: float, max_side: int = 768):
    """Yield (time_s, BGR frame) at ~fps_out from a proxy video."""
    cap = cv2.VideoCapture(str(proxy))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    step = max(1, int(round(fps / fps_out)))
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == step // 2:  # middle of each step interval
            ok, fr = cap.retrieve()
            if not ok:
                break
            h, w = fr.shape[:2]
            s = max_side / max(h, w)
            if s < 1:
                fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            yield i / fps, fr
        i += 1
    cap.release()


# ---------------------------------------------------------------------------------------------
# embedding backends


class MockEmbedder:
    """Fast-mode stand-in: 8x8 colour layout, L2-normalised. Good enough for diversity clustering."""

    name = "mock"
    dim = 192

    def embed_images(self, frames: list[np.ndarray]) -> np.ndarray:
        out = []
        for f in frames:
            v = cv2.resize(f, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)
            v = v - v.mean()
            out.append(v / (np.linalg.norm(v) + 1e-6))
        return np.stack(out)

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        return np.zeros((len(prompts), self.dim), np.float32)

    def close(self) -> None:
        pass


class SigLIPEmbedder:
    name = "siglip"

    def __init__(self, model_id: str, cfg: Cfg):
        import torch
        from transformers import AutoModel, AutoProcessor

        cache = model_cache_dir(cfg)
        cache.mkdir(parents=True, exist_ok=True)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        log.info("loading %s on %s (cache: %s)", model_id, self.device, cache)
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(cache))
        self.model = AutoModel.from_pretrained(model_id, cache_dir=str(cache), torch_dtype=dtype).to(self.device).eval()
        self.bs = int(cfg.semantic.batch_size)

    def embed_images(self, frames: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        torch = self.torch
        outs = []
        for i in range(0, len(frames), self.bs):
            imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames[i : i + self.bs]]
            px = self.processor(images=imgs, return_tensors="pt")["pixel_values"].to(self.device, self.model.dtype)
            with torch.no_grad():
                e = self.model.get_image_features(pixel_values=px)
            e = torch.nn.functional.normalize(e.float(), dim=-1)
            outs.append(e.cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 1), np.float32)

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        torch = self.torch
        inp = self.processor(text=prompts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
        with torch.no_grad():
            e = self.model.get_text_features(**inp)
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()

    def close(self) -> None:
        del self.model
        import gc

        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def make_embedder(cfg: Cfg, fast: bool):
    if fast or cfg.semantic.backend == "mock":
        return MockEmbedder()
    if cfg.semantic.backend == "siglip":
        return SigLIPEmbedder(cfg.semantic.model, cfg)
    raise ValueError(f"unknown semantic backend {cfg.semantic.backend!r} (siglip | mock)")


# ---------------------------------------------------------------------------------------------
# person detection


class PersonDetector:
    def __init__(self, backend: str, cfg: Cfg):
        self.backend = backend
        self.model: Any = None
        if backend == "yolo":
            from ultralytics import YOLO

            wdir = model_cache_dir(cfg)
            wdir.mkdir(parents=True, exist_ok=True)
            w = wdir / "yolov8n.pt"
            if not w.exists():
                url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
                log.info("downloading yolov8n weights to %s", w)
                urllib.request.urlretrieve(url, w)  # noqa: S310
            self.model = YOLO(str(w))
            self.device = 0 if _cuda() else "cpu"
        elif backend == "hog":
            self.model = cv2.HOGDescriptor()
            self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def person_area(self, frames: list[np.ndarray]) -> np.ndarray:
        """Per frame: largest-person bbox area fraction (0 if none)."""
        out = np.zeros(len(frames), np.float32)
        if self.backend == "yolo":
            for i in range(0, len(frames), 16):
                res = self.model.predict(frames[i : i + 16], classes=[0], conf=0.30, verbose=False, device=self.device, imgsz=640)
                for j, r in enumerate(res):
                    if len(r.boxes):
                        h, w = r.orig_shape
                        a = (r.boxes.xywhn[:, 2] * r.boxes.xywhn[:, 3]).max().item()
                        out[i + j] = float(a) if h and w else 0.0
        elif self.backend == "hog":
            for i, f in enumerate(frames):
                small = cv2.resize(f, (480, int(f.shape[0] * 480 / f.shape[1])))
                rects, _ = self.model.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.1)
                if len(rects):
                    out[i] = max(w * h for (_, _, w, h) in rects) / (small.shape[0] * small.shape[1])
        return out


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------------------------
# stage


def _text_sims(emb: Any, cfg: Cfg, cache: Path, fast: bool, model_name: str) -> dict[str, np.ndarray]:
    s = cfg.semantic
    groups = {"pos": s.positive_prompts, "neg": s.negative_prompts, "wide": s.wide_prompts}
    ph = params_hash({"m": model_name, "g": groups})
    f = cache / "text" / f"{emb.name}_{ph}.npz"
    if f.exists():
        with np.load(f) as z:
            return {k: z[k] for k in z.files}
    res = {k: emb.embed_text(list(v)) for k, v in groups.items()}
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, **res)
    return res


def run(an, proxies: dict[str, Path], cfg: Cfg, cache: Path, fast: bool = False) -> None:
    """Fill ``sem_raw``, ``wide_raw``, ``person``, ``emb`` on every window dict of ``an``."""
    s = cfg.semantic
    backend = "mock" if fast else s.backend
    model_name = "mock" if backend == "mock" else s.model
    person_backend = "none" if fast else s.person_backend
    ph = params_hash({"b": backend, "m": model_name, "fps": s.sample_fps, "pb": person_backend, "ph": cfg.ingest.proxy_fps, "v": 2})
    keys = sorted({w["clip_key"] for w in an.windows})
    todo = [k for k in keys if not _emb_ok(cache / k / "emb.npz", ph)]
    emb = None
    det = None
    if todo:
        emb = make_embedder(cfg, fast)
        if person_backend != "none":
            try:
                det = PersonDetector(person_backend, cfg)
            except Exception as e:  # noqa: BLE001
                log.warning("person detector unavailable (%s) - continuing without", str(e)[:150])
        for k in progress(todo, "semantic embeddings"):
            try:
                times, frames = [], []
                for t, fr in sample_frames(proxies[k], float(s.sample_fps)):
                    times.append(t)
                    frames.append(fr)
                e = emb.embed_images(frames) if frames else np.zeros((0, emb.dim if hasattr(emb, "dim") else 1), np.float32)
                pa = det.person_area(frames) if (det and frames) else np.zeros(len(frames), np.float32)
                np.savez_compressed(cache / k / "emb.npz", t=np.asarray(times, np.float32), e=e, person=pa, ph=np.asarray(ph))
            except Exception as ex:  # noqa: BLE001
                log.warning("semantic analysis failed for clip %s: %s", k, str(ex)[:200])
    # prompts -> sims (needs the text tower once; cached afterwards)
    tsims: dict[str, np.ndarray] | None = None
    if backend != "mock":
        if emb is None:
            probe_dir = cache / "text"
            ph_t = params_hash({"m": model_name, "g": {"pos": s.positive_prompts, "neg": s.negative_prompts, "wide": s.wide_prompts}})
            if not (probe_dir / f"{backend}_{ph_t}.npz").exists():
                emb = make_embedder(cfg, fast)
        tsims = _text_sims(emb, cfg, cache, fast, model_name) if emb is not None else _text_sims(None, cfg, cache, fast, model_name)
    if emb is not None:
        emb.close()
    del det
    if _cuda():
        import torch

        torch.cuda.empty_cache()

    by_clip: dict[str, dict] = {}
    for k in keys:
        f = cache / k / "emb.npz"
        if _emb_ok(f, ph):
            with np.load(f) as z:
                by_clip[k] = {"t": z["t"], "e": z["e"], "person": z["person"]}
    for w in an.windows:
        d = by_clip.get(w["clip_key"])
        w.update({"sem_raw": 0.0, "wide_raw": 0.0, "person": 0.0, "emb": None})
        if d is None or len(d["t"]) == 0:
            continue
        m = (d["t"] >= w["start"]) & (d["t"] < w["end"])
        if not m.any():
            m = np.abs(d["t"] - (w["start"] + w["end"]) / 2).argmin() == np.arange(len(d["t"]))
        e = d["e"][m]
        v = e.mean(axis=0)
        w["emb"] = (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)
        w["person"] = float((d["person"][m] > 0.004).mean())
        w["person_area"] = float(d["person"][m].max())
        if tsims is not None:
            pos = e @ tsims["pos"].T
            neg = e @ tsims["neg"].T
            w["sem_raw"] = float((pos.max(axis=1) - neg.max(axis=1)).mean())
            w["wide_raw"] = float((e @ tsims["wide"].T).max(axis=1).mean())


def _emb_ok(path: Path, ph: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            return str(z["ph"]) == ph
    except Exception:  # noqa: BLE001
        return False


def semantic_scores(windows: list[dict], cfg: Cfg) -> None:
    """Percentile-rank the raw similarity margins across all windows -> ``sem`` in 0..1 (+ person boost)."""
    if not windows:
        return
    raw = np.array([w.get("sem_raw", 0.0) for w in windows])
    wide = np.array([w.get("wide_raw", 0.0) for w in windows])

    def rank(x: np.ndarray) -> np.ndarray:
        if np.ptp(x) < 1e-9:
            return np.full(len(x), 0.5)
        order = x.argsort().argsort()
        return order / max(1, len(x) - 1)

    r, rw = rank(raw), rank(wide)
    boost = float(cfg.semantic.person_boost)
    for w, a, b in zip(windows, r, rw, strict=True):
        w["sem"] = float(np.clip(a + boost * w.get("person", 0.0), 0, 1))
        w["wide"] = float(b)
    _ = os  # noqa: F841
