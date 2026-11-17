"""Optional stage (``--vlm``): local VLM captions + 1-10 'interestingness' score for the top candidates.

Qwen2.5-VL-7B-Instruct in 4-bit (fits 12 GB) with Moondream2 as a lightweight fallback. The model is loaded
only here, after SigLIP has been released, and freed again before rendering.
"""
from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import cv2

from .config import Cfg, model_cache_dir
from .utils import log, params_hash, progress, read_json, write_json

PROMPT = (
    "These are 3 frames from a short drone video clip of a family outing. Rate how interesting and "
    "watchable the clip would be in a highlight reel (people, action, beautiful scenery, nice camera movement "
    "score high; boring, empty, cluttered or messy footage scores low). "
    'Reply with JSON only: {"caption": "<one short sentence>", "score": <integer 1-10>}'
)


def _keyframes(proxy: Path, w: dict, n: int = 3):
    from PIL import Image

    cap = cv2.VideoCapture(str(proxy))
    frames = []
    for i in range(n):
        t = w["start"] + (w["end"] - w["start"]) * (i + 1) / (n + 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if ok:
            h, ww = fr.shape[:2]
            s = 560 / max(h, ww)
            fr = cv2.resize(fr, (int(ww * s), int(h * s)))
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def _parse(text: str) -> tuple[str, float | None]:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return str(d.get("caption", "")).strip(), float(d["score"])
        except (ValueError, KeyError, TypeError):
            pass
    sc = re.search(r"(?:score\D{0,5})?(\d{1,2})\s*(?:/\s*10)?", text)
    return text.strip()[:140], float(sc.group(1)) if sc else None


class _Qwen:
    def __init__(self, cfg: Cfg):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        cache = str(model_cache_dir(cfg))
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        mid = cfg.vlm.model
        log.info("loading %s (4-bit)", mid)
        self.processor = AutoProcessor.from_pretrained(mid, cache_dir=cache, min_pixels=128 * 28 * 28, max_pixels=384 * 28 * 28)
        self.model = AutoModelForImageTextToText.from_pretrained(mid, cache_dir=cache, quantization_config=q, device_map={"": 0}, torch_dtype=torch.float16)
        self.torch = torch

    def ask(self, images) -> str:
        msgs = [{"role": "user", "content": [*({"type": "image", "image": im} for im in images), {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=90, do_sample=False)
        return self.processor.batch_decode(out[:, inp["input_ids"].shape[1] :], skip_special_tokens=True)[0]

    def close(self) -> None:
        del self.model


class _Moondream:
    def __init__(self, cfg: Cfg):
        import torch
        from transformers import AutoModelForCausalLM

        mid = cfg.vlm.fallback_model
        log.info("loading fallback %s", mid)
        self.model = AutoModelForCausalLM.from_pretrained(mid, cache_dir=str(model_cache_dir(cfg)), trust_remote_code=True,
                                                          torch_dtype=torch.float16, device_map={"": 0})
        self.torch = torch

    def ask(self, images) -> str:
        im = images[len(images) // 2]
        cap = self.model.caption(im, length="short")["caption"]
        ans = self.model.query(im, "On a scale of 1 to 10, how interesting is this drone shot for a family highlight video? Answer with a number only.")["answer"]
        return json.dumps({"caption": cap, "score": float(re.search(r"\d+", ans).group(0)) if re.search(r"\d+", ans) else 5})

    def close(self) -> None:
        del self.model


def run(an, proxies: dict[str, Path], cfg: Cfg, cache: Path, seed: int = 0) -> None:
    from .select import score_windows

    ok, _ = score_windows(an.windows, cfg)  # provisional ranking (no VLM term yet)
    top = sorted(ok, key=lambda w: -w["score"])[: int(cfg.vlm.top_k)]
    ph = params_hash({"m": cfg.vlm.model, "p": PROMPT})
    store = cache / "vlm.json"
    saved = read_json(store) or {}
    if saved.get("_params") != ph:
        saved = {"_params": ph}
    todo = [w for w in top if w["id"] not in saved]
    if todo:
        model = None
        try:
            model = _Qwen(cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("Qwen2.5-VL unavailable (%s) - falling back to Moondream2", str(e)[:200])
            try:
                model = _Moondream(cfg)
            except Exception as e2:  # noqa: BLE001
                log.error("no VLM could be loaded: %s", str(e2)[:200])
        if model is not None:
            for w in progress(todo, "VLM captions"):
                try:
                    frames = _keyframes(proxies[w["clip_key"]], w)
                    cap, sc = _parse(model.ask(frames))
                    if sc is not None:
                        saved[w["id"]] = {"caption": cap, "score": max(1.0, min(10.0, sc))}
                except Exception as e:  # noqa: BLE001
                    log.warning("VLM failed on %s: %s", w["id"], str(e)[:160])
            model.close()
            del model
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        write_json(store, saved)
    for w in an.windows:
        r = saved.get(w["id"])
        if r:
            w["caption"] = r["caption"]
            w["vlm"] = (r["score"] - 1.0) / 9.0
