"""Stage 4: technical scoring with OpenCV only (sharpness, exposure, stability, motion, junk heuristics).

Frame-level metrics are computed once per proxy and cached; per-window features are cheap aggregates.
Hard-reject thresholds are applied in :func:`evaluate` (selection time), so tuning them never needs re-analysis.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

cv2.setNumThreads(1)

FRAME_KEYS = ("sharp", "luma", "clip_low", "clip_high", "sky", "occl", "tx", "ty", "zoom", "mot", "chaos", "b", "g", "r")


def frame_metrics(proxy: str, tech: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per-frame metrics over a proxy video. Flow-based metrics at index i compare frame i-1 -> i."""
    cap = cv2.VideoCapture(proxy)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open proxy {proxy}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    fw = int(tech["flow_width"])
    sw = int(tech["sharp_width"])
    rows: list[list[float]] = []
    prev = None
    grid = None
    hann = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h0, w0 = frame.shape[:2]
        gray_s = cv2.cvtColor(cv2.resize(frame, (sw, int(h0 * sw / w0)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray_s, cv2.CV_64F).var())
        small = cv2.resize(frame, (fw, int(h0 * fw / w0)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        luma = float(gray.mean())
        gblur = cv2.GaussianBlur(gray, (0, 0), 1.5)
        clip_low = float((gray < 8).mean())
        clip_high = float((gray > 247).mean())
        # sky: low gradient and (bluish bright | bright neutral)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        smooth = (np.abs(gx) + np.abs(gy)) < 24.0
        b, g, r = (small[..., i].astype(np.int16) for i in range(3))
        v = small.max(axis=2)
        mn = small.min(axis=2)
        sat = (v.astype(np.int16) - mn) / np.maximum(v, 1)
        skylike = ((b > r + 25) & (v > 130)) | ((sat < 0.14) & (v > 175))
        sky = float((smooth & skylike).mean())
        # occlusion: big near-black, textureless blocks (finger / prop / lens cap)
        bh, bw = gray.shape[0] // 8 * 8, gray.shape[1] // 8 * 8
        blocks = gray[:bh, :bw].reshape(8, bh // 8, 8, bw // 8).transpose(0, 2, 1, 3).reshape(64, -1)
        occl = float(((blocks.std(axis=1) < 4.0) & (blocks.mean(axis=1) < 40.0)).mean())
        tx = ty = zoom = mot = chaos = 0.0
        if prev is not None and prev.shape == gblur.shape:
            flow = cv2.calcOpticalFlowFarneback(prev, gblur, None, 0.5, 5, 25, 3, 7, 1.5, 0)
            fxs, fys = flow[..., 0], flow[..., 1]
            tx, ty = float(np.median(fxs)), float(np.median(fys))
            # phase correlation is far more robust than dense flow for large global shifts (camera shake)
            if hann is None or hann.shape != gblur.shape[::-1] and hann.shape != gblur.shape:
                hann = cv2.createHanningWindow(gblur.shape[::-1], cv2.CV_32F)
            (px, py), resp = cv2.phaseCorrelate(prev.astype(np.float32), gblur.astype(np.float32), hann)
            if resp > 0.15 and np.hypot(px, py) > np.hypot(tx, ty):
                tx, ty = float(px), float(py)
            if grid is None or grid[0].shape != fxs.shape:
                hh, ww = fxs.shape
                yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
                xx -= ww / 2.0
                yy -= hh / 2.0
                grid = (xx, yy, float((xx**2 + yy**2).sum()))
            xx, yy, rr = grid
            zoom = float(((fxs - tx) * xx + (fys - ty) * yy).sum() / rr)  # log-scale change per frame
            mag = np.hypot(fxs, fys)
            mot = max(float(np.median(mag)), float(np.hypot(tx, ty)))
            chaos = float(np.median(np.hypot(fxs - tx, fys - ty)))
        mean_bgr = small.reshape(-1, 3).mean(axis=0)
        rows.append([sharp, luma, clip_low, clip_high, sky, occl, tx, ty, zoom, mot, chaos, *map(float, mean_bgr)])
        prev = gblur
    cap.release()
    arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, len(FRAME_KEYS)), np.float32)
    out = {k: arr[:, i] for i, k in enumerate(FRAME_KEYS)}
    out["fps"] = np.asarray([fps], np.float32)
    return out


def window_features(fm: dict[str, np.ndarray], start: float, end: float, clip_dur: float) -> dict[str, float] | None:
    fps = float(fm["fps"][0])
    n = len(fm["sharp"])
    i0 = max(0, int(np.ceil(start * fps - 1e-6)))
    i1 = min(n, int(np.floor(end * fps + 1e-6)))
    if i1 - i0 < 3:
        return None
    sl = slice(i0, i1)
    fl = slice(min(i0 + 1, i1 - 1), i1)  # skip first frame: its flow may cross a scene cut
    tx, ty = fm["tx"][fl].astype(np.float64), fm["ty"][fl].astype(np.float64)
    vel = np.stack([tx, ty], axis=1)
    k = min(5, len(vel))
    if k >= 3:
        ker = np.ones(k) / k
        smoothed = np.stack([np.convolve(np.pad(vel[:, j], (k // 2, k - 1 - k // 2), mode="edge"), ker, "valid") for j in range(2)], 1)
    else:
        smoothed = np.tile(vel.mean(axis=0), (len(vel), 1))
    resid = vel - smoothed
    shake = float(np.sqrt((resid**2).sum(axis=1).mean()))
    jerk = float(np.hypot(*np.diff(vel, axis=0).T).max()) if len(vel) > 1 else 0.0
    zoom_s = fm["zoom"][fl].astype(np.float64) * fps
    return {
        "sharp": float(np.median(fm["sharp"][sl])),
        "luma": float(fm["luma"][sl].mean()),
        "luma_p5": float(np.percentile(fm["luma"][sl], 5)),
        "clip_low": float(fm["clip_low"][sl].mean()),
        "clip_high": float(fm["clip_high"][sl].mean()),
        "sky": float(fm["sky"][sl].mean()),
        "occl": float(fm["occl"][sl].mean()),
        "shake": shake,
        "jerk": jerk,
        "motion": float(np.median(fm["mot"][fl]) * fps),
        "chaos": float(np.median(fm["chaos"][fl]) * fps),
        "zoom": float(zoom_s.mean()),
        "zoom_abs": float(np.abs(zoom_s).max()),
        "mean_b": float(fm["b"][sl].mean()),
        "mean_g": float(fm["g"][sl].mean()),
        "mean_r": float(fm["r"][sl].mean()),
        "dist_start": float(start),
        "dist_end": float(clip_dur - end),
    }


def global_stats(feats: list[dict[str, float]]) -> dict[str, float]:
    if not feats:
        return {"sharp_ref": 1.0}
    sharp = np.array([f["sharp"] for f in feats])
    return {"sharp_ref": float(np.median(sharp)), "sharp_p75": float(np.percentile(sharp, 75))}


def _bump(x: float, lo: float, hi: float) -> float:
    """1 inside [lo, hi], smooth fall-off outside (to 0 at lo/4 and 4*hi)."""
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return float(np.clip((x - lo / 4) / (lo - lo / 4 + 1e-9), 0, 1))
    return float(np.clip(1 - (x - hi) / (3 * hi + 1e-9), 0, 1))


def evaluate(f: dict[str, float], gs: dict[str, float], tech: dict[str, Any]) -> tuple[float, list[str]]:
    """Return (technical score 0..1, rejection reasons). Reasons empty = window is acceptable."""
    why: list[str] = []
    ref = max(gs.get("sharp_ref", 1.0), 1e-6)
    if f["sharp"] < tech["min_sharpness_abs"] or f["sharp"] < tech["min_sharpness_rel"] * ref:
        why.append("blurry")
    if f["luma"] < tech["luma_min"]:
        why.append("underexposed")
    if f["luma"] > tech["luma_max"]:
        why.append("overexposed")
    if f["clip_high"] > tech["clip_high_max"]:
        why.append("blown_highlights")
    if f["clip_low"] > tech["clip_low_max"]:
        why.append("crushed_blacks")
    if f["shake"] > tech["shake_max"]:
        why.append("shaky")
    if f["jerk"] > tech["jerk_max"]:
        why.append("sudden_jerk")
    if f["motion"] < tech["motion_min"]:
        why.append("static")
    if f["motion"] > tech["motion_max"]:
        why.append("chaotic")
    if f["zoom_abs"] > tech["zoom_rate_max"] * 1.6 or abs(f["zoom"]) > tech["zoom_rate_max"]:
        why.append("rapid_zoom")
    if f["dist_start"] < tech["edge_s"] and f["zoom"] < -tech["edge_zoom_rate"]:
        why.append("takeoff")
    if f["dist_end"] < tech["edge_s"] and f["zoom"] > tech["edge_zoom_rate"]:
        why.append("landing")
    if f["sky"] > tech["sky_max"]:
        why.append("sky_only")
    if f["occl"] > tech["occlusion_max"]:
        why.append("occluded")

    s_sharp = float(np.clip(f["sharp"] / (1.5 * ref), 0, 1) ** 0.5)
    exp_pen = abs(f["luma"] - 120.0) / 120.0 + 1.5 * f["clip_high"] + 0.8 * f["clip_low"]
    s_exp = float(np.clip(1.0 - exp_pen, 0, 1))
    s_stab = float(np.exp(-f["shake"] / max(tech["shake_max"], 1e-6)) * np.exp(-f["jerk"] / max(tech["jerk_max"] * 2, 1e-6)))
    lo, hi = tech["motion_target"]
    s_mot = _bump(f["motion"], lo, hi) * float(np.exp(-max(0.0, f["chaos"] - 0.5 * hi) / hi))
    w = tech["weights"]
    tot = sum(w.values()) or 1.0
    score = (w["sharpness"] * s_sharp + w["exposure"] * s_exp + w["stability"] * s_stab + w["motion"] * s_mot) / tot
    return float(np.clip(score, 0, 1)), why
