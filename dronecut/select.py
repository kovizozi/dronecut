"""Stage 6 + beat alignment of the edit: scoring, diversity, narrative order, cut list on the bar grid."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from . import score_semantic, score_tech
from .config import Cfg
from .music import MusicPlan
from .pins import Pins, apply_pins
from .utils import log


@dataclass
class Cut:
    window: dict
    src_start: float  # seconds into the original clip
    src_dur: float  # seconds of source used (= out_dur * speed)
    out_start: float  # seconds on the output timeline (beat/bar aligned)
    out_dur: float
    speed: float = 1.0  # source seconds per output second (<1 = slow motion)
    bars: int = 1


@dataclass
class Selection:
    cuts: list[Cut] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)  # windows with reasons
    candidates: list[dict] = field(default_factory=list)


def score_windows(windows: list[dict], cfg: Cfg, pins: Pins | None = None) -> tuple[list[dict], list[dict]]:
    """Attach tech/semantic/combined scores and rejection reasons. Returns (accepted, rejected).

    ``pins`` (must-have / never-use parts) override the quality filter: pinned windows are always accepted.
    """
    gs = score_tech.global_stats([w["feat"] for w in windows])
    score_semantic.semantic_scores(windows, cfg)
    wt = cfg.select.weights
    tot = (wt["technical"] + wt["semantic"]) or 1.0
    vlm_w = float(cfg.vlm.weight)
    ok, bad = [], []
    for w in windows:
        t, why = score_tech.evaluate(w["feat"], gs, cfg.tech)
        alt = w["feat"].get("alt")
        min_alt = float(cfg.tech.get("min_altitude_m", 0.0))
        if alt is not None and min_alt > 0 and alt < min_alt:
            why.append("ground_level")
        w["tech"], w["reasons"] = t, why
        base = (wt["technical"] * t + wt["semantic"] * w.get("sem", 0.5)) / tot
        if w.get("vlm") is not None:
            base = (1 - vlm_w) * base + vlm_w * float(w["vlm"])
        w["score"] = float(base)
    if pins:
        apply_pins(windows, pins, float(cfg.segment.max_window_s))
    for w in windows:
        (bad if w["reasons"] else ok).append(w)
    return ok, bad


def _pick_candidates(ok: list[dict], need_s: float, cfg: Cfg, rng: random.Random, bar_dur: float,
                     forced: list[dict] | None = None) -> list[dict]:
    """Greedy by score with cluster limits and a min gap between picks of the same clip.

    ``forced`` (user-pinned) windows are taken first and are not subject to any of the limits.
    """
    from sklearn.cluster import KMeans

    pool = [w for w in ok if w["end"] - w["start"] >= bar_dur - 1e-6]
    if not pool:
        return []
    embs = [w.get("emb") for w in pool]
    k = int(min(cfg.select.clusters, len(pool)))
    labels = np.zeros(len(pool), int)
    if k > 1 and all(e is not None for e in embs):
        X = np.stack(embs)
        labels = KMeans(n_clusters=k, n_init=4, random_state=rng.randrange(2**31)).fit_predict(X)
    for w, lab in zip(pool, labels, strict=True):
        w["cluster"] = int(lab)
    order = sorted(pool, key=lambda w: (-w["score"], w["id"]))
    n_est = max(4, int(np.ceil(need_s / max(bar_dur * 2, 1.0))))
    cap = max(1, int(np.ceil(cfg.select.max_per_cluster_frac * n_est * 1.5)))
    gap = float(cfg.select.min_gap_s)
    picked: list[dict] = []
    per_cluster: dict[int, int] = {}
    dur_sum = 0.0
    pool_ids = {p["id"] for p in pool}
    for w in forced or []:
        if w["id"] in pool_ids:
            picked.append(w)
            per_cluster[w["cluster"]] = per_cluster.get(w["cluster"], 0) + 1
            dur_sum += min(int(np.floor((w["end"] - w["start"]) / bar_dur + 1e-6)), 2) * bar_dur

    def ok_gap(w: dict) -> bool:
        for p in picked:
            if p["clip_key"] == w["clip_key"] and not (w["start"] >= p["end"] + gap or w["end"] <= p["start"] - gap):
                return False
        return True

    for relax in (0, 1, 2):  # progressively relax constraints only if footage is scarce
        for w in order:
            if w in picked:
                continue
            if relax < 2 and per_cluster.get(w["cluster"], 0) >= cap * (1 + relax):
                continue
            if relax < 1 and not ok_gap(w):
                continue
            if relax == 1 and any(p["clip_key"] == w["clip_key"] and p["start"] < w["end"] and w["start"] < p["end"] for p in picked):
                continue
            picked.append(w)
            per_cluster[w["cluster"]] = per_cluster.get(w["cluster"], 0) + 1
            dur_sum += min(int(np.floor((w["end"] - w["start"]) / bar_dur + 1e-6)), 2) * bar_dur
            if dur_sum >= need_s * 1.3:
                break
        if dur_sum >= need_s * 1.3:
            break
    return picked


def _order_narrative(picks: list[dict], cfg: Cfg) -> list[dict]:
    """Chronological by clip name/start, opening with a wide establishing shot, closing with a pull-away."""
    if len(picks) < 3:
        return sorted(picks, key=lambda w: (w["clip_name"], w["start"]))
    chrono = sorted(picks, key=lambda w: (w["clip_name"], w["start"]))
    opener = closer = None
    if cfg.select.open_with_wide:
        top = sorted(picks, key=lambda w: -w["score"])[: max(3, len(picks) // 3)]
        opener = max(top, key=lambda w: (w.get("wide", 0.0) * 0.6 + w["tech"] * 0.4, w["id"]))
    if cfg.select.close_with_pullaway:
        rest = [w for w in picks if w is not opener]
        pull = [w for w in rest if w["feat"]["zoom"] < -0.03]
        pool = pull or sorted(rest, key=lambda w: -w["score"])[: max(3, len(rest) // 3)]
        closer = max(pool, key=lambda w: (-w["feat"]["zoom"] + 0.5 * w.get("wide", 0.0) + 0.3 * w["score"], w["id"]))
    middle = [w for w in chrono if w is not opener and w is not closer]
    return ([opener] if opener else []) + middle + ([closer] if closer else [])


def _layout(ordered: list[dict], plan: MusicPlan, cfg: Cfg, bar_dur: float) -> list[Cut]:
    """Lay the ordered windows on the bar grid of the music plan (every cut starts and ends on a downbeat)."""
    cuts: list[Cut] = []
    bar = 0
    t_out = 0.0
    slow = float(cfg.render.slowmo_factor)
    for w in ordered:
        if bar >= plan.n_bars:
            break
        wl = w["end"] - w["start"]
        feas = int(np.floor(wl / bar_dur + 2e-3))
        if feas < 1:
            continue
        smooth = w["feat"]["shake"] < 0.35 * cfg.tech.shake_max and cfg.tech.motion_target[0] <= w["feat"]["motion"] <= cfg.tech.motion_target[1]
        speed = slow if (slow < 1.0 and smooth) else 1.0
        nb = min(plan.bars_pref[bar], feas, plan.n_bars - bar)
        if w.get("hud"):  # a HUD moment keeps the whole long shot, at normal speed
            speed, nb = 1.0, min(feas, plan.n_bars - bar)
            if nb < feas:  # not enough music left for the full-length shot: skip, the fitting step makes room
                continue
        if speed < 1.0:  # slow-mo consumes fewer source seconds; may allow longer output
            nb = min(plan.bars_pref[bar], max(1, int(np.floor(wl / speed / bar_dur + 1e-6))), plan.n_bars - bar)
        out_dur = float(plan.bar_times[bar + nb] - plan.bar_times[bar])
        src_dur = out_dur * speed
        if src_dur > wl + 2e-3:
            speed, src_dur = 1.0, out_dur
            if src_dur > wl + 2e-3:
                continue
        src_start = w["start"] + (wl - src_dur) / 2.0
        if w.get("hud"):
            src_start = w["start"]
        elif w.get("pinned") and w.get("pin_range"):  # centre the cut on the part the user asked for
            a, b = w["pin_range"]
            src_start = min(max((a + b) / 2.0 - src_dur / 2.0, w["start"]), w["end"] - src_dur)
        cuts.append(Cut(w, round(src_start, 3), round(src_dur, 4), round(t_out, 4), round(out_dur, 4), speed, nb))
        t_out += out_dur
        bar += nb
    return cuts


def _overlaps(a: dict, b: dict) -> bool:
    return a["clip_key"] == b["clip_key"] and a["start"] < b["end"] and b["start"] < a["end"]


def build_cuts(windows: list[dict], plan: MusicPlan, cfg: Cfg, seed: int = 0, pins: Pins | None = None,
               tels: dict | None = None) -> Selection:
    """Pick windows, order them, and lay them on the bar grid of the music plan (cuts land on downbeats)."""
    rng = random.Random(seed)
    ok, bad = score_windows(windows, cfg, pins)
    sel = Selection(rejected=bad, candidates=ok)
    bar_dur = float(np.median(np.diff(plan.bar_times)))
    forced = [w for w in ok if w.get("pinned")]
    too_short = [w for w in forced if w["end"] - w["start"] < bar_dur - 1e-6]
    for w in too_short:
        log.warning("must-have %s %.1f-%.1fs is shorter than one bar (%.1fs) and cannot be placed on the beat",
                    w["clip_name"], w["start"], w["end"], bar_dur)
    forced = [w for w in forced if w not in too_short]
    if cfg.hud.enabled and cfg.hud.mode == "moments" and tels:
        from .moments import find_moments

        moments = find_moments(ok, tels, bar_dur, cfg, avoid=forced)
        if moments:
            log.info("HUD moments: %s", ", ".join(f"{m['hud_kind']} ({m['clip_name'][4:18]} {m['start']:.0f}-{m['end']:.0f}s)" for m in moments))
            windows.extend(moments)  # so build.json / the report can show them
            ok = [w for w in ok if not any(_overlaps(w, m) for m in moments)] + moments
            sel.candidates = ok
            forced = forced + moments
    picks = _pick_candidates(ok, plan.duration, cfg, rng, bar_dur, forced)
    ordered = _order_narrative(picks, cfg)
    cuts = _layout(ordered, plan, cfg, bar_dur)
    # must-haves win over everything else: if the bars ran out, drop the weakest ordinary shots and re-lay
    while True:
        used = {c.window["id"] for c in cuts}
        missing = [w for w in forced if w["id"] not in used]
        spare = [w for w in ordered if w["id"] in used and not w.get("pinned") and not w.get("hud")]
        if not missing or not spare:
            break
        drop = min(spare, key=lambda w: (w["score"], w["id"]))
        ordered = [w for w in ordered if w is not drop]
        cuts = _layout(ordered, plan, cfg, bar_dur)
    used = {c.window["id"] for c in cuts}
    lost = [w for w in forced if w["id"] not in used]
    if lost:
        log.warning("%d must-have part(s) did not fit into %.0fs of music - raise --duration: %s", len(lost), plan.duration,
                    ", ".join(f"{w['clip_name']}@{w['start']:.0f}s" for w in lost[:5]))
    sel.cuts = cuts
    return sel
