"""Stage 9b: report.html - thumbnails of selected and rejected windows with scores and rejection reasons."""
from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

import cv2

from .utils import log, progress, read_json


def _thumb(proxy: Path, t: float, out: Path, width: int = 320) -> bool:
    if out.exists():
        return True
    cap = cv2.VideoCapture(str(proxy))
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return False
    h, w = fr.shape[:2]
    fr = cv2.resize(fr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), fr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return True


def _card(w: dict, rel: str, extra: str = "") -> str:
    why = ", ".join(w.get("reasons", [])) or ""
    cls = "bad" if why else "ok"
    return (
        f'<div class="card {cls}"><img loading="lazy" src="{html.escape(rel)}" alt="">'
        f'<div class="m"><b>{html.escape(w["clip_name"])}</b> {w["start"]:.1f}-{w["end"]:.1f}s{extra}<br>'
        f'score {w.get("score", 0):.2f} &middot; tech {w.get("tech", 0):.2f} &middot; sem {w.get("sem", 0):.2f}'
        + (f'<br><span class="why">{html.escape(why)}</span>' if why else "")
        + "</div></div>"
    )


def write_report(build_json: Path, cache: Path, out_html: Path, max_rejected: int = 150) -> Path:
    d = read_json(build_json)
    if not d:
        raise SystemExit(f"no build data at {build_json} - run `dronecut build` first")
    tdir = out_html.parent / "report_thumbs"
    wins = {w["id"]: w for w in d["windows"]}
    sel_ids = [c["window_id"] for c in d["cuts"]]
    selected = set(sel_ids)
    rejected = [w for w in d["windows"] if w.get("reasons")]
    unused = [w for w in d["windows"] if not w.get("reasons") and w["id"] not in selected]
    reasons = Counter(r for w in rejected for r in w["reasons"])
    to_thumb = [wins[i] for i in sel_ids] + unused[:60]
    step = max(1, len(rejected) // max_rejected)
    rej_show = sorted(rejected, key=lambda w: (w["reasons"][0], w["clip_name"], w["start"]))[::step][:max_rejected]
    to_thumb += rej_show
    for w in progress(to_thumb, "thumbnails"):
        proxy = cache / w["clip_key"] / "proxy.mp4"
        if proxy.exists():
            _thumb(proxy, (w["start"] + w["end"]) / 2, tdir / f"{w['id'].replace(':', '_')}.jpg")

    def rel(w: dict) -> str:
        return f"{tdir.name}/{w['id'].replace(':', '_')}.jpg"

    cut_cards = []
    for i, c in enumerate(d["cuts"], 1):
        w = wins[c["window_id"]]
        cut_cards.append(_card(w, rel(w), f' &rarr; <b>#{i}</b> @ {c["out_start"]:.1f}s ({c["bars"]} bar{"s" if c["bars"] > 1 else ""})'))
    m = d.get("music", {})
    tim = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.1f}s</td></tr>" for k, v in d.get("timings", {}).items())
    rc = "".join(f"<li>{html.escape(k)}: {v}</li>" for k, v in reasons.most_common())
    failed = "".join(f"<li>{html.escape(k)} &mdash; {html.escape(v)}</li>" for k, v in d.get("failed", {}).items())
    doc = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>dronecut report</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;background:#14161a;color:#e6e6e6}}
h1,h2{{font-weight:600}} .grid{{display:flex;flex-wrap:wrap;gap:10px}}
.card{{width:320px;background:#1f2228;border-radius:6px;overflow:hidden;border:2px solid #2d6a3f}}
.card.bad{{border-color:#8a3535}} .card img{{display:block;width:320px;height:auto;background:#000}}
.m{{padding:6px 8px;font-size:12px;color:#c7c7c7}} .why{{color:#ff8a80}}
table{{border-collapse:collapse}} td{{padding:2px 12px 2px 0}} code{{color:#9ad}}
</style><h1>dronecut report</h1>
<p>Output: <code>{html.escape(str(d.get("output", "")))}</code> &middot; {len(d["cuts"])} cuts &middot; {d.get("duration", 0):.1f}s
&middot; music: {html.escape(str(m.get("name", "")))} ({m.get("bpm", 0):.1f} BPM, {m.get("bars", 0)} bars) &middot; seed {d.get("seed")}</p>
<h2>Selected ({len(d["cuts"])})</h2><div class="grid">{"".join(cut_cards)}</div>
<h2>Rejected windows ({len(rejected)} of {len(d["windows"])}) &mdash; showing {len(rej_show)}</h2>
<ul>{rc}</ul><div class="grid">{"".join(_card(w, rel(w)) for w in rej_show)}</div>
<h2>Good but unused ({len(unused)}) &mdash; showing {min(60, len(unused))}</h2>
<div class="grid">{"".join(_card(w, rel(w)) for w in unused[:60])}</div>
<h2>Stage timings</h2><table>{tim}</table>
<h2>Clips that failed</h2><ul>{failed or "<li>none</li>"}</ul></html>"""
    out_html.write_text(doc, encoding="utf-8")
    log.info("report: %s", out_html)
    return out_html
