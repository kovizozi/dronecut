"""Stage 9b: report.html - thumbnails of selected and rejected windows, with a must-have / never-use picker."""
from __future__ import annotations

import html
import json
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
    state = "in" if w.get("pinned") else "ex" if "excluded" in w.get("reasons", []) else ""
    over = w.get("overridden")
    note = f'<br><span class="ovr">quality filter overridden: {html.escape(", ".join(over))}</span>' if over else ""
    cap = f'<br><i>{html.escape(w["caption"])}</i>' if w.get("caption") else ""
    return (
        f'<div class="card {cls}" data-id="{html.escape(w["id"])}" data-clip="{html.escape(w["clip_name"])}" '
        f'data-start="{w["start"]:.2f}" data-end="{w["end"]:.2f}" data-state="{state}" data-init="{state}">'
        f'<img loading="lazy" src="{html.escape(rel)}" alt="">'
        f'<div class="pick"><button class="in" title="must have in the edit">&#11088; must have</button>'
        f'<button class="ex" title="never use">&#128683; never</button></div>'
        f'<div class="m"><b>{html.escape(w["clip_name"])}</b> {w["start"]:.1f}-{w["end"]:.1f}s{extra}<br>'
        f'score {w.get("score", 0):.2f} &middot; tech {w.get("tech", 0):.2f} &middot; sem {w.get("sem", 0):.2f}'
        + (f'<br><span class="why">{html.escape(why)}</span>' if why else "")
        + note + cap + "</div></div>"
    )


JS = """
const KEY = 'dronecut-pins:' + location.pathname;
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
const cards = [...document.querySelectorAll('.card')];
function setState(c, s) { c.dataset.state = s; }
cards.forEach(c => { if (saved[c.dataset.id] !== undefined) setState(c, saved[c.dataset.id]); });
function refresh() {
  const st = {};
  cards.forEach(c => { if (c.dataset.state) st[c.dataset.id] = c.dataset.state; else st[c.dataset.id] = ''; });
  try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {}
  const ids = new Set();
  let nin = 0, nex = 0;
  cards.forEach(c => { if (c.dataset.state === 'in') { nin++; } if (c.dataset.state === 'ex') { nex++; } });
  document.getElementById('n_in').textContent = nin;
  document.getElementById('n_ex').textContent = nex;
}
cards.forEach(c => {
  c.querySelector('button.in').onclick = () => { setState(c, c.dataset.state === 'in' ? '' : 'in'); syncTwins(c); refresh(); };
  c.querySelector('button.ex').onclick = () => { setState(c, c.dataset.state === 'ex' ? '' : 'ex'); syncTwins(c); refresh(); };
});
function syncTwins(c) { cards.forEach(o => { if (o !== c && o.dataset.id === c.dataset.id) setState(o, c.dataset.state); }); }
function collect() {
  // pins from the previous build stay unless the user switched their card off; new card picks are added
  const out = {include: [], exclude: []};
  const cleared = cards.filter(c => c.dataset.init && !c.dataset.state);
  for (const kind of ['include', 'exclude']) {
    for (const p of BASE[kind] || []) {
      const want = kind === 'include' ? 'in' : 'ex';
      const drop = cleared.some(c => c.dataset.init === want && c.dataset.clip.toLowerCase().includes(String(p.clip).toLowerCase()) &&
        (p.start == null || (+c.dataset.start < p.end && p.start < +c.dataset.end)));
      if (!drop) out[kind].push(p);
    }
  }
  const seen = new Set();
  cards.forEach(c => {
    const s = c.dataset.state;
    if (!s || s === c.dataset.init) return;
    const e = {clip: c.dataset.clip, start: +c.dataset.start, end: +c.dataset.end};
    const key = s + e.clip + e.start + e.end;
    if (seen.has(key)) return; seen.add(key);
    (s === 'in' ? out.include : out.exclude).push(e);
  });
  return out;
}
document.getElementById('dl').onclick = () => {
  const blob = new Blob([JSON.stringify(collect(), null, 1)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'must_have.json'; a.click();
};
if (location.protocol.indexOf('http') === 0) {
  const sv = document.getElementById('sv'); sv.style.display = '';
  sv.onclick = async () => {
    try {
      const r = await fetch('/api/pins', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Requested-With': 'dronecut'}, body: JSON.stringify(collect())});
      const j = await r.json(); if (!r.ok) throw new Error(j.error);
      sv.textContent = 'Saved (' + j.include + ' must-have, ' + j.exclude + ' never) - now press Build again';
    } catch (e) { alert('Could not save: ' + e.message); }
  };
}
document.getElementById('clr').onclick = () => { cards.forEach(c => setState(c, '')); refresh(); };
document.getElementById('flt').oninput = e => {
  const q = e.target.value.toLowerCase();
  cards.forEach(c => { c.style.display = (!q || c.dataset.clip.toLowerCase().includes(q)) ? '' : 'none'; });
};
document.getElementById('only').onchange = e => {
  cards.forEach(c => { if (e.target.checked && !c.dataset.state) c.style.display = 'none'; else if (!e.target.checked) c.style.display = ''; });
};
refresh();
"""


def write_report(build_json: Path, cache: Path, out_html: Path, max_rejected: int = 150, max_unused: int = 400) -> Path:
    d = read_json(build_json)
    if not d:
        raise SystemExit(f"no build data at {build_json} - run `dronecut build` first")
    tdir = out_html.parent / "report_thumbs"
    wins = {w["id"]: w for w in d["windows"]}
    sel_ids = [c["window_id"] for c in d["cuts"]]
    selected = set(sel_ids)
    rejected = [w for w in d["windows"] if w.get("reasons")]
    unused = sorted((w for w in d["windows"] if not w.get("reasons") and w["id"] not in selected),
                    key=lambda w: (w["clip_name"], w["start"]))[:max_unused]
    reasons = Counter(r for w in rejected for r in w["reasons"])
    step = max(1, len(rejected) // max_rejected)
    rej_show = sorted(rejected, key=lambda w: (w["reasons"][0], w["clip_name"], w["start"]))[::step][:max_rejected]
    rej_show += [w for w in rejected if "excluded" in w["reasons"] and w not in rej_show]
    to_thumb = [wins[i] for i in sel_ids] + unused + rej_show
    for w in progress(to_thumb, "thumbnails"):
        proxy = cache / w["clip_key"] / "proxy.mp4"
        if proxy.exists():
            _thumb(proxy, (w["start"] + w["end"]) / 2, tdir / f"{w['id'].replace(':', '_')}.jpg")

    def rel(w: dict) -> str:
        return f"{tdir.name}/{w['id'].replace(':', '_')}.jpg"

    cut_cards = []
    for i, c in enumerate(d["cuts"], 1):
        w = wins[c["window_id"]]
        star = " &#11088;" if c.get("pinned") else ""
        cut_cards.append(_card(w, rel(w), f' &rarr; <b>#{i}</b>{star} @ {c["out_start"]:.1f}s ({c["bars"]} bar{"s" if c["bars"] > 1 else ""})'))
    m = d.get("music", {})
    tim = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.1f}s</td></tr>" for k, v in d.get("timings", {}).items())
    rc = "".join(f"<li>{html.escape(k)}: {v}</li>" for k, v in reasons.most_common())
    failed = "".join(f"<li>{html.escape(k)} &mdash; {html.escape(v)}</li>" for k, v in d.get("failed", {}).items())
    tele = d.get("telemetry")
    tele_s = f" &middot; HUD: {html.escape(str(tele))}" if tele else ""
    doc = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>dronecut report</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;background:#14161a;color:#e6e6e6;padding-top:56px}}
h1,h2{{font-weight:600}} .grid{{display:flex;flex-wrap:wrap;gap:10px}}
.card{{width:320px;background:#1f2228;border-radius:6px;overflow:hidden;border:2px solid #2d6a3f;position:relative}}
.card.bad{{border-color:#8a3535}} .card img{{display:block;width:320px;height:auto;background:#000}}
.card[data-state=in]{{border-color:#f5c518;box-shadow:0 0 0 2px #f5c518}}
.card[data-state=ex]{{border-color:#777;opacity:.45}}
.pick{{position:absolute;top:6px;right:6px;display:flex;gap:4px}}
.pick button{{background:#000a;color:#fff;border:1px solid #fff5;border-radius:4px;padding:2px 6px;font-size:12px;cursor:pointer}}
.card[data-state=in] button.in{{background:#f5c518;color:#000}} .card[data-state=ex] button.ex{{background:#999;color:#000}}
.m{{padding:6px 8px;font-size:12px;color:#c7c7c7}} .why{{color:#ff8a80}} .ovr{{color:#f5c518}}
table{{border-collapse:collapse}} td{{padding:2px 12px 2px 0}} code{{color:#9ad}}
#bar{{position:fixed;top:0;left:0;right:0;background:#0d0f12;border-bottom:1px solid #333;padding:8px 24px;z-index:9;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
#bar button{{cursor:pointer;padding:4px 10px}} #bar input[type=text]{{padding:3px 6px}}
</style>
<div id="bar"><span>&#11088; must have: <b id="n_in">0</b> &nbsp; &#128683; never: <b id="n_ex">0</b></span>
<button id="sv" style="display:none">Save picks to project</button><button id="dl">Download must_have.json</button><button id="clr">Clear</button>
<input id="flt" type="text" placeholder="filter by clip name"><label><input id="only" type="checkbox"> only my picks</label>
<span>Save the file next to <code>config.yaml</code>, then re-run <code>dronecut build ...</code> (it is picked up automatically).</span></div>
<h1>dronecut report</h1>
<p>Output: <code>{html.escape(str(d.get("output", "")))}</code> &middot; {len(d["cuts"])} cuts &middot; {d.get("duration", 0):.1f}s
&middot; music: {html.escape(str(m.get("name", "")))} ({m.get("bpm", 0):.1f} BPM, {m.get("bars", 0)} bars) &middot; seed {d.get("seed")}{tele_s}</p>
<p>Click &#11088; on any thumbnail to force that part into the next edit (it overrides the quality filter), or &#128683; to never use it.
Any other time range: <code>dronecut build ... --include "0010@1:05-1:12"</code>.</p>
<h2>Selected ({len(d["cuts"])})</h2><div class="grid">{"".join(cut_cards)}</div>
<h2>Good but unused ({len(unused)}) </h2>
<div class="grid">{"".join(_card(w, rel(w)) for w in unused)}</div>
<h2>Rejected windows ({len(rejected)} of {len(d["windows"])}) &mdash; showing {len(rej_show)}</h2>
<ul>{rc}</ul><div class="grid">{"".join(_card(w, rel(w)) for w in rej_show)}</div>
<h2>Stage timings</h2><table>{tim}</table>
<h2>Clips that failed</h2><ul>{failed or "<li>none</li>"}</ul>
<script>const BASE = {json.dumps(d.get("pins") or {"include": [], "exclude": []})};
{JS}</script></html>"""
    out_html.write_text(doc, encoding="utf-8")
    log.info("report: %s", out_html)
    return out_html
