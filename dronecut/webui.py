"""Local web interface (``dronecut ui``): control the whole pipeline from the browser, with local folders.

Zero extra dependencies (stdlib http.server). Binds to 127.0.0.1 only. Every request needs the per-run token (set as
a SameSite=Strict cookie when you open the printed URL), the Host header must be local, and POSTs must carry a custom
header - so no other web page or machine can drive it.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import secrets
import shutil
import string
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import yaml

from . import __version__, ingest
from . import music as music_mod
from .config import _merge, load_config
from .utils import file_key, run, which

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
AUDIO_EXT = set(music_mod.AUDIO_EXTS)
SERVE_EXT = VIDEO_EXT | AUDIO_EXT | {".jpg", ".jpeg", ".png", ".html", ".json", ".fcpxml", ".srt", ".txt", ".css", ".js"}
STAGES = ["probe", "proxy", "segment+technical", "semantic", "vlm", "music", "select", "render", "export", "report"]
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

DEFAULT_SETTINGS: dict = {
    "footage": "", "music_mode": "auto", "music_dir": "", "music_file": "", "output_dir": "output", "cache_dir": ".dronecut_cache",
    "duration": 150, "res": 1080, "seed": 1, "fast": False, "vlm": False, "skip_clips": [], "overrides": {}, "extra_yaml": "",
}


class Job:
    def __init__(self, kind: str, cmd: list[str], cwd: str):
        self.id = secrets.token_hex(4)
        self.kind, self.cmd = kind, cmd
        self.lines: list[str] = []
        self.state = "running"
        self.rc: int | None = None
        self.stage = ""
        self.started = time.time()
        self.ended: float | None = None
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", NO_COLOR="1", COLUMNS="200", TERM="dumb")
        self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout
        for raw in self.proc.stdout:
            line = ANSI.sub("", raw.rstrip("\r\n")).rstrip()
            if not line.strip():
                continue
            self.lines.append(line)
            m = re.search(r">\s+([a-z+]+)", line)
            if m and m.group(1) in STAGES:
                self.stage = m.group(1)
        self.rc = self.proc.wait()
        self.ended = time.time()
        if self.state == "running":
            self.state = "done" if self.rc == 0 else "failed"

    def cancel(self) -> None:
        if self.proc.poll() is None:
            self.state = "cancelled"
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
            else:
                self.proc.terminate()

    def info(self, since: int = 0) -> dict:
        return {"id": self.id, "kind": self.kind, "state": self.state, "rc": self.rc, "stage": self.stage,
                "stages": STAGES, "elapsed": (self.ended or time.time()) - self.started, "next": len(self.lines),
                "lines": self.lines[since:since + 2000]}


class App:
    def __init__(self, config_path: str | None):
        self.config_path = config_path
        self.token = secrets.token_urlsafe(24)
        self.cwd = Path.cwd()
        self.settings_file = self.cwd / "ui_settings.json"
        self.lock = threading.Lock()
        self.job: Job | None = None
        self.env_cache: dict | None = None
        self.settings = dict(DEFAULT_SETTINGS)
        try:
            self.settings.update(json.loads(self.settings_file.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass

    # -------------------------------------------------------------- settings / paths
    def save_settings(self, new: dict) -> None:
        with self.lock:
            for k in DEFAULT_SETTINGS:
                if k in new:
                    self.settings[k] = new[k]
            self.settings_file.write_text(json.dumps(self.settings, indent=1), encoding="utf-8")

    def _p(self, v: str) -> Path:
        p = Path(os.path.expandvars(os.path.expanduser(v or "")))
        return p if p.is_absolute() else (self.cwd / p)

    def roots(self) -> dict[str, Path]:
        s = self.settings
        music = self._p(s["music_dir"]) if s["music_dir"] else None
        if s["music_mode"] == "file" and s["music_file"]:
            music = self._p(s["music_file"]).parent
        r = {"out": self._p(s["output_dir"]), "cache": self._p(s["cache_dir"])}
        if s["footage"]:
            r["footage"] = self._p(s["footage"])
        if music:
            r["music"] = music
        return {k: v.resolve() for k, v in r.items()}

    def safe_path(self, root: str, rel: str) -> Path | None:
        base = self.roots().get(root)
        if base is None:
            return None
        p = (base / rel).resolve()
        if p != base and base not in p.parents:
            return None
        return p if p.suffix.lower() in SERVE_EXT else None

    def effective_config(self) -> dict:
        s = self.settings
        cfg = load_config(self.config_path, s.get("overrides") or {})
        if s["extra_yaml"].strip():
            _merge(cfg, yaml.safe_load(s["extra_yaml"]) or {})
        cfg["paths"]["cache_dir"] = str(self._p(s["cache_dir"]))
        cfg["paths"]["output_dir"] = str(self._p(s["output_dir"]))
        if s["music_dir"]:
            cfg["paths"]["music_dir"] = str(self._p(s["music_dir"]))
        return dict(cfg)

    def env(self) -> dict:
        if self.env_cache is None:
            e = {"ffmpeg": bool(shutil.which("ffmpeg")), "nvenc": False, "cuda": None, "version": __version__}
            try:
                e["nvenc"] = ingest.nvenc_available()
            except Exception:  # noqa: BLE001
                pass
            try:
                import torch

                e["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            except Exception:  # noqa: BLE001
                pass
            self.env_cache = e
        return self.env_cache

    # -------------------------------------------------------------- jobs
    def start_job(self, kind: str, extra: dict | None = None) -> Job:
        if self.job and self.job.proc.poll() is None:
            raise RuntimeError("a job is already running")
        s = self.settings
        cache = self._p(s["cache_dir"])
        cache.mkdir(parents=True, exist_ok=True)
        cfgfile = cache / "ui_config.yaml"
        cfgfile.write_text(yaml.safe_dump(self.effective_config(), sort_keys=False, allow_unicode=True), encoding="utf-8")
        base = [sys.executable, "-m", "dronecut"]
        common = ["--config", str(cfgfile), "--cache-dir", str(cache)]
        if kind == "music":
            if not s["music_dir"]:
                raise RuntimeError("choose a music folder first")
            cmd = [*base, "music-scan", str(self._p(s["music_dir"])), *common]
        else:
            if not s["footage"]:
                raise RuntimeError("choose the footage folder first")
            footage = str(self._p(s["footage"]))
            if kind == "analyze":
                cmd = [*base, "analyze", footage, *common, "--seed", str(s["seed"])]
            elif kind == "build":
                music = "auto" if s["music_mode"] == "auto" else str(self._p(s["music_file"]))
                if s["music_mode"] == "file" and not s["music_file"]:
                    raise RuntimeError("choose a music file (or switch to automatic)")
                cmd = [*base, "build", "--music", music, "--duration", str(s["duration"]), "--res", str(s["res"]),
                       "--seed", str(s["seed"]), "--footage", footage, "-o", str(self._p(s["output_dir"])), *common]
                for name in s["skip_clips"]:
                    cmd += ["--exclude", name]
            else:
                raise RuntimeError(f"unknown job {kind}")
            if s["fast"]:
                cmd.append("--fast")
            if s["vlm"]:
                cmd.append("--vlm")
        self.job = Job(kind, cmd, str(self.cwd))
        return self.job

    # -------------------------------------------------------------- data
    def clips(self) -> list[dict]:
        root = self.roots().get("footage")
        if not root or not root.exists():
            return []
        cfg = load_config(self.config_path)
        files = ingest.discover(root, cfg.ingest.video_exts)
        cache = self.roots()["cache"]
        out = []
        for info in ingest.probe_all(files, cache):
            p = Path(info.path)
            out.append({"name": p.name, "path": str(p), "rel": str(p.relative_to(root)) if root in p.parents else p.name,
                        "width": info.width, "height": info.height, "fps": round(info.fps, 2), "codec": info.codec,
                        "duration": round(info.duration, 1), "audio": info.has_audio, "srt": bool(info.srt_path),
                        "mb": round(p.stat().st_size / 1e6, 1), "key": info.key})
        return out

    def thumb(self, path: str) -> Path | None:
        root = self.roots().get("footage")
        p = Path(path).resolve()
        if not root or (root != p and root not in p.parents) or p.suffix.lower() not in VIDEO_EXT:
            return None
        out = self.roots()["cache"] / "ui_thumbs" / f"{file_key(p)}.jpg"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            from .probe import probe_file

            dur = probe_file(p).duration
            run([which("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{dur / 2:.2f}", "-i", str(p), "-frames:v", "1",
                 "-vf", "scale=360:-2", str(out)])
        return out

    def tracks(self) -> list[dict]:
        root = self.roots().get("music")
        if not root or not root.exists():
            return []
        cache = self.roots()["cache"]
        files = sorted((f for f in root.rglob("*") if f.suffix.lower() in AUDIO_EXT), key=lambda f: f.name.lower())[:600]
        out = []
        for f in files:
            info = music_mod.cached_info(f, cache)
            out.append({"name": f.name, "path": str(f), "rel": str(f.relative_to(root)), "mb": round(f.stat().st_size / 1e6, 1),
                        "bpm": round(info["bpm"], 1) if info else None, "duration": round(info["duration"], 1) if info else None})
        return out

    def builds(self) -> list[dict]:
        root = self.roots()["out"]
        out = []
        if root.exists():
            for bj in [root / "build.json", *sorted(root.glob("*/build.json"))]:
                if not bj.exists():
                    continue
                try:
                    d = json.loads(bj.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                folder = bj.parent
                rel = "" if folder == root else folder.name + "/"
                mp4 = folder / "dronecut_highlight.mp4"
                out.append({"dir": str(folder), "mtime": bj.stat().st_mtime, "duration": round(d.get("duration", 0), 1),
                            "cuts": len(d.get("cuts", [])), "music": d.get("music", {}).get("name", ""), "bpm": round(d.get("music", {}).get("bpm", 0), 1),
                            "seed": d.get("seed"), "pinned": sum(1 for c in d.get("cuts", []) if c.get("pinned")),
                            "mp4": f"/files/out/{rel}{mp4.name}" if mp4.exists() else None,
                            "report": f"/files/out/{rel}report.html" if (folder / "report.html").exists() else None,
                            "fcpxml": f"/files/out/{rel}dronecut_highlight.fcpxml" if (folder / "dronecut_highlight.fcpxml").exists() else None,
                            "telemetry": d.get("telemetry")})
        return sorted(out, key=lambda b: -b["mtime"])

    def pins_file(self) -> Path:
        return self.cwd / "must_have.json"

    def defaults_cfg(self) -> dict:
        """Config as the form should show it: built-in defaults + the user's config.yaml (not the UI overrides)."""
        return dict(load_config(self.config_path))


def list_dir(path: str) -> dict:
    if not path:
        items = []
        if os.name == "nt":
            items += [{"name": f"{d}:\\", "path": f"{d}:/"} for d in string.ascii_uppercase if os.path.exists(f"{d}:/")]
        else:
            items.append({"name": "/", "path": "/"})
        home = Path.home()
        items += [{"name": f"~ {home.name}", "path": str(home)}] + [{"name": n, "path": str(home / n)} for n in ("Desktop", "Videos", "Music", "Documents", "Downloads")
                                                                       if (home / n).exists()]
        return {"path": "", "parent": None, "dirs": items, "files": []}
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(path)
    dirs, files = [], []
    try:
        for e in sorted(p.iterdir(), key=lambda e: e.name.lower()):
            try:
                if e.is_dir():
                    if not e.name.startswith((".", "$")):
                        dirs.append({"name": e.name, "path": str(e)})
                elif e.suffix.lower() in VIDEO_EXT | AUDIO_EXT:
                    files.append({"name": e.name, "path": str(e), "mb": round(e.stat().st_size / 1e6, 1)})
            except OSError:
                continue
    except PermissionError:
        pass
    parent = str(p.parent) if p.parent != p else ""
    return {"path": str(p), "parent": parent, "dirs": dirs, "files": files[:500]}


def make_handler(app: App, port: int):
    index = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class H(BaseHTTPRequestHandler):
        server_version = "dronecut-ui"

        def log_message(self, *a) -> None:  # quiet
            pass

        # ---- helpers
        def _json(self, obj, code: int = 200) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _err(self, code: int, msg: str) -> None:
            self._json({"error": msg}, code)

        def _auth(self, q: dict) -> bool:
            if self.headers.get("Host", "") not in allowed_hosts:
                return False
            cookie = self.headers.get("Cookie", "")
            tok = next((c.split("=", 1)[1] for c in (x.strip() for x in cookie.split(";")) if c.startswith("dc_token=")), "")
            return secrets.compare_digest(tok, app.token) or secrets.compare_digest(q.get("token", [""])[0], app.token)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def _file(self, p: Path) -> None:
            if not p.is_file():
                return self._err(404, "not found")
            size = p.stat().st_size
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            start, end, code = 0, size - 1, 200
            rng = self.headers.get("Range")
            if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                code = 206
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                with open(p, "rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (ConnectionError, OSError):
                pass

        # ---- routing
        def do_GET(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._auth(q):
                return self._err(403, "forbidden - open the URL printed by `dronecut ui`")
            try:
                if u.path == "/":
                    data = index.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Set-Cookie", f"dc_token={app.token}; Path=/; HttpOnly; SameSite=Strict")
                    self.end_headers()
                    self.wfile.write(data)
                elif u.path == "/api/state":
                    self._json({"settings": app.settings, "config": app.defaults_cfg(), "env": app.env(),
                                "cwd": str(app.cwd), "pins": app.pins_file().exists()})
                elif u.path == "/api/fs":
                    self._json(list_dir(q.get("path", [""])[0]))
                elif u.path == "/api/clips":
                    self._json({"clips": app.clips()})
                elif u.path == "/api/tracks":
                    self._json({"tracks": app.tracks()})
                elif u.path == "/api/builds":
                    self._json({"builds": app.builds()})
                elif u.path == "/api/thumb":
                    t = app.thumb(q.get("path", [""])[0])
                    self._file(t) if t else self._err(404, "no thumbnail")
                elif u.path == "/api/job":
                    j = app.job
                    self._json(j.info(int(q.get("since", ["0"])[0])) if j else {"state": "idle", "lines": [], "next": 0, "stages": STAGES})
                elif u.path == "/api/pins":
                    f = app.pins_file()
                    self._json(json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"include": [], "exclude": []})
                elif u.path.startswith("/files/"):
                    root, _, rel = u.path[len("/files/"):].partition("/")
                    p = app.safe_path(root, unquote(rel))
                    self._file(p) if p else self._err(404, "not found")
                else:
                    self._err(404, "not found")
            except FileNotFoundError as e:
                self._err(404, str(e))
            except Exception as e:  # noqa: BLE001
                self._err(500, f"{type(e).__name__}: {e}")

        def do_POST(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            origin = self.headers.get("Origin")
            if not self._auth(q) or self.headers.get("X-Requested-With") != "dronecut" or (origin and urlparse(origin).netloc not in allowed_hosts):
                return self._err(403, "forbidden")
            try:
                body = self._body()
                if u.path == "/api/settings":
                    app.save_settings(body)
                    self._json({"ok": True})
                elif u.path == "/api/job":
                    j = app.start_job(body.get("kind", ""))
                    self._json(j.info())
                elif u.path == "/api/job/cancel":
                    if app.job:
                        app.job.cancel()
                    self._json({"ok": True})
                elif u.path == "/api/pins":
                    inc, exc = body.get("include", []), body.get("exclude", [])
                    for it in [*inc, *exc]:
                        if not isinstance(it.get("clip"), str):
                            raise ValueError("every pin needs a clip name")
                    app.pins_file().write_text(json.dumps({"include": inc, "exclude": exc}, indent=1), encoding="utf-8")
                    self._json({"ok": True, "include": len(inc), "exclude": len(exc)})
                elif u.path == "/api/open":
                    p = Path(body.get("path", "")).resolve()
                    roots = list(app.roots().values())
                    if not any(r == p or r in p.parents for r in roots):
                        return self._err(403, "path outside the project folders")
                    if os.name == "nt":
                        os.startfile(str(p))  # type: ignore[attr-defined]  # noqa: S606
                    else:
                        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(p)])
                    self._json({"ok": True})
                else:
                    self._err(404, "not found")
            except RuntimeError as e:
                self._err(409, str(e))
            except (ValueError, KeyError, TypeError) as e:
                self._err(400, str(e))
            except Exception as e:  # noqa: BLE001
                self._err(500, f"{type(e).__name__}: {e}")

    return H


def serve(config_path: str | None = None, port: int = 8765, open_browser: bool = True) -> None:
    app = App(config_path)
    srv = None
    for p in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), make_handler(app, p))
            port = p
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit(f"no free port near {port}")
    url = f"http://127.0.0.1:{port}/?token={app.token}"
    print(f"dronecut UI running at {url}\n(local only - press Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if app.job:
            app.job.cancel()
        srv.server_close()
