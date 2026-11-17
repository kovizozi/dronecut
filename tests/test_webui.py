from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from dronecut import webui


@pytest.fixture()
def server(tmp_path, media, music_file, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = webui.App(None)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: None)
    port = srv.server_address[1]
    srv.server_close()
    srv = ThreadingHTTPServer(("127.0.0.1", port), webui.make_handler(app, port))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    app.save_settings({"footage": str(media), "music_mode": "file", "music_file": str(music_file), "music_dir": str(music_file.parent),
                       "output_dir": str(tmp_path / "out"), "cache_dir": str(tmp_path / "cache"), "duration": 20, "res": 270, "fast": True,
                       "overrides": {"ingest": {"prefer_nvenc": False}, "render": {"nvenc": "off", "crf": 30}, "tech": {"workers": 2}}})
    yield app, port
    if app.job:
        app.job.cancel()
    srv.shutdown()


class Client:
    def __init__(self, app, port):
        self.app, self.port, self.cookie = app, port, f"dc_token={app.token}"

    def req(self, method, path, body=None, headers=None, cookie=True, host=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        h = dict(headers or {})
        if cookie:
            h["Cookie"] = self.cookie
        if host:
            h["Host"] = host
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h.setdefault("Content-Type", "application/json")
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        raw = r.read()
        return r.status, r.getheaders(), raw

    def get(self, path, **kw):
        s, h, raw = self.req("GET", path, **kw)
        try:
            return s, json.loads(raw)
        except ValueError:
            return s, raw

    def post(self, path, body, **kw):
        kw.setdefault("headers", {"X-Requested-With": "dronecut"})
        s, h, raw = self.req("POST", path, body, **kw)
        return s, json.loads(raw)


def test_auth_token_host_and_csrf_header(server):
    app, port = server
    c = Client(app, port)
    assert c.req("GET", "/", cookie=False)[0] == 403  # no token
    s, h, raw = c.req("GET", f"/?token={app.token}", cookie=False)
    assert s == 200 and b"dronecut" in raw and any(k == "Set-Cookie" and "HttpOnly" in v and "SameSite=Strict" in v for k, v in h)
    assert c.get("/api/state")[0] == 200
    assert c.req("GET", "/api/state", host="evil.example:80")[0] == 403  # DNS-rebinding style host
    assert c.get("/api/state", cookie=False)[0] == 403
    assert c.post("/api/settings", {}, headers={})[0] == 403  # missing X-Requested-With
    assert c.post("/api/settings", {}, headers={"X-Requested-With": "dronecut", "Origin": "http://evil.example"})[0] == 403
    assert c.post("/api/settings", {"seed": 9})[0] == 200


def test_state_settings_persist_and_fs_browser(server, tmp_path):
    app, port = server
    c = Client(app, port)
    s, st = c.get("/api/state")
    assert s == 200 and st["config"]["hud"]["style"] == "full" and "ffmpeg" in st["env"]
    c.post("/api/settings", {"seed": 42, "overrides": {"hud": {"enabled": True}}})
    saved = json.loads((tmp_path / "ui_settings.json").read_text())
    assert saved["seed"] == 42 and saved["overrides"]["hud"]["enabled"] is True
    s, r = c.get("/api/fs?path=" + str(tmp_path).replace("\\", "/"))
    assert s == 200 and r["parent"] and "ui_settings.json" not in [f["name"] for f in r["files"]]  # files are filtered to media
    s, roots = c.get("/api/fs")
    assert s == 200 and roots["dirs"]  # drives / home shortcuts
    assert c.get("/api/fs?path=/definitely/not/here")[0] == 404


def test_clips_thumbs_tracks_and_range(server, media):
    app, port = server
    c = Client(app, port)
    s, r = c.get("/api/clips")
    names = {x["name"] for x in r["clips"]}
    assert s == 200 and {"good_pan_1080.mp4", "good_pan_4k.mp4", "junk_blur.mp4"} <= names
    pan = next(x for x in r["clips"] if x["name"] == "good_pan_1080.mp4")
    assert pan["srt"] and pan["width"] == 1920
    s, jpg = c.get("/api/thumb?path=" + pan["path"].replace("\\", "/"))
    assert s == 200 and jpg[:2] == b"\xff\xd8"
    assert c.get("/api/thumb?path=C:/Windows/notepad.exe")[0] == 404
    s, h, body = c.req("GET", "/files/footage/good_pan_1080.mp4", headers={"Range": "bytes=0-99"})
    assert s == 206 and len(body) == 100 and dict(h)["Content-Range"].startswith("bytes 0-99/")
    s, r = c.get("/api/tracks")
    assert any(t["name"] == "click_120bpm.wav" for t in r["tracks"])


def test_file_route_blocks_traversal_and_foreign_types(server, tmp_path):
    app, port = server
    c = Client(app, port)
    (tmp_path / "secret.txt").write_text("x")
    assert c.get("/files/footage/..%2f..%2fsecret.txt")[0] == 404
    assert c.get("/files/footage/%2e%2e/%2e%2e/secret.txt")[0] == 404
    assert c.get("/files/nope/x.mp4")[0] == 404
    (Path(app.roots()["out"])).mkdir(exist_ok=True)
    (Path(app.roots()["out"]) / "run.exe").write_bytes(b"MZ")
    assert c.get("/files/out/run.exe")[0] == 404  # extension whitelist


def test_pins_roundtrip_validation(server, tmp_path):
    app, port = server
    c = Client(app, port)
    assert c.get("/api/pins")[1] == {"include": [], "exclude": []}
    s, r = c.post("/api/pins", {"include": [{"clip": "0010", "start": 5, "end": 11}], "exclude": [{"clip": "0013", "start": None, "end": None}]})
    assert s == 200 and r["include"] == 1
    from dronecut import pins

    p = pins.load_file(tmp_path / "must_have.json")  # the CLI reads exactly this file
    assert p.include[0].clip == "0010" and p.exclude[0].start is None
    assert c.post("/api/pins", {"include": [{"start": 1}]})[0] == 400


def test_job_analyze_then_build_runs_to_completion(server, tmp_path):
    app, port = server
    c = Client(app, port)
    s, r = c.post("/api/job", {"kind": "build"})  # build also analyses on demand
    assert s == 200 and r["state"] == "running"
    assert c.post("/api/job", {"kind": "build"})[0] == 409  # one job at a time
    t0 = time.time()
    since, log = 0, []
    while time.time() - t0 < 300:
        s, j = c.get(f"/api/job?since={since}")
        log += j["lines"]
        since = j["next"]
        if j["state"] != "running":
            break
        time.sleep(1)
    assert j["state"] == "done", "\n".join(log[-25:])
    assert any("render" in x for x in log)
    s, b = c.get("/api/builds")
    assert s == 200 and b["builds"] and b["builds"][0]["mp4"].startswith("/files/out/") and b["builds"][0]["report"]
    s, h, vid = c.req("GET", b["builds"][0]["mp4"], headers={"Range": "bytes=0-"})
    assert s in (200, 206) and len(vid) > 10_000
    s, _, rep = c.req("GET", b["builds"][0]["report"])
    assert s == 200 and b"Save picks to project" in rep
    assert c.post("/api/job", {"kind": "bogus"})[0] == 409


def test_job_requires_folders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = webui.App(None)
    with pytest.raises(RuntimeError, match="footage"):
        app.start_job("build")
