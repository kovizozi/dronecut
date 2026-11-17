"""Shared helpers: logging, subprocess wrappers, hashing, JSON caching, progress bars."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

T = TypeVar("T")
console = Console()
log = logging.getLogger("dronecut")
TIMINGS: dict[str, float] = {}


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, markup=True)],
        force=True,
    )


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time a pipeline stage; totals are kept in ``TIMINGS`` for the final summary."""
    t0 = time.perf_counter()
    log.info("[bold cyan]> %s[/]", name)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        TIMINGS[name] = TIMINGS.get(name, 0.0) + dt
        log.info("  %s done in %.1fs", name, dt)


def progress(items: Iterable[T], desc: str, total: int | None = None) -> Iterator[T]:
    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = None
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as p:
        t = p.add_task(desc, total=total)
        for it in items:
            yield it
            p.advance(t)


def which(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"'{name}' not found on PATH. Install ffmpeg (see README).")
    return path


def run(cmd: list, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    log.debug("run: %s", " ".join(cmd))
    cp = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
    )
    if check and cp.returncode != 0:
        tail = "\n".join(cp.stderr.strip().splitlines()[-12:])
        raise RuntimeError(f"command failed ({cp.returncode}): {' '.join(cmd[:6])} ...\n{tail}")
    return cp


def file_key(path: str | Path) -> str:
    p = Path(path)
    st = p.stat()
    h = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
    return h[:16]


def params_hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def read_json(path: Path) -> Any | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=float)
    tmp.replace(path)


def cached_json(path: Path, phash: str) -> Any | None:
    d = read_json(path)
    if isinstance(d, dict) and d.get("_params") == phash:
        return d["data"]
    return None


def store_json(path: Path, phash: str, data: Any) -> None:
    write_json(path, {"_params": phash, "data": data})
