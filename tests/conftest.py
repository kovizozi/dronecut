"""Shared fixtures: synthetic media is generated once per test session (never committed)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dronecut.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def media(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("media")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_media.py"), str(out), "--quick"], check=True)
    return out


@pytest.fixture()
def cfg(tmp_path):
    return load_config(
        None,
        {
            "paths": {"cache_dir": str(tmp_path / "cache"), "output_dir": str(tmp_path / "out")},
            "ingest": {"prefer_nvenc": False, "workers": 2},
            "tech": {"workers": 2},
            "render": {"nvenc": "off", "crf": 28},
        },
    )


@pytest.fixture(scope="session")
def music_file(media) -> Path:
    return media / "click_120bpm.wav"


@pytest.fixture(scope="session")
def analysis_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("shared_cache")
