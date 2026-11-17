"""Stage 7: music analysis (beats, downbeats, energy), automatic track choice and the bar grid for cutting."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Cfg
from .utils import file_key, log

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


@dataclass
class Track:
    path: str
    duration: float
    bpm: float
    beats: np.ndarray
    downbeats: np.ndarray
    energy_t: np.ndarray  # seconds
    energy: np.ndarray  # 0..1


@dataclass
class MusicPlan:
    track: Track
    start: float  # offset into the track (a downbeat)
    end: float  # bar boundary
    bar_times: np.ndarray  # absolute track times of bar edges (len n_bars+1)
    bar_energy: np.ndarray  # per bar 0..1
    bars_pref: list[int] = field(default_factory=list)  # preferred shot length in bars, per bar
    beat_dur: float = 0.5

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def n_bars(self) -> int:
        return len(self.bar_times) - 1


def _fold_beats(beats: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fix octave errors: double-time/half-time beat grids are folded into the preferred BPM range."""
    for _ in range(3):
        if len(beats) < 4:
            break
        bpm = 60.0 / float(np.median(np.diff(beats)))
        if bpm < lo:
            mids = (beats[:-1] + beats[1:]) / 2
            beats = np.sort(np.concatenate([beats, mids]))
        elif bpm > hi:
            beats = beats[::2]
        else:
            break
    return beats


def analyze_track(path: str | Path, cfg: Cfg, cache: Path | None = None) -> Track:
    import librosa

    p = Path(path)
    cf = None
    if cache is not None:
        cf = cache / "music" / f"{file_key(p)}.npz"
        if cf.exists():
            with np.load(cf) as z:
                return Track(str(p), float(z["duration"]), float(z["bpm"]), z["beats"], z["downbeats"], z["et"], z["e"])
    m = cfg.music
    sr = int(m.sr)
    y, sr = librosa.load(str(p), sr=sr, mono=True)
    dur = len(y) / sr
    hop = 512
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    _, bf = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, start_bpm=118.0, tightness=120)
    beats = librosa.frames_to_time(bf, sr=sr, hop_length=hop)
    if len(beats) < 8:
        raise ValueError(f"could not find a steady beat in {p.name}")
    beats = _fold_beats(beats, *m.bpm_range)
    fh = 128
    onf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=fh)
    onl = librosa.onset.onset_strength(y=y, sr=sr, hop_length=fh, fmax=250.0, n_mels=32)  # kick / bass emphasis
    tf = librosa.frames_to_time(np.arange(len(onf)), sr=sr, hop_length=fh)
    # steady tempo -> fit a regular grid (regression removes frame quantisation and fills sparse intros)
    idx = np.arange(len(beats))
    slope, icpt = np.polyfit(idx, beats, 1)
    resid = beats - (icpt + slope * idx)
    if np.std(resid) < 0.03:
        # phase refinement: slide a comb of period `slope` over a fine-resolution onset envelope
        offs = np.arange(0.0, slope, 0.004)
        ks = np.arange(0, int((dur - 0.5) / slope))
        sc_f = np.array([np.interp(o + ks * slope, tf, onf).sum() for o in offs])
        sc_l = np.array([np.interp(o + ks * slope, tf, onl).sum() for o in offs])
        sc = sc_f / (sc_f.max() + 1e-9) + 1.5 * sc_l / (sc_l.max() + 1e-9)
        o_best = float(offs[int(np.argmax(sc))])
        beats = o_best + slope * np.arange(0, int((dur - o_best) / slope) + 1)
        bpm = 60.0 / float(slope)
    else:
        bpm = 60.0 / float(np.median(np.diff(beats)))
    # downbeats: bar phase whose beats carry the most onset energy (accented beat one)
    bpb = int(m.beats_per_bar)
    bidx = np.clip(librosa.time_to_frames(beats, sr=sr, hop_length=fh), 0, len(onl) - 1)
    w = int(0.03 * sr / fh)
    # accented beat one: linear (not log) low-band energy keeps the amplitude difference that onset flux squashes
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=fh))
    low = S[librosa.fft_frequencies(sr=sr, n_fft=2048) < 150.0].sum(axis=0)
    strength = np.array([low[max(0, i - w) : i + w + 1].max() for i in bidx])
    phase_score = [strength[ph::bpb].mean() if len(strength[ph::bpb]) else 0.0 for ph in range(bpb)]
    phase = int(np.argmax(phase_score))
    downbeats = beats[phase::bpb]
    # energy curve: loudness + onset density, smoothed
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    n = min(len(rms), len(onset))

    def norm(x: np.ndarray) -> np.ndarray:
        k = int(sr / hop * 1.0)
        x = np.convolve(x, np.ones(k) / k, mode="same")
        return (x - x.min()) / (np.ptp(x) + 1e-9)

    e = 0.5 * norm(rms[:n]) + 0.5 * norm(onset[:n])
    e = (e - e.min()) / (np.ptp(e) + 1e-9)
    et = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)
    tr = Track(str(p), float(dur), float(bpm), beats, downbeats, et, e)
    if cf is not None:
        cf.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cf, duration=dur, bpm=bpm, beats=beats, downbeats=downbeats, et=et, e=e)
    return tr


def cached_info(path: str | Path, cache: Path) -> dict | None:
    """BPM / duration of an already analysed track (None if it has not been analysed yet)."""
    try:
        cf = cache / "music" / f"{file_key(path)}.npz"
        if not cf.exists():
            return None
        with np.load(cf) as z:
            return {"bpm": float(z["bpm"]), "duration": float(z["duration"])}
    except Exception:  # noqa: BLE001
        return None


def pick_auto(music_dir: str | Path, target: float, cfg: Cfg, cache: Path | None, seed: int = 0) -> Path:
    """Choose a track from a folder: long enough for the target and tempo inside the preferred range."""
    files = sorted(f for f in Path(music_dir).rglob("*") if f.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise SystemExit(f"no audio files found in {music_dir}")
    lo, hi = cfg.music.auto_prefer_bpm
    scored: list[tuple[float, Path]] = []
    for f in files:
        try:
            t = analyze_track(f, cfg, cache)
        except Exception as ex:  # noqa: BLE001
            log.debug("skip %s: %s", f.name, ex)
            continue
        short = max(0.0, target * 0.95 - t.duration) / target
        off = max(0.0, lo - t.bpm, t.bpm - hi) / 40.0
        scored.append((2.0 * short + off, f))
    if not scored:
        raise SystemExit("no usable music track found")
    scored.sort(key=lambda x: (x[0], x[1].name))
    best = [f for s, f in scored if s <= scored[0][0] + 0.05]
    return random.Random(seed).choice(best)


def make_plan(tr: Track, target: float, cfg: Cfg) -> MusicPlan:
    m = cfg.music
    db = tr.downbeats
    if len(db) < 2:
        raise ValueError("not enough downbeats")
    bar_dur = float(np.median(np.diff(db)))
    start = float(db[0])
    max_end = tr.duration - 0.2
    n_target = int(np.floor(target / bar_dur + 1e-6))
    edges = [start]
    k = 1
    while len(edges) <= n_target:
        nxt = db[k] if k < len(db) else edges[-1] + bar_dur  # beyond tracking range: extrapolate
        if nxt > max_end:
            break
        edges.append(float(nxt))
        k += 1
    edges_a = np.asarray(edges)
    if len(edges_a) < 3:
        raise ValueError("track too short for a meaningful edit")
    be = np.array([tr.energy[(tr.energy_t >= a) & (tr.energy_t < b)].mean() if ((tr.energy_t >= a) & (tr.energy_t < b)).any() else 0.0
                   for a, b in zip(edges_a[:-1], edges_a[1:], strict=True)])
    q_lo, q_hi = np.quantile(be, [0.33, 0.66])
    pref = [int(m.bars_high) if e > q_hi else int(m.bars_mid) if e > q_lo else int(m.bars_low) for e in be]
    return MusicPlan(tr, start, float(edges_a[-1]), edges_a, be, pref, 60.0 / tr.bpm)


def snap_to_beat(t: float, beats: np.ndarray) -> float:
    return float(beats[int(np.abs(beats - t).argmin())])
