# dronecut

Local, offline automatic highlight editor for drone footage. Point it at a folder of raw clips (e.g. 20-30 minutes
from a DJI Neo), give it a music track, and it produces a 2-3 minute highlight video **cut to the beat**, plus an
editable DaVinci Resolve timeline and an HTML report explaining every choice. No cloud APIs - everything runs on your
machine (GPU optional).

```
dronecut analyze  D:\footage\2026-10-04 
dronecut build    --music D:\music\track.mp3 --duration 150
```

Outputs (in `output/`): `dronecut_highlight.mp4`, `dronecut_highlight.fcpxml` (opens in Resolve and references your
**original** files), `report.html` (thumbnails + scores of selected / rejected windows with reasons), `build.json`.

## How it works

```mermaid
flowchart TD
    A[footage folder] --> B[1 probe: ffprobe, rotation, DJI .SRT]
    B --> C[2 proxies: 720p, NVENC or x264]
    C --> D[3 segmentation: PySceneDetect Content+Adaptive, 2-6 s sliding windows]
    D --> E[4 technical score: sharpness, exposure, shake, motion, takeoff/landing, sky, occlusion]
    D --> F[5 semantic score: SigLIP prompts + YOLO persons]
    E --> G[6 selection: score, k-means diversity, min gap, narrative order]
    F --> G
    F -. --vlm .-> V[Qwen2.5-VL 4-bit captions / 1-10 score]
    V -.-> G
    M[music file] --> N[7 beats, downbeats, energy, bar grid]
    N --> G
    G --> H[8 render from ORIGINALS: one ffmpeg pass, grade, NVENC/x264]
    H --> I[final MP4]
    G --> J[9 FCPXML 1.9 timeline]
    G --> K[report.html]
```

Every stage is cached under `.dronecut_cache/` (keyed by file path/size/mtime and the relevant settings), so
re-running only does the missing work. Hard-reject thresholds are applied at *selection* time, so tuning them in
`config.yaml` never needs re-analysis.

* **Cut points snap to the music.** Beats come from librosa; for steady tempos a regular grid is fitted by regression
  (sub-frame accurate, fills sparse intros) and the phase is locked to the kick drum. Downbeats are found from the
  low-frequency accent pattern. Each shot is 1 / 2 / 3 bars long depending on the energy of that section (short cuts
  on high energy, long on calm). The video is trimmed to exact frame counts from cumulative bar boundaries, so it
  cannot drift against the music. The edit ends on a bar boundary with a fade-out.
* **Junk rejection** (no AI): blurry, over/under-exposed, shaky or jerky, static or chaotic, rapid zoom,
  takeoff/landing (start-of-clip receding / end-of-clip approaching ground, or altitude < 1.5 m from the `.SRT`),
  sky-only, occluded (finger/propeller). All thresholds are in `config.yaml`.
* **Semantic scoring**: SigLIP (`google/siglip-so400m-patch14-384`) image embeddings vs. positive / negative text
  prompts, YOLOv8n person detection boost. `--fast` replaces both with cheap stand-ins (colour-layout embedding,
  no person detector) for CI and quick looks.
* **Diversity & narrative**: k-means clusters on the embeddings with a per-cluster cap, a minimum time gap between
  picks from the same clip, chronological order, opening with the most "wide establishing" strong shot and closing
  with a pull-away shot if there is one.
* **`--vlm`** (off by default): Qwen2.5-VL-7B-Instruct in 4-bit (fits 12 GB; Moondream2 fallback) captions the top
  ~40 candidates and gives a 1-10 interestingness score that is blended into the ranking. Models are loaded one at a
  time and VRAM is freed in between.

## Install

Requirements: Python 3.11+, **ffmpeg + ffprobe on PATH** (NVENC build recommended), optionally an NVIDIA GPU.

**Windows (native)**
```powershell
winget install Gyan.FFmpeg                       # or: choco install ffmpeg
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-ml.txt               # optional: PyTorch CUDA 12.4 + SigLIP/YOLO/VLM stack
pip install -e .
```

**WSL2 / Ubuntu**: `sudo apt install ffmpeg libsndfile1 python3-venv`, then the same pip commands. For GPU inside
WSL2 install the NVIDIA Windows driver (CUDA is passed through); NVENC availability in WSL2 depends on the ffmpeg build -
dronecut tests for it and falls back to libx264 automatically.

**CPU only**: skip `requirements-ml.txt` (or install the CPU torch wheel) and use `--fast`, or set
`semantic.backend: mock`. Everything works, just without semantic scoring.

Models download on first use into `%LOCALAPPDATA%\dronecut\models` (or `~/.cache/dronecut/models`), never into the repo.

## Usage

```
dronecut analyze <footage_dir> [--fast] [--vlm]
dronecut build --music <file|auto> --duration 150 [--res 1080|2160] [--vlm] [--seed N] [-o output_dir]
dronecut report            # regenerate report.html from the last build
dronecut clean-cache
```
`--music auto` picks from `paths.music_dir` (long enough, tempo in `music.auto_prefer_bpm`). `--seed` makes the whole
run deterministic. `--config path.yaml` selects a config (default: `./config.yaml`); start from `config.example.yaml`.

## Tuning with real footage

1. Run `dronecut build ...` then open `report.html`. Rejected windows are listed with their reason.
2. Good shots rejected as `shaky`/`sudden_jerk`? raise `tech.shake_max` / `tech.jerk_max`. Too much wobble kept? lower them.
3. `static` rejecting slow orbits? lower `tech.motion_min`. `blurry`: lower `tech.min_sharpness_rel`.
4. Takeoff/landing leaking through: raise `tech.edge_s`, lower `tech.edge_zoom_rate`; with `.SRT` files next to the clips,
   raise `tech.min_altitude_m`.
5. Not enough variety: more `select.clusters`, larger `select.min_gap_s`. Wrong mood: edit the prompts under `semantic:`.
6. Faster/slower cutting: `music.bars_high/mid/low`. Soft transitions: `render.crossfade_s: 0.4`.
7. Grade: `render.lut` takes a `.cube` file; per-clip exposure/white-balance normalisation is on by default.

## Development

```
pip install -e ".[dev]" && ruff check . && pytest -m "not gpu"
```
Tests generate synthetic clips (1080p + 4K, blurred / shaky / over- / under-exposed / sky-only junk, a multi-scene
clip) and a 120 BPM click track in a temp folder with `scripts/make_test_media.py`; nothing is committed. GPU tests
(`-m gpu`) exercise the real SigLIP / YOLO models and are skipped in CI.

## Known limitations

* Shake detection runs on 10 fps proxies: gentle high-frequency vibration is invisible, big jitter and jerks are caught.
* Takeoff/landing detection is heuristic (zoom direction near clip edges + optional SRT altitude).
* Downbeat detection assumes a steady 4/4 groove; music with free tempo falls back to the tracked beats.
* Source audio mixing (`render.mix_source_audio`) is basic ducking; use with `crossfade_s: 0`.
* HDR / D-Log footage is not tone-mapped; use `render.lut` for a log-to-Rec.709 LUT.

MIT licensed.
