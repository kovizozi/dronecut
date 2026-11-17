"""Full-model tests (SigLIP + YOLO on the GPU). Skipped in CI: run with ``pytest -m gpu``."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_full_semantic_end_to_end(media, music_file, cfg, tmp_path):
    from dronecut import export, pipeline

    cfg["semantic"]["person_backend"] = "yolo"
    cfg["render"]["nvenc"] = "auto"
    out = tmp_path / "out"
    r = pipeline.build(cfg, str(music_file), 40.0, 540, False, 1, out, footage=media, fast=False)
    assert r["mp4"].exists() and r["report"].exists()
    assert export.validate_fcpxml(r["fcpxml"]) == []
    wins = r["selection"].candidates
    assert wins and all(w.get("emb") is not None and w["emb"].shape[0] == 1152 for w in wins)  # SigLIP so400m width
    assert any(w["sem"] != 0.5 for w in wins)  # real prompt scores, not the neutral mock value
