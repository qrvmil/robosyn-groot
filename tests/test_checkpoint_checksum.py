from pathlib import Path

from tools.reload_checkpoint import sha256_model_weights


def test_checkpoint_checksum_only_tracks_model_weights(tmp_path: Path):
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"model-a")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"model-b")
    initial = sha256_model_weights(tmp_path)

    (tmp_path / "optimizer.pt").write_bytes(b"large mutable optimizer")
    (tmp_path / "robosyn_evidence").mkdir()
    (tmp_path / "robosyn_evidence/stats.json").write_text("{}")
    assert sha256_model_weights(tmp_path) == initial

    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"different")
    assert sha256_model_weights(tmp_path) != initial
