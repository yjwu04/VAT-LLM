from __future__ import annotations

import json

import pytest

from src.artifact_fingerprint import checkpoint_fingerprint


def _checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"llama"}\n')
    (checkpoint / "tokenizer.json").write_text('{"version":"1"}\n')
    (checkpoint / "tokenizer_config.json").write_text('{"legacy":false}\n')
    (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": ("model-00001-of-00001.safetensors")
                }
            }
        )
        + "\n"
    )
    return checkpoint


def test_checkpoint_fingerprint_binds_index_and_loadable_artifacts(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    original = checkpoint_fingerprint(checkpoint)
    index = checkpoint / "model.safetensors.index.json"
    index.write_text(index.read_text() + "\n")
    assert checkpoint_fingerprint(checkpoint) != original


def test_checkpoint_fingerprint_rejects_missing_index_shard(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(
        RuntimeError, match="no model weight files|missing weight shards"
    ):
        checkpoint_fingerprint(checkpoint)


@pytest.mark.parametrize("missing", ["config.json", "tokenizer.json"])
def test_checkpoint_fingerprint_rejects_incomplete_loadable_checkpoint(
    tmp_path, missing
):
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / missing).unlink()
    with pytest.raises(RuntimeError):
        checkpoint_fingerprint(checkpoint)
