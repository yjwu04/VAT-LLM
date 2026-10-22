from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from src.semantic_margin import (
    apply_token_semantic_cap,
    load_semantic_margin_artifact,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(tmp_path, values):
    margins = tmp_path / "semantic_margins.npy"
    np.save(margins, np.asarray(values, dtype=np.float32), allow_pickle=False)
    manifest = tmp_path / "semantic_margin_manifest.json"
    payload = {
        "schema_version": 1,
        "status": "complete",
        "distance": "original_base_input_embedding_fp32_l2",
        "margin_rule": "half_nearest_confirmed_incompatible_distance",
        "no_margin_value": "positive_infinity",
        "post_clipping_renormalization": False,
        "base_model": "/base/model",
        "embedding_key": "model.embed_tokens.weight",
        "embedding_shape": [len(values), 2],
        "embedding_shard_sha256": "a" * 64,
        "vocabulary_size": len(values),
        "covered_token_count": int(np.isfinite(values).sum()),
        "margins_file": margins.name,
        "margins_sha256": _sha256(margins),
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_load_semantic_margin_artifact(tmp_path):
    manifest = _artifact(tmp_path, [float("inf"), 0.5, 0.0])
    artifact = load_semantic_margin_artifact(
        manifest, expected_vocabulary_size=3
    )
    assert artifact.margins.dtype == torch.float32
    assert artifact.margins.tolist() == [float("inf"), 0.5, 0.0]
    assert artifact.covered_token_count == 2


def test_semantic_cap_is_tokenwise_and_does_not_renormalize():
    perturbation = torch.tensor(
        [[[3.0, 4.0], [0.0, 2.0], [8.0, 6.0]]], dtype=torch.float32
    )
    input_ids = torch.tensor([[1, 2, 0]])
    mask = torch.tensor([[True, True, False]])
    margins = torch.tensor([float("inf"), 2.5, float("inf")])
    result = apply_token_semantic_cap(
        perturbation, input_ids, mask, margins
    )
    assert torch.allclose(result[0, 0], torch.tensor([1.5, 2.0]))
    assert torch.equal(result[0, 1], perturbation[0, 1])
    assert torch.count_nonzero(result[0, 2]) == 0
    assert torch.linalg.vector_norm(result) < torch.linalg.vector_norm(
        perturbation[mask]
    )


def test_zero_margin_and_zero_row_are_well_defined():
    perturbation = torch.zeros(1, 2, 3, dtype=torch.bfloat16)
    input_ids = torch.tensor([[0, 1]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    margins = torch.tensor([0.0, float("inf")])
    result = apply_token_semantic_cap(
        perturbation, input_ids, mask, margins
    )
    assert result.dtype == torch.bfloat16
    assert torch.count_nonzero(result) == 0


def test_manifest_rejects_wrong_vector_hash(tmp_path):
    manifest = _artifact(tmp_path, [0.5, float("inf")])
    payload = json.loads(manifest.read_text())
    payload["margins_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA-256"):
        load_semantic_margin_artifact(manifest, expected_vocabulary_size=2)


def test_manifest_is_bound_to_current_base_embedding_shard(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    shard = base / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"original embedding shard")
    (base / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": shard.name,
                }
            }
        ),
        encoding="utf-8",
    )
    snapshot_contract = base / ".vat_model_snapshot.json"
    snapshot_contract.write_text(
        json.dumps(
            {
                "repo_id": "meta-llama/Meta-Llama-3-8B",
                "revision": "test-revision",
                "snapshot_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    manifest = _artifact(tmp_path, [0.5, float("inf")])
    payload = json.loads(manifest.read_text())
    payload["base_model"] = str(base.resolve())
    payload["embedding_shard_sha256"] = _sha256(shard)
    payload.update(
        {
            "base_model_snapshot_contract": str(snapshot_contract),
            "base_model_snapshot_contract_sha256": _sha256(snapshot_contract),
            "base_model_snapshot_sha256": "b" * 64,
            "base_model_repo_id": "meta-llama/Meta-Llama-3-8B",
            "base_model_revision": "test-revision",
        }
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    load_semantic_margin_artifact(
        manifest,
        expected_vocabulary_size=2,
        expected_embedding_shape=(2, 2),
        expected_base_model=base,
    )
    shard.write_bytes(b"different embedding shard")
    with pytest.raises(ValueError, match="differs from the training base model"):
        load_semantic_margin_artifact(
            manifest,
            expected_vocabulary_size=2,
            expected_embedding_shape=(2, 2),
            expected_base_model=base,
        )


def test_semantic_cap_rejects_invalid_margin_values():
    with pytest.raises(ValueError, match="non-negative"):
        apply_token_semantic_cap(
            torch.ones(1, 1, 2),
            torch.tensor([[0]]),
            torch.tensor([[True]]),
            torch.tensor([-0.1]),
        )
