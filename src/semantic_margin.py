"""Semantic-margin artifacts and the final token-wise Dense-VAT cap.

The semantic-margin search is an offline operation.  Training consumes one
immutable FP32 vector indexed by vocabulary ID: a finite value is ``m_t`` and
positive infinity means that no reliable semantic margin exists for the token.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


SEMANTIC_CAP_DENOMINATOR_DELTA = 1.0e-12
SEMANTIC_MARGIN_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"semantic-margin manifest must be a JSON object: {path}")
    return value


@dataclass(frozen=True)
class SemanticMarginArtifact:
    """Validated semantic margins plus their immutable provenance."""

    margins: Tensor
    manifest_path: Path
    manifest_sha256: str
    margins_sha256: str
    embedding_shard_sha256: str
    covered_token_count: int


def load_semantic_margin_artifact(
    manifest_path: str | Path,
    *,
    expected_vocabulary_size: int,
    expected_base_model: str | Path | None = None,
    expected_embedding_shape: tuple[int, int] | None = None,
) -> SemanticMarginArtifact:
    """Load a completed FP32 semantic-margin vector and fail closed on drift."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = _load_json_object(path)
    expected_fields = {
        "schema_version": SEMANTIC_MARGIN_SCHEMA_VERSION,
        "status": "complete",
        "distance": "original_base_input_embedding_fp32_l2",
        "margin_rule": "half_nearest_confirmed_incompatible_distance",
        "no_margin_value": "positive_infinity",
        "post_clipping_renormalization": False,
    }
    for name, expected in expected_fields.items():
        if manifest.get(name) != expected:
            raise ValueError(
                f"semantic-margin manifest {name}={manifest.get(name)!r}, "
                f"expected {expected!r}"
            )
    if isinstance(expected_vocabulary_size, bool) or expected_vocabulary_size < 1:
        raise ValueError("expected_vocabulary_size must be a positive integer")
    if manifest.get("vocabulary_size") != expected_vocabulary_size:
        raise ValueError(
            "semantic-margin vocabulary size does not match the training model"
        )
    if manifest.get("embedding_key") != "model.embed_tokens.weight":
        raise ValueError("semantic-margin embedding key is not the input embedding")
    embedding_shape = manifest.get("embedding_shape")
    if (
        not isinstance(embedding_shape, list)
        or len(embedding_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in embedding_shape)
        or embedding_shape[0] != expected_vocabulary_size
        or embedding_shape[1] < 1
    ):
        raise ValueError("semantic-margin embedding_shape is invalid")
    if expected_embedding_shape is not None and tuple(embedding_shape) != tuple(
        expected_embedding_shape
    ):
        raise ValueError("semantic-margin embedding shape differs from the base model")
    embedding_sha256 = manifest.get("embedding_shard_sha256")
    if (
        not isinstance(embedding_sha256, str)
        or len(embedding_sha256) != 64
        or any(character not in "0123456789abcdef" for character in embedding_sha256)
    ):
        raise ValueError("semantic-margin embedding shard SHA-256 is invalid")
    if expected_base_model is not None:
        base_model = Path(expected_base_model).expanduser().resolve()
        if Path(str(manifest.get("base_model", ""))).resolve() != base_model:
            raise ValueError("semantic-margin base model path differs from training")
        snapshot_contract = base_model / ".vat_model_snapshot.json"
        if not snapshot_contract.is_file():
            raise FileNotFoundError(snapshot_contract)
        snapshot = _load_json_object(snapshot_contract)
        if (
            manifest.get("base_model_snapshot_contract")
            != str(snapshot_contract)
            or manifest.get("base_model_snapshot_contract_sha256")
            != _sha256_file(snapshot_contract)
            or manifest.get("base_model_snapshot_sha256")
            != snapshot.get("snapshot_sha256")
            or manifest.get("base_model_repo_id") != snapshot.get("repo_id")
            or manifest.get("base_model_revision") != snapshot.get("revision")
        ):
            raise ValueError(
                "semantic-margin base tokenizer/model snapshot differs from training"
            )
        index_path = base_model / "model.safetensors.index.json"
        index = _load_json_object(index_path)
        relative = index.get("weight_map", {}).get("model.embed_tokens.weight")
        if not isinstance(relative, str) or not relative:
            raise ValueError("training base model lacks its input embedding shard")
        actual_shard = (base_model / relative).resolve()
        if not actual_shard.is_file():
            raise FileNotFoundError(actual_shard)
        if _sha256_file(actual_shard) != embedding_sha256:
            raise ValueError(
                "semantic-margin embedding shard differs from the training base model"
            )

    relative = manifest.get("margins_file")
    if not isinstance(relative, str) or not relative:
        raise ValueError("semantic-margin manifest lacks margins_file")
    margins_path = (path.parent / relative).resolve()
    if margins_path.parent != path.parent:
        raise ValueError("semantic-margin margins_file must be beside its manifest")
    if not margins_path.is_file():
        raise FileNotFoundError(margins_path)
    observed_sha256 = _sha256_file(margins_path)
    if manifest.get("margins_sha256") != observed_sha256:
        raise ValueError("semantic-margin FP32 vector SHA-256 mismatch")

    array = np.load(margins_path, allow_pickle=False)
    if array.dtype != np.dtype("float32"):
        raise ValueError(
            f"semantic margins must have dtype float32, got {array.dtype}"
        )
    if array.shape != (expected_vocabulary_size,):
        raise ValueError(
            "semantic margins must have shape [vocabulary_size], got "
            f"{array.shape}"
        )
    if np.isnan(array).any() or np.isneginf(array).any() or (array < 0).any():
        raise ValueError(
            "semantic margins must be non-negative finite values or +infinity"
        )
    covered = int(np.isfinite(array).sum())
    if manifest.get("covered_token_count") != covered:
        raise ValueError("semantic-margin covered_token_count mismatch")
    return SemanticMarginArtifact(
        margins=torch.from_numpy(np.array(array, copy=True)),
        manifest_path=path,
        manifest_sha256=_sha256_file(path),
        margins_sha256=observed_sha256,
        embedding_shard_sha256=embedding_sha256,
        covered_token_count=covered,
    )


def apply_token_semantic_cap(
    perturbation: Tensor,
    input_ids: Tensor,
    token_mask: Tensor,
    semantic_margins: Tensor,
    *,
    denominator_delta: float = SEMANTIC_CAP_DENOMINATOR_DELTA,
) -> Tensor:
    """Apply the exact final token-wise cap without group renormalization.

    For every selected token row this computes

    ``r_t = r_tilde * min(1, m_t / (||r_tilde||_2 + 1e-12))``.

    The row norm and scale are evaluated in FP32.  Unselected rows are exactly
    zero, and a ``+inf`` margin leaves a selected row unchanged.  The returned
    tensor keeps the perturbation dtype so adding it to BF16 embeddings does
    not promote the full model input.
    """

    if perturbation.ndim != 3:
        raise ValueError("perturbation must have shape [B, T, H]")
    if input_ids.shape != perturbation.shape[:2]:
        raise ValueError("input_ids must match perturbation [B, T]")
    if token_mask.shape != perturbation.shape[:2]:
        raise ValueError("token_mask must match perturbation [B, T]")
    if semantic_margins.ndim != 1 or semantic_margins.numel() < 1:
        raise ValueError("semantic_margins must have shape [V]")
    if semantic_margins.device != perturbation.device:
        raise ValueError("semantic_margins and perturbation must share a device")
    if input_ids.device != perturbation.device:
        raise ValueError("input_ids and perturbation must share a device")
    if not math.isfinite(float(denominator_delta)) or denominator_delta <= 0:
        raise ValueError("denominator_delta must be finite and positive")

    ids = input_ids.to(dtype=torch.long)
    if bool((ids < 0).any()) or bool((ids >= semantic_margins.numel()).any()):
        raise ValueError("input_ids contain an out-of-vocabulary token ID")
    margins = semantic_margins.to(dtype=torch.float32)
    if bool(torch.isnan(margins).any()) or bool(torch.isneginf(margins).any()):
        raise ValueError("semantic_margins contain NaN or negative infinity")
    if bool((margins < 0).any()):
        raise ValueError("semantic_margins must be non-negative")

    selected = token_mask.to(device=perturbation.device, dtype=torch.bool)
    result_fp32 = torch.zeros_like(perturbation, dtype=torch.float32)
    if not selected.any():
        return result_fp32.to(dtype=perturbation.dtype)

    selected_rows = perturbation[selected].float()
    selected_norms = torch.linalg.vector_norm(selected_rows, ord=2, dim=-1)
    selected_margins = margins[ids[selected]]
    finite_scales = selected_margins / (
        selected_norms + float(denominator_delta)
    )
    scales = torch.where(
        torch.isposinf(selected_margins),
        torch.ones_like(finite_scales),
        torch.minimum(torch.ones_like(finite_scales), finite_scales),
    )
    result_fp32[selected] = selected_rows * scales.unsqueeze(-1)
    return result_fp32.to(dtype=perturbation.dtype)


__all__ = [
    "SEMANTIC_CAP_DENOMINATOR_DELTA",
    "SEMANTIC_MARGIN_SCHEMA_VERSION",
    "SemanticMarginArtifact",
    "apply_token_semantic_cap",
    "load_semantic_margin_artifact",
]
