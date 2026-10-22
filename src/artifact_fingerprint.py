"""Stable content fingerprints for model/tokenizer checkpoint artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


_MODEL_SUFFIXES = {".safetensors", ".bin", ".model"}
_MODEL_METADATA = {
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "pytorch_model.bin.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


def checkpoint_artifact_files(directory: Path) -> list[Path]:
    """Return only files needed to load the trained model and tokenizer."""

    root = directory.resolve()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix in _MODEL_SUFFIXES
            or path.name in _MODEL_METADATA
            or path.name.startswith("tokenizer.")
        )
    )
    if not files:
        raise RuntimeError(f"no model/tokenizer artifacts found in {root}")
    names = {path.name for path in files}
    if "config.json" not in names:
        raise RuntimeError(f"checkpoint has no config.json: {root}")
    weight_files = [
        path
        for path in files
        if path.suffix in {".safetensors", ".bin"}
        and not path.name.endswith(".index.json")
    ]
    if not weight_files:
        raise RuntimeError(f"checkpoint has no model weight files: {root}")
    if "tokenizer.json" not in names and "tokenizer.model" not in names:
        raise RuntimeError(f"checkpoint has no tokenizer vocabulary artifact: {root}")
    if "tokenizer_config.json" not in names:
        raise RuntimeError(f"checkpoint has no tokenizer_config.json: {root}")
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = root / index_name
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"checkpoint index has no weight_map: {index_path}")
        missing = sorted(
            shard for shard in set(weight_map.values()) if not (root / shard).is_file()
        )
        if missing:
            raise RuntimeError(
                f"checkpoint index references missing weight shards: {missing}"
            )
    return files


def checkpoint_fingerprint(directory: Path) -> str:
    """Hash relative paths, sizes, and bytes for load-bearing checkpoint files."""

    root = directory.resolve()
    digest = hashlib.sha256()
    for path in checkpoint_artifact_files(root):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(32 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


__all__ = ["checkpoint_artifact_files", "checkpoint_fingerprint"]
