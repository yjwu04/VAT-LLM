from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.dataset_view import (
    canonical_dataset_content_digest,
    load_materialized_dataset,
    tree_sha,
    verify_canonical_dataset_equality,
)


class _Table:
    def __init__(self, rows: int):
        self.rows = rows

    def __len__(self) -> int:
        return self.rows


class _Dataset:
    def __init__(self, rows: int, *, indexed: bool = False):
        self.rows = rows
        self._indices = object() if indexed else None
        self.data = _Table(rows)

    def __len__(self) -> int:
        return self.rows


def test_load_materialized_dataset_binds_tree_and_logical_table(
    monkeypatch, tmp_path
):
    root = tmp_path / "materialized"
    root.mkdir()
    (root / "data.arrow").write_bytes(b"rows")
    digest, _, _ = tree_sha(root)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_from_disk=lambda _: _Dataset(50_000)),
    )

    dataset = load_materialized_dataset(
        root,
        expected_tree_sha256=digest,
        expected_rows=50_000,
    )
    assert len(dataset) == 50_000
    assert dataset._indices is None
    assert len(dataset.data) == 50_000


def test_load_materialized_dataset_rejects_tree_or_indices_view(
    monkeypatch, tmp_path
):
    root = tmp_path / "materialized"
    root.mkdir()
    (root / "data.arrow").write_bytes(b"rows")
    digest, _, _ = tree_sha(root)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            load_from_disk=lambda _: _Dataset(50_000, indexed=True)
        ),
    )
    with pytest.raises(RuntimeError, match="indices view"):
        load_materialized_dataset(
            root,
            expected_tree_sha256=digest,
            expected_rows=50_000,
        )
    with pytest.raises(RuntimeError, match="tree hash"):
        load_materialized_dataset(
            root,
            expected_tree_sha256="0" * 64,
            expected_rows=50_000,
        )


def test_canonical_digest_and_exact_equality_cover_order_and_all_columns():
    datasets = pytest.importorskip("datasets")
    source = datasets.Dataset.from_dict(
        {
            "input_ids": [[1, 2], [3], [4, 5]],
            "labels": [[-100, 2], [3], [4, 5]],
            "length": [2, 1, 2],
            "text": ["a", "b", "c"],
        }
    )
    identical = datasets.Dataset.from_dict(source[:])
    digest = canonical_dataset_content_digest(source, batch_rows=2)
    assert digest == canonical_dataset_content_digest(
        identical, batch_rows=2
    )
    verify_canonical_dataset_equality(source, identical, batch_rows=2)

    reordered = identical.select([1, 0, 2]).flatten_indices()
    with pytest.raises(ValueError, match="logical values differ"):
        verify_canonical_dataset_equality(source, reordered, batch_rows=2)
    assert canonical_dataset_content_digest(
        reordered, batch_rows=2
    )["content_sha256"] != digest["content_sha256"]

    changed_non_length_column = datasets.Dataset.from_dict(
        {
            "input_ids": [[1, 2], [3], [4, 5]],
            "labels": [[-100, 2], [999], [4, 5]],
            "length": [2, 1, 2],
            "text": ["a", "b", "c"],
        }
    )
    with pytest.raises(ValueError, match="logical values differ"):
        verify_canonical_dataset_equality(
            source, changed_non_length_column, batch_rows=2
        )
