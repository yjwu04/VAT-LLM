"""Load and verify a materialized Hugging Face Dataset logical shard.

Axolotl v0.4.0 reads several packing and token-count fields directly from
``dataset.data``.  That Arrow table is the unfiltered parent table when a
Dataset is an indices view, so a non-contiguous ``Dataset.shard`` must be
materialized before Axolotl sees it.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any


CANONICAL_DATASET_DIGEST_ALGORITHM = (
    "arrow_logical_values_all_columns_fixed_128_rows_v2"
)
CANONICAL_DATASET_DIGEST_BATCH_ROWS = 128


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha(directory: pathlib.Path) -> tuple[str, int, int]:
    directory = directory.resolve()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"materialized Dataset contains no files: {directory}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        if path.is_symlink():
            raise RuntimeError("materialized Dataset may not contain symlinks")
        relative = path.relative_to(directory).as_posix().encode()
        size = path.stat().st_size
        total_bytes += size
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest(), len(files), total_bytes


def canonical_dataset_content_digest(
    dataset: Any,
    *,
    batch_rows: int = CANONICAL_DATASET_DIGEST_BATCH_ROWS,
) -> dict[str, Any]:
    """Hash every logical value in row order using canonical Arrow slices.

    Hugging Face fingerprints describe a transform graph, not necessarily the
    logical values produced by that graph.  This digest deliberately reads the
    public logical Dataset view in fixed-size slices, removes schema metadata,
    combines Arrow chunks, and hashes normalized list offsets plus primitive
    integer values.  It deliberately avoids Arrow IPC padding bytes, whose
    contents are not part of logical equality.  Fixed slicing makes the result
    independent of the backing Arrow file's physical chunking.
    """

    import numpy as np
    import pyarrow as pa

    def update_integer_array(
        target: Any,
        array: Any,
        *,
        context: str,
    ) -> None:
        if array.null_count:
            raise ValueError(
                f"canonical Dataset digest refuses nulls in {context}"
            )
        type_name = str(array.type).encode("ascii")
        target.update(len(type_name).to_bytes(4, "big"))
        target.update(type_name)
        target.update(len(array).to_bytes(8, "big"))
        if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
            offsets = np.asarray(
                array.offsets.to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            normalized = offsets - offsets[0]
            offset_bytes = normalized.astype(">i8", copy=False).tobytes()
            target.update(len(offset_bytes).to_bytes(8, "big"))
            target.update(offset_bytes)
            start = int(offsets[0])
            count = int(offsets[-1] - offsets[0])
            update_integer_array(
                target,
                array.values.slice(start, count),
                context=f"{context}.values",
            )
            return
        if (
            pa.types.is_string(array.type)
            or pa.types.is_large_string(array.type)
            or pa.types.is_binary(array.type)
            or pa.types.is_large_binary(array.type)
        ):
            for value in array.to_pylist():
                payload = (
                    value.encode("utf-8")
                    if isinstance(value, str)
                    else bytes(value)
                )
                target.update(len(payload).to_bytes(8, "big"))
                target.update(payload)
            return
        if pa.types.is_boolean(array.type):
            payload = bytes(int(value) for value in array.to_pylist())
            target.update(len(payload).to_bytes(8, "big"))
            target.update(payload)
            return
        dtype_map = {
            pa.int8(): np.dtype("i1"),
            pa.uint8(): np.dtype("u1"),
            pa.int16(): np.dtype(">i2"),
            pa.uint16(): np.dtype(">u2"),
            pa.int32(): np.dtype(">i4"),
            pa.uint32(): np.dtype(">u4"),
            pa.int64(): np.dtype(">i8"),
            pa.uint64(): np.dtype(">u8"),
        }
        dtype = dtype_map.get(array.type)
        if dtype is None:
            raise ValueError(
                "canonical Dataset digest supports only integer or "
                f"list-of-integer columns, got {array.type} in {context}"
            )
        values = np.asarray(
            array.to_numpy(zero_copy_only=False),
        ).astype(dtype, copy=False)
        payload = values.tobytes()
        target.update(len(payload).to_bytes(8, "big"))
        target.update(payload)

    if (
        not isinstance(batch_rows, int)
        or isinstance(batch_rows, bool)
        or batch_rows <= 0
    ):
        raise ValueError("canonical Dataset digest batch_rows must be positive")
    row_count = len(dataset)
    column_names = list(getattr(dataset, "column_names", ()))
    if not column_names or len(column_names) != len(set(column_names)):
        raise ValueError("canonical Dataset digest requires unique columns")

    logical = dataset.with_format(
        "arrow",
        columns=column_names,
        output_all_columns=False,
    )
    digest = hashlib.sha256()
    marker = CANONICAL_DATASET_DIGEST_ALGORITHM.encode("ascii")
    digest.update(len(marker).to_bytes(4, "big"))
    digest.update(marker)
    digest.update(row_count.to_bytes(8, "big"))
    digest.update(len(column_names).to_bytes(4, "big"))
    for name in column_names:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)

    canonical_schema: Any | None = None
    for start in range(0, row_count, batch_rows):
        stop = min(start + batch_rows, row_count)
        table = logical[start:stop]
        if isinstance(table, pa.RecordBatch):
            table = pa.Table.from_batches([table])
        if not isinstance(table, pa.Table):
            raise ValueError("logical Dataset slice did not produce an Arrow table")
        if table.num_rows != stop - start:
            raise ValueError("logical Dataset slice row count differs")
        if table.column_names != column_names:
            raise ValueError("logical Dataset slice column order differs")
        table = table.replace_schema_metadata(None).combine_chunks()
        if canonical_schema is None:
            canonical_schema = table.schema
        elif not table.schema.equals(canonical_schema, check_metadata=False):
            raise ValueError("logical Dataset schema changes between slices")

        digest.update(start.to_bytes(8, "big"))
        digest.update(table.num_rows.to_bytes(8, "big"))
        for name in column_names:
            column = table.column(name)
            if column.num_chunks != 1:
                raise ValueError(
                    "canonical Dataset digest expected one combined chunk"
                )
            update_integer_array(
                digest,
                column.chunk(0),
                context=f"{name}[{start}:{stop}]",
            )

    if canonical_schema is None:
        raise ValueError("canonical Dataset digest refuses an empty Dataset")
    schema_bytes = canonical_schema.serialize().to_pybytes()
    return {
        "algorithm": CANONICAL_DATASET_DIGEST_ALGORITHM,
        "batch_rows": batch_rows,
        "row_count": row_count,
        "column_names": column_names,
        "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "content_sha256": digest.hexdigest(),
    }


def verify_canonical_dataset_equality(
    source: Any,
    materialized: Any,
    *,
    batch_rows: int = CANONICAL_DATASET_DIGEST_BATCH_ROWS,
) -> None:
    """Require exact logical Arrow equality in fixed row-aligned chunks."""

    import pyarrow as pa

    if len(source) != len(materialized):
        raise ValueError("materialized Dataset row count differs from source")
    if list(source.column_names) != list(materialized.column_names):
        raise ValueError("materialized Dataset column order differs from source")
    columns = list(source.column_names)
    source_logical = source.with_format(
        "arrow", columns=columns, output_all_columns=False
    )
    materialized_logical = materialized.with_format(
        "arrow", columns=columns, output_all_columns=False
    )
    for start in range(0, len(source), batch_rows):
        stop = min(start + batch_rows, len(source))
        source_table = source_logical[start:stop]
        materialized_table = materialized_logical[start:stop]
        if isinstance(source_table, pa.RecordBatch):
            source_table = pa.Table.from_batches([source_table])
        if isinstance(materialized_table, pa.RecordBatch):
            materialized_table = pa.Table.from_batches([materialized_table])
        if (
            not isinstance(source_table, pa.Table)
            or not isinstance(materialized_table, pa.Table)
            or source_table.num_rows != stop - start
            or materialized_table.num_rows != stop - start
        ):
            raise ValueError("logical Dataset equality slice is malformed")
        source_table = source_table.replace_schema_metadata(None).combine_chunks()
        materialized_table = (
            materialized_table.replace_schema_metadata(None).combine_chunks()
        )
        if not source_table.equals(
            materialized_table,
            check_metadata=False,
        ):
            raise ValueError(
                "materialized Dataset logical values differ from source at "
                f"rows [{start},{stop})"
            )


def load_materialized_dataset(
    path: str | pathlib.Path,
    *,
    expected_tree_sha256: str,
    expected_rows: int,
) -> Any:
    """Load a real, hash-bound Dataset whose Arrow table is its logical view."""

    from datasets import load_from_disk

    supplied = pathlib.Path(path)
    if supplied.is_symlink() or not supplied.resolve().is_dir():
        raise RuntimeError("materialized Dataset path must be a real directory")
    resolved = supplied.resolve()
    digest, _, _ = tree_sha(resolved)
    if digest != expected_tree_sha256:
        raise RuntimeError("materialized Dataset tree hash differs")
    dataset = load_from_disk(str(resolved))
    if getattr(dataset, "_indices", None) is not None:
        raise RuntimeError("materialized Dataset is still an indices view")
    if len(dataset) != expected_rows:
        raise RuntimeError("materialized Dataset row count differs")
    table = getattr(dataset, "data", None)
    if table is None or len(table) != expected_rows:
        raise RuntimeError(
            "materialized Dataset table does not match its logical row count"
        )
    return dataset


__all__ = [
    "CANONICAL_DATASET_DIGEST_ALGORITHM",
    "CANONICAL_DATASET_DIGEST_BATCH_ROWS",
    "canonical_dataset_content_digest",
    "verify_canonical_dataset_equality",
    "load_materialized_dataset",
    "sha256_file",
    "tree_sha",
]
