"""Runtime hooks that keep Axolotl as the sole training/data foundation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset_view import load_materialized_dataset
from .metamorphic_vat import make_metamorphic_vat_trainer
from .user_token_mask import tokenize_axolotl_llama3_prompt_with_mask
from .vat_trainer import make_question_only_vat_trainer


AXOLOTL_RECONSTRUCTION_COMMIT = "8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2"


def configure_assistant_eot_supervision(enabled: bool) -> None:
    """Set the process-local label policy used by ShareGPT strategies.

    The value is attached to the strategy class so every tokenizer instance
    created later by Axolotl sees the same validated training configuration.
    """

    from axolotl.prompt_tokenizers import ShareGPTPromptTokenizingStrategy

    ShareGPTPromptTokenizingStrategy.supervise_assistant_eot = bool(enabled)


def verify_axolotl_checkout(
    module_file: str,
    *,
    expected_commit: str = AXOLOTL_RECONSTRUCTION_COMMIT,
) -> str:
    """Fail closed unless the imported Axolotl comes from the locked checkout."""

    path = Path(module_file).resolve()
    git_root = next(
        (parent for parent in (path, *path.parents) if (parent / ".git").exists()),
        None,
    )
    if git_root is None:
        raise RuntimeError(f"cannot find Axolotl git root above {path}")
    observed = subprocess.run(
        ["git", "-C", str(git_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != expected_commit:
        raise RuntimeError(
            f"Axolotl revision mismatch: expected {expected_commit}, got {observed}"
        )
    return observed


def _pad_perturb_mask(
    features: list[dict[str, Any]],
    *,
    pad_to_multiple_of: int | None,
    padding_side: str,
) -> None:
    if not features:
        return
    mask_presence = ["perturb_mask" in feature for feature in features]
    if not any(mask_presence):
        return
    if not all(mask_presence):
        raise ValueError("perturb_mask is missing from part of a collator batch")
    source_presence = ["source_ids" in feature for feature in features]
    if any(source_presence) and not all(source_presence):
        raise ValueError("source_ids is missing from part of a collator batch")
    for feature in features:
        input_length = len(feature["input_ids"])
        for field in ("perturb_mask", "source_ids"):
            if field in feature and len(feature[field]) != input_length:
                raise ValueError(
                    f"{field} length differs from input_ids before padding"
                )
        perturb_values = np.asarray(feature["perturb_mask"]).tolist()
        if any(int(value) not in (0, 1) for value in perturb_values):
            raise ValueError("perturb_mask must be binary before padding")

    target = max(len(feature["input_ids"]) for feature in features)
    if pad_to_multiple_of:
        target = (
            (target + pad_to_multiple_of - 1) // pad_to_multiple_of
        ) * pad_to_multiple_of
    for feature in features:
        for field, pad_value in (("perturb_mask", 0), ("source_ids", -1)):
            if field not in feature:
                if field == "source_ids":
                    continue
                raise ValueError(f"{field} is missing from a collator feature")
            value = feature[field]
            remainder = [pad_value] * (target - len(value))
            if isinstance(value, list):
                feature[field] = (
                    value + remainder if padding_side == "right" else remainder + value
                )
            elif isinstance(value, torch.Tensor):
                pad = value.new_full((len(remainder),), pad_value)
                feature[field] = (
                    torch.cat((value, pad))
                    if padding_side == "right"
                    else torch.cat((pad, value))
                )
            else:
                array = np.asarray(value)
                pad = np.full(len(remainder), pad_value, dtype=array.dtype)
                feature[field] = (
                    np.concatenate((array, pad))
                    if padding_side == "right"
                    else np.concatenate((pad, array))
                )


def install_axolotl_vat_integration(*, verify_revision: bool = True) -> None:
    """Install idempotent hooks into the pinned Axolotl process.

    The hooks add one sequence field during Llama-3 ShareGPT tokenization, pad
    that field in Axolotl's existing collator, and ask the existing
    ``HFCausalTrainerBuilder`` for the VAT-aware subclass of whichever trainer
    it would otherwise select.
    """

    import axolotl
    from axolotl.core.trainer_builder import HFCausalTrainerBuilder
    from axolotl.prompt_tokenizers import ShareGPTPromptTokenizingStrategy
    from axolotl.utils.collators import DataCollatorForSeq2Seq
    from axolotl.utils.data import sft as axolotl_sft

    if verify_revision:
        verify_axolotl_checkout(axolotl.__file__)

    if not getattr(
        ShareGPTPromptTokenizingStrategy,
        "_vat_user_mask_hook_installed",
        False,
    ):
        original_tokenize_prompt = ShareGPTPromptTokenizingStrategy.tokenize_prompt

        def tokenize_prompt(self: Any, prompt: Any) -> Any:
            conversation = getattr(
                getattr(self, "prompter", None), "_conversation", None
            )
            if getattr(conversation, "name", None) == "llama3":
                return tokenize_axolotl_llama3_prompt_with_mask(self, prompt)
            return original_tokenize_prompt(self, prompt)

        ShareGPTPromptTokenizingStrategy.tokenize_prompt = tokenize_prompt
        ShareGPTPromptTokenizingStrategy._vat_user_mask_hook_installed = True

    if not getattr(DataCollatorForSeq2Seq, "_vat_mask_padding_hook_installed", False):
        original_collator_call = DataCollatorForSeq2Seq.__call__

        def collator_call(
            self: Any,
            features: list[dict[str, Any]],
            return_tensors: str | None = None,
        ) -> Any:
            _pad_perturb_mask(
                features,
                pad_to_multiple_of=self.pad_to_multiple_of,
                padding_side=self.tokenizer.padding_side,
            )
            batch = original_collator_call(
                self, features, return_tensors=return_tensors
            )
            if "perturb_mask" in batch:
                batch["perturb_mask"] = batch["perturb_mask"].to(dtype=torch.bool)
                if batch["perturb_mask"].shape != batch["input_ids"].shape:
                    raise ValueError(
                        "collated perturb_mask shape differs from input_ids"
                    )
            if "source_ids" in batch:
                batch["source_ids"] = batch["source_ids"].to(dtype=torch.long)
                if batch["source_ids"].shape != batch["input_ids"].shape:
                    raise ValueError("collated source_ids shape differs from input_ids")
            return batch

        DataCollatorForSeq2Seq.__call__ = collator_call
        DataCollatorForSeq2Seq._vat_mask_padding_hook_installed = True

    if not getattr(
        axolotl_sft.load_prepare_datasets,
        "_vat_logical_shard_materialization_hook_installed",
        False,
    ):
        original_load_prepare_datasets = axolotl_sft.load_prepare_datasets

        def load_prepare_datasets(*args: Any, **kwargs: Any) -> Any:
            train_dataset, eval_dataset, prompters = (
                original_load_prepare_datasets(*args, **kwargs)
            )
            cfg = args[1] if len(args) > 1 else kwargs.get("cfg")
            if (
                cfg is not None
                and getattr(cfg, "dataset_shard_num", None)
                and getattr(cfg, "dataset_shard_idx", None) is not None
            ):
                if eval_dataset is not None:
                    raise RuntimeError(
                        "Phase-2 materialized shard requires val_set_size=0"
                    )
                materialized_path = getattr(
                    cfg, "phase2_materialized_train_dataset_path", None
                )
                materialized_tree = getattr(
                    cfg, "phase2_materialized_train_dataset_tree_sha256", None
                )
                materialized_rows = getattr(
                    cfg, "phase2_materialized_train_dataset_rows", None
                )
                if (
                    not isinstance(materialized_path, str)
                    or not isinstance(materialized_tree, str)
                    or len(materialized_tree) != 64
                    or not isinstance(materialized_rows, int)
                    or isinstance(materialized_rows, bool)
                    or materialized_rows <= 0
                ):
                    raise RuntimeError(
                        "sharded training requires a bound materialized Dataset"
                    )
                if train_dataset is None or len(train_dataset) != materialized_rows:
                    raise RuntimeError(
                        "Axolotl logical shard row count differs from the "
                        "materialized Dataset contract"
                    )
                train_dataset = load_materialized_dataset(
                    materialized_path,
                    expected_tree_sha256=materialized_tree,
                    expected_rows=materialized_rows,
                )
            return train_dataset, eval_dataset, prompters

        load_prepare_datasets._vat_logical_shard_materialization_hook_installed = (
            True
        )
        axolotl_sft.load_prepare_datasets = load_prepare_datasets

    if not getattr(HFCausalTrainerBuilder, "_vat_trainer_hook_installed", False):
        original_get_trainer_cls = HFCausalTrainerBuilder._get_trainer_cls

        def get_trainer_cls(self: Any) -> type:
            base_trainer_cls = original_get_trainer_cls(self)
            if bool(getattr(self.cfg, "use_metamorphic_vat", False)):
                return make_metamorphic_vat_trainer(base_trainer_cls, self.cfg)
            return make_question_only_vat_trainer(base_trainer_cls, self.cfg)

        HFCausalTrainerBuilder._get_trainer_cls = get_trainer_cls
        HFCausalTrainerBuilder._vat_trainer_hook_installed = True


def install_axolotl_native_trainer_integration(
    *, verify_revision: bool = True
) -> None:
    """Install only the trainer extension, preserving native preprocessing.

    Unlike :func:`install_axolotl_vat_integration`, this deliberately does not
    patch ShareGPT tokenization, collators, or dataset preparation.  Native
    Axolotl therefore remains the sole owner of ``input_ids``, ``labels``,
    attention masks, truncation, padding, and packing.  The VAT trainer derives
    its question mask later from the already-packed native tensors.
    """

    import axolotl
    from axolotl.core.trainer_builder import HFCausalTrainerBuilder

    if verify_revision:
        verify_axolotl_checkout(axolotl.__file__)
    if not getattr(HFCausalTrainerBuilder, "_vat_trainer_hook_installed", False):
        original_get_trainer_cls = HFCausalTrainerBuilder._get_trainer_cls

        def get_trainer_cls(self: Any) -> type:
            base_trainer_cls = original_get_trainer_cls(self)
            if bool(getattr(self.cfg, "use_metamorphic_vat", False)):
                return make_metamorphic_vat_trainer(base_trainer_cls, self.cfg)
            return make_question_only_vat_trainer(base_trainer_cls, self.cfg)

        HFCausalTrainerBuilder._get_trainer_cls = get_trainer_cls
        HFCausalTrainerBuilder._vat_trainer_hook_installed = True


__all__ = [
    "AXOLOTL_RECONSTRUCTION_COMMIT",
    "configure_assistant_eot_supervision",
    "install_axolotl_vat_integration",
    "install_axolotl_native_trainer_integration",
    "verify_axolotl_checkout",
]
