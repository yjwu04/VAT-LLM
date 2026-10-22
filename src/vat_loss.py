"""Mathematical core for question-only virtual adversarial training.

The functions in this module deliberately do not own a trainer.  They provide:

* causal, assistant-only clean-to-perturbed KL divergence in FP32;
* masked Gaussian perturbation initialization;
* per-packed-sequence Frobenius normalization; and
* one-step power iteration for constructing ``r_vadv``.

All token masks use the unshifted ``[batch, sequence]`` layout.  In particular,
``perturb_mask`` marks input embeddings that may be perturbed, while
``assistant_prediction_mask(labels)`` marks logit positions that predict an
assistant token after applying the decoder-only causal shift.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .semantic_margin import apply_token_semantic_cap


_KL_ROW_CHUNK_SIZE = 32
# Relative-radius VAT uses the ``g / (||g|| + delta)`` construction in FP32.
# The legacy absolute-radius path retains delta=0 to preserve its numerical
# semantics, while relative-rho uses this explicit positive stabilizer.
RELATIVE_GRADIENT_NORM_DELTA = 1.0e-12
# Dense-VAT constants are fixed by the formulation rather than exposed as
# tunable hyperparameters.
DENSE_VAT_COUNT_FLOOR = 1.0e-4
DENSE_VAT_COUNT_EXPONENT = 1.0 / 12.0
# Direct-distance query chunking bounds the temporary output of the exact
# exhaustive vocabulary scan.  It is an implementation detail, not a model
# hyperparameter and does not change the neighbor set.
_DENSE_VAT_QUERY_CHUNK_SIZE = 16
# These are numerical identity tolerances, not admission/effect thresholds.
# They are emitted with standalone evidence so a protocol can seal the exact
# values and a separate gate can reject tolerance drift.
# Two independent FP32 positive reductions (sum vs scatter_add) over at most
# 8192 tokens have a conservative relative discrepancy just above 2^-10:
# 2*gamma_8191/(1-gamma_8191), plus the surrounding sqrt/divide/square
# roundings. 2^-9 leaves roughly a 2x implementation-identity margin while
# remaining far below any model-selection scale. A zero absolute tolerance
# preserves the identity's dimensionless scale invariance.
RHO_WEIGHTED_ENERGY_IDENTITY_RTOL = 2.0**-9
RHO_WEIGHTED_ENERGY_IDENTITY_ATOL = 0.0


@dataclass(frozen=True)
class RandomNeighborBatchStats:
    """Exact sufficient statistics for one random-neighbor microbatch."""

    conversation_count: int
    selected_token_count: int
    neighborhood_size_sum: int
    replacement_count: int

    @property
    def average_neighborhood_size(self) -> float:
        return (
            self.neighborhood_size_sum / self.selected_token_count
            if self.selected_token_count
            else 0.0
        )

    @property
    def replacement_rate(self) -> float:
        return (
            self.replacement_count / self.selected_token_count
            if self.selected_token_count
            else 0.0
        )


def assert_zero_stochastic_dropout(model: nn.Module) -> dict[str, Any]:
    """Fail closed unless the exact model has no active dropout.

    Clean, probe, and adversarial forwards are compared as numerical
    distributions.  Non-zero dropout would confound those differences unless
    masks were explicitly shared, which this protocol does not do.
    """

    config = getattr(model, "config", None)
    config_values = (
        config.to_dict()
        if config is not None and callable(getattr(config, "to_dict", None))
        else {}
    )
    nonzero_config = {
        str(key): float(value)
        for key, value in config_values.items()
        if "dropout" in str(key).lower()
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) != 0.0
    }
    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    nonzero_modules = {
        name: float(module.p)
        for name, module in model.named_modules()
        if isinstance(module, dropout_types) and float(module.p) != 0.0
    }
    if nonzero_config or nonzero_modules:
        raise RuntimeError(
            "VAT requires zero stochastic dropout or an explicit shared-mask "
            f"contract; config={nonzero_config}, modules={nonzero_modules}"
        )
    return {
        "passed": True,
        "config_dropout_fields": sorted(
            str(key)
            for key in config_values
            if "dropout" in str(key).lower()
        ),
        "dropout_module_count": sum(
            isinstance(module, dropout_types) for module in model.modules()
        ),
    }


class _ChunkedForwardKL(torch.autograd.Function):
    """Exact first-order ``KL(p || q)`` without full FP32 probability tensors.

    The clean distribution is a detached VAT target.  Computing the forward
    value in row chunks preserves FP32 softmax/log-softmax arithmetic while
    bounding temporary memory.  The analytical logit gradient of the mean
    forward KL is ``(q - p) / N``; computing it in the same row chunks avoids
    saving any full-vocabulary FP32 intermediates for backward.

    VAT only requires first-order gradients through the perturbed logits.
    Returning no clean-logit gradient also enforces the clean-target
    stop-gradient contract at this lowest level.
    """

    @staticmethod
    def forward(
        ctx: Any,
        clean_logits: Tensor,
        perturbed_logits: Tensor,
        row_chunk_size: int,
    ) -> Tensor:
        if clean_logits.ndim != 2 or perturbed_logits.ndim != 2:
            raise ValueError("chunked KL inputs must have shape [N, V]")
        if clean_logits.shape != perturbed_logits.shape:
            raise ValueError("chunked KL inputs must have identical shapes")
        if clean_logits.shape[0] == 0:
            raise ValueError("chunked KL requires at least one selected row")
        if row_chunk_size <= 0:
            raise ValueError("row_chunk_size must be positive")

        clean_target = clean_logits.detach()
        ctx.save_for_backward(clean_target, perturbed_logits)
        ctx.row_chunk_size = int(row_chunk_size)

        total = torch.zeros(
            (), dtype=torch.float32, device=perturbed_logits.device
        )
        for start in range(0, clean_target.shape[0], ctx.row_chunk_size):
            stop = min(start + ctx.row_chunk_size, clean_target.shape[0])
            clean_log_probs = F.log_softmax(
                clean_target[start:stop].float(), dim=-1
            )
            perturbed_log_probs = F.log_softmax(
                perturbed_logits[start:stop].float(), dim=-1
            )
            total.add_(
                (
                    clean_log_probs.exp()
                    * (clean_log_probs - perturbed_log_probs)
                ).sum()
            )
        return total / clean_target.shape[0]

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[None, Tensor, None]:
        clean_target, perturbed_logits = ctx.saved_tensors
        row_count = clean_target.shape[0]
        grad_perturbed = torch.empty_like(perturbed_logits)
        scale = grad_output.float() / row_count

        for start in range(0, row_count, ctx.row_chunk_size):
            stop = min(start + ctx.row_chunk_size, row_count)
            clean_probs = F.softmax(clean_target[start:stop].float(), dim=-1)
            perturbed_probs = F.softmax(
                perturbed_logits[start:stop].float(), dim=-1
            )
            chunk_gradient = (perturbed_probs - clean_probs) * scale
            grad_perturbed[start:stop].copy_(
                chunk_gradient.to(dtype=perturbed_logits.dtype)
            )
        return None, grad_perturbed, None


class _ChunkedCrossEntropy(torch.autograd.Function):
    """Official FP32-logit mean CE with bounded vocabulary temporary memory."""

    @staticmethod
    def forward(
        ctx: Any,
        logits: Tensor,
        targets: Tensor,
        row_chunk_size: int,
    ) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("chunked cross entropy logits must have shape [N, V]")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError(
                "chunked cross entropy targets must have shape [N] matching logits"
            )
        if logits.shape[0] == 0:
            raise ValueError("chunked cross entropy requires at least one target")
        if targets.dtype != torch.long:
            raise TypeError("chunked cross entropy targets must have dtype torch.long")
        if row_chunk_size <= 0:
            raise ValueError("row_chunk_size must be positive")
        if bool(((targets < 0) | (targets >= logits.shape[1])).any()):
            raise ValueError("chunked cross entropy target is outside the vocabulary")

        ctx.save_for_backward(logits, targets)
        ctx.row_chunk_size = int(row_chunk_size)
        total = torch.zeros((), dtype=torch.float32, device=logits.device)
        for start in range(0, logits.shape[0], ctx.row_chunk_size):
            stop = min(start + ctx.row_chunk_size, logits.shape[0])
            total.add_(
                F.cross_entropy(
                    logits[start:stop].float(),
                    targets[start:stop],
                    reduction="sum",
                )
            )
        return total / logits.shape[0]

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[Tensor, None, None]:
        logits, targets = ctx.saved_tensors
        row_count = logits.shape[0]
        grad_logits = torch.empty_like(logits)
        scale = grad_output.float() / row_count

        for start in range(0, row_count, ctx.row_chunk_size):
            stop = min(start + ctx.row_chunk_size, row_count)
            probabilities = F.softmax(logits[start:stop].float(), dim=-1)
            local_targets = targets[start:stop]
            rows = torch.arange(
                stop - start, device=logits.device, dtype=torch.long
            )
            probabilities[rows, local_targets] -= 1.0
            probabilities.mul_(scale)
            grad_logits[start:stop].copy_(
                probabilities.to(dtype=logits.dtype)
            )
        return grad_logits, None, None


def memory_efficient_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean causal-target CE matching FP32 Llama loss without full FP32 logits."""

    return _ChunkedCrossEntropy.apply(logits, targets, _KL_ROW_CHUNK_SIZE)


def assistant_prediction_mask(labels: Tensor, ignore_index: int = -100) -> Tensor:
    """Return logit positions whose *next* token is an assistant target.

    Hugging Face causal language-model losses compare ``logits[:, :-1]`` with
    ``labels[:, 1:]``.  The returned mask has the original ``[B, T]`` shape so
    it can be applied directly to full logits; its last column is always false.
    """

    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [B, T], got {tuple(labels.shape)}")

    mask = torch.zeros_like(labels, dtype=torch.bool)
    if labels.shape[1] > 1:
        mask[:, :-1] = labels[:, 1:].ne(ignore_index)
    return mask


def select_masked_logits(logits: Tensor, prediction_mask: Tensor) -> Tensor:
    """Gather vocabulary logits at selected causal prediction positions."""

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {tuple(logits.shape)}")
    if prediction_mask.shape != logits.shape[:2]:
        raise ValueError(
            "prediction_mask must match logits' [B, T] dimensions, got "
            f"{tuple(prediction_mask.shape)} versus {tuple(logits.shape[:2])}"
        )
    return logits[prediction_mask.to(device=logits.device, dtype=torch.bool)]


def masked_kl_divergence(
    clean_logits: Tensor,
    perturbed_logits: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Mean ``KL(clean || perturbed)`` over selected positions, in FP32.

    ``clean_logits`` are always detached inside this function, making the clean
    distribution a fixed VAT target.  Inputs can either be full ``[B, T, V]``
    tensors with a ``[B, T]`` mask or already-gathered ``[N, V]`` tensors with
    ``mask=None``.

    If no position is selected, the result is a differentiable FP32 zero tied
    to ``perturbed_logits``.  Thus outer training may call ``backward`` without
    a special case, while the clean target still receives no gradient.
    """

    if clean_logits.shape != perturbed_logits.shape:
        raise ValueError(
            "clean_logits and perturbed_logits must have identical shapes, got "
            f"{tuple(clean_logits.shape)} and {tuple(perturbed_logits.shape)}"
        )
    if clean_logits.ndim < 2:
        raise ValueError(
            "logits must include at least position and vocabulary dimensions"
        )

    if mask is not None:
        if mask.shape != clean_logits.shape[:-1]:
            raise ValueError(
                "mask must match all non-vocabulary logits dimensions, got "
                f"{tuple(mask.shape)} versus {tuple(clean_logits.shape[:-1])}"
            )
        selected = mask.to(device=perturbed_logits.device, dtype=torch.bool)
        clean_selected = clean_logits.detach()[selected]
        perturbed_selected = perturbed_logits[selected]
    else:
        clean_selected = clean_logits.detach().reshape(-1, clean_logits.shape[-1])
        perturbed_selected = perturbed_logits.reshape(-1, perturbed_logits.shape[-1])

    if perturbed_selected.shape[0] == 0:
        # Summing an empty view gives a graph-connected zero and does not risk
        # the ``inf * 0 -> nan`` failure of the common ``tensor.sum() * 0`` idiom.
        return perturbed_selected.float().reshape(-1)[:0].sum()

    return _ChunkedForwardKL.apply(
        clean_selected,
        perturbed_selected,
        _KL_ROW_CHUNK_SIZE,
    )


def assistant_only_kl(
    clean_logits: Tensor,
    perturbed_logits: Tensor,
    *,
    labels: Tensor | None = None,
    prediction_mask: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Compute causal assistant-only ``KL(clean || perturbed)`` in FP32.

    Exactly one of ``labels`` and ``prediction_mask`` must be supplied.
    ``prediction_mask`` is aligned with logit positions, whereas ``labels`` is
    shifted internally according to causal-LM loss semantics.
    """

    if (labels is None) == (prediction_mask is None):
        raise ValueError("supply exactly one of labels or prediction_mask")
    if prediction_mask is None:
        prediction_mask = assistant_prediction_mask(labels, ignore_index)
    return masked_kl_divergence(clean_logits, perturbed_logits, prediction_mask)


def group_ids_from_position_ids(
    position_ids: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Infer packed-sequence IDs from position resets.

    A new group begins at the first active token and whenever the active
    position ID is less than or equal to the previous active position ID.
    Inactive positions receive ``-1``.  IDs are local to each batch row; the
    normalization helpers additionally key groups by batch row, so equal IDs in
    different rows never share a norm.
    """

    if position_ids.ndim != 2:
        raise ValueError(
            f"position_ids must have shape [B, T], got {tuple(position_ids.shape)}"
        )
    if attention_mask is None:
        active = torch.ones_like(position_ids, dtype=torch.bool)
    else:
        if attention_mask.shape != position_ids.shape:
            raise ValueError(
                "attention_mask and position_ids must have the same shape, got "
                f"{tuple(attention_mask.shape)} and {tuple(position_ids.shape)}"
            )
        active = attention_mask.to(device=position_ids.device, dtype=torch.bool)

    starts = torch.zeros_like(active)
    if position_ids.shape[1] > 0:
        starts[:, 0] = active[:, 0]
    if position_ids.shape[1] > 1:
        previous_inactive = ~active[:, :-1]
        reset = position_ids[:, 1:] <= position_ids[:, :-1]
        starts[:, 1:] = active[:, 1:] & (previous_inactive | reset)

    ids = starts.to(torch.long).cumsum(dim=1) - 1
    return ids.masked_fill(~active, -1)


def relative_vat_group_ids(position_ids: Tensor) -> Tensor:
    """Return the production grouping geometry for relative-radius VAT.

    The locked Magpie tokenizer aliases padding to EOS, so Axolotl may emit
    attention-mask zeroes for EOS tokens inside one conversation. Those
    zeroes are not packing boundaries. Relative VAT therefore groups only
    from position-ID resets after source/position alignment is validated.
    """

    return group_ids_from_position_ids(position_ids)


def validate_packed_source_position_alignment(
    *,
    perturb_mask: Tensor,
    position_ids: Tensor,
    source_ids: Tensor,
) -> None:
    """Fail closed when packed-conversation provenance and resets disagree.

    Axolotl's locked Llama sample-packing attention path derives FlashAttention
    ``cu_seqlens`` from positions equal to zero.  ``source_ids`` are provenance
    labels, not group IDs: the same source may legitimately occur in more than
    one packed segment.  This validator therefore keeps position resets as the
    grouping authority while proving that every source transition is
    accompanied by an exact zero reset and that a non-reset segment never
    mixes source IDs.
    """

    if (
        perturb_mask.ndim != 2
        or position_ids.shape != perturb_mask.shape
        or source_ids.shape != perturb_mask.shape
    ):
        raise ValueError(
            "perturb_mask, position_ids, and source_ids must have identical "
            f"[B, T] shapes, got {tuple(perturb_mask.shape)}, "
            f"{tuple(position_ids.shape)}, and {tuple(source_ids.shape)}"
        )
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if (
        position_ids.dtype not in integer_dtypes
        or source_ids.dtype not in integer_dtypes
    ):
        raise TypeError("position_ids and source_ids must have integer dtype")
    mask = perturb_mask.to(device=position_ids.device, dtype=torch.bool)
    positions = position_ids.to(dtype=torch.long)
    sources = source_ids.to(device=position_ids.device, dtype=torch.long)
    active = sources.ge(0)
    if bool((mask & ~active).any()):
        raise ValueError("perturb_mask selects padding or missing source provenance")

    if bool((sources[~active] != -1).any()):
        raise ValueError("inactive source_ids must use the locked padding value -1")
    if bool((positions[~active] != 0).any()):
        raise ValueError("right-padding position_ids must use the locked value 0")
    # The locked collator emits active tokens followed only by right padding.
    # This vectorized prefix check avoids one CUDA synchronization per token.
    after_padding = (~active).to(torch.int32).cumsum(dim=1).gt(0)
    if bool((active & after_padding).any()):
        raise ValueError("source_ids must be an active prefix plus right padding")
    has_active = active.any(dim=1)
    if bool((has_active & positions[:, 0].ne(0)).any()):
        raise ValueError("every packed row must begin with position_id 0")
    if positions.shape[1] > 1:
        adjacent_active = active[:, 1:] & active[:, :-1]
        current_position = positions[:, 1:]
        previous_position = positions[:, :-1]
        reset = current_position.eq(0)
        if bool(
            (adjacent_active & reset & previous_position.eq(0)).any()
        ):
            raise ValueError(
                "adjacent active position_id zeroes are ambiguous with "
                "Axolotl's packed cu_seqlens contract"
            )
        if bool(
            (
                adjacent_active
                & ~reset
                & current_position.ne(previous_position + 1)
            ).any()
        ):
            raise ValueError(
                "packed position_ids must increment by one or reset exactly "
                "to zero"
            )
        if bool(
            (
                adjacent_active
                & sources[:, 1:].ne(sources[:, :-1])
                & ~reset
            ).any()
        ):
            raise ValueError("source transition without a position_id zero reset")


def _resolve_group_ids(
    token_mask: Tensor,
    *,
    group_ids: Tensor | None,
    position_ids: Tensor | None,
    attention_mask: Tensor | None,
) -> Tensor:
    if token_mask.ndim != 2:
        raise ValueError(
            f"token_mask must have shape [B, T], got {tuple(token_mask.shape)}"
        )
    if group_ids is not None and position_ids is not None:
        raise ValueError("supply at most one of group_ids or position_ids")

    if group_ids is not None:
        if group_ids.shape != token_mask.shape:
            raise ValueError(
                f"group_ids must have shape {tuple(token_mask.shape)}, "
                f"got {tuple(group_ids.shape)}"
            )
        return group_ids.to(device=token_mask.device, dtype=torch.long)

    if position_ids is not None:
        if position_ids.shape != token_mask.shape:
            raise ValueError(
                f"position_ids must have shape {tuple(token_mask.shape)}, "
                f"got {tuple(position_ids.shape)}"
            )
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=token_mask.device)
        return group_ids_from_position_ids(
            position_ids.to(device=token_mask.device),
            attention_mask,
        )

    # With no packing metadata, each batch row is one independent sequence.
    return (
        torch.arange(token_mask.shape[0], device=token_mask.device, dtype=torch.long)
        .unsqueeze(1)
        .expand_as(token_mask)
    )


def masked_frobenius_norms(
    vectors: Tensor,
    token_mask: Tensor,
    *,
    group_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return FP32 Frobenius norms and their corresponding ``[batch, group]`` keys.

    Only masked tokens participate.  Groups containing no selected token are
    absent from the result.  This helper is useful for diagnostics and tests;
    normalization itself uses the same grouping implementation.
    """

    if vectors.ndim != 3:
        raise ValueError(
            f"vectors must have shape [B, T, H], got {tuple(vectors.shape)}"
        )
    if token_mask.shape != vectors.shape[:2]:
        raise ValueError(
            f"token_mask must have shape {tuple(vectors.shape[:2])}, "
            f"got {tuple(token_mask.shape)}"
        )

    token_mask = token_mask.to(device=vectors.device)
    resolved = _resolve_group_ids(
        token_mask,
        group_ids=group_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    selected = token_mask.to(device=vectors.device, dtype=torch.bool) & resolved.ge(0)
    if not selected.any():
        return (
            torch.empty(0, device=vectors.device, dtype=torch.float32),
            torch.empty((0, 2), device=vectors.device, dtype=torch.long),
        )

    batch_ids = (
        torch.arange(vectors.shape[0], device=vectors.device, dtype=torch.long)
        .unsqueeze(1)
        .expand_as(resolved)
    )
    keys = torch.stack((batch_ids[selected], resolved[selected]), dim=-1)
    unique_keys, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)

    # Select in the source dtype before FP32 promotion.  Promoting the full
    # [B,T,H] embedding tensor would add a large transient allocation at the
    # most memory-sensitive point of VAT construction.
    token_energy = vectors[selected].float().square().sum(dim=-1)
    energy = torch.zeros(
        unique_keys.shape[0], device=vectors.device, dtype=torch.float32
    )
    energy.scatter_add_(0, inverse, token_energy)
    return energy.sqrt(), unique_keys


def normalize_masked_per_group(
    vectors: Tensor,
    token_mask: Tensor,
    norm: float | Tensor,
    *,
    norm_keys: Tensor | None = None,
    denominator_delta: float = 0.0,
    token_divisor: Tensor | None = None,
    group_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Mask and Frobenius-normalize vectors independently per packed sequence.

    Non-selected tokens are exactly zero.  A selected group whose input norm is
    zero stays zero rather than producing NaN/Inf. ``denominator_delta`` is
    added to the FP32 group norm; its zero default preserves the legacy exact
    absolute-radius behavior.  When ``token_divisor`` is supplied, each
    selected token row is additionally divided by its positive scalar value
    without a final renormalization.  This is the exact Dense-VAT operation
    ``(epsilon / a_t) * g_t / (||g||_F + delta)``.
    """

    if isinstance(norm, Tensor):
        detached_norm = norm.detach()
        if not bool(torch.isfinite(detached_norm).all()) or bool(
            (detached_norm < 0).any()
        ):
            raise ValueError("norm targets must be finite and non-negative")
    elif not math.isfinite(float(norm)) or float(norm) < 0:
        raise ValueError(f"norm must be finite and non-negative, got {norm}")
    if not math.isfinite(float(denominator_delta)) or denominator_delta < 0:
        raise ValueError(
            "denominator_delta must be finite and non-negative, got "
            f"{denominator_delta}"
        )
    if vectors.ndim != 3:
        raise ValueError(
            f"vectors must have shape [B, T, H], got {tuple(vectors.shape)}"
        )
    if token_mask.shape != vectors.shape[:2]:
        raise ValueError(
            f"token_mask must have shape {tuple(vectors.shape[:2])}, "
            f"got {tuple(token_mask.shape)}"
        )
    if token_divisor is not None and token_divisor.shape != token_mask.shape:
        raise ValueError(
            "token_divisor must match token_mask shape, got "
            f"{tuple(token_divisor.shape)} and {tuple(token_mask.shape)}"
        )

    token_mask = token_mask.to(device=vectors.device)
    resolved = _resolve_group_ids(
        token_mask,
        group_ids=group_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    selected = token_mask.to(device=vectors.device, dtype=torch.bool) & resolved.ge(0)
    result_fp32 = torch.zeros_like(vectors, dtype=torch.float32)
    if not selected.any():
        return result_fp32.to(dtype=vectors.dtype)
    selected_divisor = None
    if token_divisor is not None:
        selected_divisor = token_divisor.to(
            device=vectors.device, dtype=torch.float32
        )[selected]
        if not bool(torch.isfinite(selected_divisor).all()) or bool(
            (selected_divisor <= 0).any()
        ):
            raise ValueError(
                "selected token_divisor values must be finite and positive"
            )

    batch_ids = (
        torch.arange(vectors.shape[0], device=vectors.device, dtype=torch.long)
        .unsqueeze(1)
        .expand_as(resolved)
    )
    keys = torch.stack((batch_ids[selected], resolved[selected]), dim=-1)
    unique_keys, inverse = torch.unique(
        keys, dim=0, sorted=True, return_inverse=True
    )

    selected_vectors = vectors[selected].float()
    token_energy = selected_vectors.square().sum(dim=-1)
    group_energy = torch.zeros(
        int(inverse.max().item()) + 1,
        device=vectors.device,
        dtype=torch.float32,
    )
    group_energy.scatter_add_(0, inverse, token_energy)
    group_norm = group_energy.sqrt()
    if isinstance(norm, Tensor):
        target_norm = norm.detach().to(device=vectors.device, dtype=torch.float32)
        if target_norm.ndim == 0:
            target_norm = target_norm.expand_as(group_norm)
        elif target_norm.ndim != 1 or target_norm.shape != group_norm.shape:
            raise ValueError(
                "tensor norm must be scalar or have one value per selected "
                f"group, got {tuple(target_norm.shape)} for "
                f"{group_norm.numel()} groups"
            )
        if norm_keys is not None:
            expected_keys = norm_keys.detach().to(
                device=vectors.device, dtype=torch.long
            )
            if (
                expected_keys.shape != unique_keys.shape
                or not torch.equal(expected_keys, unique_keys)
            ):
                raise ValueError(
                    "tensor norm keys do not match normalization group keys"
                )
    else:
        if norm_keys is not None:
            raise ValueError("norm_keys are valid only with tensor norm targets")
        target_norm = torch.full_like(group_norm, float(norm))
    scale = torch.where(
        group_norm > 0,
        target_norm / (group_norm + float(denominator_delta)),
        torch.zeros_like(group_norm),
    )
    selected_result = selected_vectors * scale[inverse].unsqueeze(-1)
    if selected_divisor is not None:
        selected_result = selected_result / selected_divisor.unsqueeze(-1)
    result_fp32[selected] = selected_result
    return result_fp32.to(dtype=vectors.dtype)


def vocabulary_radius_neighbor_counts(
    input_ids: Tensor,
    vocabulary_embeddings: Tensor,
    token_mask: Tensor,
    radii: Tensor,
    *,
    radius_keys: Tensor,
    group_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
    vocabulary_chunk_size: int = 2048,
) -> Tensor:
    """Count every other vocabulary ID within each selected token's radius.

    Distances are evaluated in FP32 against the current input-embedding
    matrix.  The center token ID itself is excluded; a different vocabulary
    ID with the same embedding is still an ``other`` token and is counted.
    ``radii`` contains one ``epsilon_i`` for every packed conversation key.
    Chunking changes only peak memory, not the set being counted.  Direct
    Euclidean distance is used instead of the cancellation-prone expanded
    squared-distance identity, so duplicate BF16-derived embeddings remain
    at exactly zero distance.
    """

    if input_ids.ndim != 2 or token_mask.shape != input_ids.shape:
        raise ValueError("input_ids and token_mask must have identical [B, T] shape")
    if vocabulary_embeddings.ndim != 2:
        raise ValueError("vocabulary_embeddings must have shape [V, H]")
    if vocabulary_embeddings.device != input_ids.device:
        raise ValueError(
            "input_ids and vocabulary_embeddings must be on the same device"
        )
    if vocabulary_chunk_size < 1:
        raise ValueError("vocabulary_chunk_size must be positive")
    if radii.ndim != 1 or radius_keys.ndim != 2 or radius_keys.shape[1] != 2:
        raise ValueError("radii and radius_keys must have shapes [G] and [G, 2]")
    if radii.shape[0] != radius_keys.shape[0]:
        raise ValueError("radii and radius_keys must contain the same groups")
    if not bool(torch.isfinite(radii).all()) or bool((radii < 0).any()):
        raise ValueError("radii must be finite and non-negative")

    token_mask = token_mask.to(device=input_ids.device, dtype=torch.bool)
    resolved = _resolve_group_ids(
        token_mask,
        group_ids=group_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    selected = token_mask & resolved.ge(0)
    result = torch.zeros_like(input_ids, dtype=torch.long)
    if not selected.any():
        return result

    batch_ids = (
        torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        .unsqueeze(1)
        .expand_as(resolved)
    )
    keys = torch.stack((batch_ids[selected], resolved[selected]), dim=-1)
    unique_keys, inverse = torch.unique(
        keys, dim=0, sorted=True, return_inverse=True
    )
    expected_keys = radius_keys.detach().to(device=input_ids.device, dtype=torch.long)
    if expected_keys.shape != unique_keys.shape or not torch.equal(
        expected_keys, unique_keys
    ):
        raise ValueError("radius_keys do not match selected packed conversation keys")

    selected_ids = input_ids[selected].to(dtype=torch.long)
    vocabulary_size, hidden_size = vocabulary_embeddings.shape
    if bool((selected_ids < 0).any()) or bool((selected_ids >= vocabulary_size).any()):
        raise ValueError("selected input token ID is outside the embedding vocabulary")

    device_type = vocabulary_embeddings.device.type
    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
        # Repeated occurrences of one token in the same packed conversation
        # have the same center and the same epsilon_i.  Count that exact
        # (group, token-ID) query once, then broadcast it back to occurrences.
        query_keys = torch.stack((inverse, selected_ids), dim=-1)
        unique_query_keys, query_inverse = torch.unique(
            query_keys, dim=0, sorted=True, return_inverse=True
        )
        query_group_indices = unique_query_keys[:, 0]
        query_token_ids = unique_query_keys[:, 1]
        queries = vocabulary_embeddings.index_select(0, query_token_ids).float()
        if queries.shape[1] != hidden_size:
            raise RuntimeError("selected query hidden size differs from vocabulary")
        query_radii = radii.detach().to(
            device=queries.device, dtype=torch.float32
        )[query_group_indices]
        counts = torch.zeros(
            query_token_ids.shape[0], device=queries.device, dtype=torch.long
        )
        for start in range(0, vocabulary_size, vocabulary_chunk_size):
            stop = min(start + vocabulary_chunk_size, vocabulary_size)
            candidates = vocabulary_embeddings[start:stop].float()
            candidate_ids = torch.arange(
                start, stop, device=queries.device, dtype=torch.long
            ).unsqueeze(0)
            for query_start in range(
                0, queries.shape[0], _DENSE_VAT_QUERY_CHUNK_SIZE
            ):
                query_stop = min(
                    query_start + _DENSE_VAT_QUERY_CHUNK_SIZE,
                    queries.shape[0],
                )
                local_queries = queries[query_start:query_stop]
                distances = torch.cdist(
                    local_queries,
                    candidates,
                    p=2.0,
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
                within = distances <= query_radii[
                    query_start:query_stop
                ].unsqueeze(-1)
                within &= candidate_ids.ne(
                    query_token_ids[query_start:query_stop].unsqueeze(-1)
                )
                counts[query_start:query_stop] += within.sum(dim=-1)
        result[selected] = counts[query_inverse].to(device=result.device)
    return result


def build_random_neighbor_embeddings(
    input_embeddings: Tensor,
    input_ids: Tensor,
    vocabulary_embeddings: Tensor,
    perturb_mask: Tensor,
    *,
    rho: float,
    group_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
    vocabulary_chunk_size: int = 2048,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, RandomNeighborBatchStats]:
    """Replace one uniformly chosen question token per packed conversation.

    For selected token ``t`` in packed conversation ``i``, the search radius is
    ``rho * ||E_Q,i||_F / (n_t + 1e-4)^(1/12)``, where ``n_t`` counts every
    *other* vocabulary ID inside the unscaled conversation radius. A token ID
    is sampled uniformly from the final radius, including ``t`` itself.

    Neighborhood membership is computed without gradients in FP32. The final
    embedding gather remains in the outer graph, so perturbed-branch gradients
    flow to the sampled vocabulary row, as in a true discrete replacement.
    """

    if input_embeddings.ndim != 3:
        raise ValueError("input_embeddings must have shape [B, T, H]")
    if input_ids.shape != input_embeddings.shape[:2]:
        raise ValueError("input_ids must match input_embeddings [B, T]")
    if perturb_mask.shape != input_ids.shape:
        raise ValueError("perturb_mask must match input_ids [B, T]")
    if vocabulary_embeddings.ndim != 2:
        raise ValueError("vocabulary_embeddings must have shape [V, H]")
    if vocabulary_embeddings.shape[1] != input_embeddings.shape[2]:
        raise ValueError("input and vocabulary embedding widths differ")
    if vocabulary_embeddings.device != input_embeddings.device:
        raise ValueError("input and vocabulary embeddings must share a device")
    if not math.isfinite(float(rho)) or rho < 0:
        raise ValueError("rho must be finite and non-negative")
    if vocabulary_chunk_size < 1:
        raise ValueError("vocabulary_chunk_size must be positive")

    perturb_mask = perturb_mask.to(
        device=input_embeddings.device, dtype=torch.bool
    )
    resolved = _resolve_group_ids(
        perturb_mask,
        group_ids=group_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    eligible = perturb_mask & resolved.ge(0)
    if not eligible.any():
        return input_embeddings, RandomNeighborBatchStats(0, 0, 0, 0)

    batch_ids = (
        torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        .unsqueeze(1)
        .expand_as(resolved)
    )
    eligible_keys = torch.stack(
        (batch_ids[eligible], resolved[eligible]), dim=-1
    )
    group_keys, occurrence_to_group = torch.unique(
        eligible_keys, dim=0, sorted=True, return_inverse=True
    )

    # Uniformly choose one eligible question-token occurrence per conversation.
    eligible_flat_indices = torch.nonzero(
        eligible.reshape(-1), as_tuple=False
    ).squeeze(1)
    chosen_flat_indices: list[Tensor] = []
    for group_index in range(group_keys.shape[0]):
        occurrences = torch.nonzero(
            occurrence_to_group.eq(group_index), as_tuple=False
        ).squeeze(1)
        offset = torch.randint(
            occurrences.numel(),
            (1,),
            device=input_ids.device,
            generator=generator,
        )
        chosen_flat_indices.append(eligible_flat_indices[occurrences[offset]])
    chosen_flat = torch.stack(chosen_flat_indices).reshape(-1)
    chosen_mask = torch.zeros_like(perturb_mask)
    chosen_mask.reshape(-1)[chosen_flat] = True

    reference_norms, reference_keys = masked_frobenius_norms(
        input_embeddings.detach(),
        perturb_mask,
        group_ids=resolved,
    )
    if not torch.equal(reference_keys, group_keys):
        raise RuntimeError("random-neighbor packed group keys are inconsistent")
    base_radii = reference_norms * float(rho)
    neighbor_counts = vocabulary_radius_neighbor_counts(
        input_ids,
        vocabulary_embeddings.detach(),
        chosen_mask,
        base_radii,
        radius_keys=group_keys,
        group_ids=resolved,
        vocabulary_chunk_size=vocabulary_chunk_size,
    )
    chosen_ids = input_ids.reshape(-1).index_select(0, chosen_flat).long()
    chosen_counts = neighbor_counts.reshape(-1).index_select(0, chosen_flat)
    final_radii = base_radii / (
        chosen_counts.float()
        .add(DENSE_VAT_COUNT_FLOOR)
        .pow(DENSE_VAT_COUNT_EXPONENT)
    )

    vocabulary_size = int(vocabulary_embeddings.shape[0])
    sampled_ids = chosen_ids.clone()
    seen = torch.zeros_like(chosen_ids)
    queries = vocabulary_embeddings.detach().index_select(0, chosen_ids).float()
    device_type = vocabulary_embeddings.device.type
    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
        for start in range(0, vocabulary_size, vocabulary_chunk_size):
            stop = min(start + vocabulary_chunk_size, vocabulary_size)
            candidates = vocabulary_embeddings.detach()[start:stop].float()
            candidate_ids = torch.arange(
                start, stop, device=input_ids.device, dtype=torch.long
            )
            for query_start in range(
                0, queries.shape[0], _DENSE_VAT_QUERY_CHUNK_SIZE
            ):
                query_stop = min(
                    query_start + _DENSE_VAT_QUERY_CHUNK_SIZE,
                    queries.shape[0],
                )
                distances = torch.cdist(
                    queries[query_start:query_stop],
                    candidates,
                    p=2.0,
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
                within = distances <= final_radii[
                    query_start:query_stop
                ].unsqueeze(-1)
                for local_query in range(within.shape[0]):
                    query_index = query_start + local_query
                    local_candidates = candidate_ids[within[local_query]]
                    local_count = int(local_candidates.numel())
                    if local_count == 0:
                        continue
                    previous_count = int(seen[query_index].item())
                    total_count = previous_count + local_count
                    if previous_count == 0 or bool(
                        torch.rand(
                            (), device=input_ids.device, generator=generator
                        ) < local_count / total_count
                    ):
                        sampled_offset = torch.randint(
                            local_count,
                            (1,),
                            device=input_ids.device,
                            generator=generator,
                        )
                        sampled_ids[query_index] = local_candidates[sampled_offset]
                    seen[query_index] = total_count

    if bool((seen < 1).any()):
        raise RuntimeError("random-neighbor radius unexpectedly excluded its center")
    sampled_embeddings = vocabulary_embeddings.index_select(0, sampled_ids)
    result = input_embeddings.clone()
    result.reshape(-1, result.shape[-1]).index_copy_(
        0, chosen_flat, sampled_embeddings
    )
    selected_count = int(chosen_ids.numel())
    stats = RandomNeighborBatchStats(
        conversation_count=int(group_keys.shape[0]),
        selected_token_count=selected_count,
        neighborhood_size_sum=int(seen.sum().item()),
        replacement_count=int(sampled_ids.ne(chosen_ids).sum().item()),
    )
    return result, stats


def masked_difference_frobenius_norms(
    left: Tensor,
    right: Tensor,
    token_mask: Tensor,
    *,
    group_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """FP32 group norms of ``left - right`` without a full FP32 delta tensor."""

    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("left and right must have identical [B, T, H] shapes")
    if token_mask.shape != left.shape[:2]:
        raise ValueError("token_mask must match left/right [B, T] dimensions")
    token_mask = token_mask.to(device=left.device)
    resolved = _resolve_group_ids(
        token_mask,
        group_ids=group_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    selected = token_mask.to(device=left.device, dtype=torch.bool) & resolved.ge(0)
    if not selected.any():
        return (
            torch.empty(0, device=left.device, dtype=torch.float32),
            torch.empty((0, 2), device=left.device, dtype=torch.long),
        )
    batch_ids = (
        torch.arange(left.shape[0], device=left.device, dtype=torch.long)
        .unsqueeze(1)
        .expand_as(resolved)
    )
    keys = torch.stack((batch_ids[selected], resolved[selected]), dim=-1)
    unique_keys, inverse = torch.unique(
        keys, dim=0, sorted=True, return_inverse=True
    )
    selected_delta = left[selected].float() - right[selected].float()
    token_energy = selected_delta.square().sum(dim=-1)
    energy = torch.zeros(
        unique_keys.shape[0], device=left.device, dtype=torch.float32
    )
    energy.scatter_add_(0, inverse, token_energy)
    return energy.sqrt(), unique_keys


def verify_rho_weighted_energy_identity(
    lhs: Tensor,
    rhs: Tensor,
    group_keys: Tensor,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    """Fail unless every group satisfies the protocol's rho energy identity.

    Tolerances are required caller inputs so the mathematical helper does not
    invent an admission threshold. ``lhs`` is the embedding-energy-weighted
    sum of squared token ``c_eff`` values and ``rhs`` is the squared effective
    group radius relative to ``rho * ||E_Q||_F``.
    """

    if (
        lhs.ndim != 1
        or rhs.shape != lhs.shape
        or group_keys.shape != (lhs.shape[0], 2)
    ):
        raise ValueError("rho identity values and [batch, group] keys differ")
    if (
        not math.isfinite(float(relative_tolerance))
        or relative_tolerance < 0
        or not math.isfinite(float(absolute_tolerance))
        or absolute_tolerance < 0
    ):
        raise ValueError("rho identity tolerances must be finite and non-negative")
    if not bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all()):
        raise FloatingPointError("rho identity contains NaN or Inf")
    error = (lhs - rhs).abs()
    permitted = float(absolute_tolerance) + float(relative_tolerance) * rhs.abs()
    failed = error > permitted
    if bool(failed.any()):
        details = [
            {
                "key": key,
                "lhs": float(left),
                "rhs": float(right),
                "absolute_error": float(local_error),
                "permitted_error": float(local_permitted),
            }
            for key, left, right, local_error, local_permitted in zip(
                group_keys[failed].cpu().tolist(),
                lhs[failed].cpu().tolist(),
                rhs[failed].cpu().tolist(),
                error[failed].cpu().tolist(),
                permitted[failed].cpu().tolist(),
                strict=True,
            )
        ]
        raise RuntimeError(
            "rho weighted-energy identity verification failed: "
            f"{details}"
        )


def token_locality_statistics(
    input_embeddings: Tensor,
    nominal_perturbation: Tensor,
    perturbed_embeddings: Tensor,
    token_mask: Tensor,
    *,
    group_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
    denominator_epsilon: float = 1.0e-12,
    rho: float | None = None,
    compute_component_energy_concentration: bool = False,
) -> dict[str, Tensor]:
    """Return audited token locality and bounded per-group summaries.

    All norms, ratios, and energies are computed in FP32. ``effective`` means
    the perturbation that survives the model-compute dtype:
    ``perturbed_embeddings.float() - input_embeddings.float()``. Token keys
    have columns ``[batch, group, token]`` and group keys have columns
    ``[batch, group]``.

    ``normalized_effective_token_count`` is
    ``1 / (n * sum_t p_t**2)`` for effective energy shares ``p_t``. It is one
    for uniform energy and ``1 / n`` when one token carries all energy.
    Concentration is mathematically undefined for a zero-effective-energy
    group: those values are NaN and the explicit ``concentration_defined`` and
    ``effective_group_nonzero`` flags are false.

    The expensive top-1%-of-hidden-components statistic is disabled by
    default. Standalone admission probes may request it explicitly with
    ``compute_component_energy_concentration=True``; training diagnostics
    should retain only the bounded per-group summaries.

    The function fails closed unless every masked token has a valid group,
    nominal and effective perturbations are exactly zero outside the mask, and
    the emitted token/group coverage identities are exact.
    """

    if (
        input_embeddings.ndim != 3
        or nominal_perturbation.shape != input_embeddings.shape
        or perturbed_embeddings.shape != input_embeddings.shape
    ):
        raise ValueError(
            "input_embeddings, nominal_perturbation, and "
            "perturbed_embeddings must have identical [B, T, H] shapes"
        )
    if not (
        input_embeddings.is_floating_point()
        and nominal_perturbation.is_floating_point()
        and perturbed_embeddings.is_floating_point()
    ):
        raise TypeError("locality embeddings and perturbations must be floating point")
    if not (
        input_embeddings.device == nominal_perturbation.device
        == perturbed_embeddings.device
    ):
        raise ValueError("locality tensors must be on the same device")
    if token_mask.shape != input_embeddings.shape[:2]:
        raise ValueError("token_mask must match embedding [B, T] dimensions")
    if token_mask.dtype != torch.bool:
        raise TypeError("token_mask must have dtype torch.bool")
    if (
        not math.isfinite(float(denominator_epsilon))
        or denominator_epsilon <= 0
    ):
        raise ValueError("denominator_epsilon must be finite and positive")
    if rho is not None and (not math.isfinite(float(rho)) or rho <= 0):
        raise ValueError("rho must be finite and positive when supplied")
    if not all(
        bool(torch.isfinite(value.detach()).all())
        for value in (
            input_embeddings,
            nominal_perturbation,
            perturbed_embeddings,
        )
    ):
        raise FloatingPointError("locality inputs contain NaN or Inf")

    mask = token_mask.to(device=input_embeddings.device)
    resolved = _resolve_group_ids(
        mask,
        group_ids=group_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    if bool((mask & resolved.lt(0)).any()):
        raise ValueError("every masked locality token must have group_id >= 0")
    outside = ~mask
    if bool(nominal_perturbation[outside].ne(0).any()):
        raise ValueError("nominal perturbation must be exactly zero outside token_mask")
    outside_effective = (
        perturbed_embeddings[outside].float()
        - input_embeddings[outside].float()
    )
    if bool(outside_effective.ne(0).any()):
        raise ValueError("effective perturbation must be exactly zero outside token_mask")

    selected = mask
    empty_float = torch.empty(
        0, device=input_embeddings.device, dtype=torch.float32
    )
    if not bool(selected.any()):
        return {
            "token_keys": torch.empty(
                (0, 3), device=input_embeddings.device, dtype=torch.long
            ),
            "embedding_token_norm": empty_float,
            "nominal_token_norm": empty_float,
            "effective_token_norm": empty_float,
            "nominal_token_ratio": empty_float,
            "effective_token_ratio": empty_float,
            "effective_to_nominal_token_norm": empty_float,
            "rho_normalized_effective_token_ratio": empty_float,
            "bf16_token_survival": empty_float,
            "token_energy_fraction": empty_float,
            "group_keys": torch.empty(
                (0, 2), device=input_embeddings.device, dtype=torch.long
            ),
            "group_token_count": torch.empty(
                0, device=input_embeddings.device, dtype=torch.long
            ),
            "question_embedding_group_norm": empty_float,
            "effective_perturbation_group_norm": empty_float,
            "max_token_energy_fraction": empty_float,
            "top_10pct_token_energy_fraction": empty_float,
            "normalized_effective_token_count": empty_float,
            "nonzero_effective_token_fraction": empty_float,
            "top_1pct_component_energy_fraction": empty_float,
            "effective_group_nonzero": torch.empty(
                0, device=input_embeddings.device, dtype=torch.bool
            ),
            "concentration_defined": torch.empty(
                0, device=input_embeddings.device, dtype=torch.bool
            ),
            "component_concentration_computed": torch.empty(
                0, device=input_embeddings.device, dtype=torch.bool
            ),
            "effective_token_ratio_p95": empty_float,
            "effective_token_ratio_p99": empty_float,
            "effective_token_ratio_max": empty_float,
            "rho_normalized_effective_token_ratio_p95": empty_float,
            "rho_normalized_effective_token_ratio_p99": empty_float,
            "rho_normalized_effective_token_ratio_max": empty_float,
            "bf16_token_survival_p05": empty_float,
            "bf16_token_survival_median": empty_float,
            "rho_weighted_energy_identity_lhs": empty_float,
            "rho_weighted_energy_identity_rhs": empty_float,
            "rho_weighted_energy_identity_abs_residual": empty_float,
            "bf16_survival_identity_max_abs_residual": empty_float,
            "effective_energy_fraction_sum_residual": empty_float,
            "token_coverage_residual": torch.empty(
                0, device=input_embeddings.device, dtype=torch.long
            ),
        }

    batch_ids = (
        torch.arange(
            input_embeddings.shape[0],
            device=input_embeddings.device,
            dtype=torch.long,
        )
        .unsqueeze(1)
        .expand_as(resolved)
    )
    token_positions = (
        torch.arange(
            input_embeddings.shape[1],
            device=input_embeddings.device,
            dtype=torch.long,
        )
        .unsqueeze(0)
        .expand_as(resolved)
    )
    group_keys_for_tokens = torch.stack(
        (batch_ids[selected], resolved[selected]), dim=-1
    )
    group_keys, inverse = torch.unique(
        group_keys_for_tokens, dim=0, sorted=True, return_inverse=True
    )
    token_keys = torch.cat(
        (group_keys_for_tokens, token_positions[selected].unsqueeze(-1)), dim=-1
    )

    embedding_norm = torch.linalg.vector_norm(
        input_embeddings[selected].float(), dim=-1
    )
    nominal_norm = torch.linalg.vector_norm(
        nominal_perturbation[selected].float(), dim=-1
    )
    effective_delta = (
        perturbed_embeddings[selected].float()
        - input_embeddings[selected].float()
    )
    effective_energy = effective_delta.square().sum(dim=-1)
    effective_norm = effective_energy.sqrt()
    if bool(embedding_norm.eq(0).any()):
        raise ValueError(
            "rho-normalized locality requires nonzero embedding norm for "
            "every masked token"
        )
    embedding_denominator = embedding_norm.clamp_min(
        float(denominator_epsilon)
    )
    nominal_denominator = nominal_norm.clamp_min(float(denominator_epsilon))
    effective_ratio = effective_norm / embedding_denominator
    bf16_survival = effective_norm / (
        nominal_norm + float(denominator_epsilon)
    )
    rho_normalized_effective_ratio = (
        effective_ratio / float(rho)
        if rho is not None
        else torch.full_like(effective_ratio, float("nan"))
    )

    group_count = group_keys.shape[0]
    group_energy = torch.zeros(
        group_count, device=input_embeddings.device, dtype=torch.float32
    )
    group_energy.scatter_add_(0, inverse, effective_energy)
    group_embedding_energy = torch.zeros_like(group_energy)
    group_embedding_energy.scatter_add_(0, inverse, embedding_norm.square())
    question_embedding_group_norm = group_embedding_energy.sqrt()
    effective_perturbation_group_norm = group_energy.sqrt()
    group_token_count = torch.zeros(
        group_count, device=input_embeddings.device, dtype=torch.long
    )
    group_token_count.scatter_add_(0, inverse, torch.ones_like(inverse))
    if int(group_token_count.sum().item()) != int(mask.sum().item()):
        raise RuntimeError("token locality coverage differs from token_mask")
    if token_keys.shape[0] != int(mask.sum().item()):
        raise RuntimeError("token locality token-key coverage differs from token_mask")
    if torch.unique(token_keys, dim=0).shape[0] != token_keys.shape[0]:
        raise RuntimeError("token locality token keys are not unique")
    energy_fraction = torch.where(
        group_energy[inverse] > 0,
        effective_energy / group_energy[inverse],
        torch.zeros_like(effective_energy),
    )

    maximum_shares: list[Tensor] = []
    top_ten_percent_shares: list[Tensor] = []
    normalized_effective_counts: list[Tensor] = []
    nonzero_effective_token_fractions: list[Tensor] = []
    top_one_percent_component_shares: list[Tensor] = []
    concentration_defined_values: list[Tensor] = []
    component_concentration_computed_values: list[Tensor] = []
    effective_ratio_p95_values: list[Tensor] = []
    effective_ratio_p99_values: list[Tensor] = []
    effective_ratio_max_values: list[Tensor] = []
    rho_normalized_p95_values: list[Tensor] = []
    rho_normalized_p99_values: list[Tensor] = []
    rho_normalized_max_values: list[Tensor] = []
    survival_p05_values: list[Tensor] = []
    survival_median_values: list[Tensor] = []
    rho_identity_lhs_values: list[Tensor] = []
    rho_identity_rhs_values: list[Tensor] = []
    rho_identity_residuals: list[Tensor] = []
    survival_identity_residuals: list[Tensor] = []
    energy_identity_residuals: list[Tensor] = []
    coverage_residuals: list[Tensor] = []

    def linear_quantile(values: Tensor, probability: float) -> Tensor:
        ordered = torch.sort(values).values
        position = (ordered.numel() - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    for group_index in range(group_count):
        group_selection = inverse.eq(group_index)
        shares = energy_fraction[group_selection]
        group_effective_norm = effective_norm[group_selection]
        group_effective_ratio = effective_ratio[group_selection]
        group_rho_normalized = rho_normalized_effective_ratio[group_selection]
        group_survival = bf16_survival[group_selection]
        token_count = int(shares.numel())
        if token_count == 0:
            raise RuntimeError("token locality group unexpectedly has no tokens")
        coverage_residuals.append(
            group_token_count[group_index] - token_count
        )
        nonzero_effective_token_fractions.append(
            group_effective_norm.ne(0).float().mean()
        )
        effective_ratio_p95_values.append(
            linear_quantile(group_effective_ratio, 0.95)
        )
        effective_ratio_p99_values.append(
            linear_quantile(group_effective_ratio, 0.99)
        )
        effective_ratio_max_values.append(group_effective_ratio.max())
        survival_p05_values.append(linear_quantile(group_survival, 0.05))
        survival_median_values.append(linear_quantile(group_survival, 0.50))
        survival_identity_residuals.append(
            (
                group_survival
                * (
                    nominal_norm[group_selection]
                    + float(denominator_epsilon)
                )
                - group_effective_norm
            )
            .abs()
            .max()
        )
        if rho is not None:
            rho_normalized_p95_values.append(
                linear_quantile(group_rho_normalized, 0.95)
            )
            rho_normalized_p99_values.append(
                linear_quantile(group_rho_normalized, 0.99)
            )
            rho_normalized_max_values.append(group_rho_normalized.max())
            embedding_energy_weights = (
                embedding_norm[group_selection].square()
                / group_embedding_energy[group_index]
            )
            identity_lhs = (
                embedding_energy_weights
                * group_rho_normalized.square()
            ).sum()
            identity_rhs = (
                effective_perturbation_group_norm[group_index]
                / (
                    float(rho)
                    * question_embedding_group_norm[group_index]
                )
            ).square()
            rho_identity_lhs_values.append(identity_lhs)
            rho_identity_rhs_values.append(identity_rhs)
            rho_identity_residuals.append(
                (identity_lhs - identity_rhs).abs()
            )
        else:
            undefined = group_energy[group_index].new_tensor(float("nan"))
            rho_normalized_p95_values.append(undefined)
            rho_normalized_p99_values.append(undefined)
            rho_normalized_max_values.append(undefined)
            rho_identity_lhs_values.append(undefined)
            rho_identity_rhs_values.append(undefined)
            rho_identity_residuals.append(undefined)

        defined = group_energy[group_index] > 0
        concentration_defined_values.append(defined)
        if bool(defined):
            top_count = max(1, math.ceil(token_count * 0.10))
            maximum_shares.append(shares.max())
            top_ten_percent_shares.append(
                torch.topk(shares, top_count, sorted=False).values.sum()
            )
            normalized_effective_counts.append(
                shares.square().sum().reciprocal() / token_count
            )
            energy_identity_residuals.append(
                (shares.sum() - 1.0).abs()
            )
            if compute_component_energy_concentration:
                component_energy = (
                    effective_delta[group_selection].square().reshape(-1)
                )
                top_component_count = max(
                    1, math.ceil(int(component_energy.numel()) * 0.01)
                )
                top_one_percent_component_shares.append(
                    torch.topk(
                        component_energy,
                        top_component_count,
                        sorted=False,
                    ).values.sum()
                    / group_energy[group_index]
                )
                component_concentration_computed_values.append(
                    torch.ones_like(defined)
                )
            else:
                top_one_percent_component_shares.append(
                    group_energy[group_index].new_tensor(float("nan"))
                )
                component_concentration_computed_values.append(
                    torch.zeros_like(defined)
                )
        else:
            undefined = group_energy[group_index].new_tensor(float("nan"))
            maximum_shares.append(undefined)
            top_ten_percent_shares.append(undefined)
            normalized_effective_counts.append(undefined)
            top_one_percent_component_shares.append(undefined)
            energy_identity_residuals.append(undefined)
            component_concentration_computed_values.append(
                torch.zeros_like(defined)
            )

    result = {
        "token_keys": token_keys,
        "embedding_token_norm": embedding_norm,
        "nominal_token_norm": nominal_norm,
        "effective_token_norm": effective_norm,
        "nominal_token_ratio": nominal_norm / embedding_denominator,
        "effective_token_ratio": effective_ratio,
        "effective_to_nominal_token_norm": effective_norm / nominal_denominator,
        "rho_normalized_effective_token_ratio": (
            rho_normalized_effective_ratio
        ),
        "bf16_token_survival": bf16_survival,
        "token_energy_fraction": energy_fraction,
        "group_keys": group_keys,
        "group_token_count": group_token_count,
        "question_embedding_group_norm": question_embedding_group_norm,
        "effective_perturbation_group_norm": (
            effective_perturbation_group_norm
        ),
        "max_token_energy_fraction": torch.stack(maximum_shares),
        "top_10pct_token_energy_fraction": torch.stack(
            top_ten_percent_shares
        ),
        "normalized_effective_token_count": torch.stack(
            normalized_effective_counts
        ),
        "nonzero_effective_token_fraction": torch.stack(
            nonzero_effective_token_fractions
        ),
        "top_1pct_component_energy_fraction": torch.stack(
            top_one_percent_component_shares
        ),
        "effective_group_nonzero": group_energy.gt(0),
        "concentration_defined": torch.stack(concentration_defined_values),
        "component_concentration_computed": torch.stack(
            component_concentration_computed_values
        ),
        "effective_token_ratio_p95": torch.stack(effective_ratio_p95_values),
        "effective_token_ratio_p99": torch.stack(effective_ratio_p99_values),
        "effective_token_ratio_max": torch.stack(effective_ratio_max_values),
        "rho_normalized_effective_token_ratio_p95": torch.stack(
            rho_normalized_p95_values
        ),
        "rho_normalized_effective_token_ratio_p99": torch.stack(
            rho_normalized_p99_values
        ),
        "rho_normalized_effective_token_ratio_max": torch.stack(
            rho_normalized_max_values
        ),
        "bf16_token_survival_p05": torch.stack(survival_p05_values),
        "bf16_token_survival_median": torch.stack(survival_median_values),
        "rho_weighted_energy_identity_lhs": torch.stack(
            rho_identity_lhs_values
        ),
        "rho_weighted_energy_identity_rhs": torch.stack(
            rho_identity_rhs_values
        ),
        "rho_weighted_energy_identity_abs_residual": torch.stack(
            rho_identity_residuals
        ),
        "bf16_survival_identity_max_abs_residual": torch.stack(
            survival_identity_residuals
        ),
        "effective_energy_fraction_sum_residual": torch.stack(
            energy_identity_residuals
        ),
        "token_coverage_residual": torch.stack(coverage_residuals),
    }
    always_finite = (
        "embedding_token_norm",
        "nominal_token_norm",
        "effective_token_norm",
        "nominal_token_ratio",
        "effective_token_ratio",
        "effective_to_nominal_token_norm",
        "bf16_token_survival",
        "token_energy_fraction",
        "nonzero_effective_token_fraction",
        "effective_token_ratio_p95",
        "effective_token_ratio_p99",
        "effective_token_ratio_max",
        "bf16_token_survival_p05",
        "bf16_token_survival_median",
        "bf16_survival_identity_max_abs_residual",
    )
    if not all(bool(torch.isfinite(result[name]).all()) for name in always_finite):
        raise FloatingPointError("token locality contains NaN or Inf")
    if rho is not None and not all(
        bool(torch.isfinite(result[name]).all())
        for name in (
            "rho_normalized_effective_token_ratio",
            "rho_normalized_effective_token_ratio_p95",
            "rho_normalized_effective_token_ratio_p99",
            "rho_normalized_effective_token_ratio_max",
            "rho_weighted_energy_identity_lhs",
            "rho_weighted_energy_identity_rhs",
            "rho_weighted_energy_identity_abs_residual",
        )
    ):
        raise FloatingPointError("rho-normalized token locality contains NaN or Inf")
    defined = result["concentration_defined"]
    for name in (
        "max_token_energy_fraction",
        "top_10pct_token_energy_fraction",
        "normalized_effective_token_count",
        "effective_energy_fraction_sum_residual",
    ):
        if not bool(torch.isfinite(result[name][defined]).all()):
            raise FloatingPointError(f"defined {name} contains NaN or Inf")
        if not bool(torch.isnan(result[name][~defined]).all()):
            raise RuntimeError(f"undefined {name} must be explicitly NaN")
    component_defined = result["component_concentration_computed"]
    component_values = result["top_1pct_component_energy_fraction"]
    if not bool(torch.isfinite(component_values[component_defined]).all()):
        raise FloatingPointError(
            "defined top_1pct_component_energy_fraction contains NaN or Inf"
        )
    if not bool(torch.isnan(component_values[~component_defined]).all()):
        raise RuntimeError(
            "uncomputed top_1pct_component_energy_fraction must be NaN"
        )
    if bool(result["token_coverage_residual"].ne(0).any()):
        raise RuntimeError("token locality coverage identity residual is nonzero")
    return result


def standalone_admission_token_locality_statistics(
    input_embeddings: Tensor,
    nominal_perturbation: Tensor,
    perturbed_embeddings: Tensor,
    token_mask: Tensor,
    *,
    rho: float,
    group_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
    denominator_epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """Production interface for standalone admission component evidence.

    Unlike the training-microbatch path, this explicit interface always
    computes top-1% hidden-component energy concentration and fails if any
    nonzero effective group lacks that evidence.
    """

    if input_embeddings.dtype != torch.bfloat16:
        raise TypeError(
            "standalone admission locality requires BF16 input embeddings, "
            f"got {input_embeddings.dtype}"
        )
    if perturbed_embeddings.dtype != torch.bfloat16:
        raise TypeError(
            "standalone admission locality requires BF16 perturbed model "
            f"inputs, got {perturbed_embeddings.dtype}"
        )

    result: dict[str, Any] = token_locality_statistics(
        input_embeddings,
        nominal_perturbation,
        perturbed_embeddings,
        token_mask,
        group_ids=group_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        denominator_epsilon=denominator_epsilon,
        rho=rho,
        compute_component_energy_concentration=True,
    )
    expected = result["effective_group_nonzero"]
    if not torch.equal(result["component_concentration_computed"], expected):
        raise RuntimeError(
            "standalone admission component concentration coverage differs "
            "from nonzero effective groups"
        )
    verify_rho_weighted_energy_identity(
        result["rho_weighted_energy_identity_lhs"],
        result["rho_weighted_energy_identity_rhs"],
        result["group_keys"],
        relative_tolerance=RHO_WEIGHTED_ENERGY_IDENTITY_RTOL,
        absolute_tolerance=RHO_WEIGHTED_ENERGY_IDENTITY_ATOL,
    )
    result["runtime_rho"] = float(rho)
    result["rho_identity_tolerances"] = {
        "relative_tolerance": RHO_WEIGHTED_ENERGY_IDENTITY_RTOL,
        "absolute_tolerance": RHO_WEIGHTED_ENERGY_IDENTITY_ATOL,
    }
    independent_user_norms, independent_user_keys = masked_frobenius_norms(
        input_embeddings,
        token_mask,
        group_ids=group_ids,
        position_ids=position_ids if group_ids is None else None,
        attention_mask=attention_mask,
    )
    independent_effective_norms, independent_effective_keys = (
        masked_difference_frobenius_norms(
            perturbed_embeddings,
            input_embeddings,
            token_mask,
            group_ids=group_ids,
            position_ids=position_ids if group_ids is None else None,
            attention_mask=attention_mask,
        )
    )
    if not (
        torch.equal(result["group_keys"], independent_user_keys)
        and torch.equal(result["group_keys"], independent_effective_keys)
    ):
        raise RuntimeError(
            "standalone independent locality norm group keys differ"
        )
    result["group_records"] = locality_group_records(
        result,
        independent_user_embedding_norm=independent_user_norms,
        independent_effective_perturbation_norm=independent_effective_norms,
    )
    return result


def locality_group_records(
    statistics: Mapping[str, Any],
    *,
    microbatch_index: int | None = None,
    selection: Tensor | None = None,
    independent_user_embedding_norm: Tensor | None = None,
    independent_effective_perturbation_norm: Tensor | None = None,
) -> list[dict[str, Any]]:
    """Convert independent group locality quantities to canonical records.

    Each returned record owns its key and every quantity needed by an
    independent gate to recompute the rho-identity RHS.  This avoids treating
    the shared ordering of several parallel arrays as evidence.  The output is
    bounded by the number of packed groups and never contains token or hidden
    tensors.
    """

    required = (
        "group_keys",
        "group_token_count",
        "question_embedding_group_norm",
        "effective_perturbation_group_norm",
        "effective_group_nonzero",
        "concentration_defined",
        "component_concentration_computed",
        "max_token_energy_fraction",
        "top_10pct_token_energy_fraction",
        "normalized_effective_token_count",
        "top_1pct_component_energy_fraction",
        "rho_weighted_energy_identity_lhs",
        "rho_weighted_energy_identity_rhs",
        "rho_weighted_energy_identity_abs_residual",
        "effective_energy_fraction_sum_residual",
        "token_coverage_residual",
    )
    missing = [name for name in required if name not in statistics]
    if missing:
        raise KeyError(f"locality statistics lack group fields: {missing}")
    group_keys = statistics["group_keys"]
    if not isinstance(group_keys, Tensor) or group_keys.ndim != 2:
        raise TypeError("group_keys must be a rank-2 tensor")
    if group_keys.shape[1] != 2:
        raise ValueError("group_keys must have [batch, group] columns")
    count = int(group_keys.shape[0])
    if selection is None:
        selected = torch.ones(
            count, dtype=torch.bool, device=group_keys.device
        )
    else:
        if selection.shape != (count,) or selection.dtype != torch.bool:
            raise ValueError("group record selection must be a boolean [groups] tensor")
        selected = selection.to(device=group_keys.device)
    selected_keys = group_keys[selected]
    if torch.unique(selected_keys, dim=0).shape[0] != selected_keys.shape[0]:
        raise RuntimeError("canonical locality group keys are not unique")
    if microbatch_index is not None and (
        isinstance(microbatch_index, bool) or int(microbatch_index) < 0
    ):
        raise ValueError("microbatch_index must be a non-negative integer")
    for name, values in (
        ("independent_user_embedding_norm", independent_user_embedding_norm),
        (
            "independent_effective_perturbation_norm",
            independent_effective_perturbation_norm,
        ),
    ):
        if values is None:
            continue
        if not isinstance(values, Tensor) or values.shape != (count,):
            raise ValueError(f"{name} must have exactly {count} values")
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(f"{name} contains NaN or Inf")

    selected_indices = torch.nonzero(selected, as_tuple=False).reshape(-1).tolist()

    def scalar(name: str, index: int) -> Any:
        values = statistics[name]
        if not isinstance(values, Tensor) or values.shape != (count,):
            raise ValueError(
                f"locality group field {name} must have exactly {count} values"
            )
        value = values[index].detach().cpu().item()
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    records: list[dict[str, Any]] = []
    for index in selected_indices:
        batch_index, group_index = (
            int(value) for value in group_keys[index].detach().cpu().tolist()
        )
        key = {
            "batch_index": batch_index,
            "group_index": group_index,
        }
        if microbatch_index is not None:
            key = {
                "microbatch_index": int(microbatch_index),
                **key,
            }
        records.append(
            {
                "key": key,
                "group_token_count": scalar("group_token_count", index),
                "user_embedding_norm": (
                    float(
                        independent_user_embedding_norm[index]
                        .detach()
                        .cpu()
                        .item()
                    )
                    if independent_user_embedding_norm is not None
                    else scalar("question_embedding_group_norm", index)
                ),
                "effective_perturbation_norm": (
                    float(
                        independent_effective_perturbation_norm[index]
                        .detach()
                        .cpu()
                        .item()
                    )
                    if independent_effective_perturbation_norm is not None
                    else scalar("effective_perturbation_group_norm", index)
                ),
                "question_embedding_group_norm": scalar(
                    "question_embedding_group_norm", index
                ),
                "effective_perturbation_group_norm": scalar(
                    "effective_perturbation_group_norm", index
                ),
                "effective_group_nonzero": scalar(
                    "effective_group_nonzero", index
                ),
                "concentration_defined": scalar(
                    "concentration_defined", index
                ),
                "component_concentration_computed": scalar(
                    "component_concentration_computed", index
                ),
                "max_token_energy_fraction": scalar(
                    "max_token_energy_fraction", index
                ),
                "top_10pct_token_energy_fraction": scalar(
                    "top_10pct_token_energy_fraction", index
                ),
                "normalized_effective_token_count": scalar(
                    "normalized_effective_token_count", index
                ),
                "top_1pct_component_energy_fraction": scalar(
                    "top_1pct_component_energy_fraction", index
                ),
                "rho_weighted_energy_identity_lhs": scalar(
                    "rho_weighted_energy_identity_lhs", index
                ),
                "rho_weighted_energy_identity_rhs": scalar(
                    "rho_weighted_energy_identity_rhs", index
                ),
                "rho_weighted_energy_identity_abs_residual": scalar(
                    "rho_weighted_energy_identity_abs_residual", index
                ),
                "effective_energy_fraction_sum_residual": scalar(
                    "effective_energy_fraction_sum_residual", index
                ),
                "token_coverage_residual": scalar(
                    "token_coverage_residual", index
                ),
            }
        )
    records.sort(key=lambda record: tuple(record["key"].values()))
    return records


def initialize_masked_gaussian(
    reference_embeddings: Tensor,
    perturb_mask: Tensor,
    xi: float,
    *,
    group_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample masked Gaussian noise with per-packed-sequence norm ``xi``."""

    if not reference_embeddings.is_floating_point():
        raise TypeError("reference_embeddings must be floating point")
    noise = torch.randn(
        reference_embeddings.shape,
        dtype=reference_embeddings.dtype,
        device=reference_embeddings.device,
        generator=generator,
    )
    return normalize_masked_per_group(
        noise,
        perturb_mask,
        xi,
        group_ids=group_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )


def initialize_relative_masked_gaussian(
    reference_embeddings: Tensor,
    perturb_mask: Tensor,
    tau: float,
    *,
    group_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample a per-group relative finite-difference probe.

    For each original packed conversation ``i`` the nominal probe norm is
    ``tau * ||E_Q,i||_F``.  ``tau`` is dimensionless and the embedding norm is
    detached, so probe construction remains outside the outer optimization
    graph.
    """

    if not math.isfinite(float(tau)) or tau < 0:
        raise ValueError(f"tau must be finite and non-negative, got {tau}")
    reference_norms, reference_keys = masked_frobenius_norms(
        reference_embeddings.detach(),
        perturb_mask,
        group_ids=group_ids,
        position_ids=position_ids if group_ids is None else None,
        attention_mask=attention_mask,
    )
    noise = torch.randn(
        reference_embeddings.shape,
        dtype=reference_embeddings.dtype,
        device=reference_embeddings.device,
        generator=generator,
    )
    return normalize_masked_per_group(
        noise,
        perturb_mask,
        reference_norms * float(tau),
        norm_keys=reference_keys,
        group_ids=group_ids,
        position_ids=position_ids if group_ids is None else None,
        attention_mask=attention_mask,
    )


def _extract_logits(outputs: Any) -> Tensor:
    if isinstance(outputs, Tensor):
        return outputs
    if isinstance(outputs, Mapping):
        logits = outputs.get("logits")
    else:
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, (tuple, list)) and outputs:
            logits = outputs[0]
    if not isinstance(logits, Tensor):
        raise TypeError(
            "model output must be logits or expose a tensor-valued `.logits`"
        )
    return logits


def _assistant_logits(logits: Tensor, prediction_mask: Tensor) -> Tensor:
    """Accept either full logits or logits already gathered at assistant positions."""

    count = int(prediction_mask.sum().item())
    if logits.ndim == 3:
        return select_masked_logits(logits, prediction_mask)
    if logits.ndim == 2 and logits.shape[0] == count:
        return logits
    raise ValueError(
        "logits must be full [B, T, V] or gathered [N_assistant, V], got "
        f"{tuple(logits.shape)} with N_assistant={count}"
    )


def build_supervised_adversarial_perturbation(
    sft_loss: Tensor,
    input_embeddings: Tensor,
    perturb_mask: Tensor,
    *,
    rho: float,
    group_ids: Tensor | None = None,
    position_ids: Tensor | None = None,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Build the detached first-order inner maximizer for adversarial SFT.

    For every original packed conversation ``i`` this implements exactly::

        epsilon_i = rho * ||E_Q,i||_F
        r_i = epsilon_i * M_i grad_E L_sft / (||M_i grad_E L_sft||_F + 1e-12)

    ``torch.autograd.grad`` requests only the input-embedding activation
    gradient, so the inner maximization neither populates parameter ``.grad``
    buffers nor constructs a second-order graph.  The returned perturbation is
    zero outside the question mask and detached from both the clean loss and
    the model parameters.
    """

    if not isinstance(sft_loss, Tensor) or sft_loss.ndim != 0:
        raise ValueError("sft_loss must be one scalar tensor")
    if not bool(torch.isfinite(sft_loss.detach())):
        raise FloatingPointError("sft_loss is non-finite")
    if not math.isfinite(float(rho)) or rho < 0:
        raise ValueError(f"rho must be finite and non-negative, got {rho}")
    if input_embeddings.ndim != 3:
        raise ValueError(
            "input_embeddings must have shape [B, T, H], got "
            f"{tuple(input_embeddings.shape)}"
        )
    if not input_embeddings.requires_grad:
        raise ValueError("input_embeddings must require gradients")
    if perturb_mask.shape != input_embeddings.shape[:2]:
        raise ValueError(
            f"perturb_mask must have shape {tuple(input_embeddings.shape[:2])}, "
            f"got {tuple(perturb_mask.shape)}"
        )

    mask = perturb_mask.to(device=input_embeddings.device, dtype=torch.bool)
    if not mask.any() or rho == 0:
        return torch.zeros_like(input_embeddings).detach()

    reference_norms, reference_keys = masked_frobenius_norms(
        input_embeddings.detach(),
        mask,
        group_ids=group_ids,
        position_ids=position_ids if group_ids is None else None,
        attention_mask=attention_mask,
    )
    gradient = torch.autograd.grad(
        sft_loss,
        input_embeddings,
        retain_graph=False,
        create_graph=False,
        only_inputs=True,
    )[0]
    perturbation = normalize_masked_per_group(
        gradient,
        mask,
        reference_norms * float(rho),
        norm_keys=reference_keys,
        denominator_delta=RELATIVE_GRADIENT_NORM_DELTA,
        group_ids=group_ids,
        position_ids=position_ids if group_ids is None else None,
        attention_mask=attention_mask,
    )
    return perturbation.detach()


def build_vadv(
    model: nn.Module,
    input_embeddings: Tensor,
    clean_logits: Tensor,
    perturb_mask: Tensor,
    *,
    xi: float,
    xi_mode: str = "absolute",
    epsilon: float | None = None,
    rho: float | None = None,
    labels: Tensor | None = None,
    prediction_mask: Tensor | None = None,
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
    group_ids: Tensor | None = None,
    input_ids: Tensor | None = None,
    vocabulary_embeddings: Tensor | None = None,
    density_mode: str = "none",
    density_vocabulary_chunk_size: int = 2048,
    semantic_margins: Tensor | None = None,
    ignore_index: int = -100,
    generator: torch.Generator | None = None,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> Tensor:
    """Construct detached ``r_vadv`` with one VAT power iteration.

    The probe uses ``input_embeddings.detach() + d`` and
    :func:`torch.autograd.grad` requests a gradient only for ``d``.  Therefore
    probe gradients are not accumulated into model parameters.  Existing
    parameter ``.grad`` values, if any, are preserved.

    ``clean_logits`` may be full ``[B, T, V]`` logits or already-gathered
    assistant logits ``[N_assistant, V]``.  The returned tensor always has the
    embedding shape, is zero outside ``perturb_mask``, and is detached.
    """

    if not math.isfinite(float(xi)) or xi < 0:
        raise ValueError(f"xi must be finite and non-negative, got {xi}")
    if xi_mode not in {"absolute", "relative_question_frobenius"}:
        raise ValueError(
            "xi_mode must be 'absolute' or "
            f"'relative_question_frobenius', got {xi_mode!r}"
        )
    if density_mode not in {"none", "vocab_radius_count"}:
        raise ValueError(
            "density_mode must be 'none' or 'vocab_radius_count', got "
            f"{density_mode!r}"
        )
    if density_mode == "vocab_radius_count" and rho is None:
        raise ValueError("Dense-VAT requires the relative rho radius")
    if semantic_margins is not None and density_mode != "vocab_radius_count":
        raise ValueError("semantic margins require Dense-VAT density scaling")
    if density_vocabulary_chunk_size < 1:
        raise ValueError("density_vocabulary_chunk_size must be positive")
    if (epsilon is None) == (rho is None):
        raise ValueError("supply exactly one of epsilon or rho")
    if epsilon is not None and (
        not math.isfinite(float(epsilon)) or epsilon < 0
    ):
        raise ValueError(f"epsilon must be finite and non-negative, got {epsilon}")
    if rho is not None and (not math.isfinite(float(rho)) or rho < 0):
        raise ValueError(f"rho must be finite and non-negative, got {rho}")
    if input_embeddings.ndim != 3:
        raise ValueError(
            "input_embeddings must have shape [B, T, H], got "
            f"{tuple(input_embeddings.shape)}"
        )
    if perturb_mask.shape != input_embeddings.shape[:2]:
        raise ValueError(
            f"perturb_mask must have shape {tuple(input_embeddings.shape[:2])}, "
            f"got {tuple(perturb_mask.shape)}"
        )
    if (labels is None) == (prediction_mask is None):
        raise ValueError("supply exactly one of labels or prediction_mask")
    if prediction_mask is None:
        prediction_mask = assistant_prediction_mask(labels, ignore_index)
    if prediction_mask.shape != input_embeddings.shape[:2]:
        raise ValueError(
            f"prediction_mask must have shape {tuple(input_embeddings.shape[:2])}, "
            f"got {tuple(prediction_mask.shape)}"
        )
    prediction_mask = prediction_mask.to(
        device=input_embeddings.device, dtype=torch.bool
    )
    perturb_mask = perturb_mask.to(device=input_embeddings.device, dtype=torch.bool)

    # A probe cannot define an adversarial direction without both a perturbed
    # input token and an assistant distribution to compare.  Avoiding the
    # forward also guarantees no incidental model state/gradient changes.
    radius_is_zero = (
        epsilon == 0 if epsilon is not None else rho == 0
    )
    if (
        not perturb_mask.any()
        or not prediction_mask.any()
        or xi == 0
        or radius_is_zero
    ):
        return torch.zeros_like(input_embeddings).detach()

    initializer = (
        initialize_relative_masked_gaussian
        if xi_mode == "relative_question_frobenius"
        else initialize_masked_gaussian
    )
    d = initializer(
        input_embeddings,
        perturb_mask,
        xi,
        group_ids=group_ids,
        position_ids=position_ids if group_ids is None else None,
        attention_mask=attention_mask,
        generator=generator,
    ).detach()
    d.requires_grad_(True)

    kwargs = dict(forward_kwargs or {})
    if "inputs_embeds" in kwargs:
        raise ValueError("forward_kwargs must not contain inputs_embeds")
    kwargs["inputs_embeds"] = input_embeddings.detach() + d
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    kwargs["use_cache"] = False

    probe_outputs = model(**kwargs)
    probe_logits = _assistant_logits(_extract_logits(probe_outputs), prediction_mask)
    clean_target = _assistant_logits(clean_logits, prediction_mask).detach()
    probe_kl = masked_kl_divergence(clean_target, probe_logits)
    gradient = torch.autograd.grad(
        probe_kl,
        d,
        retain_graph=False,
        create_graph=False,
        only_inputs=True,
    )[0]
    del probe_outputs, probe_logits, probe_kl

    if rho is None:
        target_norm: float | Tensor = float(epsilon)
    else:
        # The relative radius is defined per original packed conversation:
        # epsilon_i = rho * ||E_Q,i||_F.  Detaching the embedding norm keeps
        # perturbation construction outside the outer optimization graph.
        reference_norms, reference_keys = masked_frobenius_norms(
            input_embeddings.detach(),
            perturb_mask,
            group_ids=group_ids,
            position_ids=position_ids if group_ids is None else None,
            attention_mask=attention_mask,
        )
        target_norm = reference_norms * float(rho)
    if rho is None:
        reference_keys = None

    token_divisor = None
    if density_mode == "vocab_radius_count":
        if input_ids is None or vocabulary_embeddings is None:
            raise ValueError(
                "Dense-VAT requires input_ids and vocabulary_embeddings"
            )
        neighbor_counts = vocabulary_radius_neighbor_counts(
            input_ids,
            vocabulary_embeddings,
            perturb_mask,
            target_norm,
            radius_keys=reference_keys,
            group_ids=group_ids,
            position_ids=position_ids if group_ids is None else None,
            attention_mask=attention_mask,
            vocabulary_chunk_size=density_vocabulary_chunk_size,
        )
        # Fixed Dense-VAT formula:
        # a_t = (n_t + 1e-4)^(1/12).  There is deliberately no final
        # perturbation renormalization after dividing token rows by a_t.
        token_divisor = (
            neighbor_counts.to(device=gradient.device, dtype=torch.float32)
            .add(DENSE_VAT_COUNT_FLOOR)
            .pow(DENSE_VAT_COUNT_EXPONENT)
        )

    r_vadv = normalize_masked_per_group(
        gradient,
        perturb_mask,
        target_norm,
        norm_keys=reference_keys,
        denominator_delta=(
            RELATIVE_GRADIENT_NORM_DELTA if rho is not None else 0.0
        ),
        token_divisor=token_divisor,
        group_ids=group_ids,
        position_ids=position_ids if group_ids is None else None,
        attention_mask=attention_mask,
    )
    if semantic_margins is not None:
        if input_ids is None:
            raise ValueError("semantic margins require input_ids")
        # The semantic margin is deliberately the final operation.  Dense-VAT
        # computes r_tilde unchanged above; each selected token row is then
        # capped independently, with no packed-group renormalization after it.
        r_vadv = apply_token_semantic_cap(
            r_vadv,
            input_ids,
            perturb_mask,
            semantic_margins,
        )
    r_vadv = r_vadv.detach()
    return r_vadv


__all__ = [
    "DENSE_VAT_COUNT_EXPONENT",
    "DENSE_VAT_COUNT_FLOOR",
    "RELATIVE_GRADIENT_NORM_DELTA",
    "RHO_WEIGHTED_ENERGY_IDENTITY_ATOL",
    "RHO_WEIGHTED_ENERGY_IDENTITY_RTOL",
    "assert_zero_stochastic_dropout",
    "assistant_only_kl",
    "assistant_prediction_mask",
    "build_random_neighbor_embeddings",
    "build_supervised_adversarial_perturbation",
    "build_vadv",
    "group_ids_from_position_ids",
    "initialize_masked_gaussian",
    "initialize_relative_masked_gaussian",
    "locality_group_records",
    "masked_frobenius_norms",
    "masked_difference_frobenius_norms",
    "masked_kl_divergence",
    "memory_efficient_cross_entropy",
    "normalize_masked_per_group",
    "RandomNeighborBatchStats",
    "vocabulary_radius_neighbor_counts",
    "relative_vat_group_ids",
    "select_masked_logits",
    "standalone_admission_token_locality_statistics",
    "token_locality_statistics",
    "validate_packed_source_position_alignment",
    "verify_rho_weighted_energy_identity",
]
