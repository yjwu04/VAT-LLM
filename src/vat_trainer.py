"""Axolotl trainer extension for question-only virtual adversarial SFT.

The module is intentionally a mixin/factory instead of a second training
entrypoint.  Axolotl still owns the optimizer, scheduler, data loader,
distributed wrapping, checkpointing, callbacks, and the training loop.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Mapping, MutableMapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .artifact_fingerprint import checkpoint_fingerprint
from .semantic_margin import (
    SemanticMarginArtifact,
    load_semantic_margin_artifact,
)
from .user_token_mask import native_llama3_packed_user_mask
from .vat_loss import (
    RELATIVE_GRADIENT_NORM_DELTA,
    RHO_WEIGHTED_ENERGY_IDENTITY_ATOL,
    RHO_WEIGHTED_ENERGY_IDENTITY_RTOL,
    RandomNeighborBatchStats,
    assert_zero_stochastic_dropout,
    assistant_prediction_mask,
    build_random_neighbor_embeddings,
    build_supervised_adversarial_perturbation,
    build_vadv,
    locality_group_records,
    masked_difference_frobenius_norms,
    masked_frobenius_norms,
    masked_kl_divergence,
    memory_efficient_cross_entropy,
    relative_vat_group_ids,
    token_locality_statistics,
    validate_packed_source_position_alignment,
)

PILOT_SCHEDULER_TOTAL_STEPS_ENV = "VAT_PILOT_SCHEDULER_TOTAL_STEPS"
LOCKED_PILOT_SCHEDULER_TOTAL_STEPS = 544
LAUNCH_EXECUTION_ID_ENV = "VAT_LAUNCH_EXECUTION_ID"
RUN_CONFIG_SHA256_ENV = "VAT_RUN_CONFIG_SHA256"
PHASE2_ARM_ENV = "VAT_PHASE2_ARM"
PHASE2_REQUESTED_ENDPOINT_ENV = "VAT_PHASE2_REQUESTED_STAGE_ENDPOINT"
GRADIENT_SIMILARITY_SCOPE = (
    "globally summed FP32 accumulated probe gradients of "
    "model.model.layers[-1]"
)
GRADIENT_SIMILARITY_UPDATE_SCOPE = (
    "full model via same-step recomputation of SFT + lambda_t * VAT"
)
GRADIENT_SIMILARITY_AGGREGATION = (
    "manual SUM of per-rank last-block FP32 probes after complete "
    "microbatch accumulation; common scaling cancels in cosine and norm ratio"
)
GRADIENT_SIMILARITY_SCHEDULES = frozenset(
    {
        "gradient_similarity",
        "linear_warmup_gradient_similarity",
        "cosine_warmup_gradient_similarity",
        "delayed_cosine_gradient_similarity",
    }
)


def _gradient_similarity_base_lambda(
    *,
    schedule: str,
    maximum: float,
    progress: float,
    delay_fraction: float | None = None,
    warmup_fraction: float | None = None,
) -> float:
    """Return the ungated coefficient used by a similarity-gated schedule."""

    if schedule == "gradient_similarity":
        return maximum
    if schedule == "linear_warmup_gradient_similarity":
        if warmup_fraction is None:
            raise RuntimeError(
                "linear warm-up gradient-similarity schedule lacks its phase"
            )
        return maximum * min(1.0, max(0.0, progress / warmup_fraction))
    if schedule == "cosine_warmup_gradient_similarity":
        if warmup_fraction is None:
            raise RuntimeError(
                "cosine warm-up gradient-similarity schedule lacks its phase"
            )
        ramp = min(1.0, max(0.0, progress / warmup_fraction))
        return maximum * 0.5 * (1.0 - math.cos(math.pi * ramp))
    if schedule != "delayed_cosine_gradient_similarity":
        raise RuntimeError(f"unsupported gradient-similarity schedule: {schedule}")
    if delay_fraction is None or warmup_fraction is None:
        raise RuntimeError("combined gradient-similarity schedule lacks its phase")
    if progress < delay_fraction:
        return 0.0
    ramp = (progress - delay_fraction) / warmup_fraction
    if ramp >= 1.0 - 1.0e-15:
        return maximum
    ramp = min(max(ramp, 0.0), 1.0)
    return maximum * 0.5 * (1.0 - math.cos(math.pi * ramp))


def _scheduled_relative_rho(
    *,
    schedule: str,
    maximum: float,
    progress: float,
    delay_fraction: float | None = None,
    warmup_fraction: float | None = None,
) -> float:
    """Return the exact relative radius used by a rho schedule."""

    if schedule == "constant":
        return maximum
    if warmup_fraction is None:
        raise RuntimeError("scheduled rho lacks its warm-up fraction")
    if schedule in {"sqrt_warmup", "sqrt_linear_warmup"}:
        shape = min(1.0, max(0.0, progress / warmup_fraction))
    elif schedule == "sqrt_cosine_warmup":
        ramp = min(1.0, max(0.0, progress / warmup_fraction))
        shape = 0.5 * (1.0 - math.cos(math.pi * ramp))
    elif schedule == "sqrt_delayed_cosine_warmup":
        if delay_fraction is None:
            raise RuntimeError("delayed rho schedule lacks its delay")
        if progress < delay_fraction:
            shape = 0.0
        else:
            ramp = min(
                1.0,
                max(0.0, (progress - delay_fraction) / warmup_fraction),
            )
            shape = 0.5 * (1.0 - math.cos(math.pi * ramp))
    else:
        raise RuntimeError(f"unsupported rho schedule: {schedule}")
    return maximum * math.sqrt(max(0.0, shape))


def _validate_gradient_similarity_records(
    records: object,
    *,
    expected_steps: int,
    parameter_order_sha256: str,
    gradient_accumulation_steps: int,
    world_size: int,
    total_steps: int = LOCKED_PILOT_SCHEDULER_TOTAL_STEPS,
    lambda_schedule: str = "gradient_similarity",
    vat_lambda: float = 0.03,
    lambda_delay_fraction: float | None = None,
    lambda_warmup_fraction: float | None = None,
    rho_schedule: str | None = None,
    vat_rho: float | None = None,
    rho_delay_fraction: float | None = None,
    rho_warmup_fraction: float | None = None,
) -> tuple[float, int]:
    """Validate an immutable current-step gate prefix and return EMA/count."""

    if not isinstance(records, list) or len(records) != expected_steps:
        raise RuntimeError(
            "gradient-similarity records are not a complete optimizer-step prefix"
        )
    enabled_count = 0
    ema = 0.0
    for step, record in enumerate(records):
        if not isinstance(record, Mapping) or int(
            record.get("optimizer_step", -1)
        ) != step:
            raise RuntimeError(
                "gradient-similarity records are not a complete optimizer-step "
                "prefix"
            )
        try:
            cosine = float(record["gradient_cosine"])
            observed_ema = float(record["gradient_cosine_ema"])
            effective_lambda = float(record["effective_vat_lambda"])
            sft_norm = float(record["sft_gradient_norm"])
            vat_norm = float(record["unweighted_vat_gradient_norm"])
            weighted_norm = float(record["weighted_vat_gradient_norm"])
            dot = float(record["gradient_dot_product"])
            ratio = float(record["weighted_vat_to_sft_ratio"])
            progress = float(record["training_progress"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "gradient-similarity record lacks numeric evidence"
            ) from exc
        cosine_defined = record.get("gradient_cosine_defined", True)
        if not isinstance(cosine_defined, bool):
            raise RuntimeError(
                "gradient-similarity cosine-defined evidence is invalid"
            )
        enabled = cosine_defined and cosine >= 0.0
        enabled_count += int(enabled)
        if cosine_defined:
            ema = 0.99 * ema + 0.01 * cosine
        scheduled_lambda = _gradient_similarity_base_lambda(
            schedule=lambda_schedule,
            maximum=vat_lambda,
            progress=progress,
            delay_fraction=lambda_delay_fraction,
            warmup_fraction=lambda_warmup_fraction,
        )
        expected_lambda = scheduled_lambda if enabled else 0.0
        rho_evidence_valid = True
        if rho_schedule is not None:
            if vat_rho is None:
                raise RuntimeError(
                    "gradient-similarity rho validation lacks vat_rho"
                )
            expected_rho = _scheduled_relative_rho(
                schedule=rho_schedule,
                maximum=vat_rho,
                progress=progress,
                delay_fraction=rho_delay_fraction,
                warmup_fraction=rho_warmup_fraction,
            )
            try:
                observed_rho = float(record["effective_vat_rho"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "gradient-similarity record lacks rho evidence"
                ) from exc
            rho_evidence_valid = math.isclose(
                observed_rho, expected_rho, rel_tol=0.0, abs_tol=1e-15
            )
        combined_evidence_valid = True
        if lambda_schedule in {
            "linear_warmup_gradient_similarity",
            "cosine_warmup_gradient_similarity",
            "delayed_cosine_gradient_similarity",
        }:
            combined_evidence_valid = (
                math.isclose(
                    float(record.get("scheduled_vat_lambda", float("nan"))),
                    scheduled_lambda,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and record.get("gradient_gate_enabled") is enabled
                and record.get("vat_active") is (expected_lambda > 0.0)
            )
        cosine_evidence_valid = (
            (
                cosine_defined
                and math.isfinite(cosine)
                and -1.0 <= cosine <= 1.0
                and vat_norm > 0.0
                and math.isclose(
                    cosine,
                    dot / (sft_norm * vat_norm),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                and record.get("gradient_gate_inactive_reason") is None
            )
            or (
                not cosine_defined
                and cosine == 0.0
                and vat_norm == 0.0
                and dot == 0.0
                and record.get("gradient_gate_inactive_reason")
                == "zero_effective_rho"
                and float(record.get("effective_vat_rho", float("nan")))
                == 0.0
            )
        )
        valid = (
            cosine_evidence_valid
            and math.isfinite(observed_ema)
            and math.isfinite(effective_lambda)
            and math.isfinite(sft_norm)
            and sft_norm > 0.0
            and math.isfinite(vat_norm)
            and math.isfinite(weighted_norm)
            and math.isfinite(dot)
            and math.isfinite(ratio)
            and math.isfinite(progress)
            and math.isclose(observed_ema, ema, rel_tol=0.0, abs_tol=1e-12)
            and effective_lambda == expected_lambda
            and record.get("gradient_cosine_defined", True) is cosine_defined
            and record.get("gradient_gate_enabled", enabled) is enabled
            and record.get("vat_enabled") is enabled
            and int(record.get("vat_enabled_count", -1)) == enabled_count
            and math.isclose(
                float(record.get("vat_enabled_fraction", float("nan"))),
                enabled_count / (step + 1),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                weighted_norm,
                expected_lambda * vat_norm,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and math.isclose(
                ratio,
                weighted_norm / sft_norm,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and math.isclose(
                progress,
                step / total_steps,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and record.get("scope") == GRADIENT_SIMILARITY_SCOPE
            and record.get("update_scope") == GRADIENT_SIMILARITY_UPDATE_SCOPE
            and record.get("gradient_aggregation")
            == GRADIENT_SIMILARITY_AGGREGATION
            and record.get("parameter_order_sha256")
            == parameter_order_sha256
            and int(record.get("gradient_accumulation_steps", 0))
            == gradient_accumulation_steps
            and int(record.get("world_size", 0)) == world_size
            and combined_evidence_valid
            and rho_evidence_valid
        )
        if not valid:
            raise RuntimeError("gradient-similarity evidence is invalid")
    return ema, enabled_count


def _required_lowercase_hex_environment(
    name: str,
    *,
    length: int,
) -> str:
    """Validate internal launch/config identities used for exact resume."""

    value = os.environ.get(name)
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(
            f"{name} must be exactly {length} lowercase hexadecimal characters "
            "when VAT_CONTRACT_OUTPUT is set"
        )
    return value


def _pilot_scheduler_total_steps(raw_value: str | None) -> int | None:
    """Parse the optional pilot horizon and fail closed unless it is 544."""

    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{PILOT_SCHEDULER_TOTAL_STEPS_ENV} must be integer 544"
        ) from exc
    if raw_value.strip() != str(value):
        raise ValueError(
            f"{PILOT_SCHEDULER_TOTAL_STEPS_ENV} must use canonical integer syntax"
        )
    if value != LOCKED_PILOT_SCHEDULER_TOTAL_STEPS:
        raise ValueError(
            f"{PILOT_SCHEDULER_TOTAL_STEPS_ENV} must equal locked main "
            f"horizon {LOCKED_PILOT_SCHEDULER_TOTAL_STEPS}, got {value}"
        )
    return value


@dataclass(frozen=True)
class VATRuntimeConfig:
    """Numerical and routing settings read from the Axolotl config."""

    use_vat: bool = False
    use_minmax_adversarial_sft: bool = False
    minmax_rho: float | None = None
    radius_mode: str = "absolute"
    xi: float = 1.0e-3
    xi_mode: str = "absolute"
    epsilon: float | None = 1.0
    rho: float | None = None
    rho_schedule: str = "constant"
    rho_warmup_fraction: float | None = None
    rho_delay_fraction: float | None = None
    vat_lambda: float = 1.0
    lambda_schedule: str = "constant"
    lambda_warmup_fraction: float | None = None
    lambda_delay_fraction: float | None = None
    lambda_cycles: int | None = None
    gradient_similarity_ema_decay: float = 0.99
    gradient_telemetry_interval_steps: int = 0
    power_iterations: int = 1
    perturbation_mode: str = "adversarial_direction"
    density_mode: str = "none"
    density_vocabulary_chunk_size: int = 2048
    semantic_margin_manifest: str | None = None
    base_model: str | None = None
    perturb_mask_source: str = "dataset"

    def __post_init__(self) -> None:
        if self.perturb_mask_source not in {
            "dataset",
            "native_llama3_input_ids",
        }:
            raise ValueError(
                "vat_perturb_mask_source must be 'dataset' or "
                "'native_llama3_input_ids'"
            )
        if (
            self.perturb_mask_source == "native_llama3_input_ids"
            and not self.use_vat
        ):
            raise ValueError(
                "native_llama3_input_ids perturb-mask derivation is valid only "
                "for VAT"
            )
        if self.use_vat and self.use_minmax_adversarial_sft:
            raise ValueError(
                "use_vat and use_minmax_adversarial_sft are mutually exclusive"
            )
        if self.use_minmax_adversarial_sft:
            if self.minmax_rho != 0.03:
                raise ValueError(
                    "Question-only Min-Max Adversarial SFT requires "
                    "minmax_rho=0.03"
                )
        elif self.minmax_rho is not None:
            raise ValueError(
                "minmax_rho is valid only when "
                "use_minmax_adversarial_sft=true"
            )
        if not math.isfinite(self.xi) or self.xi < 0:
            raise ValueError("vat_xi must be non-negative")
        if self.xi_mode not in {"absolute", "relative_question_frobenius"}:
            raise ValueError(
                "vat_xi_mode must be 'absolute' or "
                "'relative_question_frobenius'"
            )
        if self.radius_mode not in {"absolute", "relative"}:
            raise ValueError("vat_radius_mode must be 'absolute' or 'relative'")
        if self.radius_mode == "absolute":
            if self.epsilon is None or self.rho is not None:
                raise ValueError(
                    "absolute VAT requires exactly one radius field: "
                    "vat_epsilon present and vat_rho absent"
                )
        elif self.rho is None or self.epsilon is not None:
            raise ValueError(
                "relative VAT requires exactly one radius field: "
                "vat_rho present and vat_epsilon absent"
            )
        if self.epsilon is not None and (
            not math.isfinite(self.epsilon) or self.epsilon < 0
        ):
            raise ValueError("vat_epsilon must be finite and non-negative")
        if self.rho is not None and (
            not math.isfinite(self.rho) or self.rho < 0
        ):
            raise ValueError("vat_rho must be finite and non-negative")
        if self.rho_schedule not in {
            "constant",
            "sqrt_warmup",
            "sqrt_linear_warmup",
            "sqrt_cosine_warmup",
            "sqrt_delayed_cosine_warmup",
        }:
            raise ValueError("unsupported vat_rho_schedule")
        if self.rho_schedule == "constant":
            if (
                self.rho_warmup_fraction is not None
                or self.rho_delay_fraction is not None
            ):
                raise ValueError(
                    "constant vat_rho_schedule must not set "
                    "vat_rho_warmup_fraction or vat_rho_delay_fraction"
                )
        else:
            if self.radius_mode != "relative" or not self.use_vat:
                raise ValueError(
                    "sqrt_warmup vat_rho_schedule requires relative VAT"
                )
            if self.rho_warmup_fraction is None or not (
                math.isfinite(self.rho_warmup_fraction)
                and 0.0 < self.rho_warmup_fraction <= 1.0
            ):
                raise ValueError(
                    "sqrt_warmup requires vat_rho_warmup_fraction in (0, 1]"
                )
            if self.rho_schedule == "sqrt_delayed_cosine_warmup":
                if self.rho_delay_fraction is None or not (
                    math.isfinite(self.rho_delay_fraction)
                    and 0.0 <= self.rho_delay_fraction < 1.0
                    and self.rho_delay_fraction + self.rho_warmup_fraction
                    <= 1.0
                ):
                    raise ValueError(
                        "sqrt_delayed_cosine_warmup requires a finite "
                        "vat_rho_delay_fraction with delay + warm-up <= 1"
                    )
            elif self.rho_delay_fraction is not None:
                raise ValueError(
                    f"{self.rho_schedule} must not set vat_rho_delay_fraction"
                )
            if self.perturbation_mode != "adversarial_direction":
                raise ValueError(
                    "sqrt_warmup is supported only for adversarial-direction VAT"
                )
            if self.density_mode != "none" or self.semantic_margin_manifest is not None:
                raise ValueError(
                    "sqrt_warmup is supported only for ordinary relative VAT"
                )
        if not math.isfinite(self.vat_lambda) or self.vat_lambda < 0:
            raise ValueError("vat_lambda must be non-negative")
        if self.lambda_schedule not in {
            "constant",
            "linear_warmup",
            "delayed_linear_warmup",
            "cosine_warmup",
            "delayed_cosine_warmup",
            "gradient_similarity",
            "linear_warmup_gradient_similarity",
            "cosine_warmup_gradient_similarity",
            "delayed_cosine_gradient_similarity",
            "cyclical_cosine",
        }:
            raise ValueError(
                "unsupported vat_lambda_schedule"
            )
        if self.lambda_schedule in {"constant", "gradient_similarity"}:
            if (
                self.lambda_warmup_fraction is not None
                or self.lambda_delay_fraction is not None
                or self.lambda_cycles is not None
            ):
                raise ValueError(
                    f"{self.lambda_schedule} vat_lambda_schedule must not set "
                    "VAT warm-up/delay/cycle fields"
                )
        elif self.lambda_schedule == "cyclical_cosine":
            if (
                self.lambda_warmup_fraction is not None
                or self.lambda_delay_fraction is not None
                or self.lambda_cycles != 2
            ):
                raise ValueError(
                    "cyclical_cosine requires vat_lambda_cycles=2 and no "
                    "warm-up/delay fractions"
                )
        else:
            if self.lambda_cycles is not None:
                raise ValueError(
                    f"{self.lambda_schedule} must not set vat_lambda_cycles"
                )
            if self.lambda_warmup_fraction is None or not (
                math.isfinite(self.lambda_warmup_fraction)
                and 0.0 < self.lambda_warmup_fraction <= 1.0
            ):
                raise ValueError(
                    "VAT linear warm-up requires "
                    "vat_lambda_warmup_fraction in (0, 1]"
                )
            if self.lambda_schedule in {
                "linear_warmup",
                "cosine_warmup",
                "linear_warmup_gradient_similarity",
                "cosine_warmup_gradient_similarity",
            }:
                if self.lambda_delay_fraction is not None:
                    raise ValueError(
                        f"{self.lambda_schedule} must not set "
                        "vat_lambda_delay_fraction"
                    )
            elif self.lambda_delay_fraction is None or not (
                math.isfinite(self.lambda_delay_fraction)
                and 0.0 <= self.lambda_delay_fraction < 1.0
                and self.lambda_delay_fraction + self.lambda_warmup_fraction
                <= 1.0
            ):
                raise ValueError(
                    "delayed VAT warm-up requires a non-negative "
                    "vat_lambda_delay_fraction whose sum with the warm-up "
                    "fraction is at most 1"
                )
        if self.uses_gradient_similarity_gate:
            expected_lambda = (
                0.0375
                if self.lambda_schedule
                == "delayed_cosine_gradient_similarity"
                else 0.03
            )
            if self.vat_lambda != expected_lambda:
                raise ValueError(
                    f"{self.lambda_schedule} requires "
                    f"vat_lambda={expected_lambda}"
                )
            if self.gradient_telemetry_interval_steps != 0:
                raise ValueError(
                    f"{self.lambda_schedule} records every optimizer step and "
                    "requires vat_gradient_telemetry_interval_steps=0"
                )
            if self.lambda_schedule == "delayed_cosine_gradient_similarity" and (
                self.lambda_delay_fraction != 0.1
                or self.lambda_warmup_fraction != 0.2
            ):
                raise ValueError(
                    "delayed_cosine_gradient_similarity requires exact "
                    "10% delay and 20% ramp"
                )
            if self.lambda_schedule in {
                "linear_warmup_gradient_similarity",
                "cosine_warmup_gradient_similarity",
            } and (
                self.lambda_warmup_fraction != 0.2
                or self.lambda_delay_fraction is not None
            ):
                raise ValueError(
                    f"{self.lambda_schedule} requires an exact 20% ramp "
                    "without delay"
                )
        if not (
            math.isfinite(self.gradient_similarity_ema_decay)
            and self.gradient_similarity_ema_decay == 0.99
        ):
            raise ValueError(
                "vat_gradient_similarity_ema_decay must be exactly 0.99"
            )
        if (
            isinstance(self.gradient_telemetry_interval_steps, bool)
            or not isinstance(self.gradient_telemetry_interval_steps, int)
            or self.gradient_telemetry_interval_steps < 0
        ):
            raise ValueError(
                "vat_gradient_telemetry_interval_steps must be a non-negative integer"
            )
        if self.power_iterations != 1:
            raise ValueError(
                "the locked experiment requires exactly one VAT power iteration"
            )
        if self.perturbation_mode not in {
            "adversarial_direction",
            "random_neighbor",
            "metamorphic_text",
            "metamorphic_text_global_token_mean",
            "metamorphic_text_soft_worst",
            "metamorphic_text_uniform_sft",
        }:
            raise ValueError(
                "vat_perturbation_mode must be 'adversarial_direction' or "
                "'random_neighbor' or a supported metamorphic-text mode"
            )
        if self.density_mode not in {"none", "vocab_radius_count"}:
            raise ValueError(
                "vat_density_mode must be 'none' or 'vocab_radius_count'"
            )
        if (
            self.density_mode == "vocab_radius_count"
            and self.radius_mode != "relative"
        ):
            raise ValueError("Dense-VAT requires vat_radius_mode=relative")
        if self.perturbation_mode == "random_neighbor":
            if self.radius_mode != "relative" or self.rho != 0.03:
                raise ValueError("Random-Neighbor VAT requires relative rho=0.03")
            if self.density_mode != "vocab_radius_count":
                raise ValueError(
                    "Random-Neighbor VAT requires vocab-radius density counts"
                )
            if self.semantic_margin_manifest is not None:
                raise ValueError(
                    "Random-Neighbor VAT cannot use a semantic-margin cap"
                )
        if (
            isinstance(self.density_vocabulary_chunk_size, bool)
            or self.density_vocabulary_chunk_size < 1
        ):
            raise ValueError(
                "vat_density_vocabulary_chunk_size must be a positive integer"
            )
        if self.semantic_margin_manifest is not None:
            if (
                not isinstance(self.semantic_margin_manifest, str)
                or not self.semantic_margin_manifest.strip()
            ):
                raise ValueError(
                    "vat_semantic_margin_manifest must be a non-empty path"
                )
            if self.density_mode != "vocab_radius_count":
                raise ValueError(
                    "semantic-margin capping requires Dense-VAT density scaling"
                )
            if self.rho != 0.03:
                raise ValueError("Semantic-Margin-Capped Dense-VAT requires rho=0.03")
            if not isinstance(self.base_model, str) or not self.base_model.strip():
                raise ValueError(
                    "Semantic-Margin-Capped Dense-VAT requires base_model provenance"
                )

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | Any) -> "VATRuntimeConfig":
        """Build from a dict or Axolotl's attribute-access ``DictDefault``."""

        def read(name: str, default: Any) -> Any:
            if isinstance(cfg, Mapping):
                value = cfg.get(name, default)
            else:
                value = getattr(cfg, name, default)
            return default if value is None else value

        radius_mode = str(read("vat_radius_mode", "absolute"))
        rho = read("vat_rho", None)
        epsilon = read(
            "vat_epsilon",
            1.0 if radius_mode == "absolute" else None,
        )
        lambda_warmup_fraction = read("vat_lambda_warmup_fraction", None)
        lambda_delay_fraction = read("vat_lambda_delay_fraction", None)
        lambda_cycles = read("vat_lambda_cycles", None)
        rho_warmup_fraction = read("vat_rho_warmup_fraction", None)
        rho_delay_fraction = read("vat_rho_delay_fraction", None)
        telemetry_interval = read(
            "vat_gradient_telemetry_interval_steps", 0
        )
        if isinstance(telemetry_interval, bool):
            raise ValueError(
                "vat_gradient_telemetry_interval_steps must be a non-negative integer"
            )
        return cls(
            use_vat=bool(read("use_vat", False)),
            use_minmax_adversarial_sft=bool(
                read("use_minmax_adversarial_sft", False)
            ),
            minmax_rho=(
                None
                if read("minmax_rho", None) is None
                else float(read("minmax_rho", None))
            ),
            radius_mode=radius_mode,
            xi=float(read("vat_xi", 1.0e-3)),
            xi_mode=str(read("vat_xi_mode", "absolute")),
            epsilon=None if epsilon is None else float(epsilon),
            rho=None if rho is None else float(rho),
            rho_schedule=str(read("vat_rho_schedule", "constant")),
            rho_warmup_fraction=(
                None
                if rho_warmup_fraction is None
                else float(rho_warmup_fraction)
            ),
            rho_delay_fraction=(
                None
                if rho_delay_fraction is None
                else float(rho_delay_fraction)
            ),
            vat_lambda=float(read("vat_lambda", 1.0)),
            lambda_schedule=str(read("vat_lambda_schedule", "constant")),
            lambda_warmup_fraction=(
                None
                if lambda_warmup_fraction is None
                else float(lambda_warmup_fraction)
            ),
            lambda_delay_fraction=(
                None
                if lambda_delay_fraction is None
                else float(lambda_delay_fraction)
            ),
            lambda_cycles=(
                None if lambda_cycles is None else int(lambda_cycles)
            ),
            gradient_similarity_ema_decay=float(
                read("vat_gradient_similarity_ema_decay", 0.99)
            ),
            gradient_telemetry_interval_steps=int(telemetry_interval),
            power_iterations=int(read("vat_power_iterations", 1)),
            perturbation_mode=str(
                read("vat_perturbation_mode", "adversarial_direction")
            ),
            density_mode=str(read("vat_density_mode", "none")),
            density_vocabulary_chunk_size=int(
                read("vat_density_vocabulary_chunk_size", 2048)
            ),
            semantic_margin_manifest=read(
                "vat_semantic_margin_manifest", None
            ),
            base_model=read("base_model", None),
            perturb_mask_source=str(
                read("vat_perturb_mask_source", "dataset")
            ),
        )

    @property
    def uses_question_adversary(self) -> bool:
        return self.use_vat or self.use_minmax_adversarial_sft

    @property
    def training_objective(self) -> str:
        if self.use_minmax_adversarial_sft:
            return "question_only_minmax_adversarial_sft"
        if self.use_vat and self.perturbation_mode == "metamorphic_text":
            return "metamorphic_text_worst_candidate_vat_sft"
        if (
            self.use_vat
            and self.perturbation_mode == "metamorphic_text_global_token_mean"
        ):
            return "metamorphic_text_worst_candidate_global_token_mean_vat_sft"
        if (
            self.use_vat
            and self.perturbation_mode == "metamorphic_text_soft_worst"
        ):
            return "metamorphic_text_soft_worst_global_token_mean_vat_sft"
        if self.use_vat and self.perturbation_mode == "metamorphic_text_uniform_sft":
            return "metamorphic_text_uniform_candidate_clean_transformed_sft"
        if self.use_vat:
            return "question_only_virtual_adversarial_sft"
        return "vanilla_sft"

    @property
    def uses_gradient_similarity_gate(self) -> bool:
        return self.lambda_schedule in GRADIENT_SIMILARITY_SCHEDULES

    def effective_lambda(self, *, global_step: int, total_steps: int) -> float:
        """Return the exact coefficient for the current optimizer progress."""

        if isinstance(global_step, bool) or global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        if isinstance(total_steps, bool) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if self.lambda_schedule == "constant":
            return self.vat_lambda
        if self.uses_gradient_similarity_gate:
            raise RuntimeError(
                "gradient-similarity lambda is selected at the optimizer-step "
                "boundary, not from scalar training progress"
            )
        if self.lambda_schedule == "cyclical_cosine":
            if self.lambda_cycles != 2:  # guarded by __post_init__
                raise RuntimeError("cyclical cosine lacks its two-cycle contract")
            if total_steps % self.lambda_cycles:
                raise RuntimeError(
                    "cyclical_cosine requires total_steps divisible by cycles"
                )
            cycle_steps = total_steps // self.lambda_cycles
            cycle_step = global_step % cycle_steps
            half_cycle_steps = cycle_steps / 2.0
            if cycle_step >= half_cycle_steps:
                return self.vat_lambda
            phase = cycle_step / half_cycle_steps
            return self.vat_lambda * 0.5 * (1.0 - math.cos(math.pi * phase))
        if self.lambda_warmup_fraction is None:  # guarded by __post_init__
            raise RuntimeError("VAT warm-up lacks its fraction")
        progress = global_step / total_steps
        if self.lambda_schedule in {"linear_warmup", "cosine_warmup"}:
            ramp = min(1.0, progress / self.lambda_warmup_fraction)
            if self.lambda_schedule == "cosine_warmup":
                ramp = 0.5 * (1.0 - math.cos(math.pi * ramp))
            return self.vat_lambda * ramp
        if self.lambda_delay_fraction is None:  # guarded by __post_init__
            raise RuntimeError("delayed VAT warm-up lacks its delay fraction")
        if progress < self.lambda_delay_fraction:
            return 0.0
        ramp_progress = (
            progress - self.lambda_delay_fraction
        ) / self.lambda_warmup_fraction
        if ramp_progress >= 1.0 - 1.0e-15:
            return self.vat_lambda
        ramp_progress = max(0.0, ramp_progress)
        if self.lambda_schedule == "delayed_cosine_warmup":
            ramp_progress = 0.5 * (
                1.0 - math.cos(math.pi * ramp_progress)
            )
        return self.vat_lambda * ramp_progress

    def effective_rho(
        self,
        *,
        global_step: int,
        total_steps: int,
    ) -> float | None:
        """Return the relative radius for the current optimizer progress."""

        if isinstance(global_step, bool) or global_step < 0:
            raise ValueError("global_step must be a non-negative integer")
        if isinstance(total_steps, bool) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if self.rho is None:
            return None
        if self.rho_schedule == "constant":
            return self.rho
        if self.rho_warmup_fraction is None:  # guarded by __post_init__
            raise RuntimeError("VAT rho warm-up lacks its fraction")
        progress = global_step / total_steps
        if self.rho_schedule in {"sqrt_warmup", "sqrt_linear_warmup"}:
            shape = min(1.0, progress / self.rho_warmup_fraction)
        elif self.rho_schedule == "sqrt_cosine_warmup":
            ramp = min(1.0, progress / self.rho_warmup_fraction)
            shape = 0.5 * (1.0 - math.cos(math.pi * ramp))
        else:
            if self.rho_delay_fraction is None:  # guarded by __post_init__
                raise RuntimeError("delayed VAT rho warm-up lacks its delay")
            if progress < self.rho_delay_fraction:
                shape = 0.0
            else:
                ramp = min(
                    1.0,
                    (progress - self.rho_delay_fraction)
                    / self.rho_warmup_fraction,
                )
                shape = 0.5 * (1.0 - math.cos(math.pi * ramp))
        return self.rho * math.sqrt(max(0.0, shape))


@dataclass(frozen=True)
class ExactResumePosition:
    """Contract-derived dataloader position for an optimizer checkpoint."""

    global_step: int
    batches: int
    gradient_accumulation_steps: int
    dataloader_batches_per_epoch: int
    completed_epochs: int
    batches_in_current_epoch: int
    upstream_completed_epochs: int
    upstream_batches_in_current_epoch: int

    @property
    def upstream_update_steps_per_epoch(self) -> int:
        return max(
            self.dataloader_batches_per_epoch
            // self.gradient_accumulation_steps,
            1,
        )

    @property
    def correction_required(self) -> bool:
        return (
            self.completed_epochs != self.upstream_completed_epochs
            or self.batches_in_current_epoch
            != self.upstream_batches_in_current_epoch
        )

    def as_audit(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "contract_batches": self.batches,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "dataloader_batches_per_epoch": self.dataloader_batches_per_epoch,
            "completed_epochs": self.completed_epochs,
            "batches_in_current_epoch": self.batches_in_current_epoch,
            "upstream_completed_epochs": self.upstream_completed_epochs,
            "upstream_batches_in_current_epoch": (
                self.upstream_batches_in_current_epoch
            ),
            "correction_required": self.correction_required,
            "passed": True,
        }


def _infer_exact_resume_position(
    *,
    global_step: int,
    recorded_epoch: float,
    rank_batches: list[int],
    gradient_accumulation_steps: int,
) -> ExactResumePosition:
    """Recover the exact micro-batch position from signed checkpoint evidence.

    Transformers 4.40.2 uses ``len(dataloader) // grad_accum`` to map an
    optimizer step back to an epoch. Its loop, however, carries a trailing
    partial accumulation across the epoch boundary. When the dataloader length
    is not divisible by ``grad_accum``, the quotient therefore cannot recover
    the consumed batches. Our cumulative per-rank contract can.
    """

    if global_step < 1:
        raise RuntimeError("resume checkpoint global_step must be positive")
    if (
        not math.isfinite(recorded_epoch)
        or recorded_epoch <= 0
        or not rank_batches
    ):
        raise RuntimeError("resume checkpoint lacks a valid epoch/batch position")
    if gradient_accumulation_steps < 1:
        raise RuntimeError("gradient_accumulation_steps must be positive")
    if any(value < 1 for value in rank_batches):
        raise RuntimeError("resume contract batch counts must be positive")
    if len(set(rank_batches)) != 1:
        raise RuntimeError(
            "resume contract ranks consumed different numbers of micro-batches"
        )

    batches = rank_batches[0]
    dataloader_batches = int(round(batches / recorded_epoch))
    if dataloader_batches < 1 or not math.isclose(
        recorded_epoch,
        batches / dataloader_batches,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "trainer_state epoch is inconsistent with cumulative contract batches"
        )

    if dataloader_batches <= gradient_accumulation_steps:
        # Transformers explicitly flushes the final partial accumulation only
        # when the *entire* epoch has at most grad_accum batches. The small
        # fixed pilot shard therefore performs one update per short epoch.
        if batches % dataloader_batches != 0:
            raise RuntimeError(
                "short-dataloader resume checkpoint is not at an epoch boundary"
            )
        expected_global_step = batches // dataloader_batches
    else:
        # For normal-sized dataloaders the 4.40.2 loop carries an incomplete
        # accumulation into the next epoch, so every saved optimizer step
        # corresponds to exactly grad_accum cumulative micro-batches.
        if batches % gradient_accumulation_steps != 0:
            raise RuntimeError(
                "resume contract is not at a complete accumulation boundary"
            )
        expected_global_step = batches // gradient_accumulation_steps
    if global_step != expected_global_step:
        raise RuntimeError(
            "resume contract batch count is inconsistent with global_step: "
            f"observed {global_step}, expected {expected_global_step}"
        )

    completed_epochs, batches_in_current_epoch = divmod(
        batches, dataloader_batches
    )
    update_steps_per_epoch = max(
        dataloader_batches // gradient_accumulation_steps,
        1,
    )
    upstream_completed_epochs, upstream_update_offset = divmod(
        global_step, update_steps_per_epoch
    )
    upstream_batches_in_current_epoch = (
        upstream_update_offset * gradient_accumulation_steps
    )
    return ExactResumePosition(
        global_step=global_step,
        batches=batches,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_batches_per_epoch=dataloader_batches,
        completed_epochs=completed_epochs,
        batches_in_current_epoch=batches_in_current_epoch,
        upstream_completed_epochs=upstream_completed_epochs,
        upstream_batches_in_current_epoch=upstream_batches_in_current_epoch,
    )


class _ExactResumeBatchOffset:
    """One-use carrier for Trainer's ``remainder * grad_accum`` expression."""

    def __init__(self, batches: int, gradient_accumulation_steps: int) -> None:
        self.batches = batches
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.used = False

    def __mul__(self, other: Any) -> int:
        if int(other) != self.gradient_accumulation_steps:
            raise RuntimeError(
                "Transformers resume multiplication changed unexpectedly"
            )
        if self.used:
            raise RuntimeError("exact resume batch offset was consumed twice")
        self.used = True
        return self.batches


class _ExactResumeGlobalStep(int):
    """Int-compatible step that corrects only Trainer's two resume operations."""

    def __new__(
        cls,
        position: ExactResumePosition,
    ) -> "_ExactResumeGlobalStep":
        value = int.__new__(cls, position.global_step)
        value.position = position
        value.floor_division_used = False
        value.modulo_used = False
        value.offset_carrier = None
        return value

    def __floordiv__(self, other: Any) -> int:
        if (
            not self.floor_division_used
            and int(other) == self.position.upstream_update_steps_per_epoch
        ):
            self.floor_division_used = True
            return self.position.completed_epochs
        return int(self) // other

    def __mod__(self, other: Any) -> Any:
        if (
            not self.modulo_used
            and int(other) == self.position.upstream_update_steps_per_epoch
        ):
            self.modulo_used = True
            carrier = _ExactResumeBatchOffset(
                self.position.batches_in_current_epoch,
                self.position.gradient_accumulation_steps,
            )
            self.offset_carrier = carrier
            return carrier
        return int(self) % other

    @property
    def correction_consumed(self) -> bool:
        return bool(
            self.floor_division_used
            and self.modulo_used
            and self.offset_carrier is not None
            and self.offset_carrier.used
        )


def _call_with_exact_resume_position(
    function: Any,
    *,
    checkpoint: Path,
    position: ExactResumePosition,
    save_steps: int,
) -> Any:
    """Scope the old-Trainer resume corrections to one upstream train call.

    Transformers 4.40.2 replaces the freshly initialized ``TrainerState``
    with the checkpoint JSON after deriving the current segment's logging,
    evaluation, and save intervals.  Consequently, a staged resume silently
    inherits the prior segment's ``save_steps`` even when the current
    ``TrainingArguments`` request a different exact endpoint.  Restore the
    current, already validated save cadence on every scoped state load while
    leaving the immutable checkpoint JSON untouched.
    """

    from transformers.trainer_callback import TrainerState

    original_descriptor = inspect.getattr_static(TrainerState, "load_from_json")
    original_loader = TrainerState.load_from_json
    wrapped_steps: list[_ExactResumeGlobalStep] = []
    state_path = (checkpoint / "trainer_state.json").resolve()
    if isinstance(save_steps, bool) or not isinstance(save_steps, int):
        raise TypeError("exact resume save_steps must be an integer")
    if save_steps < 1:
        raise ValueError("exact resume save_steps must be positive")

    @classmethod
    def exact_load_from_json(cls: type, json_path: str):
        del cls
        state = original_loader(json_path)
        if Path(json_path).resolve() == state_path:
            if int(state.global_step) != position.global_step:
                raise RuntimeError(
                    "trainer state changed after exact resume position audit"
                )
            wrapped = _ExactResumeGlobalStep(position)
            state.global_step = wrapped
            state.save_steps = save_steps
            wrapped_steps.append(wrapped)
        return state

    setattr(TrainerState, "load_from_json", exact_load_from_json)
    succeeded = False
    try:
        result = function()
        succeeded = True
        return result
    finally:
        setattr(TrainerState, "load_from_json", original_descriptor)
        if succeeded:
            consumed = [
                wrapped for wrapped in wrapped_steps if wrapped.correction_consumed
            ]
            if len(consumed) != 1:
                raise RuntimeError(
                    "Transformers did not consume the exact resume correction once"
                )


def _underlying_module(model: nn.Module) -> nn.Module:
    """Return the causal-LM module below common distributed wrappers."""

    current = model
    seen: set[int] = set()
    while hasattr(current, "module") and id(current) not in seen:
        seen.add(id(current))
        nested = getattr(current, "module")
        if not isinstance(nested, nn.Module):
            break
        current = nested
    return current


def _underlying_optimizer(optimizer: Any) -> Any:
    """Unwrap Accelerate's optimizer wrapper without assuming its class."""

    current = optimizer
    seen: set[int] = set()
    while hasattr(current, "optimizer") and id(current) not in seen:
        seen.add(id(current))
        nested = getattr(current, "optimizer")
        if nested is None:
            break
        current = nested
    return current


def _enforce_vat_non_reentrant_checkpointing(model: nn.Module) -> bool:
    """Replace active reentrant checkpointing with VAT-compatible checkpointing."""

    causal_lm = _underlying_module(model)
    if not bool(getattr(causal_lm, "is_gradient_checkpointing", False)):
        return False
    enable = getattr(causal_lm, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise RuntimeError(
            "VAT model reports active gradient checkpointing but cannot configure it"
        )
    enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    active_modules = [
        module
        for module in causal_lm.modules()
        if bool(getattr(module, "gradient_checkpointing", False))
    ]
    if not active_modules:
        raise RuntimeError(
            "VAT gradient checkpointing was requested but no active modules were found"
        )
    incompatible = []
    for module in active_modules:
        checkpoint_function = getattr(module, "_gradient_checkpointing_func", None)
        keywords = getattr(checkpoint_function, "keywords", None)
        if not isinstance(keywords, dict) or keywords.get("use_reentrant") is not False:
            incompatible.append(type(module).__name__)
    if incompatible:
        raise RuntimeError(
            "VAT requires use_reentrant=False gradient checkpointing; incompatible "
            f"modules: {sorted(set(incompatible))}"
        )
    return True


def _call_with_vat_non_reentrant_checkpointing(
    function: Any,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, int]:
    """Force Axolotl's direct checkpoint calls to the non-reentrant engine."""

    import torch.utils.checkpoint as checkpoint_module

    original_checkpoint = checkpoint_module.checkpoint
    checkpoint_calls = 0

    @functools.wraps(original_checkpoint)
    def non_reentrant_checkpoint(*checkpoint_args: Any, **checkpoint_kwargs: Any):
        nonlocal checkpoint_calls
        requested = checkpoint_kwargs.get("use_reentrant", False)
        if requested is not False:
            raise RuntimeError("VAT forbids reentrant activation checkpointing")
        checkpoint_kwargs["use_reentrant"] = False
        checkpoint_calls += 1
        return original_checkpoint(*checkpoint_args, **checkpoint_kwargs)

    checkpoint_module.checkpoint = non_reentrant_checkpoint
    try:
        return function(*args, **kwargs), checkpoint_calls
    finally:
        checkpoint_module.checkpoint = original_checkpoint


def _repage_loaded_optimizer_state(
    optimizer: Any,
    *,
    get_paged: Any | None = None,
) -> int:
    """Restore loaded bitsandbytes moment tensors to paged allocations.

    bitsandbytes 0.43.1 initializes large ``state1``/``state2`` buffers with
    ``get_paged`` during a fresh run, but its generic ``load_state_dict`` moves
    checkpoint tensors to ordinary CUDA allocations. Re-create only those two
    buffers as paged tensors and copy their exact contents.
    """

    optimizer = _underlying_optimizer(optimizer)
    if not bool(getattr(optimizer, "is_paged", False)):
        return 0
    if get_paged is None:
        from bitsandbytes import functional as bnb_functional

        get_paged = bnb_functional.get_paged
    page_manager = getattr(optimizer, "page_mng", None)
    paged_tensors = getattr(page_manager, "paged_tensors", None)
    if not isinstance(paged_tensors, list):
        raise RuntimeError("paged optimizer has no bitsandbytes page manager")

    def is_registered_paged(value: Tensor) -> bool:
        # ``is_paged`` is serialized as a Python tensor attribute. A tensor
        # loaded by torch.load can therefore claim to be paged even though its
        # storage is an ordinary CUDA allocation. Membership by identity in
        # the live GlobalPageManager is the authoritative runtime invariant.
        return bool(getattr(value, "is_paged", False)) and any(
            value is registered for registered in paged_tensors
        )

    converted = 0
    for parameter, state in optimizer.state.items():
        if not isinstance(parameter, Tensor) or not isinstance(state, MutableMapping):
            continue
        for key in ("state1", "state2"):
            value = state.get(key)
            if (
                not isinstance(value, Tensor)
                or value.numel() < 100_000
                or is_registered_paged(value)
            ):
                continue
            paged = get_paged(
                *value.shape,
                dtype=value.dtype,
                device=parameter.device,
            )
            paged.copy_(value)
            state[key] = paged
            paged_tensors.append(paged)
            converted += 1

    remaining = [
        (key, tuple(value.shape))
        for state in optimizer.state.values()
        if isinstance(state, Mapping)
        for key in ("state1", "state2")
        if isinstance((value := state.get(key)), Tensor)
        and value.numel() >= 100_000
        and not is_registered_paged(value)
    ]
    if remaining:
        raise RuntimeError(
            f"large paged optimizer state remained on ordinary CUDA storage: {remaining[:4]}"
        )
    return converted


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
        raise TypeError("model output must expose tensor-valued logits")
    return logits


def _decoder_and_head(causal_lm: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Resolve the decoder and output projection of a Llama-style causal LM."""

    decoder = getattr(causal_lm, "model", None)
    output_head = getattr(causal_lm, "lm_head", None)
    if not isinstance(decoder, nn.Module) or not isinstance(output_head, nn.Module):
        raise TypeError(
            "compact VAT logits require a Llama-style model exposing `.model` "
            "and `.lm_head` modules"
        )
    return decoder, output_head


def install_compact_assistant_forward(causal_lm: nn.Module) -> None:
    """Add a private assistant-position-only forward mode to one model.

    Calls without ``vat_prediction_mask`` execute the model's original bound
    ``forward`` unchanged.  This is the path used by Vanilla SFT and eval.
    Calls with the mask execute the same decoder, gather the selected hidden
    states, and apply the LM head only to those states.  The root module is
    still invoked through DDP, so parameter gradient synchronization remains
    owned by the standard distributed wrapper.
    """

    causal_lm = _underlying_module(causal_lm)
    if getattr(causal_lm, "_vat_compact_forward_installed", False):
        return
    _decoder_and_head(causal_lm)
    original_forward = causal_lm.forward

    @functools.wraps(original_forward)
    def compact_forward(
        self: nn.Module,
        *args: Any,
        vat_prediction_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if vat_prediction_mask is None:
            return original_forward(*args, **kwargs)
        if args:
            raise TypeError("compact VAT forward accepts keyword arguments only")
        if kwargs.get("labels") is not None:
            raise ValueError("compact VAT forward computes logits, not model CE")

        decoder, output_head = _decoder_and_head(self)
        decoder_keys = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "use_cache",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "cache_position",
        }
        decoder_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in decoder_keys and value is not None
        }
        # Axolotl v0.4.0 / Transformers 4.40 does not universally expose
        # cache_position. Retry without it when the pinned decoder rejects it.
        try:
            decoder_outputs = decoder(**decoder_kwargs)
        except TypeError:
            if "cache_position" not in decoder_kwargs:
                raise
            decoder_kwargs.pop("cache_position")
            decoder_outputs = decoder(**decoder_kwargs)

        if isinstance(decoder_outputs, Mapping):
            hidden_states = decoder_outputs.get("last_hidden_state")
        else:
            hidden_states = getattr(decoder_outputs, "last_hidden_state", None)
        if hidden_states is None and isinstance(decoder_outputs, (tuple, list)):
            hidden_states = decoder_outputs[0]
        if not isinstance(hidden_states, Tensor) or hidden_states.ndim != 3:
            raise TypeError("decoder must return [B, T, H] last_hidden_state")

        mask = vat_prediction_mask.to(device=hidden_states.device, dtype=torch.bool)
        if mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                "vat_prediction_mask must match decoder [B, T], got "
                f"{tuple(mask.shape)} versus {tuple(hidden_states.shape[:2])}"
            )
        selected_hidden = hidden_states[mask]

        pretraining_tp = int(
            getattr(getattr(self, "config", None), "pretraining_tp", 1)
        )
        if pretraining_tp > 1 and hasattr(output_head, "weight"):
            # Match Transformers' LlamaForCausalLM tensor-parallel LM-head path.
            split_size = output_head.weight.shape[0] // pretraining_tp
            head_slices = output_head.weight.split(split_size, dim=0)
            logits = torch.cat(
                [
                    F.linear(selected_hidden, weight_slice)
                    for weight_slice in head_slices
                ],
                dim=-1,
            )
        else:
            logits = output_head(selected_hidden)

        return SimpleNamespace(
            # Preserve the model's BF16 logits. VAT KL promotes one bounded
            # row chunk at a time to FP32, avoiding a full [N, V] FP32 copy.
            logits=logits,
            past_key_values=getattr(decoder_outputs, "past_key_values", None),
            hidden_states=None,
            attentions=None,
        )

    # ``original_forward`` is already bound, while ``MethodType`` below binds
    # the wrapper once more.  Without an explicit unbound-style signature,
    # ``inspect.signature(causal_lm.forward)`` drops the original first
    # argument (``input_ids``). Transformers then removes that column from the
    # eval dataset as "unused", even though the delegated Vanilla/eval path
    # needs it. Preserve every original forward column for Trainer pruning.
    original_signature = inspect.signature(original_forward)
    setattr(
        compact_forward,
        "__signature__",
        original_signature.replace(
            parameters=[
                inspect.Parameter(
                    "self",
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ),
                *original_signature.parameters.values(),
            ]
        ),
    )
    causal_lm.forward = MethodType(compact_forward, causal_lm)
    causal_lm._vat_compact_forward_installed = True
    causal_lm._vat_original_forward = original_forward


def _shifted_targets(labels: Tensor, prediction_mask: Tensor) -> Tensor:
    """Gather label ``t+1`` for each selected prediction position ``t``."""

    shifted = torch.full_like(labels, -100)
    if labels.shape[1] > 1:
        shifted[:, :-1] = labels[:, 1:]
    return shifted[prediction_mask]


class QuestionOnlyVATTrainerMixin:
    """Mixin inserted immediately before Axolotl's original trainer class."""

    vat_runtime_config = VATRuntimeConfig()

    def _native_llama3_batch_masks(
        self,
        *,
        input_ids: Tensor,
        position_ids: Tensor | None,
        attention_mask: Tensor | None,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Add VAT-only metadata without touching native Axolotl labels."""

        if not isinstance(position_ids, Tensor) or not isinstance(
            attention_mask, Tensor
        ):
            raise KeyError(
                "native Axolotl VAT requires position_ids and attention_mask"
            )
        cached = getattr(self, "_native_llama3_mask_tokens", None)
        if cached is None:
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                tokenizer = getattr(self, "processing_class", None)
            if tokenizer is None or not callable(tokenizer):
                raise RuntimeError(
                    "native Axolotl VAT trainer has no callable tokenizer"
                )
            encoded = tokenizer(
                "<|start_header_id|>user<|end_header_id|>\n\n",
                add_special_tokens=False,
            )
            header_ids = encoded.get("input_ids")
            if hasattr(header_ids, "tolist"):
                header_ids = header_ids.tolist()
            if (
                not isinstance(header_ids, list)
                or not header_ids
                or any(isinstance(value, list) for value in header_ids)
            ):
                raise RuntimeError("cannot tokenize native Llama-3 user header")
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if not isinstance(eot_id, int) or eot_id < 0:
                raise RuntimeError("native Llama-3 tokenizer lacks <|eot_id|>")
            special_ids = tuple(
                int(value) for value in getattr(tokenizer, "all_special_ids", ())
            )
            cached = (tuple(int(value) for value in header_ids), eot_id, special_ids)
            self._native_llama3_mask_tokens = cached
        perturb_mask, source_ids = native_llama3_packed_user_mask(
            input_ids,
            position_ids,
            attention_mask,
            user_header_ids=cached[0],
            eot_token_id=cached[1],
            special_token_ids=cached[2],
        )
        if bool((perturb_mask & labels.ne(-100)).any()):
            raise RuntimeError(
                "native Axolotl VAT mask overlaps a supervised SFT label"
            )
        return perturb_mask, source_ids

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        model = kwargs.get("model")
        if model is None and args:
            model = args[0]
        if not isinstance(model, nn.Module):
            raise TypeError("VAT trainer requires a torch model")
        contract_path = os.environ.get("VAT_CONTRACT_OUTPUT")
        if contract_path:
            # Validate before model patching, Trainer construction, CUDA cache
            # reset, or any training/probe work.
            launch_execution_id = _required_lowercase_hex_environment(
                LAUNCH_EXECUTION_ID_ENV,
                length=32,
            )
            run_config_sha256 = _required_lowercase_hex_environment(
                RUN_CONFIG_SHA256_ENV,
                length=64,
            )
        else:
            launch_execution_id = None
            run_config_sha256 = None
        install_compact_assistant_forward(model)
        self._semantic_margin_artifact: SemanticMarginArtifact | None = None
        self._semantic_margin_device_tensors: dict[str, Tensor] = {}
        if self.vat_runtime_config.semantic_margin_manifest is not None:
            embedding_layer = _underlying_module(model).get_input_embeddings()
            embedding_shape = tuple(int(value) for value in embedding_layer.weight.shape)
            if len(embedding_shape) != 2:
                raise ValueError("input embedding weight must have shape [V, H]")
            self._semantic_margin_artifact = load_semantic_margin_artifact(
                self.vat_runtime_config.semantic_margin_manifest,
                expected_vocabulary_size=embedding_shape[0],
                expected_base_model=self.vat_runtime_config.base_model,
                expected_embedding_shape=embedding_shape,
            )
        self._vat_metric_sums: MutableMapping[str, float] = {}
        self._vat_metric_count = 0
        self._vat_total_training_steps: int | None = None
        self._vat_gradient_telemetry_records: list[dict[str, Any]] = []
        self._vat_gradient_telemetry_seen_steps: set[int] = set()
        self._vat_gradient_telemetry_pending: dict[str, float] | None = None
        self._vat_gradient_similarity_records: list[dict[str, Any]] = []
        self._vat_gradient_similarity_ema = 0.0
        self._vat_gradient_similarity_enabled_count = 0
        self._vat_gradient_similarity_cached_batches: list[
            dict[str, Tensor]
        ] = []
        self._vat_gradient_similarity_rng_states: list[
            tuple[Tensor, Tensor | None]
        ] = []
        self._vat_gradient_similarity_window_initial_rng: (
            tuple[Tensor, Tensor | None] | None
        ) = None
        self._vat_gradient_similarity_analysis_final_rng: (
            tuple[Tensor, Tensor | None] | None
        ) = None
        # Probe-first gating stores only the last Transformer block's two FP32
        # fields on CPU.  Once the current-step gate is known, the complete
        # accumulation window is recomputed and the ordinary full-model
        # backward path receives SFT + lambda_t * VAT directly.
        self._vat_gradient_similarity_sft_grads: list[Tensor] | None = None
        self._vat_gradient_similarity_vat_grads: list[Tensor] | None = None
        self._vat_gradient_similarity_parameter_sha256: str | None = None
        self._vat_gradient_similarity_replay = False
        self._vat_gradient_similarity_analysis = False
        self._vat_gradient_similarity_gate_lambda: float | None = None
        self._random_neighbor_totals: dict[str, int] = {
            "conversation_count": 0,
            "selected_token_count": 0,
            "neighborhood_size_sum": 0,
            "replacement_count": 0,
        }
        super().__init__(*args, **kwargs)
        if self.vat_runtime_config.uses_question_adversary:
            self._vat_zero_dropout_audit = assert_zero_stochastic_dropout(
                _underlying_module(model)
            )
        else:
            self._vat_zero_dropout_audit = None
        set_signature_columns = getattr(self, "_set_signature_columns_if_needed", None)
        if callable(set_signature_columns):
            set_signature_columns()
        signature_columns = getattr(self, "_signature_columns", None)
        if isinstance(signature_columns, list):
            for field in ("perturb_mask", "source_ids"):
                if field not in signature_columns:
                    signature_columns.append(field)

        self._contract_path = contract_path
        self._launch_execution_id = launch_execution_id
        self._run_config_sha256 = run_config_sha256
        self._contract_hash_state = bytes(32)
        self._contract_target_tokens = 0
        self._contract_batches = 0
        self._contract_nan_or_inf = False
        self._contract_memory_samples: list[int] = []
        self._contract_train_started = 0.0
        self._contract_prior_wall_seconds = 0.0
        self._contract_resumed_from: str | None = None
        self._pilot_scheduler_contract: dict[str, int] | None = None
        self._resume_optimizer_audit: dict[str, Any] | None = None
        self._resume_data_audit: dict[str, Any] | None = None
        self._exact_resume_position: ExactResumePosition | None = None
        self._exact_resume_checkpoint: Path | None = None
        self._vat_non_reentrant_checkpointing = False
        self._vat_checkpoint_compatibility_calls = 0
        self._pilot_grad_output = os.environ.get("VAT_PILOT_GRAD_OUTPUT")
        self._pilot_grad_done = False
        self._phase2_diagnostic_output = os.environ.get(
            "VAT_PHASE2_DIAGNOSTIC_OUTPUT"
        )
        self._phase2_input_embeddings_dtype: str | None = None
        self._phase2_locality_integrity: dict[str, int | bool] = {
            "audited_microbatches": 0,
            "masked_groups_valid": True,
            "nominal_zero_outside_mask": True,
            "effective_zero_outside_mask": True,
            "token_group_coverage_consistent": True,
            "component_topk_disabled_during_training": True,
            "bounded_per_group_summary": True,
            "raw_token_values_serialized": False,
            "rho_identity_recomputable_from_group_fields": True,
            "canonical_group_records_emitted": True,
            "independent_and_token_group_norms_match": True,
            "undefined_concentration_serialized_as_null": True,
        }
        self._phase2_microbatch_audit: list[dict[str, int]] = []
        self._phase2_locality: dict[str, list[float]] = {
            "user_embedding_norm": [],
            "target_perturbation_norm": [],
            "nominal_perturbation_norm": [],
            "effective_perturbation_norm": [],
            "effective_r_to_user_embedding_norm": [],
            "nominal_r_to_user_embedding_norm": [],
            "effective_to_nominal_r_norm": [],
            "effective_nonzero_component_fraction": [],
        }
        self._phase2_locality_counts: dict[str, int] = {
            "eligible_user_assistant_groups": 0,
            "eligible_user_tokens": 0,
            "ineligible_user_without_causal_assistant_groups": 0,
            "zero_nominal_eligible_groups": 0,
            "zero_effective_eligible_groups": 0,
        }
        self._phase2_token_locality: dict[str, list[Any]] = {
            # Bounded per-group summaries only. This remains O(groups), not
            # O(tokens * hidden_size), across a 160-step replacement pilot.
            "microbatch_index": [],
            "batch_index": [],
            "group_index": [],
            "group_token_count": [],
            "question_embedding_group_norm": [],
            "effective_perturbation_group_norm": [],
            "effective_token_ratio_p95": [],
            "effective_token_ratio_p99": [],
            "effective_token_ratio_max": [],
            "rho_normalized_effective_token_ratio_p95": [],
            "rho_normalized_effective_token_ratio_p99": [],
            "rho_normalized_effective_token_ratio_max": [],
            "bf16_token_survival_p05": [],
            "bf16_token_survival_median": [],
            "nonzero_effective_token_fraction": [],
            "effective_group_nonzero": [],
            "concentration_defined": [],
            "rho_weighted_energy_identity_lhs": [],
            "rho_weighted_energy_identity_rhs": [],
            "rho_weighted_energy_identity_abs_residual": [],
            "bf16_survival_identity_max_abs_residual": [],
            "effective_energy_fraction_sum_residual": [],
            "token_coverage_residual": [],
            "group_records": [],
        }
        self._phase2_energy_concentration: dict[
            str, list[float | int]
        ] = {
            "microbatch_index": [],
            "batch_index": [],
            "group_index": [],
            "group_token_count": [],
            "max_token_energy_fraction": [],
            "top_10pct_token_energy_fraction": [],
            "normalized_effective_token_count": [],
            "effective_group_nonzero": [],
            "concentration_defined": [],
            "component_concentration_computed": [],
        }
        if self.vat_runtime_config.uses_question_adversary:
            training_args = getattr(self, "args", None)
            requested_checkpointing = bool(
                getattr(training_args, "gradient_checkpointing", False)
            ) or bool(
                getattr(_underlying_module(model), "is_gradient_checkpointing", False)
            )
            if bool(getattr(training_args, "gradient_checkpointing", False)):
                # Transformers enables checkpointing inside _inner_training_loop,
                # after Trainer construction. Set the exact kwargs it will read.
                training_args.gradient_checkpointing_kwargs = {
                    "use_reentrant": False
                }
            if bool(
                getattr(_underlying_module(model), "is_gradient_checkpointing", False)
            ):
                self._vat_non_reentrant_checkpointing = (
                    _enforce_vat_non_reentrant_checkpointing(model)
                )
            self._vat_checkpointing_requested = requested_checkpointing
        else:
            self._vat_checkpointing_requested = False
        self._restore_contract_prefix()
        self._phase2_segment_start_batches = int(self._contract_batches)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _parameter_identity(named: list[tuple[str, nn.Parameter]]) -> str:
        identity = hashlib.sha256()
        for name, parameter in named:
            identity.update(name.encode("utf-8"))
            identity.update(str(tuple(parameter.shape)).encode("ascii"))
            identity.update(str(parameter.dtype).encode("ascii"))
        return identity.hexdigest()

    def _last_transformer_block_parameters(
        self, model: nn.Module
    ) -> tuple[list[str], list[nn.Parameter]]:
        """Return the fixed, trainable last-block parameter vector for probing."""

        causal_lm = _underlying_module(model)
        decoder = getattr(causal_lm, "model", None)
        layers = getattr(decoder, "layers", None)
        if layers is None or not len(layers):
            raise RuntimeError(
                "gradient_similarity requires model.model.layers[-1]"
            )
        last_block_parameter_ids = {
            id(parameter)
            for parameter in layers[-1].parameters()
            if parameter.requires_grad
        }
        named = [
            (name, parameter)
            for name, parameter in causal_lm.named_parameters()
            if parameter.requires_grad and id(parameter) in last_block_parameter_ids
        ]
        if not named:
            raise RuntimeError(
                "gradient_similarity last Transformer block has no trainable "
                "parameters"
            )
        observed = self._parameter_identity(named)
        if self._vat_gradient_similarity_parameter_sha256 is None:
            self._vat_gradient_similarity_parameter_sha256 = observed
        elif self._vat_gradient_similarity_parameter_sha256 != observed:
            raise RuntimeError(
                "gradient_similarity last-block parameter identity changed"
            )
        return [name for name, _ in named], [parameter for _, parameter in named]

    @staticmethod
    def _accumulate_gradient_vector(
        accumulator: list[Tensor] | None,
        gradients: tuple[Tensor | None, ...],
        parameters: list[nn.Parameter],
    ) -> list[Tensor]:
        """Accumulate a last-block probe vector in FP32 CPU storage."""

        if accumulator is None:
            accumulator = [
                torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
                for parameter in parameters
            ]
        if len(accumulator) != len(parameters) or len(gradients) != len(parameters):
            raise RuntimeError("gradient_similarity probe layout changed")
        for index, (target, gradient) in enumerate(
            zip(accumulator, gradients, strict=True)
        ):
            if gradient is None:
                raise RuntimeError(
                    "gradient_similarity probe parameter has no component "
                    f"gradient at index {index}"
                )
            detached = gradient.detach()
            if not bool(torch.isfinite(detached).all()):
                raise FloatingPointError(
                    "gradient_similarity probe contains a non-finite gradient"
                )
            target.add_(detached.to(device="cpu", dtype=torch.float32))
        return accumulator

    def _global_gradient_similarity_gate(
        self,
        *,
        optimizer_step: int,
        training_progress: float,
        effective_rho: float | None = None,
    ) -> float:
        """Select this step's gate from globally summed last-block probes."""

        sft = self._vat_gradient_similarity_sft_grads
        vat = self._vat_gradient_similarity_vat_grads
        if sft is None or vat is None or len(sft) != len(vat):
            raise RuntimeError("gradient_similarity accumulation is incomplete")
        distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        reduction_device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        block_dot = 0.0
        sft_squared = 0.0
        vat_squared = 0.0
        for sft_local, vat_local in zip(sft, vat, strict=True):
            # NCCL cannot reduce CPU tensors, so move one parameter-sized
            # FP32 probe at a time to the current rank's GPU.  Copy the
            # globally summed result back over the existing CPU probe before
            # allocating the VAT transfer.  This keeps the transient device
            # peak to one last-block parameter rather than two probe vectors.
            sft_global = sft_local.to(reduction_device, non_blocking=False)
            if distributed:
                torch.distributed.all_reduce(
                    sft_global, op=torch.distributed.ReduceOp.SUM
                )
            if sft_global.data_ptr() != sft_local.data_ptr():
                sft_local.copy_(sft_global, non_blocking=False)
            del sft_global
            vat_global = vat_local.to(reduction_device, non_blocking=False)
            if distributed:
                torch.distributed.all_reduce(
                    vat_global, op=torch.distributed.ReduceOp.SUM
                )
            if vat_global.data_ptr() != vat_local.data_ptr():
                vat_local.copy_(vat_global, non_blocking=False)
            del vat_global

            # Accumulate statistics in bounded FP64 CPU chunks.  The two
            # persistent tensors remain precisely the FP32 last-block probes;
            # no full-vector FP64 copies are retained.
            sft_flat = sft_local.reshape(-1)
            vat_flat = vat_local.reshape(-1)
            chunk_elements = 1 << 20
            for start in range(0, sft_flat.numel(), chunk_elements):
                stop = min(start + chunk_elements, sft_flat.numel())
                sft_chunk = sft_flat[start:stop].double()
                vat_chunk = vat_flat[start:stop].double()
                block_dot += float(torch.sum(sft_chunk * vat_chunk).item())
                sft_squared += float(torch.sum(sft_chunk * sft_chunk).item())
                vat_squared += float(torch.sum(vat_chunk * vat_chunk).item())
        sft_norm = math.sqrt(sft_squared)
        vat_norm = math.sqrt(vat_squared)
        denominator = sft_norm * vat_norm
        if sft_norm <= 0.0:
            raise FloatingPointError(
                "gradient_similarity requires a non-zero SFT block gradient"
            )
        scheduled_lambda = _gradient_similarity_base_lambda(
            schedule=self.vat_runtime_config.lambda_schedule,
            maximum=self.vat_runtime_config.vat_lambda,
            progress=training_progress,
            delay_fraction=self.vat_runtime_config.lambda_delay_fraction,
            warmup_fraction=self.vat_runtime_config.lambda_warmup_fraction,
        )
        cosine_defined = effective_rho != 0.0 and denominator > 0.0
        inactive_reason: str | None = None
        if effective_rho == 0.0:
            if vat_norm != 0.0 or block_dot != 0.0:
                raise FloatingPointError(
                    "zero-rho gradient-similarity probe must be exactly zero"
                )
            cosine = 0.0
            enabled = False
            inactive_reason = "zero_effective_rho"
        elif cosine_defined:
            cosine = max(-1.0, min(1.0, block_dot / denominator))
            if not math.isfinite(cosine):
                raise FloatingPointError(
                    "gradient_similarity cosine is non-finite"
                )
            enabled = cosine >= 0.0
        else:
            raise FloatingPointError(
                "gradient_similarity requires a non-zero VAT block gradient "
                "whenever the perturbation radius is active"
            )
        effective_lambda = scheduled_lambda if enabled else 0.0
        vat_active = effective_lambda > 0.0
        weighted_ratio = effective_lambda * vat_norm / sft_norm
        decay = self.vat_runtime_config.gradient_similarity_ema_decay
        if cosine_defined:
            self._vat_gradient_similarity_ema = (
                decay * self._vat_gradient_similarity_ema
                + (1.0 - decay) * cosine
            )
        self._vat_gradient_similarity_enabled_count += int(enabled)
        record = {
            "optimizer_step": optimizer_step,
            "training_progress": float(training_progress),
            "gradient_cosine": cosine,
            "gradient_cosine_defined": cosine_defined,
            "gradient_cosine_ema": self._vat_gradient_similarity_ema,
            "gradient_gate_inactive_reason": inactive_reason,
            "effective_vat_rho": effective_rho,
            "scheduled_vat_lambda": scheduled_lambda,
            "effective_vat_lambda": effective_lambda,
            "gradient_gate_enabled": enabled,
            "vat_active": vat_active,
            "vat_enabled": enabled,
            "vat_enabled_count": self._vat_gradient_similarity_enabled_count,
            "vat_enabled_fraction": (
                self._vat_gradient_similarity_enabled_count
                / float(optimizer_step + 1)
            ),
            "sft_gradient_norm": sft_norm,
            "unweighted_vat_gradient_norm": vat_norm,
            "weighted_vat_gradient_norm": effective_lambda * vat_norm,
            "gradient_dot_product": block_dot,
            "weighted_vat_to_sft_ratio": weighted_ratio,
            "scope": GRADIENT_SIMILARITY_SCOPE,
            "update_scope": GRADIENT_SIMILARITY_UPDATE_SCOPE,
            "gradient_accumulation_steps": int(
                self.args.gradient_accumulation_steps
            ),
            "world_size": (
                torch.distributed.get_world_size()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 1
            ),
            "parameter_order_sha256": (
                self._vat_gradient_similarity_parameter_sha256
            ),
            "gradient_aggregation": GRADIENT_SIMILARITY_AGGREGATION,
        }
        self._vat_gradient_similarity_records.append(record)
        self._vat_gradient_telemetry_pending = {
            "vat_gradient_similarity_cosine": cosine,
            "vat_gradient_similarity_cosine_defined": float(cosine_defined),
            "vat_gradient_similarity_ema": self._vat_gradient_similarity_ema,
            "vat_scheduled_lambda": scheduled_lambda,
            "vat_effective_lambda": effective_lambda,
            "vat_active": float(vat_active),
            "vat_enabled_fraction": record["vat_enabled_fraction"],
            "vat_gradient_weighted_vat_to_sft_ratio": weighted_ratio,
        }
        return effective_lambda

    def _process_index(self) -> int:
        accelerator = getattr(self, "accelerator", None)
        return int(getattr(accelerator, "process_index", 0))

    def _world_size(self) -> int:
        accelerator = getattr(self, "accelerator", None)
        accelerator_world_size = int(
            getattr(accelerator, "num_processes", 1)
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            distributed_world_size = torch.distributed.get_world_size()
            if accelerator_world_size not in {1, distributed_world_size}:
                raise RuntimeError(
                    "Accelerate and torch.distributed disagree on world size"
                )
            return distributed_world_size
        return accelerator_world_size

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Any | None = None,
    ) -> Any:
        """Keep staged pilots on the locked uninterrupted 544-step LR curve."""

        requested_steps = int(num_training_steps)
        pilot_total_steps = _pilot_scheduler_total_steps(
            os.environ.get(PILOT_SCHEDULER_TOTAL_STEPS_ENV)
        )
        effective_steps = (
            pilot_total_steps
            if pilot_total_steps is not None
            else requested_steps
        )
        if requested_steps <= 0:
            raise ValueError("scheduler training steps must be positive")
        if pilot_total_steps is not None and requested_steps > pilot_total_steps:
            raise ValueError(
                "pilot endpoint exceeds locked scheduler horizon: "
                f"{requested_steps} > {pilot_total_steps}"
            )
        self._pilot_scheduler_contract = (
            {
                "requested_stage_endpoint": requested_steps,
                "effective_total_steps": effective_steps,
            }
            if pilot_total_steps is not None
            else None
        )
        self._vat_total_training_steps = effective_steps
        return super().create_scheduler(
            effective_steps,
            optimizer=optimizer,
        )

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        """Load upstream state and restore bitsandbytes' paged-state invariant.

        Transformers 4.40.2 loads optimizer tensors directly onto every GPU
        for DDP resumes. bitsandbytes 0.43.1 then retains large Adam moments as
        ordinary CUDA tensors instead of the paged buffers used on a fresh
        run. Re-paging exact state values prevents resume-only OOM without
        changing model, optimizer, scheduler, or RNG values.
        """

        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is not None and torch.cuda.is_available():
            wrapped_optimizer = getattr(self, "optimizer", None)
            optimizer = _underlying_optimizer(wrapped_optimizer)
            converted = _repage_loaded_optimizer_state(optimizer)
            state_count = len(getattr(optimizer, "state", {}))
            configured_optimizer = str(
                getattr(getattr(self, "args", None), "optim", "")
            ).lower()
            expects_paged = "paged" in configured_optimizer
            optimizer_is_paged = bool(getattr(optimizer, "is_paged", False))
            audit_passed = (
                not expects_paged
                or (optimizer_is_paged and state_count > 0 and converted > 0)
            )
            self._resume_optimizer_audit = {
                "configured_optimizer": configured_optimizer,
                "expects_paged": expects_paged,
                "underlying_optimizer": (
                    f"{type(optimizer).__module__}.{type(optimizer).__name__}"
                ),
                "optimizer_is_paged": optimizer_is_paged,
                "state_count": state_count,
                "repaged_tensors": converted,
                "passed": audit_passed,
            }
            print(
                "VAT resume optimizer audit: "
                f"rank={self._process_index()} "
                f"wrapped={type(wrapped_optimizer).__module__}."
                f"{type(wrapped_optimizer).__name__} "
                f"underlying={type(optimizer).__module__}.{type(optimizer).__name__} "
                f"is_paged={optimizer_is_paged} "
                f"state_count={state_count} repaged_tensors={converted} "
                f"passed={audit_passed}",
                flush=True,
            )
            if not audit_passed:
                raise RuntimeError(
                    "paged optimizer resume audit failed: "
                    f"configured_optimizer={configured_optimizer!r}, "
                    f"is_paged={optimizer_is_paged}, state_count={state_count}, "
                    f"repaged_tensors={converted}"
                )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _restore_contract_prefix(self) -> None:
        if not self._contract_path or os.environ.get("VAT_CONTRACT_RESUME") != "1":
            return
        resume_checkpoint = os.environ.get("VAT_RESUME_CHECKPOINT")
        if not resume_checkpoint:
            raise RuntimeError("resume contract requires VAT_RESUME_CHECKPOINT")
        checkpoint = Path(resume_checkpoint).resolve()
        path = checkpoint / ".vat_trainer_contract.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"resume checkpoint lacks cumulative VAT contract: {path}"
            )
        previous = json.loads(path.read_text(encoding="utf-8"))
        trainer_state_path = checkpoint / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise FileNotFoundError(
                f"resume checkpoint lacks trainer state: {trainer_state_path}"
            )
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        recorded_checkpoint = Path(previous.get("final_checkpoint", "")).resolve()
        if recorded_checkpoint != checkpoint:
            raise RuntimeError(
                "resume contract checkpoint differs from requested checkpoint"
            )
        expected_fingerprint = previous.get("final_checkpoint_fingerprint")
        if expected_fingerprint != checkpoint_fingerprint(checkpoint):
            raise RuntimeError("resume checkpoint model/tokenizer fingerprint changed")
        rank = self._process_index()
        ranks = previous.get("ranks", [])
        current_world_size = self._world_size()
        if not isinstance(ranks, list) or len(ranks) != current_world_size:
            raise RuntimeError(
                "resume contract world size differs from current distributed run"
            )
        match = next(
            (entry for entry in ranks if int(entry.get("rank", -1)) == rank),
            None,
        )
        if match is None:
            raise RuntimeError(f"prior contract has no rank {rank} stream")
        current_semantic_margin_identity = self._semantic_margin_identity()
        previous_semantic_margin_identity = {
            name: match.get(name)
            for name in current_semantic_margin_identity
        }
        if previous_semantic_margin_identity != current_semantic_margin_identity:
            raise RuntimeError(
                "resume semantic-margin identity differs from the checkpoint contract"
            )
        rank_batches = [
            int(entry.get("batches", -1)) for entry in previous.get("ranks", [])
        ]
        if int(previous.get("global_step", -1)) != int(
            trainer_state.get("global_step", -2)
        ):
            raise RuntimeError(
                "resume trainer state global_step differs from cumulative contract"
            )
        position = _infer_exact_resume_position(
            global_step=int(trainer_state["global_step"]),
            recorded_epoch=float(trainer_state["epoch"]),
            rank_batches=rank_batches,
            gradient_accumulation_steps=int(
                getattr(self.args, "gradient_accumulation_steps", 0)
            ),
        )
        self._exact_resume_position = position
        self._exact_resume_checkpoint = checkpoint
        self._resume_data_audit = position.as_audit()
        print(
            "VAT exact resume data audit: "
            f"rank={rank} global_step={position.global_step} "
            f"batches={position.batches} "
            f"dataloader_batches={position.dataloader_batches_per_epoch} "
            f"exact_epochs={position.completed_epochs} "
            f"exact_skip={position.batches_in_current_epoch} "
            f"upstream_epochs={position.upstream_completed_epochs} "
            f"upstream_skip={position.upstream_batches_in_current_epoch} "
            f"correction_required={position.correction_required} passed=True",
            flush=True,
        )
        self._contract_hash_state = bytes.fromhex(match["rolling_order_sha256"])
        self._contract_target_tokens = int(match["target_tokens"])
        self._contract_batches = int(match["batches"])
        previous_random = match.get("random_neighbor_totals", {})
        if self.vat_runtime_config.perturbation_mode == "random_neighbor":
            if not isinstance(previous_random, Mapping):
                raise RuntimeError("resume contract lacks random-neighbor totals")
            for name in self._random_neighbor_totals:
                value = int(previous_random.get(name, -1))
                if value < 0:
                    raise RuntimeError(
                        f"resume contract lacks valid random-neighbor {name}"
                    )
                self._random_neighbor_totals[name] = value
        previous_telemetry = match.get("vat_gradient_telemetry", [])
        if not isinstance(previous_telemetry, list):
            raise RuntimeError("resume contract has invalid VAT gradient telemetry")
        restored_records: list[dict[str, Any]] = []
        restored_steps: set[int] = set()
        for record in previous_telemetry:
            if not isinstance(record, Mapping):
                raise RuntimeError(
                    "resume contract has a malformed VAT gradient telemetry record"
                )
            step = int(record.get("optimizer_step", -1))
            if step < 0 or step in restored_steps:
                raise RuntimeError(
                    "resume contract has duplicate/invalid VAT telemetry steps"
                )
            interval = (
                self.vat_runtime_config.gradient_telemetry_interval_steps
            )
            if interval <= 0 or step % interval:
                raise RuntimeError(
                    "resume contract VAT telemetry cadence differs from config"
                )
            if step >= int(trainer_state.get("global_step", -1)):
                raise RuntimeError(
                    "resume contract contains telemetry at/after its checkpoint"
                )
            restored_steps.add(step)
            restored_records.append(dict(record))
        if (
            self.vat_runtime_config.gradient_telemetry_interval_steps > 0
            and int(trainer_state.get("global_step", 0)) > 0
            and not restored_records
        ):
            raise RuntimeError(
                "resume checkpoint lacks required VAT gradient telemetry"
            )
        self._vat_gradient_telemetry_records = restored_records
        self._vat_gradient_telemetry_seen_steps = restored_steps
        if self.vat_runtime_config.uses_gradient_similarity_gate:
            previous_similarity = match.get(
                "vat_gradient_similarity_records", []
            )
            parameter_sha = match.get(
                "vat_gradient_similarity_parameter_order_sha256"
            )
            if not isinstance(parameter_sha, str) or len(parameter_sha) != 64:
                raise RuntimeError(
                    "resume contract lacks gradient-similarity parameter identity"
                )
            has_rho_evidence = bool(previous_similarity) and all(
                isinstance(record, Mapping)
                and "effective_vat_rho" in record
                for record in previous_similarity
            )
            ema, enabled_count = _validate_gradient_similarity_records(
                previous_similarity,
                expected_steps=int(trainer_state.get("global_step", 0)),
                parameter_order_sha256=parameter_sha,
                gradient_accumulation_steps=int(
                    self.args.gradient_accumulation_steps
                ),
                world_size=current_world_size,
                lambda_schedule=self.vat_runtime_config.lambda_schedule,
                vat_lambda=self.vat_runtime_config.vat_lambda,
                lambda_delay_fraction=(
                    self.vat_runtime_config.lambda_delay_fraction
                ),
                lambda_warmup_fraction=(
                    self.vat_runtime_config.lambda_warmup_fraction
                ),
                rho_schedule=(
                    self.vat_runtime_config.rho_schedule
                    if self.vat_runtime_config.rho is not None
                    and has_rho_evidence
                    else None
                ),
                vat_rho=self.vat_runtime_config.rho,
                rho_delay_fraction=self.vat_runtime_config.rho_delay_fraction,
                rho_warmup_fraction=(
                    self.vat_runtime_config.rho_warmup_fraction
                ),
            )
            self._vat_gradient_similarity_ema = ema
            self._vat_gradient_similarity_enabled_count = enabled_count
            self._vat_gradient_similarity_parameter_sha256 = parameter_sha
            self._vat_gradient_similarity_records = [
                dict(record) for record in previous_similarity
            ]
        self._contract_prior_wall_seconds = float(previous.get("wall_seconds", 0.0))

    @staticmethod
    def _tensor_digest(hasher: Any, name: str, tensor: Tensor | None) -> None:
        hasher.update(name.encode("ascii"))
        if tensor is None:
            hasher.update(b"NONE")
            return
        value = tensor.detach().to(device="cpu").contiguous()
        hasher.update(str(tuple(value.shape)).encode("ascii"))
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(value.numpy().tobytes())

    def _track_contract_batch(
        self,
        *,
        input_ids: Tensor,
        labels: Tensor,
        position_ids: Tensor | None,
        source_ids: Tensor | None,
    ) -> None:
        batch = hashlib.sha256()
        self._tensor_digest(batch, "source_ids", source_ids)
        self._tensor_digest(batch, "input_ids", input_ids)
        self._tensor_digest(batch, "labels", labels)
        self._tensor_digest(batch, "position_ids", position_ids)
        self._contract_hash_state = hashlib.sha256(
            self._contract_hash_state + batch.digest()
        ).digest()
        self._contract_target_tokens += int(
            assistant_prediction_mask(labels).sum().item()
        )
        self._contract_batches += 1

    def _track_contract_loss(self, loss: Tensor) -> None:
        if not bool(torch.isfinite(loss.detach()).all()):
            self._contract_nan_or_inf = True
            raise FloatingPointError("training loss contains NaN or Inf")

    def _vat_optimizer_progress(self) -> tuple[int, int, float]:
        state = getattr(self, "state", None)
        global_step = int(getattr(state, "global_step", 0))
        total_steps = self._vat_total_training_steps
        if total_steps is None:
            candidate = int(getattr(state, "max_steps", 0))
            total_steps = candidate if candidate > 0 else None
        if total_steps is None:
            if (
                self.vat_runtime_config.lambda_schedule != "constant"
                or self.vat_runtime_config.rho_schedule != "constant"
            ):
                raise RuntimeError(
                    "scheduled VAT lambda/rho requires the scheduler horizon"
                )
            # Unit-level trainers do not construct a scheduler.  Constant
            # coefficients are horizon-independent, so one synthetic step is
            # sufficient to expose a canonical zero progress value.
            total_steps = 1
        progress = min(1.0, global_step / total_steps)
        return global_step, total_steps, progress

    def _effective_vat_lambda(self) -> tuple[float, float]:
        global_step, total_steps, progress = self._vat_optimizer_progress()
        if self.vat_runtime_config.uses_gradient_similarity_gate:
            if self._vat_gradient_similarity_gate_lambda is None:
                raise RuntimeError(
                    "gradient_similarity compute_loss requires a same-step gate"
                )
            return self._vat_gradient_similarity_gate_lambda, progress
        return (
            self.vat_runtime_config.effective_lambda(
                global_step=global_step,
                total_steps=total_steps,
            ),
            progress,
        )

    def _effective_vat_rho(self) -> tuple[float | None, float]:
        global_step, total_steps, progress = self._vat_optimizer_progress()
        return (
            self.vat_runtime_config.effective_rho(
                global_step=global_step,
                total_steps=total_steps,
            ),
            progress,
        )

    def training_step(
        self,
        model: nn.Module,
        inputs: Mapping[str, Tensor],
    ) -> Tensor:
        """Run exact same-step gradient-similarity gating when requested.

        The first pass stores only separate FP32 SFT/VAT probes for the final
        Transformer block.  Once their globally summed current-step cosine
        selects ``lambda_t``, the complete accumulation window is replayed
        once and ordinary full-model backward receives
        ``SFT + lambda_t * VAT``.  No full-model component gradient buffers
        are retained.
        """

        if not self.vat_runtime_config.uses_gradient_similarity_gate:
            return super().training_step(model, inputs)
        model.train()
        prepared = self._prepare_inputs(inputs)
        if not self._vat_gradient_similarity_cached_batches:
            self._vat_gradient_similarity_window_initial_rng = (
                torch.get_rng_state().clone(),
                (
                    torch.cuda.get_rng_state().clone()
                    if torch.cuda.is_available()
                    else None
                ),
            )
        analysis_rng = (
            torch.get_rng_state().clone(),
            (
                torch.cuda.get_rng_state().clone()
                if torch.cuda.is_available()
                else None
            ),
        )
        cached = {
            key: value.detach().clone() if isinstance(value, Tensor) else value
            for key, value in prepared.items()
        }
        self._vat_gradient_similarity_cached_batches.append(cached)
        self._vat_gradient_similarity_rng_states.append(
            analysis_rng
        )
        _, probe_parameters = self._last_transformer_block_parameters(model)
        self._vat_gradient_similarity_analysis = True
        probe_model = _underlying_module(model)
        try:
            # ``autograd.grad`` does not populate parameter ``.grad`` and is
            # not delegated to DDP's reducer.  Run the probe on the underlying
            # causal-LM module, then aggregate only the completed last-block
            # FP32 vectors below.  The update replay still uses the DDP model.
            with self.compute_loss_context_manager():
                analysis = self.compute_loss(
                    probe_model, prepared, return_outputs=True
                )
            if not isinstance(analysis, tuple) or not isinstance(
                analysis[1], Mapping
            ):
                raise RuntimeError(
                    "gradient_similarity analysis requires component losses"
                )
            outputs = analysis[1]
            sft_loss = outputs.get("sft_loss")
            vat_loss = outputs.get("vat_loss")
            if not isinstance(sft_loss, Tensor) or not isinstance(
                vat_loss, Tensor
            ):
                raise RuntimeError(
                    "gradient_similarity analysis lacks SFT/VAT component "
                    "losses"
                )
            sft_probe_gradients = torch.autograd.grad(
                sft_loss,
                probe_parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            self._vat_gradient_similarity_sft_grads = (
                self._accumulate_gradient_vector(
                    self._vat_gradient_similarity_sft_grads,
                    sft_probe_gradients,
                    probe_parameters,
                )
            )
            del sft_probe_gradients
            analysis_rho, _ = self._effective_vat_rho()
            if analysis_rho == 0.0:
                # The VAT objective has no direction at radius zero.  Avoid a
                # numerically meaningless derivative of KL(p || p) and store
                # an exact zero probe for the schedule-inactive step.
                vat_probe_gradients = tuple(
                    torch.zeros_like(parameter)
                    for parameter in probe_parameters
                )
            else:
                vat_probe_gradients = torch.autograd.grad(
                    vat_loss,
                    probe_parameters,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
            self._vat_gradient_similarity_vat_grads = (
                self._accumulate_gradient_vector(
                    self._vat_gradient_similarity_vat_grads,
                    vat_probe_gradients,
                    probe_parameters,
                )
            )
        finally:
            self._vat_gradient_similarity_analysis = False
        zero_loss = analysis[0].detach().new_zeros(())
        del (
            vat_probe_gradients,
            sft_loss,
            vat_loss,
            outputs,
            analysis,
        )
        accumulation_steps = int(self.args.gradient_accumulation_steps)
        cached_count = len(self._vat_gradient_similarity_cached_batches)
        if cached_count < accumulation_steps:
            return zero_loss
        if cached_count != accumulation_steps:
            raise RuntimeError(
                "gradient_similarity cached more than one accumulation window"
            )
        self._vat_gradient_similarity_analysis_final_rng = (
            torch.get_rng_state().clone(),
            (
                torch.cuda.get_rng_state().clone()
                if torch.cuda.is_available()
                else None
            ),
        )
        step, _, progress = self._vat_optimizer_progress()
        effective_rho, rho_progress = self._effective_vat_rho()
        if not math.isclose(
            rho_progress, progress, rel_tol=0.0, abs_tol=0.0
        ):
            raise RuntimeError(
                "gradient-similarity rho and lambda used different progress"
            )
        effective_lambda = self._global_gradient_similarity_gate(
            optimizer_step=step,
            training_progress=progress,
            effective_rho=effective_rho,
        )
        self._vat_gradient_similarity_gate_lambda = effective_lambda
        replay_losses: list[Tensor] = []
        self._vat_gradient_similarity_replay = True
        try:
            if any(
                parameter.grad is not None
                for parameter in _underlying_module(model).parameters()
            ):
                raise RuntimeError(
                    "gradient_similarity probe pass unexpectedly populated "
                    "full-model .grad"
                )
            for index, (replay_batch, rng_states) in enumerate(
                zip(
                    self._vat_gradient_similarity_cached_batches,
                    self._vat_gradient_similarity_rng_states,
                    strict=True,
                )
            ):
                torch.set_rng_state(rng_states[0])
                if rng_states[1] is not None:
                    torch.cuda.set_rng_state(rng_states[1])
                no_sync = getattr(model, "no_sync", None)
                sync_context = (
                    no_sync()
                    if callable(no_sync) and index < accumulation_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    with self.compute_loss_context_manager():
                        replay_loss = self.compute_loss(
                            model, replay_batch, return_outputs=False
                        )
                    if not isinstance(replay_loss, Tensor):
                        raise RuntimeError(
                            "gradient_similarity replay lacks combined loss"
                        )
                    if self.args.n_gpu > 1:
                        replay_loss = replay_loss.mean()
                    if self.use_apex:
                        raise RuntimeError(
                            "gradient_similarity does not support Apex AMP"
                        )
                    self.accelerator.backward(replay_loss)
                replay_losses.append(replay_loss.detach())
            total_for_logging = torch.stack(replay_losses).sum() / accumulation_steps
        finally:
            self._vat_gradient_similarity_replay = False
            self._vat_gradient_similarity_gate_lambda = None
            self._vat_gradient_similarity_cached_batches.clear()
            self._vat_gradient_similarity_rng_states.clear()
            self._vat_gradient_similarity_sft_grads = None
            self._vat_gradient_similarity_vat_grads = None
            self._vat_gradient_similarity_window_initial_rng = None
        final_rng = self._vat_gradient_similarity_analysis_final_rng
        self._vat_gradient_similarity_analysis_final_rng = None
        if final_rng is not None:
            torch.set_rng_state(final_rng[0])
            if final_rng[1] is not None:
                torch.cuda.set_rng_state(final_rng[1])
        return total_for_logging

    def _maybe_record_gradient_telemetry(
        self,
        *,
        sft_loss: Tensor,
        vat_loss: Tensor,
        input_embeddings: Tensor,
        effective_lambda: float,
        effective_rho: float | None,
        training_progress: float,
    ) -> None:
        """Measure SFT/VAT interaction without changing the outer objective."""

        interval = self.vat_runtime_config.gradient_telemetry_interval_steps
        if interval <= 0:
            return
        step = int(getattr(getattr(self, "state", None), "global_step", 0))
        if step % interval or step in self._vat_gradient_telemetry_seen_steps:
            return

        sft_gradient = torch.autograd.grad(
            sft_loss,
            input_embeddings,
            retain_graph=True,
            create_graph=False,
            only_inputs=True,
        )[0].float()
        weighted_vat_loss = effective_lambda * vat_loss
        weighted_vat_gradient = torch.autograd.grad(
            weighted_vat_loss,
            input_embeddings,
            retain_graph=True,
            create_graph=False,
            only_inputs=True,
        )[0].float()
        local_statistics = torch.stack(
            (
                torch.sum(sft_gradient * sft_gradient),
                torch.sum(weighted_vat_gradient * weighted_vat_gradient),
                torch.sum(sft_gradient * weighted_vat_gradient),
            )
        ).to(dtype=torch.float64)
        if not bool(torch.isfinite(local_statistics).all()):
            raise FloatingPointError(
                "VAT gradient telemetry contains a non-finite local statistic"
            )
        global_statistics = local_statistics.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                global_statistics,
                op=torch.distributed.ReduceOp.SUM,
            )
        if not bool(torch.isfinite(global_statistics).all()):
            raise FloatingPointError(
                "VAT gradient telemetry contains a non-finite global statistic"
            )

        sft_squared, vat_squared, gradient_dot = (
            float(value.item()) for value in global_statistics
        )
        sft_norm = math.sqrt(max(0.0, sft_squared))
        weighted_vat_norm = math.sqrt(max(0.0, vat_squared))
        ratio = weighted_vat_norm / sft_norm if sft_norm > 0.0 else None
        denominator = sft_norm * weighted_vat_norm
        cosine = (
            max(-1.0, min(1.0, gradient_dot / denominator))
            if denominator > 0.0
            else None
        )
        record: dict[str, Any] = {
            "optimizer_step": step,
            "training_progress": float(training_progress),
            "effective_vat_lambda": float(effective_lambda),
            "effective_vat_rho": (
                float(effective_rho) if effective_rho is not None else None
            ),
            "sft_gradient_norm": sft_norm,
            "weighted_vat_gradient_norm": weighted_vat_norm,
            "weighted_vat_to_sft_ratio": ratio,
            "gradient_cosine": cosine,
            "scope": (
                "global concatenation across DDP ranks of shared "
                "input-embedding activation gradients"
            ),
            "local_sft_gradient_squared_norm": float(
                local_statistics[0].item()
            ),
            "local_weighted_vat_gradient_squared_norm": float(
                local_statistics[1].item()
            ),
            "local_gradient_dot_product": float(local_statistics[2].item()),
        }
        self._vat_gradient_telemetry_records.append(record)
        self._vat_gradient_telemetry_seen_steps.add(step)
        pending = {
            "vat_gradient_measurement_step": float(step),
            "vat_gradient_sft_norm": sft_norm,
            "vat_gradient_weighted_vat_norm": weighted_vat_norm,
        }
        if ratio is not None:
            pending["vat_gradient_weighted_vat_to_sft_ratio"] = ratio
        if cosine is not None:
            pending["vat_gradient_cosine"] = cosine
        self._vat_gradient_telemetry_pending = pending
        del sft_gradient, weighted_vat_gradient

    def _write_pilot_gradient_diagnostic(
        self,
        *,
        sft_loss: Tensor,
        vat_loss: Tensor,
        input_embeddings: Tensor,
    ) -> None:
        if not self._pilot_grad_output or self._pilot_grad_done:
            return
        ce_gradient = torch.autograd.grad(
            sft_loss,
            input_embeddings,
            retain_graph=True,
            create_graph=False,
            only_inputs=True,
        )[0]
        weighted_vat = self.vat_runtime_config.vat_lambda * vat_loss
        vat_gradient = torch.autograd.grad(
            weighted_vat,
            input_embeddings,
            retain_graph=True,
            create_graph=False,
            only_inputs=True,
        )[0]
        ce_norm = float(torch.linalg.vector_norm(ce_gradient.float()).item())
        vat_norm = float(torch.linalg.vector_norm(vat_gradient.float()).item())
        ratio = vat_norm / ce_norm if ce_norm > 0 else float("inf")
        local = {
            "rank": self._process_index(),
            "ce_gradient_norm": ce_norm,
            "weighted_vat_gradient_norm": vat_norm,
            "vat_to_ce_ratio": ratio,
        }
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ranks: list[Any] = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(ranks, local)
        else:
            ranks = [local]

        if not all(
            math.isfinite(float(entry["vat_to_ce_ratio"]))
            and float(entry["vat_to_ce_ratio"]) > 0.0
            for entry in ranks
        ):
            raise FloatingPointError(
                "first-batch VAT/CE activation ratio is zero or nonfinite"
            )
        is_world_zero = getattr(self, "is_world_process_zero", lambda: True)
        if is_world_zero():
            path = Path(self._pilot_grad_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 2,
                "measurement_provenance": self._measurement_provenance(
                    "first_batch_of_segment"
                ),
                "scope": (
                    "shared input-embedding activation on the first batch of "
                    "the requested training segment"
                ),
                "decision_role": (
                    "finite_nonzero_diagnostic_only; the raw parameter-norm "
                    "ratio is not an upper safety or dominance claim"
                ),
                "vat_lambda": self.vat_runtime_config.vat_lambda,
                "ranks": ranks,
            }
            encoded = (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            try:
                with path.open("x", encoding="utf-8") as handle:
                    handle.write(encoded)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"refuse to overwrite Phase 2 activation diagnostic: {path}"
                ) from exc
        self._pilot_grad_done = True

    def _measurement_provenance(self, scope: str) -> dict[str, Any]:
        """Record the readable arm and endpoint for one runtime measurement."""

        arm = os.environ.get(PHASE2_ARM_ENV)
        if not isinstance(arm, str) or not arm or arm.strip() != arm:
            raise RuntimeError(f"{PHASE2_ARM_ENV} must be one nonempty arm name")
        raw_endpoint = os.environ.get(PHASE2_REQUESTED_ENDPOINT_ENV)
        try:
            endpoint = int(raw_endpoint or "")
        except ValueError as exc:
            raise RuntimeError(
                f"{PHASE2_REQUESTED_ENDPOINT_ENV} must be a positive integer"
            ) from exc
        if str(endpoint) != raw_endpoint or endpoint <= 0:
            raise RuntimeError(
                f"{PHASE2_REQUESTED_ENDPOINT_ENV} must use canonical positive "
                "integer syntax"
            )
        launch_execution_id = self._launch_execution_id
        config_sha256 = self._run_config_sha256
        if launch_execution_id is None or config_sha256 is None:
            raise RuntimeError(
                "Phase 2 measurements require VAT_CONTRACT_OUTPUT and exact "
                "launch/config identities"
            )
        return {
            "arm": arm,
            "config_sha256": config_sha256,
            "launch_execution_id": launch_execution_id,
            "requested_stage_endpoint": endpoint,
            "measurement_scope": scope,
        }

    def _phase2_contract_provenance(self) -> dict[str, Any] | None:
        names = (
            PHASE2_ARM_ENV,
            PHASE2_REQUESTED_ENDPOINT_ENV,
        )
        present = [os.environ.get(name) is not None for name in names]
        if not any(present):
            return None
        if not all(present):
            missing = [
                name for name, is_present in zip(names, present, strict=True)
                if not is_present
            ]
            raise RuntimeError(
                f"incomplete Phase 2 launch provenance; missing {missing}"
            )
        return self._measurement_provenance("training_segment")

    def _record_phase2_locality(
        self,
        *,
        input_embeddings: Tensor,
        perturbed_embeddings: Tensor,
        nominal_perturbation: Tensor,
        perturb_mask: Tensor,
        prediction_mask: Tensor,
        group_ids: Tensor | None,
        position_ids: Tensor | None,
        effective_rho: float | None = None,
    ) -> None:
        if not self._phase2_diagnostic_output:
            return
        # Preserve the direct diagnostic-call contract used by the existing
        # constant-rho runs and tests.  Scheduled runs always pass the
        # optimizer-step value explicitly (including an exact 0.0 at step 0).
        if (
            effective_rho is None
            and self.vat_runtime_config.radius_mode == "relative"
        ):
            effective_rho = self.vat_runtime_config.rho
        if input_embeddings.dtype != torch.bfloat16:
            raise TypeError(
                "replacement locality diagnostics require BF16 model input "
                f"embeddings, got {input_embeddings.dtype}"
            )
        observed_dtype = str(input_embeddings.dtype)
        if (
            self._phase2_input_embeddings_dtype is not None
            and self._phase2_input_embeddings_dtype != observed_dtype
        ):
            raise RuntimeError(
                "input embedding dtype changed within Phase 2 diagnostics"
            )
        self._phase2_input_embeddings_dtype = observed_dtype
        mask = perturb_mask.detach().to(dtype=torch.bool)
        if group_ids is None:
            if position_ids is not None:
                resolved_group_ids = relative_vat_group_ids(position_ids)
            else:
                resolved_group_ids = (
                    torch.arange(
                        mask.shape[0], device=mask.device, dtype=torch.long
                    )
                    .unsqueeze(1)
                    .expand_as(mask)
                )
        else:
            resolved_group_ids = group_ids
        user_norms, user_keys = masked_frobenius_norms(
            input_embeddings.detach(),
            mask,
            group_ids=resolved_group_ids,
        )
        effective_norms, effective_keys = masked_difference_frobenius_norms(
            perturbed_embeddings.detach(),
            input_embeddings.detach(),
            mask,
            group_ids=resolved_group_ids,
        )
        nominal_norms, nominal_keys = masked_frobenius_norms(
            nominal_perturbation.detach(),
            mask,
            group_ids=resolved_group_ids,
        )
        if not (
            torch.equal(user_keys, effective_keys)
            and torch.equal(user_keys, nominal_keys)
        ):
            raise RuntimeError("Phase 2 locality group keys differ")
        # A correctly isolated packed group with user tokens but no assistant
        # target has no local KL signal and may legitimately produce g=0.  It
        # is recorded separately rather than being misclassified as a failed
        # rho construction or BF16 collapse.
        _, assistant_keys = masked_frobenius_norms(
            input_embeddings.detach(),
            prediction_mask.detach().to(dtype=torch.bool),
            group_ids=resolved_group_ids,
        )
        assistant_key_set = {
            (int(key[0]), int(key[1])) for key in assistant_keys.cpu().tolist()
        }
        mask_cpu = mask.cpu()
        prediction_cpu = prediction_mask.detach().to(
            device="cpu", dtype=torch.bool
        )
        group_cpu = resolved_group_ids.cpu()
        eligible = torch.tensor(
            [
                (
                    (int(key[0]), int(key[1])) in assistant_key_set
                    and int(
                        torch.nonzero(
                            prediction_cpu[int(key[0])]
                            & group_cpu[int(key[0])].eq(int(key[1])),
                            as_tuple=False,
                        )[-1, 0]
                    )
                    >= int(
                        torch.nonzero(
                            mask_cpu[int(key[0])]
                            & group_cpu[int(key[0])].eq(int(key[1])),
                            as_tuple=False,
                        )[0, 0]
                    )
                )
                for key in user_keys.cpu().tolist()
            ],
            device=user_norms.device,
            dtype=torch.bool,
        )
        self._phase2_locality_counts[
            "eligible_user_assistant_groups"
        ] += int(eligible.sum().item())
        self._phase2_locality_counts[
            "ineligible_user_without_causal_assistant_groups"
        ] += int((~eligible).sum().item())
        token_stats = token_locality_statistics(
            input_embeddings.detach(),
            nominal_perturbation.detach(),
            perturbed_embeddings.detach(),
            mask,
            group_ids=resolved_group_ids,
            rho=(effective_rho if effective_rho and effective_rho > 0.0 else None),
            compute_component_energy_concentration=False,
        )
        if not torch.equal(token_stats["group_keys"], user_keys):
            raise RuntimeError("Phase 2 token-locality group keys differ")
        if bool(token_stats["token_coverage_residual"].ne(0).any()):
            raise RuntimeError("Phase 2 token-locality coverage identity failed")
        if not torch.equal(
            token_stats["effective_group_nonzero"],
            effective_norms.ne(0),
        ):
            raise RuntimeError(
                "Phase 2 token/group effective-norm coverage differs"
            )
        if not (
            torch.allclose(
                token_stats["question_embedding_group_norm"],
                user_norms,
                rtol=RHO_WEIGHTED_ENERGY_IDENTITY_RTOL,
                atol=RHO_WEIGHTED_ENERGY_IDENTITY_ATOL,
            )
            and torch.allclose(
                token_stats["effective_perturbation_group_norm"],
                effective_norms,
                rtol=RHO_WEIGHTED_ENERGY_IDENTITY_RTOL,
                atol=RHO_WEIGHTED_ENERGY_IDENTITY_ATOL,
            )
        ):
            raise RuntimeError(
                "Phase 2 independent and token-locality group norms exceed "
                "the sealed FP32 reduction tolerance"
            )
        if not torch.equal(
            token_stats["concentration_defined"],
            token_stats["effective_group_nonzero"],
        ):
            raise RuntimeError(
                "Phase 2 concentration definition differs from group support"
            )
        if bool(token_stats["component_concentration_computed"].any()):
            raise RuntimeError(
                "training diagnostics must not compute component top-k"
            )
        microbatch_index = int(self._contract_batches)
        eligible_group_count = int(eligible.sum().item())
        expected_microbatch_index = (
            int(self._phase2_segment_start_batches)
            + len(self._phase2_microbatch_audit)
            + 1
        )
        if microbatch_index != expected_microbatch_index:
            raise RuntimeError(
                "Phase 2 microbatch audit is not contiguous with the "
                "cumulative trainer contract"
            )
        self._phase2_microbatch_audit.append(
            {
                "microbatch_index": microbatch_index,
                "eligible_group_count": eligible_group_count,
            }
        )
        self._phase2_locality_integrity["audited_microbatches"] = int(
            self._phase2_locality_integrity["audited_microbatches"]
        ) + 1
        if not bool(eligible.any()):
            return
        independent_user_norms = user_norms
        independent_effective_norms = effective_norms
        user_norms = user_norms[eligible]
        effective_norms = effective_norms[eligible]
        nominal_norms = nominal_norms[eligible]
        eligible_keys = user_keys[eligible]
        nonzero_component_fractions: list[float] = []
        for batch_index, group_index in eligible_keys.cpu().tolist():
            selected = (
                resolved_group_ids[int(batch_index)].eq(int(group_index))
                & mask[int(batch_index)]
            )
            selected_values = (
                perturbed_embeddings.detach()[int(batch_index)][selected].float()
                - input_embeddings.detach()[int(batch_index)][selected].float()
            )
            if selected_values.numel() == 0:
                raise RuntimeError("eligible locality group unexpectedly empty")
            nonzero_component_fractions.append(
                float(selected_values.ne(0).float().mean().item())
            )
        token_count = int(
            token_stats["group_token_count"][eligible].sum().item()
        )
        self._phase2_locality_counts["eligible_user_tokens"] += token_count
        self._phase2_locality_counts[
            "zero_nominal_eligible_groups"
        ] += int(nominal_norms.eq(0).sum().item())
        self._phase2_locality_counts[
            "zero_effective_eligible_groups"
        ] += int(effective_norms.eq(0).sum().item())
        user_ratios = torch.where(
            user_norms > 0,
            effective_norms / user_norms,
            torch.zeros_like(user_norms),
        )
        nominal_user_ratios = torch.where(
            user_norms > 0,
            nominal_norms / user_norms,
            torch.zeros_like(user_norms),
        )
        effective_nominal_ratios = torch.where(
            nominal_norms > 0,
            effective_norms / nominal_norms,
            torch.zeros_like(nominal_norms),
        )
        if not (
            torch.isfinite(user_ratios).all()
            and torch.isfinite(nominal_user_ratios).all()
            and torch.isfinite(effective_nominal_ratios).all()
        ):
            raise FloatingPointError("Phase 2 locality contains NaN or Inf")
        target_norms = (
            user_norms * float(effective_rho)
            if effective_rho is not None
            else torch.full_like(
                user_norms,
                float(self.vat_runtime_config.epsilon or 0.0),
            )
        )
        self._phase2_locality["user_embedding_norm"].extend(
            user_norms.cpu().tolist()
        )
        self._phase2_locality["target_perturbation_norm"].extend(
            target_norms.cpu().tolist()
        )
        self._phase2_locality["nominal_perturbation_norm"].extend(
            nominal_norms.cpu().tolist()
        )
        self._phase2_locality["effective_perturbation_norm"].extend(
            effective_norms.cpu().tolist()
        )
        self._phase2_locality[
            "effective_r_to_user_embedding_norm"
        ].extend(user_ratios.cpu().tolist())
        self._phase2_locality[
            "nominal_r_to_user_embedding_norm"
        ].extend(nominal_user_ratios.cpu().tolist())
        self._phase2_locality[
            "effective_to_nominal_r_norm"
        ].extend(effective_nominal_ratios.cpu().tolist())
        self._phase2_locality[
            "effective_nonzero_component_fraction"
        ].extend(nonzero_component_fractions)
        group_keys = token_stats["group_keys"][eligible].cpu()
        group_count = int(group_keys.shape[0])
        self._phase2_token_locality["microbatch_index"].extend(
            [microbatch_index] * group_count
        )
        self._phase2_token_locality["batch_index"].extend(
            group_keys[:, 0].tolist()
        )
        self._phase2_token_locality["group_index"].extend(
            group_keys[:, 1].tolist()
        )
        self._phase2_token_locality["group_token_count"].extend(
            token_stats["group_token_count"][eligible].cpu().tolist()
        )
        for name in (
            "question_embedding_group_norm",
            "effective_perturbation_group_norm",
            "effective_token_ratio_p95",
            "effective_token_ratio_p99",
            "effective_token_ratio_max",
            "bf16_token_survival_p05",
            "bf16_token_survival_median",
            "nonzero_effective_token_fraction",
            "bf16_survival_identity_max_abs_residual",
            "token_coverage_residual",
        ):
            self._phase2_token_locality[name].extend(
                token_stats[name][eligible].cpu().tolist()
            )
        for name in (
            "rho_normalized_effective_token_ratio_p95",
            "rho_normalized_effective_token_ratio_p99",
            "rho_normalized_effective_token_ratio_max",
            "rho_weighted_energy_identity_lhs",
            "rho_weighted_energy_identity_rhs",
            "rho_weighted_energy_identity_abs_residual",
        ):
            self._phase2_token_locality[name].extend(
                [
                    float(value) if math.isfinite(float(value)) else None
                    for value in token_stats[name][eligible].cpu().tolist()
                ]
            )
        for name in ("effective_group_nonzero", "concentration_defined"):
            self._phase2_token_locality[name].extend(
                token_stats[name][eligible].cpu().tolist()
            )
        energy_identity = token_stats[
            "effective_energy_fraction_sum_residual"
        ][eligible].cpu()
        concentration_defined = token_stats["concentration_defined"][
            eligible
        ].cpu()
        self._phase2_token_locality[
            "effective_energy_fraction_sum_residual"
        ].extend(
            [
                float(value) if bool(defined) else None
                for value, defined in zip(
                    energy_identity.tolist(),
                    concentration_defined.tolist(),
                    strict=True,
                )
            ]
        )
        self._phase2_energy_concentration["microbatch_index"].extend(
            [microbatch_index] * group_count
        )
        self._phase2_energy_concentration["batch_index"].extend(
            group_keys[:, 0].tolist()
        )
        self._phase2_energy_concentration["group_index"].extend(
            group_keys[:, 1].tolist()
        )
        self._phase2_energy_concentration["group_token_count"].extend(
            token_stats["group_token_count"][eligible].cpu().tolist()
        )
        for name in (
            "effective_group_nonzero",
            "concentration_defined",
            "component_concentration_computed",
        ):
            self._phase2_energy_concentration[name].extend(
                token_stats[name][eligible].cpu().tolist()
            )
        for name in (
            "max_token_energy_fraction",
            "top_10pct_token_energy_fraction",
            "normalized_effective_token_count",
        ):
            values = token_stats[name][eligible].cpu().tolist()
            self._phase2_energy_concentration[name].extend(
                [
                    float(value) if bool(defined) else None
                    for value, defined in zip(
                        values,
                        concentration_defined.tolist(),
                        strict=True,
                    )
                ]
            )
        self._phase2_token_locality["group_records"].extend(
            locality_group_records(
                token_stats,
                microbatch_index=microbatch_index,
                selection=eligible,
                independent_user_embedding_norm=independent_user_norms,
                independent_effective_perturbation_norm=(
                    independent_effective_norms
                ),
            )
        )

    def _write_phase2_diagnostic(self) -> None:
        if not self._phase2_diagnostic_output:
            return
        local = {
            "rank": self._process_index(),
            "input_embeddings_dtype": self._phase2_input_embeddings_dtype,
            "integrity": self._phase2_locality_integrity,
            "microbatch_audit": self._phase2_microbatch_audit,
            "locality": self._phase2_locality,
            "token_locality": self._phase2_token_locality,
            "energy_concentration": self._phase2_energy_concentration,
            "counts": self._phase2_locality_counts,
        }
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ranks: list[Any] = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(ranks, local)
        else:
            ranks = [local]
        is_world_zero = getattr(self, "is_world_process_zero", lambda: True)
        if not is_world_zero():
            return
        input_embedding_dtypes = {
            rank.get("input_embeddings_dtype") for rank in ranks
        }
        if input_embedding_dtypes != {"torch.bfloat16"}:
            raise RuntimeError(
                "replacement locality diagnostics require BF16 embeddings "
                f"on every rank, got {sorted(str(value) for value in input_embedding_dtypes)}"
            )
        path = Path(self._phase2_diagnostic_output)
        payload = {
            "schema_version": 5,
            "measurement_provenance": self._measurement_provenance(
                "all_microbatches_in_segment"
            ),
            "replacement_locality_schema": {
                "version": 5,
                "representation": "bounded_per_group",
                "canonical_group_key": [
                    "microbatch_index",
                    "batch_index",
                    "group_index",
                ],
                "canonical_group_records_field": "group_records",
                "rho_identity": "embedding_energy_weighted_token_c_eff_squared",
                "rho_identity_fields": [
                    "question_embedding_group_norm",
                    "effective_perturbation_group_norm",
                    "rho_weighted_energy_identity_lhs",
                    "rho_weighted_energy_identity_rhs",
                    "rho_weighted_energy_identity_abs_residual",
                ],
                "legacy_rho_identity_max_abs_residual_emitted": False,
                "rho_identity_tolerances": {
                    "relative_tolerance": (
                        RHO_WEIGHTED_ENERGY_IDENTITY_RTOL
                    ),
                    "absolute_tolerance": (
                        RHO_WEIGHTED_ENERGY_IDENTITY_ATOL
                    ),
                },
                "undefined_concentration_json_value": None,
                "training_component_topk_computed": False,
            },
            "parameterization": (
                "relative_question_frobenius_token_density_scaled"
                if self.vat_runtime_config.density_mode
                == "vocab_radius_count"
                else (
                    "relative_question_frobenius"
                    if self.vat_runtime_config.radius_mode == "relative"
                    else "absolute_frobenius"
                )
            ),
            "scope": (
                "actual BF16 perturbation locality on every pilot microbatch; "
                "adversarial/random controls run in a standalone no-optimizer "
                "probe to avoid increasing training peak memory"
            ),
            "vat_xi": self.vat_runtime_config.xi,
            "vat_xi_mode": self.vat_runtime_config.xi_mode,
            "vat_epsilon": self.vat_runtime_config.epsilon,
            "vat_rho": self.vat_runtime_config.rho,
            "radius_mode": self.vat_runtime_config.radius_mode,
            "gradient_norm_delta": (
                RELATIVE_GRADIENT_NORM_DELTA
                if self.vat_runtime_config.radius_mode == "relative"
                else 0.0
            ),
            "vat_lambda": self.vat_runtime_config.vat_lambda,
            "power_iterations": self.vat_runtime_config.power_iterations,
            "vat_density_mode": self.vat_runtime_config.density_mode,
            "vat_density_vocabulary_chunk_size": (
                self.vat_runtime_config.density_vocabulary_chunk_size
            ),
            "vat_semantic_margin_manifest": (
                self.vat_runtime_config.semantic_margin_manifest
            ),
            "semantic_margin_manifest_sha256": (
                self._semantic_margin_artifact.manifest_sha256
                if self._semantic_margin_artifact is not None
                else None
            ),
            "semantic_margin_values_sha256": (
                self._semantic_margin_artifact.margins_sha256
                if self._semantic_margin_artifact is not None
                else None
            ),
            "semantic_margin_embedding_shard_sha256": (
                self._semantic_margin_artifact.embedding_shard_sha256
                if self._semantic_margin_artifact is not None
                else None
            ),
            "semantic_margin_covered_token_count": (
                self._semantic_margin_artifact.covered_token_count
                if self._semantic_margin_artifact is not None
                else None
            ),
            "input_embeddings_dtype": "torch.bfloat16",
            "ranks": ranks,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refuse to overwrite Phase 2 diagnostic: {path}"
            ) from exc

    def _semantic_margin_identity(self) -> dict[str, Any]:
        artifact = self._semantic_margin_artifact
        return {
            "vat_semantic_margin_manifest": (
                str(artifact.manifest_path)
                if artifact is not None
                else self.vat_runtime_config.semantic_margin_manifest
            ),
            "semantic_margin_manifest_sha256": (
                artifact.manifest_sha256 if artifact is not None else None
            ),
            "semantic_margin_values_sha256": (
                artifact.margins_sha256 if artifact is not None else None
            ),
            "semantic_margin_embedding_shard_sha256": (
                artifact.embedding_shard_sha256 if artifact is not None else None
            ),
            "semantic_margin_covered_token_count": (
                artifact.covered_token_count if artifact is not None else None
            ),
        }

    def _local_contract(self, wall_seconds: float) -> dict[str, Any]:
        samples = self._contract_memory_samples[5:]
        memory_slope = 0.0
        memory_net_growth = 0
        positive_step_fraction = 0.0
        if len(samples) > 1:
            x_mean = (len(samples) - 1) / 2
            y_mean = sum(samples) / len(samples)
            denominator = sum((index - x_mean) ** 2 for index in range(len(samples)))
            memory_slope = (
                sum(
                    (index - x_mean) * (value - y_mean)
                    for index, value in enumerate(samples)
                )
                / denominator
            )
            memory_net_growth = samples[-1] - samples[0]
            meaningful_increases = sum(
                current - previous > 1 * 1024 * 1024
                for previous, current in zip(samples, samples[1:])
            )
            positive_step_fraction = meaningful_increases / (len(samples) - 1)
        sustained_growth = (
            memory_net_growth > 64 * 1024 * 1024 and positive_step_fraction >= 0.8
        )
        return {
            "rank": self._process_index(),
            "launch_execution_id": self._launch_execution_id,
            "config_sha256": self._run_config_sha256,
            "phase2_execution_provenance": (
                self._phase2_contract_provenance()
            ),
            "training_objective": self.vat_runtime_config.training_objective,
            "use_minmax_adversarial_sft": (
                self.vat_runtime_config.use_minmax_adversarial_sft
            ),
            "minmax_rho": self.vat_runtime_config.minmax_rho,
            "rolling_order_sha256": self._contract_hash_state.hex(),
            "target_tokens": self._contract_target_tokens,
            "batches": self._contract_batches,
            "nan_or_inf_detected": self._contract_nan_or_inf,
            "peak_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
            "peak_memory_reserved_bytes": (
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
            ),
            "steady_state_memory_samples": len(samples),
            # Preserve the raw post-warmup series so the promotion gate can
            # independently recompute every slope/growth decision instead of
            # trusting trainer-derived summary fields.
            "steady_state_memory_allocated_bytes": samples,
            "steady_state_memory_regression_slope_bytes_per_optimizer_step": (
                memory_slope
            ),
            "steady_state_memory_net_growth_bytes": memory_net_growth,
            "steady_state_memory_positive_step_fraction": positive_step_fraction,
            "sustained_memory_growth_detected": sustained_growth,
            "resume_optimizer_audit": self._resume_optimizer_audit,
            "resume_data_audit": self._resume_data_audit,
            "pilot_scheduler": self._pilot_scheduler_contract,
            "vat_non_reentrant_gradient_checkpointing": (
                self._vat_non_reentrant_checkpointing
            ),
            "vat_checkpoint_compatibility_calls": (
                self._vat_checkpoint_compatibility_calls
            ),
            "vat_perturbation_mode": (
                self.vat_runtime_config.perturbation_mode
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_density_mode": (
                self.vat_runtime_config.density_mode
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_density_vocabulary_chunk_size": (
                self.vat_runtime_config.density_vocabulary_chunk_size
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_lambda_max": (
                self.vat_runtime_config.vat_lambda
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_lambda_schedule": (
                self.vat_runtime_config.lambda_schedule
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_lambda_warmup_fraction": (
                self.vat_runtime_config.lambda_warmup_fraction
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_lambda_delay_fraction": (
                self.vat_runtime_config.lambda_delay_fraction
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_lambda_cycles": (
                self.vat_runtime_config.lambda_cycles
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_rho_max": (
                self.vat_runtime_config.rho
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_rho_schedule": (
                self.vat_runtime_config.rho_schedule
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_rho_warmup_fraction": (
                self.vat_runtime_config.rho_warmup_fraction
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_rho_delay_fraction": (
                self.vat_runtime_config.rho_delay_fraction
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_gradient_similarity_ema_decay": (
                self.vat_runtime_config.gradient_similarity_ema_decay
                if self.vat_runtime_config.uses_gradient_similarity_gate
                else None
            ),
            "vat_gradient_similarity_parameter_order_sha256": (
                self._vat_gradient_similarity_parameter_sha256
            ),
            "vat_gradient_similarity_records": list(
                self._vat_gradient_similarity_records
            ),
            "vat_gradient_telemetry_interval_steps": (
                self.vat_runtime_config.gradient_telemetry_interval_steps
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_gradient_telemetry": (
                list(self._vat_gradient_telemetry_records)
                if self.vat_runtime_config.use_vat
                else []
            ),
            **self._semantic_margin_identity(),
            "random_neighbor_totals": dict(self._random_neighbor_totals),
            "random_neighbor_average_neighborhood_size": (
                self._random_neighbor_totals["neighborhood_size_sum"]
                / self._random_neighbor_totals["selected_token_count"]
                if self._random_neighbor_totals["selected_token_count"]
                else None
            ),
            "random_neighbor_replacement_rate": (
                self._random_neighbor_totals["replacement_count"]
                / self._random_neighbor_totals["selected_token_count"]
                if self._random_neighbor_totals["selected_token_count"]
                else None
            ),
            "wall_seconds": wall_seconds,
        }

    def _semantic_margins_for(self, device: torch.device) -> Tensor | None:
        artifact = self._semantic_margin_artifact
        if artifact is None:
            return None
        key = str(device)
        result = self._semantic_margin_device_tensors.get(key)
        if result is None:
            result = artifact.margins.to(device=device, dtype=torch.float32)
            self._semantic_margin_device_tensors[key] = result
        return result

    def _write_training_contract(
        self,
        wall_seconds: float,
        *,
        completed: bool,
        checkpoint_override: Path | None = None,
    ) -> None:
        if not self._contract_path:
            return
        cumulative_wall = self._contract_prior_wall_seconds + wall_seconds
        local = self._local_contract(cumulative_wall)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ranks: list[Any] = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(ranks, local)
        else:
            ranks = [local]

        is_world_zero = getattr(self, "is_world_process_zero", lambda: True)
        if not is_world_zero():
            return
        combined_hash = hashlib.sha256()
        for rank in sorted(ranks, key=lambda entry: entry["rank"]):
            combined_hash.update(int(rank["rank"]).to_bytes(4, "big"))
            combined_hash.update(bytes.fromhex(rank["rolling_order_sha256"]))

        output_dir = Path(self.args.output_dir).resolve()
        if checkpoint_override is not None:
            final_checkpoint = checkpoint_override.resolve()
        else:
            checkpoints = sorted(
                output_dir.glob("checkpoint-*"),
                key=lambda item: int(item.name.rsplit("-", 1)[-1]),
            )
            if not checkpoints:
                raise RuntimeError("training contract requires a saved checkpoint")
            final_checkpoint = checkpoints[-1].resolve()
        final_fingerprint = checkpoint_fingerprint(final_checkpoint)
        total_target_tokens = sum(int(rank["target_tokens"]) for rank in ranks)
        scheduler_contracts = [rank["pilot_scheduler"] for rank in ranks]
        if any(value != scheduler_contracts[0] for value in scheduler_contracts):
            raise RuntimeError("distributed ranks used different scheduler horizons")
        launch_execution_ids = {
            rank.get("launch_execution_id") for rank in ranks
        }
        config_sha256_values = {rank.get("config_sha256") for rank in ranks}
        phase2_provenance_values = {
            json.dumps(
                rank.get("phase2_execution_provenance"),
                sort_keys=True,
                separators=(",", ":"),
            )
            for rank in ranks
        }
        if launch_execution_ids != {self._launch_execution_id}:
            raise RuntimeError(
                "distributed ranks used different launch execution IDs"
            )
        if config_sha256_values != {self._run_config_sha256}:
            raise RuntimeError("distributed ranks used different run config SHAs")
        if len(phase2_provenance_values) != 1:
            raise RuntimeError(
                "distributed ranks used different Phase 2 execution provenance"
            )
        phase2_execution_provenance = ranks[0].get(
            "phase2_execution_provenance"
        )
        semantic_margin_identities = {
            json.dumps(
                {
                    name: rank.get(name)
                    for name in self._semantic_margin_identity()
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for rank in ranks
        }
        if len(semantic_margin_identities) != 1:
            raise RuntimeError(
                "distributed ranks used different semantic-margin artifacts"
            )
        semantic_margin_identity = {
            name: ranks[0].get(name)
            for name in self._semantic_margin_identity()
        }
        global_telemetry_fields = (
            "optimizer_step",
            "training_progress",
            "effective_vat_lambda",
            "effective_vat_rho",
            "sft_gradient_norm",
            "weighted_vat_gradient_norm",
            "weighted_vat_to_sft_ratio",
            "gradient_cosine",
            "scope",
        )
        rank_global_telemetry = [
            [
                {name: record.get(name) for name in global_telemetry_fields}
                for record in rank.get("vat_gradient_telemetry", [])
            ]
            for rank in ranks
        ]
        if any(
            records != rank_global_telemetry[0]
            for records in rank_global_telemetry[1:]
        ):
            raise RuntimeError(
                "distributed ranks recorded different global VAT gradient telemetry"
            )
        gradient_telemetry = rank_global_telemetry[0]
        rank_similarity_records = [
            rank.get("vat_gradient_similarity_records", []) for rank in ranks
        ]
        if any(
            records != rank_similarity_records[0]
            for records in rank_similarity_records[1:]
        ):
            raise RuntimeError(
                "distributed ranks recorded different gradient-similarity "
                "decisions"
            )
        gradient_similarity_records = rank_similarity_records[0]
        finite_ratios = [
            float(record["weighted_vat_to_sft_ratio"])
            for record in gradient_telemetry
            if record.get("weighted_vat_to_sft_ratio") is not None
        ]
        contract = {
            "schema_version": 1,
            "launch_execution_id": self._launch_execution_id,
            "config_sha256": self._run_config_sha256,
            "phase2_execution_provenance": phase2_execution_provenance,
            "training_objective": self.vat_runtime_config.training_objective,
            "use_minmax_adversarial_sft": (
                self.vat_runtime_config.use_minmax_adversarial_sft
            ),
            "minmax_rho": self.vat_runtime_config.minmax_rho,
            "semantic_margin_identity": semantic_margin_identity,
            "completed": completed,
            "run_name": getattr(self.args, "run_name", None),
            "sample_order_sha256": combined_hash.hexdigest(),
            "sample_order_definition": (
                "rank-ordered rolling SHA256 over source content ID, packed "
                "input_ids, labels, and position_ids for every consumed batch"
            ),
            "target_tokens": total_target_tokens,
            "global_step": int(self.state.global_step),
            "completed_epochs": float(self.state.epoch or 0.0),
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_fingerprint": final_fingerprint,
            "resumed_from": self._contract_resumed_from,
            "wall_seconds": cumulative_wall,
            "target_tokens_per_second": (
                total_target_tokens / cumulative_wall if cumulative_wall > 0 else None
            ),
            "nan_or_inf_detected": any(
                bool(rank["nan_or_inf_detected"]) for rank in ranks
            ),
            "loss_history": list(self.state.log_history),
            "vat_ce_gradient_ratio": (
                max(finite_ratios) if finite_ratios else None
            ),
            "vat_ce_gradient_ratio_note": (
                "maximum periodic weighted-VAT/SFT norm ratio over the shared "
                "input-embedding activation-gradient scope"
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_lambda_max": (
                self.vat_runtime_config.vat_lambda
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_lambda_schedule": (
                self.vat_runtime_config.lambda_schedule
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_lambda_warmup_fraction": (
                self.vat_runtime_config.lambda_warmup_fraction
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_lambda_delay_fraction": (
                self.vat_runtime_config.lambda_delay_fraction
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_lambda_cycles": (
                self.vat_runtime_config.lambda_cycles
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_gradient_similarity_ema_decay": (
                self.vat_runtime_config.gradient_similarity_ema_decay
                if self.vat_runtime_config.uses_gradient_similarity_gate
                else None
            ),
            "vat_gradient_similarity_parameter_order_sha256": (
                self._vat_gradient_similarity_parameter_sha256
            ),
            "vat_gradient_similarity_records": (
                gradient_similarity_records
            ),
            "vat_gradient_telemetry_interval_steps": (
                self.vat_runtime_config.gradient_telemetry_interval_steps
                if self.vat_runtime_config.use_vat
                else None
            ),
            "vat_gradient_telemetry": gradient_telemetry,
            "pilot_scheduler": scheduler_contracts[0],
            "ranks": ranks,
        }
        path = Path(self._contract_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                contract,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        checkpoint_contract = final_checkpoint / ".vat_trainer_contract.json"
        checkpoint_temporary = checkpoint_contract.with_suffix(".json.tmp")
        checkpoint_temporary.write_text(
            json.dumps(
                contract,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoint_temporary.replace(checkpoint_contract)

    def _save_checkpoint(
        self, model: nn.Module, trial: Any, metrics: dict[str, float] | None = None
    ) -> Any:
        if self.vat_runtime_config.uses_gradient_similarity_gate and (
            self._vat_gradient_similarity_cached_batches
            or self._vat_gradient_similarity_rng_states
            or self._vat_gradient_similarity_sft_grads is not None
            or self._vat_gradient_similarity_vat_grads is not None
            or self._vat_gradient_similarity_analysis
            or self._vat_gradient_similarity_replay
            or self._vat_gradient_similarity_gate_lambda is not None
        ):
            raise RuntimeError(
                "gradient_similarity checkpoint requested with non-empty "
                "optimizer-window gradient buffers"
            )
        result = super()._save_checkpoint(model, trial, metrics=metrics)
        wait_for_everyone = getattr(
            getattr(self, "accelerator", None), "wait_for_everyone", None
        )
        if callable(wait_for_everyone):
            wait_for_everyone()
        elapsed = (
            time.perf_counter() - self._contract_train_started
            if self._contract_train_started
            else 0.0
        )
        checkpoint = (
            Path(self.args.output_dir) / f"checkpoint-{int(self.state.global_step)}"
        )
        self._write_training_contract(
            elapsed,
            completed=False,
            checkpoint_override=checkpoint,
        )
        return result

    def train(self, *args: Any, **kwargs: Any) -> Any:
        self._contract_resumed_from = (
            str(kwargs.get("resume_from_checkpoint"))
            if kwargs.get("resume_from_checkpoint")
            else None
        )
        self._contract_train_started = time.perf_counter()
        if self._exact_resume_position is not None:
            if self._exact_resume_checkpoint is None:
                raise RuntimeError("exact resume checkpoint was not recorded")
            result = _call_with_exact_resume_position(
                lambda: super(QuestionOnlyVATTrainerMixin, self).train(
                    *args, **kwargs
                ),
                checkpoint=self._exact_resume_checkpoint,
                position=self._exact_resume_position,
                save_steps=int(self.args.save_steps),
            )
        else:
            result = super().train(*args, **kwargs)
        wall_seconds = time.perf_counter() - self._contract_train_started
        self._write_phase2_diagnostic()
        self._write_training_contract(wall_seconds, completed=True)
        return result

    def _record_vat_metrics(
        self,
        *,
        sft_loss: Tensor,
        vat_loss: Tensor,
        total_loss: Tensor,
        effective_lambda: float,
        effective_rho: float | None,
        training_progress: float,
        perturb_mask: Tensor,
        prediction_mask: Tensor,
        random_neighbor_stats: RandomNeighborBatchStats | None = None,
    ) -> None:
        values = {
            "sft_loss": float(sft_loss.detach().float().item()),
            "vat_loss": float(vat_loss.detach().float().item()),
            "vat_weighted_loss": float(
                (vat_loss.detach().float() * effective_lambda).item()
            ),
            "vat_total_loss": float(total_loss.detach().float().item()),
            "vat_effective_lambda": float(effective_lambda),
            "vat_effective_rho": (
                float(effective_rho) if effective_rho is not None else 0.0
            ),
            "vat_training_progress": float(training_progress),
            "vat_user_tokens": float(perturb_mask.detach().sum().item()),
            "vat_assistant_tokens": float(prediction_mask.detach().sum().item()),
        }
        if random_neighbor_stats is not None:
            values.update(
                {
                    "random_neighbor_average_size": (
                        random_neighbor_stats.average_neighborhood_size
                    ),
                    "random_neighbor_replacement_rate": (
                        random_neighbor_stats.replacement_rate
                    ),
                }
            )
            for name in self._random_neighbor_totals:
                self._random_neighbor_totals[name] += int(
                    getattr(random_neighbor_stats, name)
                )
        for name, value in values.items():
            self._vat_metric_sums[name] = self._vat_metric_sums.get(name, 0.0) + value
        self._vat_metric_count += 1

    def _record_minmax_metrics(
        self,
        *,
        clean_sft_loss: Tensor,
        adversarial_sft_loss: Tensor,
        reference_embeddings: Tensor,
        perturbation: Tensor,
        perturb_mask: Tensor,
        prediction_mask: Tensor,
        group_ids: Tensor,
    ) -> None:
        reference_norms, reference_keys = masked_frobenius_norms(
            reference_embeddings.detach(),
            perturb_mask,
            group_ids=group_ids,
        )
        perturbation_norms, perturbation_keys = masked_frobenius_norms(
            perturbation.detach(),
            perturb_mask,
            group_ids=group_ids,
        )
        if not torch.equal(reference_keys, perturbation_keys):
            raise RuntimeError("Min-Max reference and perturbation groups differ")
        positive = reference_norms.gt(0)
        relative_norm = (
            perturbation_norms[positive] / reference_norms[positive]
            if positive.any()
            else torch.empty(
                0,
                device=reference_norms.device,
                dtype=torch.float32,
            )
        )
        values = {
            "minmax_clean_sft_loss": float(
                clean_sft_loss.detach().float().item()
            ),
            "minmax_adversarial_sft_loss": float(
                adversarial_sft_loss.detach().float().item()
            ),
            "minmax_total_loss": float(
                adversarial_sft_loss.detach().float().item()
            ),
            "minmax_user_tokens": float(perturb_mask.detach().sum().item()),
            "minmax_assistant_tokens": float(
                prediction_mask.detach().sum().item()
            ),
            "minmax_relative_perturbation_norm_mean": (
                float(relative_norm.mean().item())
                if relative_norm.numel()
                else 0.0
            ),
        }
        for name, value in values.items():
            self._vat_metric_sums[name] = self._vat_metric_sums.get(name, 0.0) + value
        self._vat_metric_count += 1

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> Any:
        # ``logging_steps=1`` calls this after each optimizer step, after the
        # active forward/backward graph has been released. Sampling here makes
        # every point the same lifecycle phase and prevents variable packed
        # sequence lengths from masquerading as persistent memory growth.
        if "loss" in logs and torch.cuda.is_available():
            self._contract_memory_samples.append(torch.cuda.memory_allocated())
        if self._vat_metric_count:
            divisor = float(self._vat_metric_count)
            for name, total in self._vat_metric_sums.items():
                logs.setdefault(name, total / divisor)
            self._vat_metric_sums.clear()
            self._vat_metric_count = 0
        if "loss" in logs and self._vat_gradient_telemetry_pending is not None:
            for name, value in self._vat_gradient_telemetry_pending.items():
                logs.setdefault(name, value)
            self._vat_gradient_telemetry_pending = None
        return super().log(logs, *args, **kwargs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Mapping[str, Tensor],
        return_outputs: bool = False,
    ) -> Tensor | tuple[Tensor, Any]:
        batch = dict(inputs)
        perturb_mask = batch.pop("perturb_mask", None)
        source_ids = batch.pop("source_ids", None)
        batch.pop("perturb_group_ids", None)
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        position_ids = batch.get("position_ids")
        cfg = self.vat_runtime_config
        if (
            model.training
            and isinstance(input_ids, Tensor)
            and isinstance(labels, Tensor)
            and perturb_mask is None
            and cfg.uses_question_adversary
            and cfg.perturb_mask_source == "native_llama3_input_ids"
        ):
            perturb_mask, source_ids = self._native_llama3_batch_masks(
                input_ids=input_ids,
                position_ids=(
                    position_ids if isinstance(position_ids, Tensor) else None
                ),
                attention_mask=(
                    batch.get("attention_mask")
                    if isinstance(batch.get("attention_mask"), Tensor)
                    else None
                ),
                labels=labels,
            )
        if (
            model.training
            and isinstance(input_ids, Tensor)
            and isinstance(labels, Tensor)
            and not self._vat_gradient_similarity_replay
        ):
            self._track_contract_batch(
                input_ids=input_ids,
                labels=labels,
                position_ids=(
                    position_ids if isinstance(position_ids, Tensor) else None
                ),
                source_ids=source_ids if isinstance(source_ids, Tensor) else None,
            )

        # This exact delegation is the Vanilla SFT equivalence contract. Eval
        # also reports ordinary CE so both experiments have comparable curves.
        if not cfg.uses_question_adversary or not model.training:
            delegated = super().compute_loss(
                model, batch, return_outputs=return_outputs
            )
            if model.training:
                delegated_loss = delegated[0] if return_outputs else delegated
                self._track_contract_loss(delegated_loss)
            return delegated

        if perturb_mask is None:
            raise KeyError(
                "question adversarial training requires a packing-aligned "
                "`perturb_mask` batch field"
            )
        if not isinstance(input_ids, Tensor) or not isinstance(labels, Tensor):
            raise KeyError("VAT training requires tensor input_ids and labels")
        if perturb_mask.shape != input_ids.shape:
            raise ValueError(
                f"perturb_mask shape {tuple(perturb_mask.shape)} does not match "
                f"input_ids {tuple(input_ids.shape)}"
            )
        if self._vat_checkpointing_requested:
            self._vat_non_reentrant_checkpointing = (
                _enforce_vat_non_reentrant_checkpointing(model)
            )
            if not self._vat_non_reentrant_checkpointing:
                raise RuntimeError(
                    "question-adversarial gradient checkpointing was requested "
                    "but is not active "
                    "at the first training loss"
                )

        prediction_mask = assistant_prediction_mask(labels)
        if not prediction_mask.any():
            raise ValueError("training batch has no assistant target tokens")

        attention_mask = batch.get("attention_mask")
        # The historical Magpie config aliases pad_token to EOS. Axolotl
        # therefore emits attention=0 for an EOS between turns. Packing groups
        # are defined by position resets, not those internal zeroes.
        if cfg.use_minmax_adversarial_sft or cfg.radius_mode == "relative":
            if not isinstance(position_ids, Tensor) or not isinstance(
                source_ids, Tensor
            ):
                raise KeyError(
                    "per-conversation relative perturbations require "
                    "position_ids and source_ids so "
                    "packed conversation groups can be verified fail-closed"
                )
            validate_packed_source_position_alignment(
                perturb_mask=perturb_mask,
                position_ids=position_ids,
                source_ids=source_ids,
            )
        group_ids = (
            relative_vat_group_ids(position_ids)
            if isinstance(position_ids, Tensor)
            else None
        )
        embedding_layer = _underlying_module(model).get_input_embeddings()
        input_embeddings = embedding_layer(input_ids)

        compact_kwargs: dict[str, Any] = {
            "inputs_embeds": input_embeddings,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
            "vat_prediction_mask": prediction_mask,
        }
        clean_outputs, checkpoint_calls = (
            _call_with_vat_non_reentrant_checkpointing(model, **compact_kwargs)
        )
        self._vat_checkpoint_compatibility_calls += checkpoint_calls
        clean_logits = _extract_logits(clean_outputs)
        targets = _shifted_targets(labels, prediction_mask)
        # Match the official Llama FP32-logit CE exactly while promoting only
        # a bounded row chunk at a time instead of all assistant logits.
        sft_loss = memory_efficient_cross_entropy(clean_logits, targets)

        if cfg.use_minmax_adversarial_sft:
            if cfg.minmax_rho is None:
                raise RuntimeError("Min-Max Adversarial SFT lacks minmax_rho")
            if group_ids is None:
                raise RuntimeError(
                    "Min-Max Adversarial SFT requires packed group IDs"
                )
            clean_sft_loss = sft_loss.detach()
            perturbation = build_supervised_adversarial_perturbation(
                sft_loss,
                input_embeddings,
                perturb_mask,
                rho=cfg.minmax_rho,
                group_ids=group_ids,
                attention_mask=attention_mask,
            )
            # The clean forward exists only to approximate the inner
            # maximization.  Recompute embeddings so the sole outer graph and
            # sole optimized objective are the perturbed-input SFT loss.
            del clean_outputs, clean_logits, sft_loss, input_embeddings
            outer_input_embeddings = embedding_layer(input_ids)
            perturbed_embeddings = outer_input_embeddings + perturbation
            adversarial_outputs, checkpoint_calls = (
                _call_with_vat_non_reentrant_checkpointing(
                    model,
                    inputs_embeds=perturbed_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    vat_prediction_mask=prediction_mask,
                )
            )
            self._vat_checkpoint_compatibility_calls += checkpoint_calls
            adversarial_logits = _extract_logits(adversarial_outputs)
            adversarial_sft_loss = memory_efficient_cross_entropy(
                adversarial_logits,
                targets,
            )
            self._record_minmax_metrics(
                clean_sft_loss=clean_sft_loss,
                adversarial_sft_loss=adversarial_sft_loss,
                reference_embeddings=outer_input_embeddings,
                perturbation=perturbation,
                perturb_mask=perturb_mask,
                prediction_mask=prediction_mask,
                group_ids=group_ids,
            )
            self._track_contract_loss(adversarial_sft_loss)
            if return_outputs:
                return adversarial_sft_loss, {
                    "loss": adversarial_sft_loss,
                    "logits": adversarial_logits,
                    "minmax_clean_sft_loss": clean_sft_loss,
                    "minmax_rho": cfg.minmax_rho,
                }
            return adversarial_sft_loss

        clean_target = clean_logits.detach()
        del clean_outputs

        effective_rho, rho_training_progress = self._effective_vat_rho()

        random_neighbor_stats: RandomNeighborBatchStats | None = None
        if cfg.perturbation_mode == "random_neighbor":
            if effective_rho is None:
                raise RuntimeError("Random-Neighbor VAT requires relative rho")
            perturbed_embeddings, random_neighbor_stats = (
                build_random_neighbor_embeddings(
                    input_embeddings,
                    input_ids,
                    embedding_layer.weight,
                    perturb_mask,
                    rho=effective_rho,
                    group_ids=group_ids,
                    position_ids=(position_ids if group_ids is None else None),
                    attention_mask=attention_mask,
                    vocabulary_chunk_size=(
                        cfg.density_vocabulary_chunk_size
                    ),
                )
            )
        else:
            r_vadv, checkpoint_calls = (
                _call_with_vat_non_reentrant_checkpointing(
                    build_vadv,
                    model,
                    input_embeddings,
                    clean_target,
                    perturb_mask,
                    labels=labels,
                    xi=cfg.xi,
                    xi_mode=cfg.xi_mode,
                    epsilon=cfg.epsilon,
                    rho=effective_rho,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    group_ids=group_ids,
                    input_ids=input_ids,
                    vocabulary_embeddings=embedding_layer.weight,
                    density_mode=cfg.density_mode,
                    density_vocabulary_chunk_size=(
                        cfg.density_vocabulary_chunk_size
                    ),
                    semantic_margins=self._semantic_margins_for(input_ids.device),
                    forward_kwargs={"vat_prediction_mask": prediction_mask},
                )
            )
            self._vat_checkpoint_compatibility_calls += checkpoint_calls
            perturbed_embeddings = input_embeddings + r_vadv
            if not self._vat_gradient_similarity_analysis:
                self._record_phase2_locality(
                    input_embeddings=input_embeddings,
                    perturbed_embeddings=perturbed_embeddings,
                    nominal_perturbation=r_vadv,
                    perturb_mask=perturb_mask,
                    prediction_mask=prediction_mask,
                    group_ids=group_ids,
                    position_ids=position_ids,
                    effective_rho=effective_rho,
                )
        adv_outputs, checkpoint_calls = (
            _call_with_vat_non_reentrant_checkpointing(
                model,
                inputs_embeds=perturbed_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                vat_prediction_mask=prediction_mask,
            )
        )
        self._vat_checkpoint_compatibility_calls += checkpoint_calls
        adv_logits = _extract_logits(adv_outputs)
        vat_loss = masked_kl_divergence(clean_target, adv_logits)
        if (
            cfg.uses_gradient_similarity_gate
            and self._vat_gradient_similarity_analysis
        ):
            effective_lambda = 0.0
            _, _, training_progress = self._vat_optimizer_progress()
        else:
            effective_lambda, training_progress = self._effective_vat_lambda()
        if not math.isclose(
            rho_training_progress,
            training_progress,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError("VAT lambda/rho schedules used different progress")
        total_loss = sft_loss + effective_lambda * vat_loss
        if not self._vat_gradient_similarity_analysis:
            self._maybe_record_gradient_telemetry(
                sft_loss=sft_loss,
                vat_loss=vat_loss,
                input_embeddings=input_embeddings,
                effective_lambda=effective_lambda,
                effective_rho=effective_rho,
                training_progress=training_progress,
            )
            self._write_pilot_gradient_diagnostic(
                sft_loss=sft_loss,
                vat_loss=vat_loss,
                input_embeddings=input_embeddings,
            )

            self._record_vat_metrics(
                sft_loss=sft_loss,
                vat_loss=vat_loss,
                total_loss=total_loss,
                effective_lambda=effective_lambda,
                effective_rho=effective_rho,
                training_progress=training_progress,
                perturb_mask=perturb_mask,
                prediction_mask=prediction_mask,
                random_neighbor_stats=random_neighbor_stats,
            )
            self._track_contract_loss(total_loss)
        if return_outputs:
            return total_loss, {
                "loss": total_loss,
                "logits": clean_logits,
                "sft_loss": sft_loss,
                "vat_loss": vat_loss,
            }
        return total_loss


def make_question_only_vat_trainer(
    base_trainer_cls: type,
    cfg: Mapping[str, Any] | Any,
) -> type:
    """Create the one trainer class used by both experiment configurations."""

    runtime = VATRuntimeConfig.from_mapping(cfg)

    class QuestionOnlyVATTrainer(QuestionOnlyVATTrainerMixin, base_trainer_cls):
        vat_runtime_config = runtime

    QuestionOnlyVATTrainer.__name__ = "QuestionOnlyVATTrainer"
    QuestionOnlyVATTrainer.__qualname__ = "QuestionOnlyVATTrainer"
    return QuestionOnlyVATTrainer


__all__ = [
    "ExactResumePosition",
    "LAUNCH_EXECUTION_ID_ENV",
    "QuestionOnlyVATTrainerMixin",
    "RUN_CONFIG_SHA256_ENV",
    "VATRuntimeConfig",
    "_call_with_exact_resume_position",
    "_infer_exact_resume_position",
    "_repage_loaded_optimizer_state",
    "install_compact_assistant_forward",
    "make_question_only_vat_trainer",
]
