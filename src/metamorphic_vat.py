"""Native-Axolotl trainer path for worst-candidate Metamorphic-VAT."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .metamorphic_candidates import FAMILY_ORDER
from .vat_loss import (
    assistant_prediction_mask,
    masked_kl_divergence,
    memory_efficient_cross_entropy,
    relative_vat_group_ids,
    validate_packed_source_position_alignment,
)
from .vat_trainer import (
    QuestionOnlyVATTrainerMixin,
    VATRuntimeConfig,
    _call_with_vat_non_reentrant_checkpointing,
    _enforce_vat_non_reentrant_checkpointing,
    _extract_logits,
    _shifted_targets,
)


SOFT_WORST_CALIBRATION_PROTOCOL = "metavat_soft_worst_tau_v1"
SOFT_WORST_CALIBRATION_SUBSET_SIZE = 512
SOFT_WORST_CALIBRATION_SEED = 42


@dataclass(frozen=True)
class MetamorphicRuntimeConfig:
    candidate_db: str
    sequence_len: int = 8192
    objective: str = "vat_per_conversation_mean"
    sampling_seed: int = 42
    soft_worst_tau: float | None = None
    soft_worst_calibration: str | None = None

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | Any) -> "MetamorphicRuntimeConfig":
        def read(name: str, default: Any = None) -> Any:
            if isinstance(cfg, Mapping):
                value = cfg.get(name, default)
            else:
                value = getattr(cfg, name, default)
            return default if value is None else value

        if not bool(read("use_metamorphic_vat", False)):
            raise ValueError("Metamorphic trainer requires use_metamorphic_vat=true")
        if not bool(read("native_axolotl_preprocessing", False)):
            raise ValueError("Metamorphic-VAT requires native Axolotl preprocessing")
        perturbation_mode = str(read("vat_perturbation_mode", ""))
        objectives = {
            "metamorphic_text": "vat_per_conversation_mean",
            "metamorphic_text_global_token_mean": "vat_global_token_mean",
            "metamorphic_text_soft_worst": "vat_soft_worst_global_token_mean",
            "metamorphic_text_uniform_sft": "uniform_candidate_sft",
        }
        if perturbation_mode not in objectives:
            raise ValueError(
                "Metamorphic training requires a supported metamorphic-text mode"
            )
        if str(read("vat_perturb_mask_source", "")) != "native_llama3_input_ids":
            raise ValueError("Metamorphic-VAT requires the native Llama-3 mask source")
        objective = objectives[perturbation_mode]
        if objective != "uniform_candidate_sft":
            if str(read("vat_lambda_schedule", "")) != "linear_warmup":
                raise ValueError("Metamorphic-VAT requires the 20% linear lambda warm-up")
            if not math.isclose(float(read("vat_lambda", -1.0)), 0.03, rel_tol=0.0, abs_tol=0.0):
                raise ValueError("Metamorphic-VAT requires vat_lambda=0.03")
            if not math.isclose(
                float(read("vat_lambda_warmup_fraction", -1.0)),
                0.2,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError("Metamorphic-VAT requires a 20% warm-up")
        sampling_seed = int(read("metamorphic_sampling_seed", 42))
        if sampling_seed != 42:
            raise ValueError("the locked MetaSFT sampler requires seed=42")
        candidate_db = read("metamorphic_candidate_db")
        if not isinstance(candidate_db, str) or not candidate_db.strip():
            raise ValueError("metamorphic_candidate_db must be a non-empty path")
        sequence_len = int(read("sequence_len", 8192))
        if sequence_len != 8192:
            raise ValueError("the locked Native experiment requires sequence_len=8192")
        calibration_path: str | None = None
        soft_worst_tau: float | None = None
        if objective == "vat_soft_worst_global_token_mean":
            raw_calibration_path = read("metamorphic_soft_worst_calibration")
            if (
                not isinstance(raw_calibration_path, str)
                or not raw_calibration_path.strip()
            ):
                raise ValueError(
                    "MetaVAT-v3 requires metamorphic_soft_worst_calibration"
                )
            calibration = Path(raw_calibration_path)
            if not calibration.is_file():
                raise FileNotFoundError(
                    f"MetaVAT-v3 calibration artifact is missing: {calibration}"
                )
            payload = json.loads(calibration.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("status") != "passed"
                or payload.get("protocol") != SOFT_WORST_CALIBRATION_PROTOCOL
                or int(payload.get("subset_size", -1))
                != SOFT_WORST_CALIBRATION_SUBSET_SIZE
                or int(payload.get("seed", -1)) != SOFT_WORST_CALIBRATION_SEED
                or int(payload.get("candidate_family_count", -1))
                != len(FAMILY_ORDER)
                or tuple(payload.get("candidate_family_order", ()))
                != tuple(FAMILY_ORDER)
                or Path(str(payload.get("candidate_db", ""))).resolve()
                != Path(candidate_db).resolve()
            ):
                raise ValueError("MetaVAT-v3 calibration artifact contract differs")
            soft_worst_tau = float(payload.get("tau", float("nan")))
            if not math.isfinite(soft_worst_tau) or soft_worst_tau <= 0.0:
                raise ValueError("MetaVAT-v3 calibrated tau must be finite and positive")
            calibration_path = str(calibration)
        return cls(
            candidate_db=candidate_db,
            sequence_len=sequence_len,
            objective=objective,
            sampling_seed=sampling_seed,
            soft_worst_tau=soft_worst_tau,
            soft_worst_calibration=calibration_path,
        )


@dataclass(frozen=True)
class CandidateSequence:
    source_row: int
    family: str
    subtype: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    candidate_text: str
    message_index: int
    modified_span: Mapping[str, Any]
    seed: int

    @property
    def target_ids(self) -> tuple[int, ...]:
        return tuple(value for value in self.labels if value != -100)

    def __post_init__(self) -> None:
        if self.source_row < 0:
            raise ValueError("candidate source_row must be non-negative")
        if self.family not in FAMILY_ORDER:
            raise ValueError(f"unknown candidate family: {self.family}")
        if not self.input_ids or not (
            len(self.input_ids) == len(self.labels) == len(self.attention_mask)
        ):
            raise ValueError("candidate token fields must be non-empty and aligned")
        if any(value not in (0, 1) for value in self.attention_mask):
            raise ValueError("candidate attention mask must be binary")
        if not self.target_ids:
            raise ValueError("candidate has no supervised answer targets")


def _decode_int_blob(blob: bytes, length: int) -> tuple[int, ...]:
    if len(blob) != length * 4:
        raise ValueError("int32 artifact blob has an invalid byte length")
    return tuple(struct.unpack(f"<{length}i", blob))


def _clean_key(input_ids: Sequence[int], labels: Sequence[int]) -> str:
    state = hashlib.sha256()
    state.update(struct.pack(f"<{len(input_ids)}i", *(int(v) for v in input_ids)))
    state.update(b"\x00labels\x00")
    state.update(struct.pack(f"<{len(labels)}i", *(int(v) for v in labels)))
    return state.hexdigest()


class CandidateStore:
    """Per-rank read-only lookup over a completed candidate artifact."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Metamorphic candidate DB is missing: {self.path}")
        uri = f"file:{self.path}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        metadata = {
            str(key): json.loads(value)
            for key, value in self.connection.execute("SELECT key,value FROM metadata")
        }
        expected_rows = int(metadata.get("expected_rows", -1))
        if (
            int(metadata.get("schema_version", -1)) != 1
            or metadata.get("status") != "passed"
            or expected_rows <= 0
            or int(metadata.get("completed_rows", -1)) != expected_rows
            or int(metadata.get("candidate_count", -1)) != 3 * expected_rows
            or int(metadata.get("families_per_row", -1)) != 3
            or metadata.get("answer_targets_preserved") is not True
        ):
            raise RuntimeError("Metamorphic candidate DB is not a complete passed artifact")
        self.expected_rows = expected_rows

    @functools.lru_cache(maxsize=16)
    def lookup(
        self,
        clean_input_ids: tuple[int, ...],
        clean_labels: tuple[int, ...],
    ) -> tuple[CandidateSequence, ...]:
        key = _clean_key(clean_input_ids, clean_labels)
        row = self.connection.execute(
            """SELECT source_row,input_ids,labels,sequence_length
               FROM clean_rows WHERE clean_key=? ORDER BY source_row LIMIT 1""",
            (key,),
        ).fetchone()
        if row is None:
            raise KeyError("packed clean conversation is absent from candidate DB")
        source_row, stored_ids, stored_labels, stored_length = row
        if (
            _decode_int_blob(stored_ids, int(stored_length)) != clean_input_ids
            or _decode_int_blob(stored_labels, int(stored_length)) != clean_labels
        ):
            raise RuntimeError("candidate lookup key matched different clean tokens")
        records = self.connection.execute(
            """SELECT family,subtype,candidate_text,message_index,modified_span_json,
                      seed,input_ids,labels,attention_mask,sequence_length
               FROM candidates WHERE source_row=?""",
            (int(source_row),),
        ).fetchall()
        by_family: dict[str, CandidateSequence] = {}
        for record in records:
            (
                family, subtype, text, message_index, span_json, seed,
                input_blob, label_blob, attention_blob, length,
            ) = record
            length = int(length)
            by_family[str(family)] = CandidateSequence(
                source_row=int(source_row),
                family=str(family),
                subtype=str(subtype),
                input_ids=_decode_int_blob(input_blob, length),
                labels=_decode_int_blob(label_blob, length),
                attention_mask=tuple(int(value) for value in bytes(attention_blob)),
                candidate_text=str(text),
                message_index=int(message_index),
                modified_span=json.loads(span_json),
                seed=int(seed),
            )
        if set(by_family) != set(FAMILY_ORDER):
            raise RuntimeError("candidate row does not contain exactly the three families")
        return tuple(by_family[family] for family in FAMILY_ORDER)


@dataclass(frozen=True)
class CleanConversation:
    batch_index: int
    start: int
    end: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    target_ids: tuple[int, ...]
    logit_start: int
    logit_end: int


@dataclass(frozen=True)
class PackedCandidateBatch:
    input_ids: Tensor
    labels: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    prediction_mask: Tensor
    target_ids: tuple[tuple[int, ...], ...]
    logit_offsets: tuple[tuple[int, int], ...]


def _clean_conversations(
    *,
    input_ids: Tensor,
    labels: Tensor,
    position_ids: Tensor,
    source_ids: Tensor,
) -> tuple[CleanConversation, ...]:
    groups = relative_vat_group_ids(position_ids)
    prediction_mask = assistant_prediction_mask(labels)
    shifted = torch.full_like(labels, -100)
    shifted[:, :-1] = labels[:, 1:]
    conversations: list[CleanConversation] = []
    offset = 0
    for batch_index in range(input_ids.shape[0]):
        active = source_ids[batch_index].ge(0)
        observed_groups = groups[batch_index][active].tolist()
        for group_id in dict.fromkeys(int(value) for value in observed_groups):
            positions = torch.nonzero(
                active & groups[batch_index].eq(group_id), as_tuple=False
            ).flatten()
            if positions.numel() == 0:
                continue
            start = int(positions[0].item())
            end = int(positions[-1].item()) + 1
            expected = torch.arange(start, end, device=positions.device)
            if not torch.equal(positions, expected):
                raise RuntimeError("packed conversation tokens are not contiguous")
            local_positions = position_ids[batch_index, start:end]
            if not torch.equal(
                local_positions,
                torch.arange(end - start, device=local_positions.device),
            ):
                raise RuntimeError("packed conversation positions do not restart at zero")
            local_prediction = prediction_mask[batch_index, start:end]
            targets = shifted[batch_index, start:end][local_prediction]
            if targets.numel() == 0:
                raise RuntimeError("packed conversation has no answer target")
            count = int(targets.numel())
            conversations.append(
                CleanConversation(
                    batch_index=batch_index,
                    start=start,
                    end=end,
                    input_ids=tuple(int(value) for value in input_ids[batch_index, start:end].tolist()),
                    labels=tuple(int(value) for value in labels[batch_index, start:end].tolist()),
                    target_ids=tuple(int(value) for value in targets.tolist()),
                    logit_start=offset,
                    logit_end=offset + count,
                )
            )
            offset += count
    if not conversations:
        raise RuntimeError("batch contains no active packed conversations")
    if offset != int(prediction_mask.sum().item()):
        raise RuntimeError("conversation answer offsets do not cover clean logits")
    return tuple(conversations)


def pack_candidate_sequences(
    records: Sequence[CandidateSequence],
    *,
    sequence_len: int,
    pad_token_id: int,
    device: torch.device,
) -> PackedCandidateBatch:
    """Pack full candidate conversations with position resets, preserving order."""

    if not records:
        raise ValueError("cannot pack an empty candidate list")
    rows: list[dict[str, list[int]]] = []
    current = {"input_ids": [], "labels": [], "attention_mask": [], "position_ids": []}
    for record in records:
        length = len(record.input_ids)
        if length > sequence_len:
            raise ValueError("candidate exceeds locked sequence length")
        if current["input_ids"] and len(current["input_ids"]) + length > sequence_len:
            rows.append(current)
            current = {"input_ids": [], "labels": [], "attention_mask": [], "position_ids": []}
        current["input_ids"].extend(record.input_ids)
        current["labels"].extend(record.labels)
        current["attention_mask"].extend(record.attention_mask)
        current["position_ids"].extend(range(length))
    if current["input_ids"]:
        rows.append(current)
    width = max(len(row["input_ids"]) for row in rows)
    tensors: dict[str, Tensor] = {}
    for name, pad in (
        ("input_ids", pad_token_id),
        ("labels", -100),
        ("attention_mask", 0),
        ("position_ids", 0),
    ):
        values = [row[name] + [pad] * (width - len(row[name])) for row in rows]
        tensors[name] = torch.tensor(values, dtype=torch.long, device=device)
    prediction_mask = assistant_prediction_mask(tensors["labels"])
    flattened_targets = _shifted_targets(tensors["labels"], prediction_mask)
    expected_targets = tuple(record.target_ids for record in records)
    offsets: list[tuple[int, int]] = []
    offset = 0
    for targets in expected_targets:
        offsets.append((offset, offset + len(targets)))
        offset += len(targets)
    if offset != int(flattened_targets.numel()):
        raise RuntimeError("packed candidate target offsets do not cover logits")
    observed: list[tuple[int, ...]] = []
    for start, end in offsets:
        observed.append(tuple(int(value) for value in flattened_targets[start:end].tolist()))
    if tuple(observed) != expected_targets:
        raise RuntimeError("candidate packing changed answer-token order")
    return PackedCandidateBatch(
        input_ids=tensors["input_ids"],
        labels=tensors["labels"],
        attention_mask=tensors["attention_mask"],
        position_ids=tensors["position_ids"],
        prediction_mask=prediction_mask,
        target_ids=expected_targets,
        logit_offsets=tuple(offsets),
    )


def select_worst_candidate(scores: Tensor) -> Tensor:
    """Return one family index per sample, with deterministic first-max ties."""

    if scores.ndim != 2 or scores.shape[1] != len(FAMILY_ORDER):
        raise ValueError("scores must have shape [samples, 3]")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("candidate scores must be finite")
    return scores.argmax(dim=1)


def soft_worst_aggregation(scores: Tensor, *, tau: float) -> Tensor:
    """Compute the stable per-conversation soft worst-case value.

    The division by the number of families is part of the requested objective,
    so this is ``tau * (logsumexp(k / tau) - log(3))`` rather than the common
    unnormalised smooth maximum.
    """

    if scores.ndim != 2 or scores.shape[1] != len(FAMILY_ORDER):
        raise ValueError("scores must have shape [samples, 3]")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("candidate scores must be finite")
    tau_value = float(tau)
    if not math.isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError("tau must be finite and positive")
    return tau_value * (
        torch.logsumexp(scores.float() / tau_value, dim=1)
        - math.log(len(FAMILY_ORDER))
    )


def token_weighted_soft_worst_loss(
    scores: Tensor,
    token_counts: Tensor,
    *,
    tau: float,
) -> Tensor:
    """Apply global assistant-token weighting to soft worst-case values."""

    if token_counts.ndim != 1 or token_counts.shape[0] != scores.shape[0]:
        raise ValueError("token_counts must contain one value per sample")
    counts = token_counts.to(device=scores.device, dtype=torch.float32)
    if not bool(torch.isfinite(counts).all()) or bool((counts <= 0).any()):
        raise ValueError("token_counts must be finite and positive")
    values = soft_worst_aggregation(scores, tau=tau)
    return (counts * values).sum() / counts.sum()


def select_uniform_family_indices(
    source_rows: Sequence[int], *, seed: int = 42
) -> tuple[int, ...]:
    """Deterministically sample one of three families with exact row balance.

    The offline artifact uses contiguous source-row identities.  Cycling the
    family by source row is therefore a reproducible uniform sampler over the
    full candidate pool (counts differ by at most one when the row count is not
    divisible by three), while remaining invariant to packing and DDP rank.
    """

    if seed != 42:
        raise ValueError("the locked MetaSFT sampler requires seed=42")
    rows = tuple(int(value) for value in source_rows)
    if any(value < 0 for value in rows):
        raise ValueError("source rows must be non-negative")
    return tuple((value + seed) % len(FAMILY_ORDER) for value in rows)


class MetamorphicVATTrainerMixin(QuestionOnlyVATTrainerMixin):
    metamorphic_runtime_config: MetamorphicRuntimeConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._metamorphic_store = CandidateStore(
            self.metamorphic_runtime_config.candidate_db
        )
        super().__init__(*args, **kwargs)

    def _candidate_logits(
        self,
        model: nn.Module,
        packed: PackedCandidateBatch,
    ) -> Tensor:
        outputs, checkpoint_calls = _call_with_vat_non_reentrant_checkpointing(
            model,
            input_ids=packed.input_ids,
            attention_mask=packed.attention_mask,
            position_ids=packed.position_ids,
            use_cache=False,
            vat_prediction_mask=packed.prediction_mask,
        )
        self._vat_checkpoint_compatibility_calls += checkpoint_calls
        return _extract_logits(outputs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Mapping[str, Tensor],
        return_outputs: bool = False,
    ) -> Tensor | tuple[Tensor, Any]:
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        batch = dict(inputs)
        batch.pop("perturb_mask", None)
        batch.pop("source_ids", None)
        batch.pop("perturb_group_ids", None)
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        position_ids = batch.get("position_ids")
        attention_mask = batch.get("attention_mask")
        if not all(isinstance(value, Tensor) for value in (input_ids, labels, position_ids, attention_mask)):
            raise KeyError("Metamorphic-VAT requires native packed input/label/position/attention tensors")
        assert isinstance(input_ids, Tensor)
        assert isinstance(labels, Tensor)
        assert isinstance(position_ids, Tensor)
        assert isinstance(attention_mask, Tensor)
        perturb_mask, source_ids = self._native_llama3_batch_masks(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        validate_packed_source_position_alignment(
            perturb_mask=perturb_mask,
            position_ids=position_ids,
            source_ids=source_ids,
        )
        self._track_contract_batch(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            source_ids=source_ids,
        )
        if self._vat_checkpointing_requested:
            self._vat_non_reentrant_checkpointing = _enforce_vat_non_reentrant_checkpointing(model)
            if not self._vat_non_reentrant_checkpointing:
                raise RuntimeError("Metamorphic-VAT gradient checkpointing is not active")

        prediction_mask = assistant_prediction_mask(labels)
        if not bool(prediction_mask.any()):
            raise RuntimeError("Metamorphic-VAT batch has no answer targets")
        clean_outputs, checkpoint_calls = _call_with_vat_non_reentrant_checkpointing(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            vat_prediction_mask=prediction_mask,
        )
        self._vat_checkpoint_compatibility_calls += checkpoint_calls
        clean_logits = _extract_logits(clean_outputs)
        clean_targets = _shifted_targets(labels, prediction_mask)
        sft_loss = memory_efficient_cross_entropy(clean_logits, clean_targets)
        clean_target = clean_logits.detach()
        conversations = _clean_conversations(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            source_ids=source_ids,
        )
        candidate_rows = tuple(
            self._metamorphic_store.lookup(conversation.input_ids, conversation.labels)
            for conversation in conversations
        )
        for conversation, candidates in zip(conversations, candidate_rows):
            for candidate in candidates:
                if candidate.target_ids != conversation.target_ids:
                    raise RuntimeError("candidate answer tokens do not align with clean answer index")

        tokenizer = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise RuntimeError("Metamorphic-VAT requires a valid native pad token")
        if self.metamorphic_runtime_config.objective == "uniform_candidate_sft":
            family_indices = select_uniform_family_indices(
                [candidates[0].source_row for candidates in candidate_rows],
                seed=self.metamorphic_runtime_config.sampling_seed,
            )
            for candidates in candidate_rows:
                if len({candidate.source_row for candidate in candidates}) != 1:
                    raise RuntimeError("candidate families disagree on source_row")
            selected = [
                candidate_rows[index][family_index]
                for index, family_index in enumerate(family_indices)
            ]
            selected_packed = pack_candidate_sequences(
                selected,
                sequence_len=self.metamorphic_runtime_config.sequence_len,
                pad_token_id=pad_token_id,
                device=input_ids.device,
            )
            selected_logits = self._candidate_logits(model, selected_packed)
            transformed_targets = _shifted_targets(
                selected_packed.labels, selected_packed.prediction_mask
            )
            if transformed_targets.numel() != clean_targets.numel():
                raise RuntimeError("MetaSFT clean/transformed target counts differ")
            transformed_sft_loss = memory_efficient_cross_entropy(
                selected_logits, transformed_targets
            )
            total_loss = 0.5 * sft_loss + 0.5 * transformed_sft_loss
            selection_counts = torch.bincount(
                torch.tensor(family_indices, dtype=torch.long),
                minlength=len(FAMILY_ORDER),
            )
            sample_count = float(len(conversations))
            custom_metrics = {
                "metasft_clean_sft_loss": float(sft_loss.detach().float().item()),
                "metasft_transformed_sft_loss": float(
                    transformed_sft_loss.detach().float().item()
                ),
                "metasft_total_loss": float(total_loss.detach().float().item()),
                "metasft_conversations": sample_count,
                "metasft_character_noise_rate": float(selection_counts[0]) / sample_count,
                "metasft_tokenization_rate": float(selection_counts[1]) / sample_count,
                "metasft_code_switch_rate": float(selection_counts[2]) / sample_count,
            }
            for name, value in custom_metrics.items():
                self._vat_metric_sums[name] = self._vat_metric_sums.get(name, 0.0) + value
            self._vat_metric_count += 1
            self._track_contract_loss(total_loss)
            if return_outputs:
                return total_loss, {
                    "loss": total_loss,
                    "logits": clean_logits,
                    "clean_sft_loss": sft_loss,
                    "transformed_sft_loss": transformed_sft_loss,
                    "sampled_family_indices": family_indices,
                }
            return total_loss

        if (
            self.metamorphic_runtime_config.objective
            == "vat_soft_worst_global_token_mean"
        ):
            tau = self.metamorphic_runtime_config.soft_worst_tau
            if tau is None:
                raise RuntimeError("MetaVAT-v3 calibrated tau is unavailable")
            score_columns: list[Tensor] = []
            # A direct graph over three full candidate forwards exceeds the
            # memory envelope of the locked four-GPU run. Checkpoint each
            # complete family forward: the exact three KL values participate
            # in log-sum-exp, while activations are replayed sequentially in
            # backward rather than retained simultaneously.
            for family_index in range(len(FAMILY_ORDER)):
                packed = pack_candidate_sequences(
                    [candidates[family_index] for candidates in candidate_rows],
                    sequence_len=self.metamorphic_runtime_config.sequence_len,
                    pad_token_id=pad_token_id,
                    device=input_ids.device,
                )
                offsets = packed.logit_offsets
                candidate_target_rows = packed.target_ids
                observed_checkpoint_calls: list[int] = []

                def family_scores(
                    candidate_input_ids: Tensor,
                    candidate_attention_mask: Tensor,
                    candidate_position_ids: Tensor,
                    candidate_prediction_mask: Tensor,
                    *,
                    _offsets: tuple[tuple[int, int], ...] = offsets,
                    _candidate_targets: tuple[tuple[int, ...], ...]
                    = candidate_target_rows,
                ) -> Tensor:
                    outputs, checkpoint_calls = (
                        _call_with_vat_non_reentrant_checkpointing(
                            model,
                            input_ids=candidate_input_ids,
                            attention_mask=candidate_attention_mask,
                            position_ids=candidate_position_ids,
                            use_cache=False,
                            vat_prediction_mask=candidate_prediction_mask,
                        )
                    )
                    if not observed_checkpoint_calls:
                        observed_checkpoint_calls.append(checkpoint_calls)
                    candidate_logits = _extract_logits(outputs)
                    values = []
                    for conversation, (start, end), candidate_targets in zip(
                        conversations, _offsets, _candidate_targets
                    ):
                        if candidate_targets != conversation.target_ids:
                            raise RuntimeError(
                                "soft-worst forward lost answer-index alignment"
                            )
                        values.append(
                            masked_kl_divergence(
                                clean_target[
                                    conversation.logit_start : conversation.logit_end
                                ],
                                candidate_logits[start:end],
                            )
                        )
                    return torch.stack(values)

                score_column = activation_checkpoint(
                    family_scores,
                    packed.input_ids,
                    packed.attention_mask,
                    packed.position_ids,
                    packed.prediction_mask,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                if not observed_checkpoint_calls:
                    raise RuntimeError(
                        "soft-worst candidate forward did not execute"
                    )
                self._vat_checkpoint_compatibility_calls += (
                    observed_checkpoint_calls[0]
                )
                score_columns.append(score_column)

            score_matrix = torch.stack(score_columns, dim=1)
            token_counts = torch.tensor(
                [
                    conversation.logit_end - conversation.logit_start
                    for conversation in conversations
                ],
                device=score_matrix.device,
                dtype=torch.float32,
            )
            metamorphic_kl = token_weighted_soft_worst_loss(
                score_matrix,
                token_counts,
                tau=tau,
            )
            effective_lambda, training_progress = self._effective_vat_lambda()
            total_loss = sft_loss + effective_lambda * metamorphic_kl
            self._record_vat_metrics(
                sft_loss=sft_loss,
                vat_loss=metamorphic_kl,
                total_loss=total_loss,
                effective_lambda=effective_lambda,
                effective_rho=None,
                training_progress=training_progress,
                perturb_mask=perturb_mask,
                prediction_mask=prediction_mask,
            )
            detached_scores = score_matrix.detach().float()
            soft_weights = torch.softmax(detached_scores / tau, dim=1)
            entropy = -(
                soft_weights * soft_weights.clamp_min(torch.finfo(torch.float32).tiny).log()
            ).sum(dim=1)
            gaps = torch.topk(detached_scores, k=2, dim=1).values
            gaps = gaps[:, 0] - gaps[:, 1]
            soft_values = soft_worst_aggregation(detached_scores, tau=tau)
            custom_metrics = {
                "metamorphic_conversations": float(len(conversations)),
                "metamorphic_search_requires_grad": 1.0,
                "metamorphic_candidate_kl_mean": float(
                    detached_scores.mean().item()
                ),
                "metamorphic_candidate_kl_max_mean": float(
                    detached_scores.max(dim=1).values.mean().item()
                ),
                "metamorphic_candidate_gap_median": float(gaps.median().item()),
                "metamorphic_soft_worst_value_mean": float(
                    soft_values.mean().item()
                ),
                "metamorphic_soft_worst_tau": float(tau),
                "metamorphic_soft_worst_effective_candidates": float(
                    entropy.exp().mean().item()
                ),
                "metamorphic_soft_character_noise_weight": float(
                    soft_weights[:, 0].mean().item()
                ),
                "metamorphic_soft_tokenization_weight": float(
                    soft_weights[:, 1].mean().item()
                ),
                "metamorphic_soft_code_switch_weight": float(
                    soft_weights[:, 2].mean().item()
                ),
                "metamorphic_soft_worst_aggregation": 1.0,
                "metamorphic_global_token_mean_reduction": 1.0,
            }
            for name, value in custom_metrics.items():
                self._vat_metric_sums[name] = (
                    self._vat_metric_sums.get(name, 0.0) + value
                )
            self._track_contract_loss(total_loss)
            if return_outputs:
                return total_loss, {
                    "loss": total_loss,
                    "logits": clean_logits,
                    "sft_loss": sft_loss,
                    "metamorphic_kl": metamorphic_kl,
                    "candidate_scores": score_matrix,
                    "soft_candidate_weights": soft_weights,
                }
            return total_loss

        score_columns = []
        # The three search forwards are deliberately graph-free.  Exactly one
        # selected-candidate forward below participates in outer minimization.
        with torch.no_grad():
            for family_index in range(len(FAMILY_ORDER)):
                packed = pack_candidate_sequences(
                    [candidates[family_index] for candidates in candidate_rows],
                    sequence_len=self.metamorphic_runtime_config.sequence_len,
                    pad_token_id=pad_token_id,
                    device=input_ids.device,
                )
                candidate_logits = self._candidate_logits(model, packed)
                sample_scores = []
                for conversation, (start, end), candidate_targets in zip(
                    conversations, packed.logit_offsets, packed.target_ids
                ):
                    if candidate_targets != conversation.target_ids:
                        raise RuntimeError("search forward lost answer-index alignment")
                    sample_scores.append(
                        masked_kl_divergence(
                            clean_target[conversation.logit_start : conversation.logit_end],
                            candidate_logits[start:end],
                        )
                    )
                score_columns.append(torch.stack(sample_scores))
        score_matrix = torch.stack(score_columns, dim=1)
        if score_matrix.requires_grad:
            raise RuntimeError("candidate search unexpectedly retained an autograd graph")
        worst_indices = select_worst_candidate(score_matrix)
        selected = [
            candidate_rows[index][int(worst_indices[index].item())]
            for index in range(len(conversations))
        ]
        selected_packed = pack_candidate_sequences(
            selected,
            sequence_len=self.metamorphic_runtime_config.sequence_len,
            pad_token_id=pad_token_id,
            device=input_ids.device,
        )
        selected_logits = self._candidate_logits(model, selected_packed)
        per_sample_kl = []
        for conversation, (start, end), candidate_targets in zip(
            conversations,
            selected_packed.logit_offsets,
            selected_packed.target_ids,
        ):
            if candidate_targets != conversation.target_ids:
                raise RuntimeError("selected forward lost answer-index alignment")
            per_sample_kl.append(
                masked_kl_divergence(
                    clean_target[conversation.logit_start : conversation.logit_end],
                    selected_logits[start:end],
                )
            )
        if self.metamorphic_runtime_config.objective == "vat_global_token_mean":
            # Both compact tensors concatenate conversations in the same order;
            # the validations above bind every slice to identical answer IDs.
            # A single KL call therefore implements the exact global
            # assistant-token mean used by continuous VAT.
            metamorphic_kl = masked_kl_divergence(clean_target, selected_logits)
        else:
            metamorphic_kl = torch.stack(per_sample_kl).mean()
        effective_lambda, training_progress = self._effective_vat_lambda()
        total_loss = sft_loss + effective_lambda * metamorphic_kl
        self._record_vat_metrics(
            sft_loss=sft_loss,
            vat_loss=metamorphic_kl,
            total_loss=total_loss,
            effective_lambda=effective_lambda,
            effective_rho=None,
            training_progress=training_progress,
            perturb_mask=perturb_mask,
            prediction_mask=prediction_mask,
        )
        selection_counts = torch.bincount(
            worst_indices.detach().cpu(), minlength=len(FAMILY_ORDER)
        )
        sample_count = float(len(conversations))
        custom_metrics = {
            "metamorphic_conversations": sample_count,
            "metamorphic_search_requires_grad": 0.0,
            "metamorphic_selected_character_noise_rate": float(selection_counts[0]) / sample_count,
            "metamorphic_selected_tokenization_rate": float(selection_counts[1]) / sample_count,
            "metamorphic_selected_code_switch_rate": float(selection_counts[2]) / sample_count,
            "metamorphic_candidate_kl_mean": float(score_matrix.mean().item()),
            "metamorphic_candidate_kl_max_mean": float(score_matrix.max(dim=1).values.mean().item()),
            "metamorphic_global_token_mean_reduction": float(
                self.metamorphic_runtime_config.objective == "vat_global_token_mean"
            ),
        }
        for name, value in custom_metrics.items():
            self._vat_metric_sums[name] = self._vat_metric_sums.get(name, 0.0) + value
        self._track_contract_loss(total_loss)
        if return_outputs:
            return total_loss, {
                "loss": total_loss,
                "logits": clean_logits,
                "sft_loss": sft_loss,
                "metamorphic_kl": metamorphic_kl,
                "candidate_scores": score_matrix,
                "worst_candidate_indices": worst_indices,
            }
        return total_loss


def make_metamorphic_vat_trainer(
    base_trainer_cls: type,
    cfg: Mapping[str, Any] | Any,
) -> type:
    runtime = VATRuntimeConfig.from_mapping(cfg)
    metamorphic_runtime = MetamorphicRuntimeConfig.from_mapping(cfg)
    if not runtime.use_vat or runtime.perturbation_mode not in {
        "metamorphic_text",
        "metamorphic_text_global_token_mean",
        "metamorphic_text_soft_worst",
        "metamorphic_text_uniform_sft",
    }:
        raise ValueError("Metamorphic trainer requires a metamorphic-text mode")

    class MetamorphicVATTrainer(
        MetamorphicVATTrainerMixin,
        base_trainer_cls,
    ):
        vat_runtime_config = runtime
        metamorphic_runtime_config = metamorphic_runtime

    MetamorphicVATTrainer.__name__ = "MetamorphicVATTrainer"
    MetamorphicVATTrainer.__qualname__ = "MetamorphicVATTrainer"
    return MetamorphicVATTrainer


__all__ = [
    "CandidateSequence",
    "CandidateStore",
    "MetamorphicRuntimeConfig",
    "MetamorphicVATTrainerMixin",
    "PackedCandidateBatch",
    "SOFT_WORST_CALIBRATION_PROTOCOL",
    "SOFT_WORST_CALIBRATION_SEED",
    "SOFT_WORST_CALIBRATION_SUBSET_SIZE",
    "make_metamorphic_vat_trainer",
    "pack_candidate_sequences",
    "select_uniform_family_indices",
    "select_worst_candidate",
    "soft_worst_aggregation",
    "token_weighted_soft_worst_loss",
]
