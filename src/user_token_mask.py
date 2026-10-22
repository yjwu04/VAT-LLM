"""Conversation-aware user-token masks for question-only VAT.

The mask is derived from structural provenance while rendering a conversation,
not by searching token IDs or rendered text for message contents.  It is then
treated as a normal sequence field so that truncation, padding, and packing use
the same operations as ``input_ids`` and ``labels``.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable


IGNORE_INDEX = -100
SEQUENCE_FIELDS = ("input_ids", "labels", "attention_mask", "perturb_mask")


class ConversationMaskError(ValueError):
    """Base error for invalid conversation-mask inputs."""


class TemplateMismatchError(ConversationMaskError):
    """The configured tokenizer does not use the expected Llama-3 template."""


class AlignmentError(ConversationMaskError):
    """Axolotl's tokenized sample is not aligned with the mask encoding."""


class TokenRegion(str, Enum):
    """Semantic provenance for a token."""

    SPECIAL = "special"
    ROLE_HEADER = "role_header"
    SEPARATOR = "separator"
    SYSTEM_BODY = "system_body"
    USER_BODY = "user_body"
    ASSISTANT_BODY = "assistant_body"
    ASSISTANT_EOT = "assistant_eot"
    MIXED = "mixed"
    PADDING = "padding"


_BODY_REGION = {
    "system": TokenRegion.SYSTEM_BODY,
    "user": TokenRegion.USER_BODY,
    "assistant": TokenRegion.ASSISTANT_BODY,
}
_ROLE_ALIASES = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
}


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class CharacterSpan:
    start: int
    end: int
    region: TokenRegion
    message_index: int | None = None


@dataclass(frozen=True)
class RenderedConversation:
    text: str
    spans: tuple[CharacterSpan, ...]
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class ConversationFeatures:
    """Aligned sequence fields plus optional provenance used by pilot tests."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    perturb_mask: tuple[int, ...]
    token_regions: tuple[TokenRegion, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.input_ids),
            len(self.labels),
            len(self.attention_mask),
            len(self.perturb_mask),
            len(self.token_regions),
        }
        if len(lengths) != 1:
            raise ConversationMaskError("all token-level fields must have equal length")
        if any(value not in (0, 1) for value in self.perturb_mask):
            raise ConversationMaskError("perturb_mask must be binary")
        if any(value not in (0, 1) for value in self.attention_mask):
            raise ConversationMaskError("attention_mask must be binary before packing")
        for perturb, label in zip(self.perturb_mask, self.labels):
            if perturb and label != IGNORE_INDEX:
                raise ConversationMaskError(
                    "a user perturbation position cannot also be an SFT target"
                )

    def as_dict(self) -> dict[str, list[int]]:
        """Return the four fields consumed by preprocessing/collation."""

        return {
            "input_ids": list(self.input_ids),
            "labels": list(self.labels),
            "attention_mask": list(self.attention_mask),
            "perturb_mask": list(self.perturb_mask),
        }


@runtime_checkable
class OffsetTokenizer(Protocol):
    """Minimal tokenizer interface required by the Llama-3 adapter.

    A fast Hugging Face tokenizer satisfies this protocol.  Offsets are
    essential: they preserve whole-template BPE tokenization while retaining
    structural message provenance.
    """

    pad_token_id: int | None
    truncation_side: str
    padding_side: str

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        return_attention_mask: bool,
    ) -> Mapping[str, Any]: ...

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any: ...


@runtime_checkable
class ConversationMaskAdapter(Protocol):
    """Adapter boundary for a tokenizer/chat-template-specific implementation."""

    def encode(
        self,
        messages: Sequence[Mapping[str, Any] | Message],
        tokenizer: OffsetTokenizer,
        *,
        validate_template: bool = True,
    ) -> ConversationFeatures: ...


def normalize_sharegpt_messages(
    messages: Sequence[Mapping[str, Any] | Message],
) -> tuple[Message, ...]:
    """Normalize HF ``role/content`` and ShareGPT ``from/value`` messages."""

    normalized: list[Message] = []
    for index, raw_message in enumerate(messages):
        if isinstance(raw_message, Message):
            role_value = raw_message.role
            content = raw_message.content
        elif isinstance(raw_message, Mapping):
            role_value = raw_message.get("role", raw_message.get("from"))
            content = raw_message.get("content", raw_message.get("value"))
        else:
            raise ConversationMaskError(f"message {index} is not a mapping")

        if not isinstance(role_value, str):
            raise ConversationMaskError(f"message {index} has no string role")
        role = _ROLE_ALIASES.get(role_value.strip().lower())
        if role is None:
            raise ConversationMaskError(
                f"message {index} has unsupported role {role_value!r}"
            )
        if not isinstance(content, str):
            raise ConversationMaskError(f"message {index} has no string content")
        normalized.append(Message(role=role, content=content))
    return tuple(normalized)


class Llama3ShareGPTMaskAdapter:
    """Structural adapter for Meta-Llama-3's canonical instruct template.

    Canonical message layout::

        <|start_header_id|>{role}<|end_header_id|>\\n\\n
        {content}<|eot_id|>

    The complete rendered template is tokenized once.  Offset mappings assign a
    token to a body only when the complete token lies inside that body's span.
    A token crossing a body/structure boundary is conservatively ``MIXED`` and
    is never perturbed or used as an assistant target.
    """

    begin_of_text = "<|begin_of_text|>"
    end_of_text = "<|end_of_text|>"
    start_header = "<|start_header_id|>"
    end_header = "<|end_header_id|>"
    end_of_turn = "<|eot_id|>"
    header_separator = "\n\n"

    def __init__(
        self,
        *,
        append_eos: bool = False,
        supervise_assistant_eot: bool = False,
    ) -> None:
        """Configure optional EOS handling and assistant-EOT supervision."""

        self.append_eos = append_eos
        self.supervise_assistant_eot = supervise_assistant_eot

    def render(
        self, messages: Sequence[Mapping[str, Any] | Message]
    ) -> RenderedConversation:
        normalized = normalize_sharegpt_messages(messages)
        chunks: list[str] = []
        spans: list[CharacterSpan] = []
        cursor = 0

        def append(
            text: str, region: TokenRegion, message_index: int | None = None
        ) -> None:
            nonlocal cursor
            start = cursor
            chunks.append(text)
            cursor += len(text)
            if text:
                spans.append(
                    CharacterSpan(
                        start=start,
                        end=cursor,
                        region=region,
                        message_index=message_index,
                    )
                )

        append(self.begin_of_text, TokenRegion.SPECIAL)
        for message_index, message in enumerate(normalized):
            append(self.start_header, TokenRegion.ROLE_HEADER, message_index)
            append(message.role, TokenRegion.ROLE_HEADER, message_index)
            append(self.end_header, TokenRegion.ROLE_HEADER, message_index)
            append(self.header_separator, TokenRegion.SEPARATOR, message_index)
            append(message.content, _BODY_REGION[message.role], message_index)
            eot_region = (
                TokenRegion.ASSISTANT_EOT
                if self.supervise_assistant_eot and message.role == "assistant"
                else TokenRegion.SEPARATOR
            )
            append(self.end_of_turn, eot_region, message_index)
        if self.append_eos:
            append(self.end_of_text, TokenRegion.SPECIAL)

        return RenderedConversation(
            text="".join(chunks),
            spans=tuple(spans),
            messages=normalized,
        )

    def encode(
        self,
        messages: Sequence[Mapping[str, Any] | Message],
        tokenizer: OffsetTokenizer,
        *,
        validate_template: bool = True,
    ) -> ConversationFeatures:
        rendered = self.render(messages)
        hf_messages = [
            {"role": message.role, "content": message.content}
            for message in rendered.messages
        ]
        if validate_template:
            template_text = rendered.text
            if self.append_eos:
                template_text = template_text[: -len(self.end_of_text)]
            self._validate_template(tokenizer, hf_messages, template_text)

        try:
            encoded = tokenizer(
                rendered.text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
            )
        except (TypeError, NotImplementedError) as exc:
            raise ConversationMaskError(
                "the Llama-3 mask adapter requires a fast tokenizer with "
                "return_offsets_mapping support"
            ) from exc

        input_ids = _flatten_1d(encoded.get("input_ids"), "input_ids")
        offsets = _flatten_offsets(encoded.get("offset_mapping"))
        if len(input_ids) != len(offsets):
            raise ConversationMaskError(
                "tokenizer returned different input_ids and offset_mapping lengths"
            )

        regions = _regions_from_offsets(offsets, rendered.spans)
        perturb_mask = tuple(int(region is TokenRegion.USER_BODY) for region in regions)
        labels = tuple(
            token_id
            if region in {TokenRegion.ASSISTANT_BODY, TokenRegion.ASSISTANT_EOT}
            else IGNORE_INDEX
            for token_id, region in zip(input_ids, regions)
        )
        return ConversationFeatures(
            input_ids=tuple(input_ids),
            labels=labels,
            attention_mask=(1,) * len(input_ids),
            perturb_mask=perturb_mask,
            token_regions=regions,
        )

    @staticmethod
    def _validate_template(
        tokenizer: OffsetTokenizer,
        messages: Sequence[Mapping[str, str]],
        expected_text: str,
    ) -> None:
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_template):
            raise TemplateMismatchError(
                "tokenizer has no apply_chat_template; pass validate_template=False "
                "only after independently pinning and verifying the template"
            )
        actual = apply_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if actual != expected_text:
            raise TemplateMismatchError(
                "configured chat template differs from canonical Llama-3. "
                "Provide a ConversationMaskAdapter for the pinned Axolotl template "
                "instead of aligning by string search."
            )


def build_conversation_features(
    messages: Sequence[Mapping[str, Any] | Message],
    tokenizer: OffsetTokenizer,
    *,
    adapter: ConversationMaskAdapter | None = None,
    validate_template: bool = True,
    max_length: int | None = None,
    truncation_side: str | None = None,
    padding_length: int | None = None,
    padding_side: str | None = None,
    pad_token_id: int | None = None,
) -> ConversationFeatures:
    """Build and synchronously truncate/pad all conversation sequence fields."""

    selected_adapter = adapter or Llama3ShareGPTMaskAdapter()
    features = selected_adapter.encode(
        messages, tokenizer, validate_template=validate_template
    )

    if max_length is not None:
        if max_length < 0:
            raise ConversationMaskError("max_length must be non-negative")
        side = truncation_side or getattr(tokenizer, "truncation_side", "right")
        features = truncate_features(features, max_length, side=side)

    if padding_length is not None:
        side = padding_side or getattr(tokenizer, "padding_side", "right")
        selected_pad_id = (
            pad_token_id
            if pad_token_id is not None
            else getattr(tokenizer, "pad_token_id", None)
        )
        if selected_pad_id is None:
            raise ConversationMaskError(
                "padding requires pad_token_id (Llama setups often set it to EOS)"
            )
        features = pad_features(
            features,
            padding_length,
            pad_token_id=int(selected_pad_id),
            side=side,
        )
    return features


def truncate_features(
    features: ConversationFeatures, max_length: int, *, side: str = "right"
) -> ConversationFeatures:
    """Apply exactly one truncation slice to every sequence field."""

    if max_length < 0:
        raise ConversationMaskError("max_length must be non-negative")
    if side not in ("left", "right"):
        raise ConversationMaskError("truncation side must be 'left' or 'right'")
    length = len(features.input_ids)
    if length <= max_length:
        return features
    token_slice = (
        slice(0, max_length) if side == "right" else slice(length - max_length, length)
    )
    return _slice_features(features, token_slice)


def pad_features(
    features: ConversationFeatures,
    length: int,
    *,
    pad_token_id: int,
    side: str = "right",
) -> ConversationFeatures:
    """Pad all sequence fields while forcing padding perturbations to zero."""

    if side not in ("left", "right"):
        raise ConversationMaskError("padding side must be 'left' or 'right'")
    padding = length - len(features.input_ids)
    if padding < 0:
        raise ConversationMaskError(
            "padding length is shorter than the unpadded sequence"
        )
    if padding == 0:
        return features

    pad_ids = (pad_token_id,) * padding
    pad_labels = (IGNORE_INDEX,) * padding
    pad_attention = (0,) * padding
    pad_perturb = (0,) * padding
    pad_regions = (TokenRegion.PADDING,) * padding
    if side == "right":
        return ConversationFeatures(
            input_ids=features.input_ids + pad_ids,
            labels=features.labels + pad_labels,
            attention_mask=features.attention_mask + pad_attention,
            perturb_mask=features.perturb_mask + pad_perturb,
            token_regions=features.token_regions + pad_regions,
        )
    return ConversationFeatures(
        input_ids=pad_ids + features.input_ids,
        labels=pad_labels + features.labels,
        attention_mask=pad_attention + features.attention_mask,
        perturb_mask=pad_perturb + features.perturb_mask,
        token_regions=pad_regions + features.token_regions,
    )


def attach_perturb_mask_to_axolotl_sample(
    tokenized_sample: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any] | Message],
    tokenizer: OffsetTokenizer,
    *,
    adapter: ConversationMaskAdapter | None = None,
    validate_template: bool = True,
    truncation_side: str | None = None,
    validate_labels: bool = False,
) -> dict[str, Any]:
    """Attach a mask before Axolotl packing, with strict token-ID alignment.

    ``tokenized_sample`` may already be truncated and/or padded.  Packing must
    happen *after* this function so that Axolotl can concatenate
    ``perturb_mask`` using the same boundaries as its other sequence fields.
    Non-binary attention masks are rejected because they usually indicate that
    packing already happened.
    """

    if "input_ids" not in tokenized_sample:
        raise AlignmentError("tokenized sample has no input_ids")
    sample_ids = _flatten_1d(tokenized_sample["input_ids"], "input_ids")
    raw_attention = tokenized_sample.get("attention_mask")
    if raw_attention is None:
        attention = [1] * len(sample_ids)
    else:
        attention = _flatten_1d(raw_attention, "attention_mask")
        if len(attention) != len(sample_ids):
            raise AlignmentError("attention_mask length differs from input_ids")
        if any(value not in (0, 1) for value in attention):
            raise AlignmentError(
                "attach perturb_mask before sample packing; a non-binary "
                "attention_mask cannot be aligned as padding"
            )

    active_positions = [index for index, value in enumerate(attention) if value]
    _validate_contiguous_padding(active_positions, len(sample_ids))
    full = (adapter or Llama3ShareGPTMaskAdapter()).encode(
        messages, tokenizer, validate_template=validate_template
    )
    active_length = len(active_positions)
    if active_length > len(full.input_ids):
        raise AlignmentError(
            "Axolotl sample has more active tokens than the source conversation"
        )
    side = truncation_side or getattr(tokenizer, "truncation_side", "right")
    expected = truncate_features(full, active_length, side=side)
    active_ids = [sample_ids[index] for index in active_positions]
    if active_ids != list(expected.input_ids):
        raise AlignmentError(
            "Axolotl input_ids do not match the structural Llama-3 encoding "
            "under the configured truncation side"
        )

    aligned_mask = [0] * len(sample_ids)
    expected_labels = [IGNORE_INDEX] * len(sample_ids)
    for source_index, destination_index in enumerate(active_positions):
        aligned_mask[destination_index] = expected.perturb_mask[source_index]
        expected_labels[destination_index] = expected.labels[source_index]

    if validate_labels and "labels" in tokenized_sample:
        labels = _flatten_1d(tokenized_sample["labels"], "labels")
        if labels != expected_labels:
            raise AlignmentError(
                "Axolotl labels differ from assistant-body-only labels "
                "(headers, separators, user/system, and padding must be -100)"
            )

    result = dict(tokenized_sample)
    result["perturb_mask"] = _restore_container_type(
        aligned_mask, tokenized_sample["input_ids"]
    )
    return result


class AxolotlUserTokenMaskTransform:
    """Callable dataset-map adapter to run after tokenization, before packing."""

    def __init__(
        self,
        tokenizer: OffsetTokenizer,
        *,
        conversation_field: str = "conversations",
        adapter: ConversationMaskAdapter | None = None,
        validate_template: bool = True,
        validate_labels: bool = False,
        truncation_side: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.conversation_field = conversation_field
        self.adapter = adapter
        self.validate_template = validate_template
        self.validate_labels = validate_labels
        self.truncation_side = truncation_side

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        if self.conversation_field not in sample:
            raise AlignmentError(
                f"sample has no {self.conversation_field!r} conversation field"
            )
        return attach_perturb_mask_to_axolotl_sample(
            sample,
            sample[self.conversation_field],
            self.tokenizer,
            adapter=self.adapter,
            validate_template=self.validate_template,
            truncation_side=self.truncation_side,
            validate_labels=self.validate_labels,
        )


def reference_pack_sequence_fields(
    samples: Iterable[Mapping[str, Sequence[int]]],
    sequence_length: int,
    *,
    pad_token_id: int | None = None,
    pad_to_sequence_length: bool = False,
    drop_last: bool = False,
) -> list[dict[str, list[int]]]:
    """Small reference packer used to test the four-field invariant.

    Production should let the pinned Axolotl sample packer own boundaries and
    register ``perturb_mask`` as an additional sequence field.  This helper
    deliberately has no position-ID or block-attention policy; it only proves
    that concatenation/chunking/padding preserve mask alignment.
    """

    if sequence_length <= 0:
        raise ConversationMaskError("sequence_length must be positive")
    flattened: dict[str, list[int]] = {field: [] for field in SEQUENCE_FIELDS}
    for sample_index, sample in enumerate(samples):
        missing = [field for field in SEQUENCE_FIELDS if field not in sample]
        if missing:
            raise ConversationMaskError(
                f"sample {sample_index} is missing fields: {', '.join(missing)}"
            )
        lengths = {len(sample[field]) for field in SEQUENCE_FIELDS}
        if len(lengths) != 1:
            raise ConversationMaskError(
                f"sample {sample_index} sequence fields have different lengths"
            )
        for field in SEQUENCE_FIELDS:
            flattened[field].extend(int(value) for value in sample[field])

    total = len(flattened["input_ids"])
    packed: list[dict[str, list[int]]] = []
    for start in range(0, total, sequence_length):
        end = min(start + sequence_length, total)
        if end - start < sequence_length and drop_last:
            break
        chunk = {field: flattened[field][start:end] for field in SEQUENCE_FIELDS}
        if end - start < sequence_length and pad_to_sequence_length:
            if pad_token_id is None:
                raise ConversationMaskError(
                    "pad_to_sequence_length requires pad_token_id"
                )
            amount = sequence_length - (end - start)
            chunk["input_ids"].extend([pad_token_id] * amount)
            chunk["labels"].extend([IGNORE_INDEX] * amount)
            chunk["attention_mask"].extend([0] * amount)
            chunk["perturb_mask"].extend([0] * amount)
        packed.append(chunk)
    return packed


def causal_prediction_mask(labels: Any, *, ignore_index: int = IGNORE_INDEX) -> Any:
    """Return the assistant mask aligned with ``logits[..., :-1, :]``.

    Decoder-only causal loss compares ``logits[..., :-1, :]`` with
    ``labels[..., 1:]``.  This helper prevents accidentally applying a
    same-position assistant mask to prediction logits.
    """

    try:
        import torch
    except ImportError:  # pragma: no cover - the project training env has torch
        torch = None

    if torch is not None and isinstance(labels, torch.Tensor):
        return labels[..., 1:].ne(ignore_index)

    values = list(labels)
    if values and isinstance(values[0], (list, tuple)):
        return [[value != ignore_index for value in row[1:]] for row in values]
    return [value != ignore_index for value in values[1:]]


def native_llama3_packed_user_mask(
    input_ids: Any,
    position_ids: Any,
    attention_mask: Any,
    *,
    user_header_ids: Sequence[int],
    eot_token_id: int,
    special_token_ids: Sequence[int],
) -> tuple[Any, Any]:
    """Derive question-body masks from native packed Axolotl tensors.

    This is intentionally a trainer-time operation: Axolotl owns tokenization,
    labels, truncation, padding, and packing without a tokenizer hook.  The
    canonical Llama-3 user header opens a question span and its structural EOT
    closes it.  Position resets supply packed-source provenance; an
    ``attention_mask=0`` token at position zero is the beginning of right
    padding (ordinary assistant EOS tokens have nonzero positions).

    The returned ``source_ids`` are local provenance labels suitable for the
    existing relative-radius alignment check.  They are not persisted into or
    used to modify the native dataset.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training requires torch
        raise ConversationMaskError("native packed masking requires torch") from exc

    if not all(isinstance(value, torch.Tensor) for value in (
        input_ids,
        position_ids,
        attention_mask,
    )):
        raise TypeError("native packed masking requires tensor inputs")
    if input_ids.ndim != 2 or position_ids.shape != input_ids.shape:
        raise AlignmentError(
            "native input_ids and position_ids must have identical [B, T] shapes"
        )
    if attention_mask.shape != input_ids.shape:
        raise AlignmentError(
            "native attention_mask must have the same shape as input_ids"
        )
    header = tuple(int(value) for value in user_header_ids)
    if not header:
        raise AlignmentError("native Llama-3 user header tokenization is empty")
    if len(header) > input_ids.shape[1]:
        raise AlignmentError("native batch is shorter than the Llama-3 user header")

    device = input_ids.device
    batch_size, sequence_length = input_ids.shape
    event_positions = torch.full(
        (batch_size, sequence_length), -1, device=device, dtype=torch.long
    )
    event_is_user = torch.zeros(
        (batch_size, sequence_length), device=device, dtype=torch.bool
    )
    indexes = torch.arange(sequence_length, device=device, dtype=torch.long)
    indexes = indexes.unsqueeze(0).expand(batch_size, -1)

    # Every packed source starts at position zero.  This clears any state from
    # the preceding source before a later user header can open a new span.
    resets = position_ids.eq(0)
    event_positions[resets] = indexes[resets]

    pattern = torch.tensor(header, device=device, dtype=input_ids.dtype)
    windows = input_ids.unfold(1, len(header), 1)
    header_starts = windows.eq(pattern).all(dim=-1).nonzero(as_tuple=False)
    if header_starts.numel():
        body_positions = header_starts[:, 1] + len(header)
        in_bounds = body_positions.lt(sequence_length)
        rows = header_starts[in_bounds, 0]
        columns = body_positions[in_bounds]
        event_positions[rows, columns] = columns
        event_is_user[rows, columns] = True

    # EOT wins if an empty question makes its body start collide with EOT.
    eot = input_ids.eq(int(eot_token_id))
    event_positions[eot] = indexes[eot]
    event_is_user[eot] = False

    latest_event = event_positions.cummax(dim=1).values
    gathered = event_is_user.gather(1, latest_event.clamp_min(0))
    perturb_mask = gathered & latest_event.ge(0)

    # The official setup aliases the pad token to EOS.  A zero-attention token
    # at nonzero position is therefore a real appended assistant EOS, while a
    # zero-attention token at reset position starts the right-padding suffix.
    padding_start = resets & attention_mask.eq(0)
    padding_reached = padding_start.to(torch.long).cumsum(dim=1).gt(0)
    active = ~padding_reached
    real_starts = resets & active
    source_ids = real_starts.to(torch.long).cumsum(dim=1) - 1
    source_ids = source_ids.masked_fill(~active, -1)

    if special_token_ids:
        special = torch.zeros_like(perturb_mask)
        for token_id in {int(value) for value in special_token_ids}:
            special |= input_ids.eq(token_id)
        perturb_mask &= ~special
    perturb_mask &= active

    if bool((perturb_mask & source_ids.lt(0)).any()):
        raise AlignmentError("native question mask selects right padding")
    if not bool(perturb_mask.any()):
        raise AlignmentError("native packed batch has no user/question body tokens")
    return perturb_mask, source_ids


def _axolotl_token_ids(
    strategy: Any,
    text: str,
    *,
    add_eos_token: bool = False,
    strip_bos_token: bool,
) -> list[int]:
    """Return IDs from the pinned Axolotl path for one structural piece."""

    tokenized = strategy._tokenize(  # pylint: disable=protected-access
        text,
        add_eos_token=add_eos_token,
        strip_bos_token=strip_bos_token,
    )
    return [int(token_id) for token_id in tokenized["input_ids"]]


def tokenize_axolotl_llama3_prompt_with_mask(
    strategy: Any,
    prompt: Mapping[str, Any],
) -> dict[str, list[int]]:
    """Tokenize ShareGPT with Axolotl IDs and configured assistant targets.

    The pinned reconstruction commit tokenizes Llama-3 one role/content turn at
    a time. This hook follows that exact input path and changes only metadata:
    ``perturb_mask`` selects complete user-body tokens, while labels retain
    complete assistant-body token IDs. Headers, system text, EOT/EOS/BOS,
    mixed boundary tokens, and padding are excluded from both objectives.
    When ``strategy.supervise_assistant_eot`` is true, the structural EOT that
    terminates each assistant turn is additionally retained as a label.  It is
    never part of ``perturb_mask``.  Any Axolotl-appended EOS remains ignored.
    """

    conversation = strategy.prompter._conversation.copy()  # pylint: disable=protected-access
    if conversation.name != "llama3":
        raise ConversationMaskError(
            "question-only VAT mask hook is restricted to conversation=llama3"
        )
    separator = getattr(conversation, "stop_str", None) or "<|eot_id|>"
    if separator != "<|eot_id|>":
        raise TemplateMismatchError(
            f"expected Llama-3 EOT separator, got {separator!r}"
        )

    result: dict[str, list[int]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "perturb_mask": [],
    }
    thread = strategy.get_conversation_thread(prompt)
    for part in strategy.prompter.build_prompt(thread):
        if not isinstance(part, tuple) or len(part) != 2:
            raise ConversationMaskError(
                f"expected Axolotl role/content tuple, got {part!r}"
            )
        role, content = part
        empty_role = role.strip() == ""
        rendered_role: str | None = None
        if not empty_role:
            header_start = "<|start_header_id|>"
            header_end = "<|end_header_id|>"
            if not role.startswith(header_start) or header_end not in role:
                raise TemplateMismatchError(
                    f"unrecognized Llama-3 role header: {role!r}"
                )
            header_end_index = role.index(header_end, len(header_start))
            rendered_role = role[len(header_start) : header_end_index].strip().lower()
            structural_suffix = role[header_end_index + len(header_end) :]
            if structural_suffix != "\n\n":
                raise TemplateMismatchError(
                    "Llama-3 role header does not end with the canonical separator"
                )
            if rendered_role not in {"system", "user", "assistant"}:
                raise ConversationMaskError(
                    f"unsupported rendered Llama-3 role: {rendered_role!r}"
                )
        input_turn = rendered_role == "user"
        output_turn = rendered_role == "assistant"

        turn = role + content
        strip_bos = not empty_role
        tokenized = strategy._tokenize(  # pylint: disable=protected-access
            turn,
            add_eos_token=bool(output_turn),
            strip_bos_token=strip_bos,
        )
        ids = [int(token_id) for token_id in tokenized["input_ids"]]

        body_start_token = 0
        body_end_token = 0
        if not empty_role:
            if not content.endswith(separator):
                raise TemplateMismatchError(
                    "Axolotl Llama-3 content does not end with structural EOT"
                )
            role_ids = _axolotl_token_ids(
                strategy,
                role,
                strip_bos_token=True,
            )
            base_turn_ids = _axolotl_token_ids(
                strategy,
                turn,
                strip_bos_token=True,
            )
            separator_ids = _axolotl_token_ids(
                strategy,
                separator,
                strip_bos_token=True,
            )
            if not role_ids or not separator_ids:
                raise AlignmentError("Llama-3 role or EOT tokenization is empty")
            if base_turn_ids[: len(role_ids)] != role_ids:
                raise AlignmentError(
                    "Llama-3 role tokenization is not an exact turn prefix; "
                    "a role/body boundary token would be ambiguous"
                )
            if ids[: len(base_turn_ids)] != base_turn_ids:
                raise AlignmentError(
                    "Axolotl turn IDs differ before its optional assistant EOS"
                )
            trailing_ids = ids[len(base_turn_ids) :]
            if trailing_ids not in ([], [int(strategy.tokenizer.eos_token_id)]):
                raise AlignmentError(
                    "Axolotl appended unexpected tokens after the Llama-3 turn"
                )

            body_start_token = len(role_ids)
            if base_turn_ids[-len(separator_ids) :] == separator_ids:
                body_end_token = len(base_turn_ids) - len(separator_ids)
            elif len(base_turn_ids) == strategy.max_length:
                # Right truncation can remove the structural EOT. Exact prefix
                # matching above proves all remaining tokens follow the role.
                body_end_token = len(base_turn_ids)
            else:
                raise AlignmentError(
                    "Llama-3 EOT tokenization is not an exact turn suffix"
                )
            if body_end_token < body_start_token:
                raise AlignmentError("Llama-3 structural token boundaries overlap")

        special_ids = {
            int(token_id)
            for token_id in getattr(strategy.tokenizer, "all_special_ids", ())
        }
        get_added_vocab = getattr(strategy.tokenizer, "get_added_vocab", None)
        if callable(get_added_vocab):
            special_ids.update(int(token_id) for token_id in get_added_vocab().values())
        complete_body = [
            int(
                not empty_role
                and body_start_token <= index < body_end_token
                and token_id not in special_ids
            )
            for index, token_id in enumerate(ids)
        ]
        perturb = [int(input_turn and is_body) for is_body in complete_body]
        supervise_assistant_eot = bool(
            getattr(strategy, "supervise_assistant_eot", False)
        )
        assistant_eot = [
            int(
                output_turn
                and supervise_assistant_eot
                and body_end_token <= index < len(base_turn_ids)
            )
            for index in range(len(ids))
        ]
        labels = [
            token_id
            if output_turn and (is_body or is_assistant_eot)
            else IGNORE_INDEX
            for token_id, is_body, is_assistant_eot in zip(
                ids, complete_body, assistant_eot
            )
        ]

        result["input_ids"].extend(ids)
        # Match Axolotl's parse_tokenized_to_result exactly. In the official
        # config pad_token is <|end_of_text|>, so its deliberately appended EOS
        # also receives attention value 0 in the historical input pipeline.
        result["attention_mask"].extend(
            int(token_id != strategy.tokenizer.pad_token_id) for token_id in ids
        )
        result["labels"].extend(labels)
        result["perturb_mask"].extend(perturb)

    lengths = {len(values) for values in result.values()}
    if len(lengths) != 1:
        raise AlignmentError("Axolotl Llama-3 sequence fields are not aligned")
    canonical_prompt = json.dumps(
        prompt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    source_id = int.from_bytes(hashlib.sha256(canonical_prompt).digest()[:8], "big") & (
        (1 << 63) - 1
    )
    result["source_ids"] = [source_id] * len(result["input_ids"])
    return result


def _regions_from_offsets(
    offsets: Sequence[tuple[int, int]], spans: Sequence[CharacterSpan]
) -> tuple[TokenRegion, ...]:
    starts = [span.start for span in spans]
    regions: list[TokenRegion] = []
    for start, end in offsets:
        if end <= start:
            regions.append(TokenRegion.SPECIAL)
            continue
        span_index = bisect_right(starts, start) - 1
        if span_index >= 0:
            span = spans[span_index]
            if span.start <= start and end <= span.end:
                regions.append(span.region)
                continue
        regions.append(TokenRegion.MIXED)
    return tuple(regions)


def _slice_features(
    features: ConversationFeatures, token_slice: slice
) -> ConversationFeatures:
    return ConversationFeatures(
        input_ids=features.input_ids[token_slice],
        labels=features.labels[token_slice],
        attention_mask=features.attention_mask[token_slice],
        perturb_mask=features.perturb_mask[token_slice],
        token_regions=features.token_regions[token_slice],
    )


def _flatten_1d(value: Any, name: str) -> list[int]:
    if value is None:
        raise ConversationMaskError(f"tokenizer returned no {name}")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ConversationMaskError(f"{name} must be a one-dimensional sequence")
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if any(isinstance(item, (list, tuple)) for item in value):
        raise ConversationMaskError(f"{name} must be one-dimensional")
    return [int(item) for item in value]


def _flatten_offsets(value: Any) -> list[tuple[int, int]]:
    if value is None:
        raise ConversationMaskError(
            "tokenizer returned no offset_mapping; use a fast tokenizer"
        )
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
        and value[0]
        and isinstance(value[0][0], (list, tuple))
    ):
        value = value[0]
    try:
        return [(int(start), int(end)) for start, end in value]
    except (TypeError, ValueError) as exc:
        raise ConversationMaskError(
            "offset_mapping must contain (start, end) pairs"
        ) from exc


def _validate_contiguous_padding(
    active_positions: Sequence[int], total_length: int
) -> None:
    if not active_positions:
        return
    expected = list(range(active_positions[0], active_positions[-1] + 1))
    if list(active_positions) != expected:
        raise AlignmentError(
            "active attention positions are not contiguous; attach the mask "
            "before packing or block-attention transformation"
        )
    if active_positions[0] != 0 and active_positions[-1] != total_length - 1:
        raise AlignmentError("padding may occur on only one side")


def _restore_container_type(values: Sequence[int], exemplar: Any) -> Any:
    if hasattr(exemplar, "new_tensor"):
        return exemplar.new_tensor(values)
    if isinstance(exemplar, tuple):
        return tuple(values)
    return list(values)
