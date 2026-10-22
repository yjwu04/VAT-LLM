from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from types import SimpleNamespace

import pytest
import torch

from src.user_token_mask import (
    IGNORE_INDEX,
    AlignmentError,
    AxolotlUserTokenMaskTransform,
    ConversationMaskError,
    Llama3ShareGPTMaskAdapter,
    TemplateMismatchError,
    TokenRegion,
    attach_perturb_mask_to_axolotl_sample,
    build_conversation_features,
    causal_prediction_mask,
    native_llama3_packed_user_mask,
    reference_pack_sequence_fields,
    tokenize_axolotl_llama3_prompt_with_mask,
)


def test_native_packed_mask_preserves_native_labels_and_packed_sources():
    # Two native packed conversations followed by EOS-valued right padding.
    # Real assistant EOS tokens also have attention zero, but their positions
    # are nonzero; this is the historical Axolotl/Llama-3 behavior.
    user_header = [10, 11, 12]
    first = [1, 20, 21, 99, 10, 11, 12, 30, 31, 99, 40, 41, 99, 2]
    second = [1, 10, 11, 12, 60, 77, 99, 40, 61, 99, 2]
    padding = [2, 2, 2]
    input_ids = torch.tensor([first + second + padding])
    position_ids = torch.tensor(
        [list(range(len(first))) + list(range(len(second))) + [0, 0, 0]]
    )
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, len(first) - 1] = 0
    attention_mask[0, len(first) + len(second) - 1 :] = 0
    native_labels = torch.tensor(
        [[-100] * 11 + first[11:] + [-100] * 8 + second[8:] + [-100] * 3]
    )
    labels_before = native_labels.clone()

    perturb_mask, source_ids = native_llama3_packed_user_mask(
        input_ids,
        position_ids,
        attention_mask,
        user_header_ids=user_header,
        eot_token_id=99,
        special_token_ids=[1, 2, 10, 11, 12, 77, 99],
    )

    selected = input_ids[perturb_mask].tolist()
    assert selected == [30, 31, 60]
    assert torch.equal(native_labels, labels_before)
    assert not (perturb_mask & native_labels.ne(IGNORE_INDEX)).any()
    assert source_ids[0, : len(first)].eq(0).all()
    assert source_ids[0, len(first) : len(first) + len(second)].eq(1).all()
    assert source_ids[0, -3:].eq(-1).all()


SPECIAL_TOKENS = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
)


class FakeLlama3FastTokenizer:
    """Dependency-free tokenizer double with Hugging Face-style offsets."""

    pad_token_id = 0
    truncation_side = "right"
    padding_side = "right"

    def __init__(self, *, ordinary_chunk_size: int = 1) -> None:
        self.ordinary_chunk_size = ordinary_chunk_size
        self._special_ids = {
            token: index + 1 for index, token in enumerate(SPECIAL_TOKENS)
        }

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        assert not add_generation_prompt
        rendered = ["<|begin_of_text|>"]
        for message in conversation:
            rendered.extend(
                (
                    "<|start_header_id|>",
                    message["role"],
                    "<|end_header_id|>",
                    "\n\n",
                    message["content"],
                    "<|eot_id|>",
                )
            )
        text = "".join(rendered)
        if not tokenize:
            return text
        return self(
            text,
            add_special_tokens=False,
            return_offsets_mapping=False,
            return_attention_mask=False,
        )["input_ids"]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
        return_attention_mask: bool,
    ):
        assert not add_special_tokens
        del return_attention_mask
        input_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(text):
            special = next(
                (token for token in SPECIAL_TOKENS if text.startswith(token, cursor)),
                None,
            )
            if special is not None:
                end = cursor + len(special)
                input_ids.append(self._special_ids[special])
                offsets.append((cursor, end))
                cursor = end
                continue

            next_special = min(
                (
                    position
                    for token in SPECIAL_TOKENS
                    if (position := text.find(token, cursor)) >= 0
                ),
                default=len(text),
            )
            end = min(cursor + self.ordinary_chunk_size, next_special)
            piece = text[cursor:end]
            input_ids.append(
                100 + sum((index + 1) * ord(char) for index, char in enumerate(piece))
            )
            offsets.append((cursor, end))
            cursor = end

        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class FakeAxolotlTokenizer(FakeLlama3FastTokenizer):
    """Tokenizer double for the pinned Axolotl per-turn tokenization API."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    @property
    def all_special_ids(self):
        return list(self._special_ids.values())

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        return_attention_mask: bool = True,
        **_kwargs,
    ):
        result = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
            return_attention_mask=return_attention_mask,
        )
        result["attention_mask"] = [1] * len(result["input_ids"])
        return result


class FakeAxolotlLlama3Strategy:
    """Small behavioral double for commit 8a20a7b's Llama-3 strategy."""

    max_length = 512

    def __init__(self) -> None:
        self.tokenizer = FakeAxolotlTokenizer()
        conversation = SimpleNamespace(
            name="llama3",
            roles=("user", "assistant"),
            stop_str="<|eot_id|>",
        )
        conversation.copy = lambda: copy(conversation)
        self.prompter = SimpleNamespace(
            _conversation=conversation,
            roles=None,
            build_prompt=lambda _thread: [
                ("", "<|begin_of_text|>"),
                (
                    "<|start_header_id|>user<|end_header_id|>\n\n",
                    "Question<|eot_id|>",
                ),
                (
                    "<|start_header_id|>assistant<|end_header_id|>\n\n",
                    "Answer<|eot_id|>",
                ),
            ],
        )

    @staticmethod
    def get_conversation_thread(prompt):
        return prompt["conversations"]

    def _tokenize(self, turn, *, add_eos_token, strip_bos_token):
        tokenized = self.tokenizer(
            turn,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            return_offsets_mapping=False,
        )
        ids = list(tokenized["input_ids"])
        attention = list(tokenized["attention_mask"])
        if (
            ids
            and ids[-1] != self.tokenizer.eos_token_id
            and len(ids) < self.max_length
            and add_eos_token
        ):
            ids.append(self.tokenizer.eos_token_id)
            attention.append(1)
        if ids and ids[0] == self.tokenizer.bos_token_id and strip_bos_token:
            ids = ids[1:]
            attention = attention[1:]
        return {"input_ids": ids, "attention_mask": attention}


@pytest.fixture
def tokenizer() -> FakeLlama3FastTokenizer:
    return FakeLlama3FastTokenizer()


def _positions(features, region: TokenRegion) -> list[int]:
    return [
        index
        for index, token_region in enumerate(features.token_regions)
        if token_region is region
    ]


def test_single_turn_masks_only_user_body_and_labels_only_assistant_body(tokenizer):
    messages = [
        {"from": "human", "value": "user"},
        {"from": "gpt", "value": "OK"},
    ]
    features = build_conversation_features(messages, tokenizer)

    user_positions = _positions(features, TokenRegion.USER_BODY)
    assistant_positions = _positions(features, TokenRegion.ASSISTANT_BODY)
    assert len(user_positions) == len("user")
    assert len(assistant_positions) == len("OK")
    assert [features.perturb_mask[index] for index in user_positions] == [1] * 4
    assert all(
        features.perturb_mask[index] == 0
        for index in range(len(features.input_ids))
        if index not in user_positions
    )
    assert all(
        features.labels[index] == features.input_ids[index]
        for index in assistant_positions
    )
    assert all(
        features.labels[index] == IGNORE_INDEX
        for index in range(len(features.input_ids))
        if index not in assistant_positions
    )


def test_optional_assistant_eot_is_labeled_but_never_perturbed(tokenizer):
    features = build_conversation_features(
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "R"},
            {"role": "assistant", "content": "B"},
        ],
        tokenizer,
        adapter=Llama3ShareGPTMaskAdapter(supervise_assistant_eot=True),
    )
    eot_positions = _positions(features, TokenRegion.ASSISTANT_EOT)
    assert len(eot_positions) == 2
    assert all(features.perturb_mask[index] == 0 for index in eot_positions)
    assert all(
        features.labels[index] == features.input_ids[index]
        for index in eot_positions
    )
    assert sum(features.perturb_mask) == 2


def test_multiturn_system_headers_and_separators_never_leak(tokenizer):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "same"},
        {"role": "assistant", "content": "same"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": "assistant"},
    ]
    features = build_conversation_features(messages, tokenizer)

    assert sum(features.perturb_mask) == len("same") + len("user")
    for index, region in enumerate(features.token_regions):
        if region in {
            TokenRegion.SPECIAL,
            TokenRegion.ROLE_HEADER,
            TokenRegion.SEPARATOR,
            TokenRegion.SYSTEM_BODY,
            TokenRegion.ASSISTANT_BODY,
        }:
            assert features.perturb_mask[index] == 0
        if region is not TokenRegion.ASSISTANT_BODY:
            assert features.labels[index] == IGNORE_INDEX


def test_optional_axolotl_eos_is_special_and_never_masked_or_labeled(tokenizer):
    features = build_conversation_features(
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ],
        tokenizer,
        adapter=Llama3ShareGPTMaskAdapter(append_eos=True),
    )
    assert features.token_regions[-1] is TokenRegion.SPECIAL
    assert features.perturb_mask[-1] == 0
    assert features.labels[-1] == IGNORE_INDEX


def test_whole_template_tokenization_conservatively_excludes_boundary_token():
    tokenizer = FakeLlama3FastTokenizer(ordinary_chunk_size=3)
    features = build_conversation_features(
        [
            {"role": "user", "content": "X"},
            {"role": "assistant", "content": "Y"},
        ],
        tokenizer,
    )

    # The tokenizer merges "\n\nX" and "\n\nY". Those tokens cross a
    # separator/body boundary and must be excluded from both objectives.
    mixed = _positions(features, TokenRegion.MIXED)
    assert len(mixed) == 2
    assert all(features.perturb_mask[index] == 0 for index in mixed)
    assert all(features.labels[index] == IGNORE_INDEX for index in mixed)


def test_right_and_left_truncation_apply_identical_slice_to_all_fields(tokenizer):
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "tail"},
    ]
    full = build_conversation_features(messages, tokenizer)
    second_user_start = [
        index
        for index, region in enumerate(full.token_regions)
        if region is TokenRegion.USER_BODY
    ][len("first")]
    max_length = second_user_start + 2

    right = build_conversation_features(
        messages, tokenizer, max_length=max_length, truncation_side="right"
    )
    left = build_conversation_features(
        messages, tokenizer, max_length=max_length, truncation_side="left"
    )
    assert right.as_dict() == {
        field: values[:max_length] for field, values in full.as_dict().items()
    }
    assert left.as_dict() == {
        field: values[-max_length:] for field, values in full.as_dict().items()
    }
    assert right.perturb_mask[-2:] == (1, 1)


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_padding_is_zero_for_attention_labels_and_perturbation(tokenizer, padding_side):
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
    unpadded = build_conversation_features(messages, tokenizer)
    padded = build_conversation_features(
        messages,
        tokenizer,
        padding_length=len(unpadded.input_ids) + 4,
        padding_side=padding_side,
    )
    pad_slice = slice(0, 4) if padding_side == "left" else slice(-4, None)
    assert padded.input_ids[pad_slice] == (tokenizer.pad_token_id,) * 4
    assert padded.labels[pad_slice] == (IGNORE_INDEX,) * 4
    assert padded.attention_mask[pad_slice] == (0,) * 4
    assert padded.perturb_mask[pad_slice] == (0,) * 4
    assert padded.token_regions[pad_slice] == (TokenRegion.PADDING,) * 4


def test_reference_packing_preserves_mask_at_sample_and_chunk_boundaries(tokenizer):
    first = build_conversation_features(
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "A"},
        ],
        tokenizer,
    ).as_dict()
    second_features = build_conversation_features(
        [
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "B"},
        ],
        tokenizer,
    )
    second = second_features.as_dict()
    second_user_end = max(_positions(second_features, TokenRegion.USER_BODY)) + 1
    sequence_length = len(first["input_ids"]) + second_user_end - 1

    packed = reference_pack_sequence_fields(
        [first, second],
        sequence_length,
        pad_token_id=tokenizer.pad_token_id,
        pad_to_sequence_length=True,
    )
    expected_mask = first["perturb_mask"] + second["perturb_mask"]
    observed_mask = [
        value
        for chunk in packed
        for value, attention in zip(chunk["perturb_mask"], chunk["attention_mask"])
        if attention
    ]
    assert observed_mask == expected_mask
    assert packed[0]["perturb_mask"][-1] == 1
    assert all(
        not perturb
        for chunk in packed
        for perturb, attention in zip(chunk["perturb_mask"], chunk["attention_mask"])
        if not attention
    )


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_axolotl_adapter_attaches_to_truncated_padded_sample(tokenizer, padding_side):
    messages = [
        {"from": "human", "value": "question"},
        {"from": "gpt", "value": "answer"},
    ]
    tokenizer.padding_side = padding_side
    features = build_conversation_features(
        messages,
        tokenizer,
        max_length=25,
        padding_length=29,
        padding_side=padding_side,
    )
    sample = {
        "input_ids": list(features.input_ids),
        "labels": list(features.labels),
        "attention_mask": list(features.attention_mask),
        "conversations": messages,
        "unrelated": "preserved",
    }
    transformed = AxolotlUserTokenMaskTransform(tokenizer, validate_labels=True)(sample)
    assert transformed["perturb_mask"] == list(features.perturb_mask)
    assert transformed["unrelated"] == "preserved"
    assert "perturb_mask" not in sample


def test_axolotl_adapter_rejects_token_or_label_mismatch(tokenizer):
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
    features = build_conversation_features(messages, tokenizer)
    sample = features.as_dict()
    sample["input_ids"][1] += 99
    with pytest.raises(AlignmentError, match="input_ids do not match"):
        attach_perturb_mask_to_axolotl_sample(sample, messages, tokenizer)

    sample = features.as_dict()
    assistant_position = _positions(features, TokenRegion.ASSISTANT_BODY)[0]
    sample["labels"][assistant_position] = IGNORE_INDEX
    with pytest.raises(AlignmentError, match="labels differ"):
        attach_perturb_mask_to_axolotl_sample(
            sample, messages, tokenizer, validate_labels=True
        )


def test_axolotl_adapter_rejects_post_packing_attention(tokenizer):
    messages = [{"role": "user", "content": "Q"}]
    features = build_conversation_features(messages, tokenizer)
    sample = features.as_dict()
    sample["attention_mask"][1] = 2
    with pytest.raises(AlignmentError, match="before sample packing"):
        attach_perturb_mask_to_axolotl_sample(sample, messages, tokenizer)


def test_template_mismatch_fails_instead_of_fuzzy_search(tokenizer):
    tokenizer.apply_chat_template = lambda *args, **kwargs: "different"
    with pytest.raises(TemplateMismatchError, match="differs"):
        build_conversation_features([{"role": "user", "content": "Q"}], tokenizer)


def test_padding_without_pad_id_fails_clearly(tokenizer):
    tokenizer.pad_token_id = None
    with pytest.raises(ConversationMaskError, match="pad_token_id"):
        build_conversation_features(
            [{"role": "user", "content": "Q"}],
            tokenizer,
            padding_length=20,
        )


def test_causal_shift_uses_labels_at_t_plus_one(tokenizer):
    torch = pytest.importorskip("torch")
    features = build_conversation_features(
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "AB"},
        ],
        tokenizer,
    )
    labels = torch.tensor(features.labels).unsqueeze(0)
    prediction_mask = causal_prediction_mask(labels)
    expected = torch.tensor(
        [
            [
                region is TokenRegion.ASSISTANT_BODY
                for region in features.token_regions[1:]
            ]
        ]
    )
    assert prediction_mask.shape[-1] == labels.shape[-1] - 1
    assert torch.equal(prediction_mask, expected)

    target_positions = prediction_mask[0].nonzero().flatten().tolist()
    assistant_positions = _positions(features, TokenRegion.ASSISTANT_BODY)
    assert target_positions == [position - 1 for position in assistant_positions]


def test_causal_shift_supports_python_batches():
    labels = [[IGNORE_INDEX, 10, 11], [IGNORE_INDEX, IGNORE_INDEX, 12]]
    assert causal_prediction_mask(labels) == [
        [True, True],
        [False, True],
    ]


def test_pinned_axolotl_path_preserves_inputs_and_masks_only_message_bodies():
    strategy = FakeAxolotlLlama3Strategy()
    prompt = {
        "conversations": [
            {"from": "human", "value": "Question"},
            {"from": "gpt", "value": "Answer"},
        ]
    }
    result = tokenize_axolotl_llama3_prompt_with_mask(strategy, prompt)

    expected_ids: list[int] = []
    expected_attention: list[int] = []
    parts = strategy.prompter.build_prompt(prompt["conversations"])
    for role, content in parts:
        tokenized = strategy._tokenize(
            role + content,
            add_eos_token="assistant" in role,
            strip_bos_token=bool(role),
        )
        expected_ids.extend(tokenized["input_ids"])
        expected_attention.extend(
            int(token_id != strategy.tokenizer.pad_token_id)
            for token_id in tokenized["input_ids"]
        )

    assert result["input_ids"] == expected_ids
    assert result["attention_mask"] == expected_attention
    assert len({*map(len, result.values())}) == 1
    assert len(set(result["source_ids"])) == 1

    tokens = [
        next(
            (
                special
                for special, token_id in strategy.tokenizer._special_ids.items()
                if token_id == current_id
            ),
            None,
        )
        for current_id in result["input_ids"]
    ]
    structural_positions = [
        index for index, token in enumerate(tokens) if token is not None
    ]
    assert all(result["perturb_mask"][index] == 0 for index in structural_positions)
    assert all(
        result["labels"][index] == IGNORE_INDEX for index in structural_positions
    )
    assert sum(result["perturb_mask"]) == len("Question")
    assert sum(label != IGNORE_INDEX for label in result["labels"]) == len("Answer")
    # Axolotl appends EOS after the assistant turn, but the official
    # pad_token=end_of_text configuration gives that historical token attention 0.
    assert result["input_ids"][-1] == strategy.tokenizer.eos_token_id
    assert result["attention_mask"][-1] == 0
    assert result["perturb_mask"][-1] == 0
    assert result["labels"][-1] == IGNORE_INDEX


def test_pinned_axolotl_path_optionally_supervises_only_assistant_eot():
    strategy = FakeAxolotlLlama3Strategy()
    strategy.supervise_assistant_eot = True
    result = tokenize_axolotl_llama3_prompt_with_mask(
        strategy,
        {
            "conversations": [
                {"from": "human", "value": "Question"},
                {"from": "gpt", "value": "Answer"},
            ]
        },
    )
    eot_id = strategy.tokenizer._special_ids["<|eot_id|>"]
    supervised_eot = [
        index
        for index, (token_id, label) in enumerate(
            zip(result["input_ids"], result["labels"], strict=True)
        )
        if token_id == eot_id and label == eot_id
    ]
    assert len(supervised_eot) == 1
    assert all(result["perturb_mask"][index] == 0 for index in supervised_eot)
    assert result["labels"][-1] == IGNORE_INDEX
    assert sum(label != IGNORE_INDEX for label in result["labels"]) == len(
        "Answer"
    ) + 1


def test_pinned_axolotl_path_never_treats_third_system_role_as_user():
    strategy = FakeAxolotlLlama3Strategy()
    strategy.prompter._conversation.roles = ("user", "assistant", "system")
    strategy.prompter.build_prompt = lambda _thread: [
        ("", "<|begin_of_text|>"),
        (
            "<|start_header_id|>system<|end_header_id|>\n\n",
            "System policy<|eot_id|>",
        ),
        (
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "Question<|eot_id|>",
        ),
        (
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "Answer<|eot_id|>",
        ),
    ]
    result = tokenize_axolotl_llama3_prompt_with_mask(
        strategy,
        {
            "conversations": [
                {"from": "system", "value": "System policy"},
                {"from": "human", "value": "Question"},
                {"from": "gpt", "value": "Answer"},
            ]
        },
    )

    # The fake tokenizer maps ordinary characters one-to-one. Only the user
    # body is selected even when Axolotl exposes system as conversation role 3.
    assert sum(result["perturb_mask"]) == len("Question")
    assert sum(label != IGNORE_INDEX for label in result["labels"]) == len("Answer")


@pytest.mark.parametrize(
    "role",
    [
        "prefix user suffix",
        "<|start_header_id|>tool<|end_header_id|>\n\n",
        "<|start_header_id|>user<|end_header_id|>\n",
    ],
)
def test_pinned_axolotl_path_fails_closed_on_noncanonical_role_headers(role):
    strategy = FakeAxolotlLlama3Strategy()
    strategy.prompter.build_prompt = lambda _thread: [
        ("", "<|begin_of_text|>"),
        (role, "Question<|eot_id|>"),
    ]
    with pytest.raises((ConversationMaskError, TemplateMismatchError)):
        tokenize_axolotl_llama3_prompt_with_mask(
            strategy,
            {"conversations": [{"from": "human", "value": "Question"}]},
        )


def test_pinned_axolotl_source_id_is_stable_and_prompt_sensitive():
    strategy = FakeAxolotlLlama3Strategy()
    first = tokenize_axolotl_llama3_prompt_with_mask(
        strategy,
        {"conversations": [{"from": "human", "value": "one"}]},
    )
    repeated = tokenize_axolotl_llama3_prompt_with_mask(
        strategy,
        {"conversations": [{"from": "human", "value": "one"}]},
    )
    changed = tokenize_axolotl_llama3_prompt_with_mask(
        strategy,
        {"conversations": [{"from": "human", "value": "two"}]},
    )

    assert first["source_ids"][0] == repeated["source_ids"][0]
    assert first["source_ids"][0] != changed["source_ids"][0]
    assert 0 <= first["source_ids"][0] < 2**63
