from __future__ import annotations

import json
import sqlite3
import struct

import pytest
import torch

from src.metamorphic_candidates import (
    CHARACTER_SUBTYPES,
    FAMILY_CHARACTER_NOISE,
    FAMILY_CODE_SWITCH,
    FAMILY_ORDER,
    FAMILY_TOKENIZATION,
    TOKENIZATION_SUBTYPES,
    _eligible_words,
    generate_family_edit,
    protected_spans,
    validate_single_local_edit,
)
from src.metamorphic_vat import (
    CandidateSequence,
    MetamorphicRuntimeConfig,
    SOFT_WORST_CALIBRATION_PROTOCOL,
    SOFT_WORST_CALIBRATION_SEED,
    SOFT_WORST_CALIBRATION_SUBSET_SIZE,
    pack_candidate_sequences,
    select_uniform_family_indices,
    select_worst_candidate,
    soft_worst_aggregation,
    token_weighted_soft_worst_loss,
)
from src.vat_loss import masked_kl_divergence


def test_one_local_edit_and_protected_constraints() -> None:
    text = 'Return exactly three items; do not alter x = y + 1. Explain the main reason.'
    for index, (family, subtype) in enumerate(
        (
            (FAMILY_CHARACTER_NOISE, "adjacent_swap"),
            (FAMILY_TOKENIZATION, "intra_word_space"),
            (FAMILY_CODE_SWITCH, "semantic_equivalent_translation"),
        )
    ):
        edit = generate_family_edit(text, family=family, subtype=subtype, seed=100 + index)
        validate_single_local_edit(text, edit)
        assert "exactly three" in edit.candidate_text
        assert "do not" in edit.candidate_text
        assert "x = y + 1" in edit.candidate_text
        assert edit.candidate_text == (
            text[: edit.span_start] + edit.replacement + text[edit.span_end :]
        )


def test_outer_quote_is_formatting_but_embedded_quote_is_protected() -> None:
    wrapped = '"What are some interesting facts about Pyongyang?"'
    edit = generate_family_edit(
        wrapped,
        family=FAMILY_CODE_SWITCH,
        subtype="semantic_equivalent_translation",
        seed=42,
    )
    validate_single_local_edit(wrapped, edit)
    assert edit.replacement in {"哪些是", "有趣", "事实"}
    embedded = 'Explain the phrase "Return exactly three items" briefly.'
    quote_start = embedded.index('"')
    quote_end = embedded.rindex('"') + 1
    assert any(start <= quote_start and end >= quote_end for start, end in protected_spans(embedded))


def test_apostrophes_are_not_mistaken_for_quoted_material() -> None:
    text = "What's the best way to start writing a children's book?"
    edit = generate_family_edit(
        text,
        family=FAMILY_CHARACTER_NOISE,
        subtype="visual_substitution",
        seed=11,
    )
    validate_single_local_edit(text, edit)


def test_formula_protection_does_not_consume_malformed_natural_text() -> None:
    formula = "Do not alter x = y + 1."
    start = formula.index("x")
    end = formula.index(".")
    assert any(left <= start and right >= end for left, right in protected_spans(formula))
    malformed = 'Is it true that=""rich people have more health problems?'
    assert any(match.group(0) == "rich" for match in _eligible_words(malformed))


def test_short_word_supports_tokenization_disruption() -> None:
    edit = generate_family_edit(
        "Who am I?",
        family=FAMILY_TOKENIZATION,
        subtype="intra_word_space",
        seed=7,
    )
    validate_single_local_edit("Who am I?", edit)


def test_formula_only_question_uses_non_destructive_boundary_fallbacks() -> None:
    formula = r"$\int_{-\infty}^{\infty} e^{-x^2}\,dx$"
    edits = (
        generate_family_edit(
            formula,
            family=FAMILY_CHARACTER_NOISE,
            subtype="adjacent_swap",
            seed=1,
        ),
        generate_family_edit(
            formula,
            family=FAMILY_TOKENIZATION,
            subtype="intra_word_space",
            seed=2,
        ),
        generate_family_edit(
            formula,
            family=FAMILY_CODE_SWITCH,
            subtype="semantic_equivalent_translation",
            seed=3,
        ),
    )
    for edit in edits:
        validate_single_local_edit(formula, edit)
        assert formula in edit.candidate_text
    assert edits[2].subtype == "bilingual_politeness_cue"


def test_dataset_index_balances_declared_subtypes() -> None:
    assert [CHARACTER_SUBTYPES[index % 4] for index in range(8)] == list(CHARACTER_SUBTYPES) * 2
    assert [TOKENIZATION_SUBTYPES[index % 3] for index in range(6)] == list(TOKENIZATION_SUBTYPES) * 2


def _candidate(family: str, ids: tuple[int, ...], labels: tuple[int, ...]) -> CandidateSequence:
    return CandidateSequence(
        source_row=0,
        family=family,
        subtype="test",
        input_ids=ids,
        labels=labels,
        attention_mask=(1,) * len(ids),
        candidate_text="readable candidate",
        message_index=0,
        modified_span={"start": 1, "end": 2, "original": "a", "replacement": "b"},
        seed=42,
    )


def test_length_changing_candidates_align_by_answer_index() -> None:
    first = _candidate(FAMILY_CHARACTER_NOISE, (1, 2, 3, 4), (-100, -100, 20, 21))
    second = _candidate(FAMILY_TOKENIZATION, (5, 6, 7, 8, 9), (-100, -100, -100, 30, 31))
    packed = pack_candidate_sequences(
        (first, second), sequence_len=8, pad_token_id=0, device=torch.device("cpu")
    )
    assert packed.position_ids.tolist() == [[0, 1, 2, 3, 0], [0, 1, 2, 3, 4]]
    assert packed.target_ids == ((20, 21), (30, 31))
    assert packed.logit_offsets == ((0, 2), (2, 4))


def test_per_sample_argmax_and_final_loss_math() -> None:
    scores = torch.tensor([[0.2, 0.7, 0.4], [0.9, 0.1, 0.3]])
    assert select_worst_candidate(scores).tolist() == [1, 0]
    clean_a = torch.randn(3, 11, requires_grad=True)
    clean_b = torch.randn(2, 11, requires_grad=True)
    chosen_a = torch.randn(3, 11, requires_grad=True)
    chosen_b = torch.randn(2, 11, requires_grad=True)
    kl = torch.stack(
        (
            masked_kl_divergence(clean_a, chosen_a),
            masked_kl_divergence(clean_b, chosen_b),
        )
    ).mean()
    sft = (clean_a.square().mean() + clean_b.square().mean()) / 2
    total = sft + 0.03 * kl
    total.backward()
    assert chosen_a.grad is not None and chosen_a.grad.abs().sum() > 0
    assert chosen_b.grad is not None and chosen_b.grad.abs().sum() > 0
    assert torch.allclose(total.detach(), sft.detach() + 0.03 * kl.detach())


def test_global_token_mean_differs_from_per_conversation_mean() -> None:
    clean_a = torch.tensor([[3.0, -1.0], [2.0, -0.5], [1.0, 0.0]])
    clean_b = torch.tensor([[0.0, 1.0]])
    chosen_a = torch.tensor([[0.0, 1.0], [0.2, 0.8], [0.5, 0.5]])
    chosen_b = torch.tensor([[1.0, 0.0]])
    global_mean = masked_kl_divergence(
        torch.cat((clean_a, clean_b)), torch.cat((chosen_a, chosen_b))
    )
    per_conversation = torch.stack(
        (
            masked_kl_divergence(clean_a, chosen_a),
            masked_kl_divergence(clean_b, chosen_b),
        )
    ).mean()
    weighted = (
        3 * masked_kl_divergence(clean_a, chosen_a)
        + masked_kl_divergence(clean_b, chosen_b)
    ) / 4
    assert torch.allclose(global_mean, weighted)
    assert not torch.allclose(global_mean, per_conversation)


def test_soft_worst_value_and_gradient_match_requested_formula() -> None:
    scores = torch.tensor(
        [[0.2, 0.7, 0.4], [0.9, 0.1, 0.3]], requires_grad=True
    )
    counts = torch.tensor([3.0, 1.0])
    tau = 0.15
    values = soft_worst_aggregation(scores, tau=tau)
    expected_values = tau * torch.log(
        torch.exp(scores / tau).mean(dim=1)
    )
    assert torch.allclose(values, expected_values)
    loss = token_weighted_soft_worst_loss(scores, counts, tau=tau)
    loss.backward()
    expected_gradient = (
        torch.softmax(scores.detach() / tau, dim=1)
        * (counts / counts.sum()).unsqueeze(1)
    )
    assert torch.allclose(scores.grad, expected_gradient)
    assert torch.all(values <= scores.detach().max(dim=1).values)
    assert torch.all(values >= scores.detach().min(dim=1).values)


@pytest.mark.parametrize("tau", (0.0, -0.1, float("nan"), float("inf")))
def test_soft_worst_rejects_invalid_tau(tau: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        soft_worst_aggregation(torch.ones(2, 3), tau=tau)


def test_metasft_family_sampler_is_deterministic_and_balanced() -> None:
    indices = select_uniform_family_indices(range(100_000), seed=42)
    counts = [indices.count(index) for index in range(3)]
    assert max(counts) - min(counts) <= 1
    assert indices == select_uniform_family_indices(range(100_000), seed=42)


def test_metamorphic_runtime_contract(tmp_path) -> None:
    config = {
        "use_metamorphic_vat": True,
        "native_axolotl_preprocessing": True,
        "vat_perturbation_mode": "metamorphic_text",
        "vat_perturb_mask_source": "native_llama3_input_ids",
        "vat_lambda": 0.03,
        "vat_lambda_schedule": "linear_warmup",
        "vat_lambda_warmup_fraction": 0.2,
        "metamorphic_candidate_db": "/tmp/candidates.sqlite3",
        "sequence_len": 8192,
    }
    runtime = MetamorphicRuntimeConfig.from_mapping(config)
    assert runtime.candidate_db.endswith("candidates.sqlite3")
    bad = dict(config, vat_lambda=0.031)
    with pytest.raises(ValueError):
        MetamorphicRuntimeConfig.from_mapping(bad)
    v2 = dict(config, vat_perturbation_mode="metamorphic_text_global_token_mean")
    assert MetamorphicRuntimeConfig.from_mapping(v2).objective == "vat_global_token_mean"
    metasft = dict(
        config,
        vat_perturbation_mode="metamorphic_text_uniform_sft",
        metamorphic_sampling_seed=42,
    )
    assert MetamorphicRuntimeConfig.from_mapping(metasft).objective == "uniform_candidate_sft"
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "protocol": SOFT_WORST_CALIBRATION_PROTOCOL,
                "subset_size": SOFT_WORST_CALIBRATION_SUBSET_SIZE,
                "seed": SOFT_WORST_CALIBRATION_SEED,
                "candidate_family_count": 3,
                "candidate_family_order": list(FAMILY_ORDER),
                "candidate_db": "/tmp/candidates.sqlite3",
                "tau": 0.125,
            }
        ),
        encoding="utf-8",
    )
    v3 = dict(
        config,
        vat_perturbation_mode="metamorphic_text_soft_worst",
        metamorphic_soft_worst_calibration=str(calibration_path),
    )
    runtime_v3 = MetamorphicRuntimeConfig.from_mapping(v3)
    assert runtime_v3.objective == "vat_soft_worst_global_token_mean"
    assert runtime_v3.soft_worst_tau == pytest.approx(0.125)
