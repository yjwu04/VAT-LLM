from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from src.semantic_margin import apply_token_semantic_cap
from src.vat_loss import (
    DENSE_VAT_COUNT_EXPONENT,
    DENSE_VAT_COUNT_FLOOR,
    assert_zero_stochastic_dropout,
    assistant_only_kl,
    assistant_prediction_mask,
    build_random_neighbor_embeddings,
    build_supervised_adversarial_perturbation,
    build_vadv,
    group_ids_from_position_ids,
    relative_vat_group_ids,
    initialize_masked_gaussian,
    initialize_relative_masked_gaussian,
    masked_kl_divergence,
    masked_frobenius_norms,
    masked_difference_frobenius_norms,
    normalize_masked_per_group,
    memory_efficient_cross_entropy,
    RELATIVE_GRADIENT_NORM_DELTA,
    RHO_WEIGHTED_ENERGY_IDENTITY_ATOL,
    RHO_WEIGHTED_ENERGY_IDENTITY_RTOL,
    standalone_admission_token_locality_statistics,
    token_locality_statistics,
    validate_packed_source_position_alignment,
    verify_rho_weighted_energy_identity,
    vocabulary_radius_neighbor_counts,
)


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 7, hidden_size: int = 5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.block = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.forward_calls = 0

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del attention_mask, position_ids, use_cache
        self.forward_calls += 1
        hidden = torch.tanh(self.block(inputs_embeds))
        return SimpleNamespace(logits=self.lm_head(self.norm(hidden)))


class VatLossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_zero_dropout_contract_audits_config_and_modules(self) -> None:
        model = TinyCausalLM()
        model.config = SimpleNamespace(
            to_dict=lambda: {
                "attention_dropout": 0.0,
                "hidden_dropout": 0.0,
            }
        )
        model.dropout = nn.Dropout(0.0)
        result = assert_zero_stochastic_dropout(model)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["config_dropout_fields"],
            ["attention_dropout", "hidden_dropout"],
        )
        self.assertEqual(result["dropout_module_count"], 1)

    def test_zero_dropout_contract_rejects_config_or_module_probability(
        self,
    ) -> None:
        config_model = TinyCausalLM()
        config_model.config = SimpleNamespace(
            to_dict=lambda: {"attention_dropout": 0.1}
        )
        with self.assertRaisesRegex(RuntimeError, "zero stochastic dropout"):
            assert_zero_stochastic_dropout(config_model)

        module_model = TinyCausalLM()
        module_model.config = SimpleNamespace(
            to_dict=lambda: {"attention_dropout": 0.0}
        )
        module_model.dropout = nn.Dropout(0.2)
        with self.assertRaisesRegex(RuntimeError, "zero stochastic dropout"):
            assert_zero_stochastic_dropout(module_model)

    def test_supervised_adversarial_perturbation_matches_packed_formula(self) -> None:
        embeddings = torch.tensor(
            [
                [
                    [3.0, 4.0],
                    [8.0, 15.0],
                    [5.0, 12.0],
                    [0.0, 7.0],
                ]
            ],
            requires_grad=True,
        )
        coefficient = nn.Parameter(
            torch.tensor(
                [
                    [
                        [2.0, -1.0],
                        [9.0, 9.0],
                        [-3.0, 4.0],
                        [1.0, 2.0],
                    ]
                ]
            )
        )
        mask = torch.tensor([[1, 0, 1, 1]], dtype=torch.bool)
        groups = torch.tensor([[0, 0, 1, 1]])
        loss = (embeddings * coefficient).sum()

        perturbation = build_supervised_adversarial_perturbation(
            loss,
            embeddings,
            mask,
            rho=0.03,
            group_ids=groups,
        )

        expected = torch.zeros_like(embeddings)
        for group in (0, 1):
            selected = mask & groups.eq(group)
            embedding_norm = torch.linalg.vector_norm(
                embeddings.detach()[selected].float()
            )
            gradient = coefficient.detach()[selected].float()
            gradient_norm = torch.linalg.vector_norm(gradient)
            expected[selected] = (
                0.03
                * embedding_norm
                * gradient
                / (gradient_norm + 1.0e-12)
            )
        torch.testing.assert_close(perturbation, expected)
        self.assertFalse(perturbation.requires_grad)
        self.assertEqual(perturbation[~mask].count_nonzero().item(), 0)
        self.assertIsNone(coefficient.grad)

        observed_norms, observed_keys = masked_frobenius_norms(
            perturbation,
            mask,
            group_ids=groups,
        )
        reference_norms, reference_keys = masked_frobenius_norms(
            embeddings.detach(),
            mask,
            group_ids=groups,
        )
        torch.testing.assert_close(observed_keys, reference_keys)
        torch.testing.assert_close(observed_norms, reference_norms * 0.03)

    def test_dense_neighbor_counts_are_exact_per_group_and_exclude_only_self(
        self,
    ) -> None:
        vocabulary = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 0.0],
            ]
        )
        input_ids = torch.tensor([[0, 1, 0, 4]])
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        groups = torch.tensor([[0, 0, 1, 1]])
        counts = vocabulary_radius_neighbor_counts(
            input_ids,
            vocabulary,
            mask,
            torch.tensor([1.0, 0.0]),
            radius_keys=torch.tensor([[0, 0], [0, 1]]),
            group_ids=groups,
            vocabulary_chunk_size=2,
        )
        # Radius comparisons are inclusive.  For ID 0 at radius 1, IDs 1, 2,
        # and duplicate-vector ID 4 are neighbors.  At radius 0, only the
        # other duplicate ID remains.  Exactly the center ID is excluded.
        self.assertEqual(counts.tolist(), [[3, 3, 1, 1]])

    def test_dense_neighbor_counts_keep_high_dimensional_duplicate_at_zero(
        self,
    ) -> None:
        torch.manual_seed(101)
        center = torch.randn(4096, dtype=torch.bfloat16).float()
        boundary = center.clone()
        boundary[0] += 0.01
        vocabulary = torch.stack((center, center.clone(), boundary))
        radius = torch.linalg.vector_norm(boundary - center).reshape(1)
        counts = vocabulary_radius_neighbor_counts(
            torch.tensor([[0]]),
            vocabulary,
            torch.tensor([[True]]),
            radius,
            radius_keys=torch.tensor([[0, 0]]),
            vocabulary_chunk_size=2,
        )
        # The duplicate ID at distance zero and the inclusive boundary ID are
        # both counted; only center ID 0 itself is excluded.
        self.assertEqual(counts.item(), 2)

    def test_random_neighbor_replaces_one_question_token_per_packed_group(
        self,
    ) -> None:
        vocabulary = nn.Parameter(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.4, 0.0],
                    [0.8, 0.0],
                    [1.2, 0.0],
                    [4.0, 0.0],
                ]
            )
        )
        input_ids = torch.tensor([[0, 1, 2, 3, 4]])
        input_embeddings = (
            vocabulary.detach()[input_ids].clone().requires_grad_(True)
        )
        mask = torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.bool)
        groups = torch.tensor([[0, 0, 0, 1, 1]])
        generator = torch.Generator().manual_seed(23)

        perturbed, stats = build_random_neighbor_embeddings(
            input_embeddings,
            input_ids,
            vocabulary,
            mask,
            rho=1.0,
            group_ids=groups,
            vocabulary_chunk_size=2,
            generator=generator,
        )

        self.assertEqual(stats.conversation_count, 2)
        self.assertEqual(stats.selected_token_count, 2)
        self.assertGreaterEqual(stats.neighborhood_size_sum, 2)
        self.assertLessEqual(stats.replacement_count, 2)
        # Exactly one position per packed group is routed through the sampled
        # vocabulary embedding, even when the uniformly sampled ID is self.
        perturbed.sum().backward()
        input_grad_is_zero = input_embeddings.grad.abs().sum(dim=-1).eq(0)
        self.assertEqual(input_grad_is_zero.tolist(), [[False, True, False, True, False]])
        self.assertEqual(int(vocabulary.grad.abs().sum(dim=-1).ne(0).sum()), 2)
        torch.testing.assert_close(perturbed[0, 2], input_embeddings[0, 2])

    def test_dense_token_divisor_has_no_final_group_renormalization(self) -> None:
        vectors = torch.tensor([[[3.0, 4.0], [0.0, 12.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        result = normalize_masked_per_group(
            vectors,
            mask,
            26.0,
            denominator_delta=1.0e-12,
            token_divisor=torch.tensor([[2.0, 4.0]]),
        )
        torch.testing.assert_close(
            result,
            torch.tensor([[[3.0, 4.0], [0.0, 6.0]]]),
        )
        self.assertNotAlmostEqual(float(result.norm().item()), 26.0)

    def test_dense_vadv_matches_frozen_formula_token_by_token(self) -> None:
        model = TinyCausalLM(vocab_size=7, hidden_size=5)
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        embeddings = model.get_input_embeddings()(ids)
        clean = model(inputs_embeds=embeddings).logits.detach()
        mask = torch.tensor([[0, 1, 1, 0, 1, 0]], dtype=torch.bool)
        groups = torch.tensor([[0, 0, 0, 1, 1, 1]])
        labels = torch.tensor([[-100, -100, 2, -100, -100, 5]])
        rho = 0.5
        seed = 29
        ordinary = build_vadv(
            model,
            embeddings,
            clean,
            mask,
            labels=labels,
            group_ids=groups,
            xi=0.1,
            rho=rho,
            generator=torch.Generator().manual_seed(seed),
        )
        dense = build_vadv(
            model,
            embeddings,
            clean,
            mask,
            labels=labels,
            group_ids=groups,
            input_ids=ids,
            vocabulary_embeddings=model.get_input_embeddings().weight,
            density_mode="vocab_radius_count",
            density_vocabulary_chunk_size=3,
            xi=0.1,
            rho=rho,
            generator=torch.Generator().manual_seed(seed),
        )
        embedding_norms, keys = masked_frobenius_norms(
            embeddings,
            mask,
            group_ids=groups,
        )
        counts = vocabulary_radius_neighbor_counts(
            ids,
            model.get_input_embeddings().weight,
            mask,
            embedding_norms * rho,
            radius_keys=keys,
            group_ids=groups,
            vocabulary_chunk_size=3,
        )
        divisors = (
            counts.float()
            .add(DENSE_VAT_COUNT_FLOOR)
            .pow(DENSE_VAT_COUNT_EXPONENT)
        )
        expected = torch.zeros_like(ordinary)
        expected[mask] = ordinary[mask] / divisors[mask].unsqueeze(-1)
        torch.testing.assert_close(dense, expected, rtol=1e-5, atol=1e-6)
        self.assertFalse(dense.requires_grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters())
        )

    def test_semantic_margin_is_only_the_final_tokenwise_dense_cap(self) -> None:
        model = TinyCausalLM(vocab_size=7, hidden_size=5)
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        embeddings = model.get_input_embeddings()(ids)
        clean = model(inputs_embeds=embeddings).logits.detach()
        mask = torch.tensor([[0, 1, 1, 0, 1, 0]], dtype=torch.bool)
        groups = torch.tensor([[0, 0, 0, 1, 1, 1]])
        labels = torch.tensor([[-100, -100, 2, -100, -100, 5]])
        seed = 31
        dense = build_vadv(
            model,
            embeddings,
            clean,
            mask,
            labels=labels,
            group_ids=groups,
            input_ids=ids,
            vocabulary_embeddings=model.get_input_embeddings().weight,
            density_mode="vocab_radius_count",
            density_vocabulary_chunk_size=3,
            xi=0.1,
            rho=0.03,
            generator=torch.Generator().manual_seed(seed),
        )
        margins = torch.full((7,), float("inf"))
        margins[1] = 0.001
        margins[2] = 0.002
        semantic = build_vadv(
            model,
            embeddings,
            clean,
            mask,
            labels=labels,
            group_ids=groups,
            input_ids=ids,
            vocabulary_embeddings=model.get_input_embeddings().weight,
            density_mode="vocab_radius_count",
            density_vocabulary_chunk_size=3,
            semantic_margins=margins,
            xi=0.1,
            rho=0.03,
            generator=torch.Generator().manual_seed(seed),
        )
        expected = apply_token_semantic_cap(dense, ids, mask, margins)
        torch.testing.assert_close(semantic, expected, rtol=0, atol=0)
        self.assertLessEqual(float(semantic[0, 1].float().norm()), 0.001001)
        self.assertLessEqual(float(semantic[0, 2].float().norm()), 0.002001)
        torch.testing.assert_close(semantic[0, 4], dense[0, 4], rtol=0, atol=0)

    def test_rho_identity_tolerance_covers_fp32_reduction_bound(self) -> None:
        unit_roundoff = 2.0**-24
        gamma = (8191 * unit_roundoff) / (1.0 - 8191 * unit_roundoff)
        two_reductions = (2.0 * gamma) / (1.0 - gamma)
        surrounding_roundings = 16.0 * unit_roundoff
        conservative_bound = two_reductions + surrounding_roundings
        self.assertGreater(conservative_bound, 2.0**-10)
        self.assertLess(conservative_bound, RHO_WEIGHTED_ENERGY_IDENTITY_RTOL)
        self.assertEqual(RHO_WEIGHTED_ENERGY_IDENTITY_RTOL, 2.0**-9)
        self.assertEqual(RHO_WEIGHTED_ENERGY_IDENTITY_ATOL, 0.0)

        keys = torch.tensor([[0, 0]])
        for scale in (2.0**-20, 1.0, 2.0**20):
            rhs = torch.tensor([scale], dtype=torch.float64)
            inside = rhs * (1.0 + RHO_WEIGHTED_ENERGY_IDENTITY_RTOL * 0.99)
            verify_rho_weighted_energy_identity(
                inside,
                rhs,
                keys,
                relative_tolerance=RHO_WEIGHTED_ENERGY_IDENTITY_RTOL,
                absolute_tolerance=RHO_WEIGHTED_ENERGY_IDENTITY_ATOL,
            )
            outside = rhs * (1.0 + RHO_WEIGHTED_ENERGY_IDENTITY_RTOL * 1.01)
            with self.assertRaisesRegex(RuntimeError, "weighted-energy identity"):
                verify_rho_weighted_energy_identity(
                    outside,
                    rhs,
                    keys,
                    relative_tolerance=RHO_WEIGHTED_ENERGY_IDENTITY_RTOL,
                    absolute_tolerance=RHO_WEIGHTED_ENERGY_IDENTITY_ATOL,
                )

    def test_assistant_only_kl_uses_causal_shift_and_detaches_clean(self) -> None:
        clean = torch.zeros(1, 5, 3, requires_grad=True)
        perturbed = torch.zeros(1, 5, 3, requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 2, -100]])

        # Assistant target labels at token positions 2 and 3 are predicted by
        # logit positions 1 and 2. Changes at 0, 3, and 4 must be ignored.
        with torch.no_grad():
            perturbed[0, 0] = torch.tensor([8.0, -8.0, 0.0])
            perturbed[0, 1] = torch.tensor([2.0, 0.0, 0.0])
            perturbed[0, 2] = torch.tensor([0.0, 2.0, 0.0])
            perturbed[0, 3] = torch.tensor([9.0, -9.0, 0.0])

        mask = assistant_prediction_mask(labels)
        self.assertEqual(mask.tolist(), [[False, True, True, False, False]])
        loss = assistant_only_kl(clean, perturbed, labels=labels)

        expected = (
            F.kl_div(
                F.log_softmax(perturbed.detach()[mask].float(), dim=-1),
                F.softmax(clean.detach()[mask].float(), dim=-1),
                reduction="none",
            )
            .sum(-1)
            .mean()
        )
        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, expected)

        loss.backward()
        self.assertIsNone(clean.grad)
        self.assertIsNotNone(perturbed.grad)
        self.assertTrue(
            torch.equal(perturbed.grad[~mask], torch.zeros_like(perturbed.grad[~mask]))
        )

    def test_kl_is_fp32_finite_for_low_precision_large_logits(self) -> None:
        clean = torch.tensor(
            [[[10000.0, -10000.0, 0.0], [-10000.0, 10000.0, 0.0]]],
            dtype=torch.bfloat16,
        )
        perturbed = clean.clone().requires_grad_(True)
        labels = torch.tensor([[-100, 1]])
        loss = assistant_only_kl(clean, perturbed, labels=labels)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_chunked_kl_value_and_gradient_match_fp32_reference(self) -> None:
        # More rows than the production chunk size exercises multiple chunks.
        clean = torch.randn(97, 23, requires_grad=True)
        perturbed = torch.randn(97, 23, requires_grad=True)
        reference_perturbed = perturbed.detach().clone().requires_grad_(True)

        loss = masked_kl_divergence(clean, perturbed)
        reference = (
            F.kl_div(
                F.log_softmax(reference_perturbed, dim=-1),
                F.softmax(clean.detach(), dim=-1),
                reduction="none",
            )
            .sum(dim=-1)
            .mean()
        )
        loss.backward()
        reference.backward()

        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, reference, rtol=1e-6, atol=1e-6)
        self.assertIsNone(clean.grad)
        torch.testing.assert_close(
            perturbed.grad,
            reference_perturbed.grad,
            rtol=1e-5,
            atol=1e-7,
        )

    def test_chunked_cross_entropy_matches_fp32_value_and_gradient(self) -> None:
        # Exercise multiple row chunks and the analytical softmax-one-hot
        # backward used by the BF16 compact trainer path.
        logits = torch.randn(97, 23, requires_grad=True)
        targets = torch.randint(0, 23, (97,))
        reference_logits = logits.detach().clone().requires_grad_(True)

        loss = memory_efficient_cross_entropy(logits, targets)
        reference = F.cross_entropy(reference_logits.float(), targets)
        loss.backward()
        reference.backward()

        self.assertEqual(loss.dtype, torch.float32)
        torch.testing.assert_close(loss, reference, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            logits.grad,
            reference_logits.grad,
            rtol=1e-5,
            atol=1e-7,
        )

    def test_chunked_cross_entropy_bf16_matches_official_fp32_semantics(
        self,
    ) -> None:
        bf16_values = torch.randn(65, 257).to(torch.bfloat16)
        logits = bf16_values.clone().requires_grad_(True)
        targets = torch.randint(0, 257, (65,))
        reference_logits = bf16_values.float().requires_grad_(True)
        upstream = torch.tensor(1.7)

        loss = memory_efficient_cross_entropy(logits, targets)
        reference = F.cross_entropy(reference_logits, targets)
        (loss * upstream).backward()
        (reference * upstream).backward()

        torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            logits.grad,
            reference_logits.grad.to(torch.bfloat16),
            rtol=8e-3,
            atol=5e-7,
        )

    def test_empty_assistant_mask_returns_differentiable_zero(self) -> None:
        clean = torch.randn(2, 4, 6, requires_grad=True)
        perturbed = torch.randn(2, 4, 6, requires_grad=True)
        labels = torch.full((2, 4), -100)
        loss = assistant_only_kl(clean, perturbed, labels=labels)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNone(clean.grad)
        self.assertIsNotNone(perturbed.grad)
        self.assertEqual(torch.count_nonzero(perturbed.grad).item(), 0)

    def test_group_ids_and_position_resets_normalize_each_pack_independently(
        self,
    ) -> None:
        vectors = torch.ones(2, 6, 2)
        user_mask = torch.tensor(
            [
                [1, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        explicit_groups = torch.tensor(
            [
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, -1, -1],
            ]
        )
        normalized = normalize_masked_per_group(
            vectors, user_mask, 3.0, group_ids=explicit_groups
        )
        norms, keys = masked_frobenius_norms(
            normalized, user_mask, group_ids=explicit_groups
        )
        self.assertEqual(keys.tolist(), [[0, 0], [0, 1], [1, 0]])
        torch.testing.assert_close(norms, torch.full((3,), 3.0))
        self.assertEqual(torch.count_nonzero(normalized[~user_mask]).item(), 0)

        position_ids = torch.tensor(
            [
                [0, 1, 2, 0, 1, 2],
                [0, 1, 2, 3, 0, 0],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0],
            ]
        )
        inferred = group_ids_from_position_ids(position_ids, attention_mask)
        self.assertEqual(
            inferred.tolist(),
            [[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, -1, -1]],
        )
        normalized_from_positions = normalize_masked_per_group(
            vectors,
            user_mask,
            3.0,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        torch.testing.assert_close(normalized_from_positions, normalized)

    def test_relative_vat_grouping_ignores_internal_eos_attention_zeroes(self) -> None:
        position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1]])
        attention_mask = torch.tensor([[1, 1, 0, 1, 1, 1, 1, 1]])
        production = relative_vat_group_ids(position_ids)
        attention_split = group_ids_from_position_ids(
            position_ids, attention_mask
        )
        self.assertEqual(
            production.tolist(),
            [[0, 0, 0, 0, 0, 0, 1, 1]],
        )
        self.assertEqual(
            attention_split.tolist(),
            [[0, 0, -1, 1, 1, 1, 2, 2]],
        )
        user_mask = torch.tensor(
            [[False, True, False, False, True, False, False, True]]
        )
        self.assertEqual(production[user_mask].tolist(), [0, 0, 1])
        self.assertEqual(attention_split[user_mask].tolist(), [0, 1, 2])

    def test_gaussian_initialization_masks_and_normalizes_every_group(self) -> None:
        embeddings = torch.empty(1, 7, 4)
        mask = torch.tensor([[0, 1, 1, 0, 1, 0, 1]], dtype=torch.bool)
        groups = torch.tensor([[0, 0, 0, 0, 1, 1, 2]])
        generator = torch.Generator().manual_seed(123)
        d = initialize_masked_gaussian(
            embeddings,
            mask,
            1.0e-3,
            group_ids=groups,
            generator=generator,
        )
        norms, keys = masked_frobenius_norms(d, mask, group_ids=groups)
        self.assertEqual(keys.tolist(), [[0, 0], [0, 1], [0, 2]])
        torch.testing.assert_close(
            norms, torch.full((3,), 1.0e-3), rtol=1e-5, atol=1e-8
        )
        self.assertEqual(torch.count_nonzero(d[~mask]).item(), 0)

    def test_group_specific_target_norms_support_relative_radius(self) -> None:
        vectors = torch.ones(1, 6, 3)
        mask = torch.tensor([[1, 1, 0, 1, 0, 1]], dtype=torch.bool)
        groups = torch.tensor([[0, 0, 0, 1, 1, 1]])
        targets = torch.tensor([0.25, 0.75], dtype=torch.float32)
        normalized = normalize_masked_per_group(
            vectors,
            mask,
            targets,
            norm_keys=torch.tensor([[0, 0], [0, 1]]),
            group_ids=groups,
        )
        norms, keys = masked_frobenius_norms(
            normalized,
            mask,
            group_ids=groups,
        )
        self.assertEqual(keys.tolist(), [[0, 0], [0, 1]])
        torch.testing.assert_close(norms, targets)
        self.assertEqual(torch.count_nonzero(normalized[~mask]).item(), 0)
        with self.assertRaisesRegex(ValueError, "keys do not match"):
            normalize_masked_per_group(
                vectors,
                mask,
                targets,
                norm_keys=torch.tensor([[0, 1], [0, 0]]),
                group_ids=groups,
            )

    def test_masked_difference_norms_match_explicit_fp32_delta(self) -> None:
        left = torch.randn(2, 6, 5).to(torch.bfloat16)
        right = torch.randn(2, 6, 5).to(torch.bfloat16)
        mask = torch.tensor(
            [[1, 1, 0, 1, 0, 1], [1, 0, 1, 0, 0, 0]],
            dtype=torch.bool,
        )
        groups = torch.tensor(
            [[0, 0, 0, 1, 1, 1], [0, 0, 2, 2, -1, -1]]
        )
        observed, keys = masked_difference_frobenius_norms(
            left,
            right,
            mask,
            group_ids=groups,
        )
        expected, expected_keys = masked_frobenius_norms(
            left.float() - right.float(),
            mask,
            group_ids=groups,
        )
        torch.testing.assert_close(keys, expected_keys)
        torch.testing.assert_close(observed, expected)

    def test_token_locality_reports_raw_ratios_and_group_energy(self) -> None:
        embeddings = torch.tensor(
            [
                [
                    [3.0, 4.0],
                    [3.0, 4.0],
                    [9.0, 12.0],
                    [3.0, 4.0],
                    [3.0, 4.0],
                    [3.0, 4.0],
                ]
            ]
        )
        nominal = torch.tensor(
            [
                [
                    [0.3, 0.4],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.5, 0.0],
                    [0.0, 0.5],
                    [-0.5, 0.0],
                ]
            ]
        )
        effective = torch.tensor(
            [
                [
                    [3.0, 0.0],
                    [4.0, 0.0],
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [-2.0, 0.0],
                ]
            ]
        )
        mask = torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.bool)
        groups = torch.tensor([[0, 0, 0, 1, 1, 1]])

        stats = token_locality_statistics(
            embeddings,
            nominal,
            embeddings + effective,
            mask,
            group_ids=groups,
            rho=0.1,
            compute_component_energy_concentration=True,
        )

        self.assertEqual(
            stats["token_keys"].tolist(),
            [[0, 0, 0], [0, 0, 1], [0, 1, 3], [0, 1, 4], [0, 1, 5]],
        )
        self.assertEqual(stats["group_keys"].tolist(), [[0, 0], [0, 1]])
        self.assertEqual(stats["group_token_count"].tolist(), [2, 3])
        torch.testing.assert_close(
            stats["embedding_token_norm"], torch.full((5,), 5.0)
        )
        torch.testing.assert_close(
            stats["nominal_token_ratio"],
            torch.tensor([0.1, 0.2, 0.1, 0.1, 0.1]),
        )
        torch.testing.assert_close(
            stats["effective_token_ratio"],
            torch.tensor([0.6, 0.8, 0.4, 0.4, 0.4]),
        )
        torch.testing.assert_close(
            stats["rho_normalized_effective_token_ratio"],
            torch.tensor([6.0, 8.0, 4.0, 4.0, 4.0]),
        )
        torch.testing.assert_close(
            stats["bf16_token_survival"],
            torch.tensor([6.0, 4.0, 4.0, 4.0, 4.0]),
        )
        torch.testing.assert_close(
            stats["token_energy_fraction"],
            torch.tensor([0.36, 0.64, 1 / 3, 1 / 3, 1 / 3]),
        )
        torch.testing.assert_close(
            stats["max_token_energy_fraction"],
            torch.tensor([0.64, 1 / 3]),
        )
        torch.testing.assert_close(
            stats["top_10pct_token_energy_fraction"],
            torch.tensor([0.64, 1 / 3]),
        )
        torch.testing.assert_close(
            stats["normalized_effective_token_count"],
            torch.tensor([1.0 / (2.0 * (0.36**2 + 0.64**2)), 1.0]),
        )
        torch.testing.assert_close(
            stats["nonzero_effective_token_fraction"],
            torch.tensor([1.0, 1.0]),
        )
        torch.testing.assert_close(
            stats["top_1pct_component_energy_fraction"],
            torch.tensor([0.64, 1 / 3]),
        )
        self.assertEqual(stats["effective_group_nonzero"].tolist(), [True, True])
        self.assertEqual(stats["concentration_defined"].tolist(), [True, True])
        self.assertEqual(
            stats["component_concentration_computed"].tolist(), [True, True]
        )
        self.assertEqual(stats["token_coverage_residual"].tolist(), [0, 0])
        torch.testing.assert_close(
            stats["rho_weighted_energy_identity_lhs"],
            torch.tensor([50.0, 16.0]),
        )
        torch.testing.assert_close(
            stats["rho_weighted_energy_identity_rhs"],
            torch.tensor([50.0, 16.0]),
        )
        torch.testing.assert_close(
            stats["rho_weighted_energy_identity_abs_residual"],
            torch.zeros(2),
            atol=1e-5,
            rtol=0,
        )
        torch.testing.assert_close(
            stats["bf16_survival_identity_max_abs_residual"],
            torch.zeros(2),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            stats["effective_energy_fraction_sum_residual"],
            torch.zeros(2),
            atol=1e-6,
            rtol=0,
        )

    def test_token_energy_concentration_distinguishes_single_and_uniform(
        self,
    ) -> None:
        embeddings = torch.ones(1, 8, 2)
        nominal = torch.ones_like(embeddings)
        effective = torch.zeros_like(embeddings)
        effective[0, 0, 0] = 2.0
        effective[0, 4:, 0] = 1.0
        mask = torch.ones(1, 8, dtype=torch.bool)
        groups = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])

        stats = token_locality_statistics(
            embeddings,
            nominal,
            embeddings + effective,
            mask,
            group_ids=groups,
            rho=0.01,
            compute_component_energy_concentration=True,
        )

        torch.testing.assert_close(
            stats["max_token_energy_fraction"], torch.tensor([1.0, 0.25])
        )
        torch.testing.assert_close(
            stats["top_10pct_token_energy_fraction"],
            torch.tensor([1.0, 0.25]),
        )
        torch.testing.assert_close(
            stats["normalized_effective_token_count"],
            torch.tensor([0.25, 1.0]),
        )
        torch.testing.assert_close(
            stats["nonzero_effective_token_fraction"],
            torch.tensor([0.25, 1.0]),
        )
        torch.testing.assert_close(
            stats["top_1pct_component_energy_fraction"],
            torch.tensor([1.0, 0.25]),
        )

    def test_token_locality_marks_zero_effective_concentration_undefined(
        self,
    ) -> None:
        embeddings = torch.ones(1, 3, 4)
        nominal = torch.zeros_like(embeddings)
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        groups = torch.tensor([[0, 0, -1]])

        stats = token_locality_statistics(
            embeddings,
            nominal,
            embeddings.clone(),
            mask,
            group_ids=groups,
            rho=0.01,
            compute_component_energy_concentration=True,
        )

        self.assertEqual(stats["effective_group_nonzero"].tolist(), [False])
        self.assertEqual(stats["concentration_defined"].tolist(), [False])
        self.assertEqual(
            stats["component_concentration_computed"].tolist(), [False]
        )
        torch.testing.assert_close(
            stats["nonzero_effective_token_fraction"], torch.tensor([0.0])
        )
        for name in (
            "max_token_energy_fraction",
            "top_10pct_token_energy_fraction",
            "normalized_effective_token_count",
            "top_1pct_component_energy_fraction",
            "effective_energy_fraction_sum_residual",
        ):
            self.assertTrue(torch.isnan(stats[name]).all(), name)

    def test_token_locality_fails_closed_on_mask_group_or_delta_leakage(
        self,
    ) -> None:
        embeddings = torch.ones(1, 3, 2)
        mask = torch.tensor([[1, 0, 0]], dtype=torch.bool)
        nominal = torch.zeros_like(embeddings)

        with self.assertRaisesRegex(ValueError, "group_id >= 0"):
            token_locality_statistics(
                embeddings,
                nominal,
                embeddings,
                mask,
                group_ids=torch.tensor([[-1, -1, -1]]),
                rho=0.01,
            )

        nominal_leak = nominal.clone()
        nominal_leak[0, 1, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "nominal perturbation"):
            token_locality_statistics(
                embeddings,
                nominal_leak,
                embeddings,
                mask,
                group_ids=torch.tensor([[0, -1, -1]]),
                rho=0.01,
            )

        perturbed_leak = embeddings.clone()
        perturbed_leak[0, 2, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "effective perturbation"):
            token_locality_statistics(
                embeddings,
                nominal,
                perturbed_leak,
                mask,
                group_ids=torch.tensor([[0, -1, -1]]),
                rho=0.01,
            )

    def test_component_topk_requires_explicit_probe_switch(self) -> None:
        embeddings = torch.ones(1, 2, 4)
        nominal = torch.zeros_like(embeddings)
        nominal[0, :, 0] = 1.0
        mask = torch.ones(1, 2, dtype=torch.bool)
        stats = token_locality_statistics(
            embeddings,
            nominal,
            embeddings + nominal,
            mask,
            group_ids=torch.zeros(1, 2, dtype=torch.long),
            rho=0.01,
        )
        self.assertEqual(
            stats["component_concentration_computed"].tolist(), [False]
        )
        self.assertTrue(
            torch.isnan(stats["top_1pct_component_energy_fraction"]).all()
        )
        admission = standalone_admission_token_locality_statistics(
            embeddings.to(torch.bfloat16),
            nominal.to(torch.bfloat16),
            (embeddings + nominal).to(torch.bfloat16),
            mask,
            group_ids=torch.zeros(1, 2, dtype=torch.long),
            rho=0.01,
        )
        self.assertEqual(
            admission["component_concentration_computed"].tolist(), [True]
        )
        self.assertEqual(
            admission["top_1pct_component_energy_fraction"].tolist(), [0.5]
        )
        for name in (
            "group_keys",
            "effective_group_nonzero",
            "concentration_defined",
            "component_concentration_computed",
            "top_1pct_component_energy_fraction",
            "rho_weighted_energy_identity_lhs",
            "rho_weighted_energy_identity_rhs",
            "rho_weighted_energy_identity_abs_residual",
            "group_records",
            "runtime_rho",
            "rho_identity_tolerances",
        ):
            self.assertIn(name, admission)
        self.assertEqual(
            admission["group_records"][0]["key"],
            {"batch_index": 0, "group_index": 0},
        )
        self.assertEqual(
            admission["group_records"][0]["user_embedding_norm"],
            admission["group_records"][0][
                "question_embedding_group_norm"
            ],
        )
        self.assertEqual(
            admission["group_records"][0]["effective_perturbation_norm"],
            admission["group_records"][0][
                "effective_perturbation_group_norm"
            ],
        )

    def test_standalone_admission_requires_actual_bf16_model_inputs(self) -> None:
        mask = torch.ones(1, 2, dtype=torch.bool)
        groups = torch.zeros(1, 2, dtype=torch.long)
        nominal = torch.full((1, 2, 4), 0.125, dtype=torch.bfloat16)
        bf16 = torch.ones(1, 2, 4, dtype=torch.bfloat16)
        result = standalone_admission_token_locality_statistics(
            bf16,
            nominal,
            (bf16.float() + nominal.float()).to(torch.bfloat16),
            mask,
            group_ids=groups,
            rho=0.01,
        )
        self.assertEqual(len(result["group_records"]), 1)

        with self.assertRaisesRegex(TypeError, "BF16 input embeddings"):
            standalone_admission_token_locality_statistics(
                bf16.float(),
                nominal.float(),
                bf16.float() + nominal.float(),
                mask,
                group_ids=groups,
                rho=0.01,
            )
        with self.assertRaisesRegex(TypeError, "BF16 perturbed model inputs"):
            standalone_admission_token_locality_statistics(
                bf16,
                nominal,
                bf16.float() + nominal.float(),
                mask,
                group_ids=groups,
                rho=0.01,
            )

    def test_rho_weighted_energy_identity_rejects_wrong_evidence(self) -> None:
        embeddings = torch.tensor(
            [[[1.0, 0.0], [3.0, 0.0], [2.0, 0.0], [4.0, 0.0]]]
        )
        effective = torch.tensor(
            [[[0.1, 0.0], [0.6, 0.0], [0.4, 0.0], [0.4, 0.0]]]
        )
        mask = torch.ones(1, 4, dtype=torch.bool)
        groups = torch.tensor([[0, 0, 1, 1]])
        stats = token_locality_statistics(
            embeddings,
            effective,
            embeddings + effective,
            mask,
            group_ids=groups,
            rho=0.1,
        )
        lhs = stats["rho_weighted_energy_identity_lhs"]
        rhs = stats["rho_weighted_energy_identity_rhs"]
        keys = stats["group_keys"]
        verify_rho_weighted_energy_identity(
            lhs,
            rhs,
            keys,
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-6,
        )

        wrong_weight_lhs = lhs.clone()
        wrong_weight_lhs[0] += 0.25
        wrong_group_lhs = lhs.flip(0)
        wrong_rhs = rhs * 0.5
        for label, candidate_lhs, candidate_rhs in (
            ("weight", wrong_weight_lhs, rhs),
            ("group", wrong_group_lhs, rhs),
            ("rhs", lhs, wrong_rhs),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    RuntimeError, "weighted-energy identity"
                ):
                    verify_rho_weighted_energy_identity(
                        candidate_lhs,
                        candidate_rhs,
                        keys,
                        relative_tolerance=1.0e-6,
                        absolute_tolerance=1.0e-6,
                    )

    def test_bf16_token_locality_rejects_mask_leakage(self) -> None:
        embeddings = torch.ones(1, 3, 4, dtype=torch.bfloat16)
        mask = torch.tensor([[1, 0, 0]], dtype=torch.bool)
        groups = torch.tensor([[0, -1, -1]])
        nominal = torch.zeros_like(embeddings)
        nominal[0, 1, 0] = 0.125
        with self.assertRaisesRegex(ValueError, "nominal perturbation"):
            token_locality_statistics(
                embeddings,
                nominal,
                embeddings,
                mask,
                group_ids=groups,
                rho=0.01,
            )

        nominal.zero_()
        perturbed = embeddings.clone()
        perturbed[0, 2, 0] = (
            perturbed[0, 2, 0].float() + 0.125
        ).to(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "effective perturbation"):
            token_locality_statistics(
                embeddings,
                nominal,
                perturbed,
                mask,
                group_ids=groups,
                rho=0.01,
            )

    def test_relative_denominator_delta_matches_declared_formula(self) -> None:
        vectors = torch.tensor([[[3.0, 4.0]]])
        mask = torch.tensor([[1]], dtype=torch.bool)
        normalized = normalize_masked_per_group(
            vectors,
            mask,
            2.0,
            denominator_delta=1.0,
        )
        # ||2 * g / (||g|| + 1)|| = 2 * 5 / 6.
        torch.testing.assert_close(
            torch.linalg.vector_norm(normalized.float()),
            torch.tensor(10.0 / 6.0),
        )
        self.assertEqual(RELATIVE_GRADIENT_NORM_DELTA, 1.0e-12)

    def test_relative_bf16_effective_radius_is_measured_not_assumed(self) -> None:
        generator = torch.Generator().manual_seed(91)
        embeddings = (
            torch.randn(1, 32, 128, generator=generator) * 0.02
        ).to(torch.bfloat16)
        gradient = torch.randn(
            embeddings.shape, generator=generator
        ).to(torch.bfloat16)
        mask = torch.ones(1, 32, dtype=torch.bool)
        embedding_norm, _ = masked_frobenius_norms(embeddings, mask)
        effective_ratios = []
        for rho in (0.001, 0.003, 0.010, 0.030):
            nominal = normalize_masked_per_group(
                gradient,
                mask,
                embedding_norm * rho,
                denominator_delta=RELATIVE_GRADIENT_NORM_DELTA,
            )
            effective = (
                (embeddings + nominal).to(torch.bfloat16).float()
                - embeddings.float()
            )
            effective_norm, _ = masked_frobenius_norms(effective, mask)
            ratio = float((effective_norm / embedding_norm).item())
            self.assertGreater(ratio, 0.0)
            effective_ratios.append(ratio)
        self.assertEqual(effective_ratios, sorted(effective_ratios))

    def test_packed_source_position_contract_accepts_repeated_source_reset(
        self,
    ) -> None:
        validate_packed_source_position_alignment(
            perturb_mask=torch.tensor([[0, 1, 0, 1, 0, 0]], dtype=torch.bool),
            position_ids=torch.tensor([[0, 1, 2, 0, 1, 0]]),
            source_ids=torch.tensor([[7, 7, 7, 7, 7, -1]]),
        )

    def test_packed_source_position_contract_rejects_silent_group_merge(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "transition without"):
            validate_packed_source_position_alignment(
                perturb_mask=torch.tensor(
                    [[0, 1, 0, 1, 0, 0]], dtype=torch.bool
                ),
                position_ids=torch.tensor([[0, 1, 2, 3, 4, 0]]),
                source_ids=torch.tensor([[7, 7, 7, 8, 8, -1]]),
            )
        with self.assertRaisesRegex(ValueError, "active prefix"):
            validate_packed_source_position_alignment(
                perturb_mask=torch.zeros(1, 4, dtype=torch.bool),
                position_ids=torch.tensor([[0, 0, 0, 1]]),
                source_ids=torch.tensor([[7, -1, 8, 8]]),
            )
        with self.assertRaisesRegex(ValueError, "padding"):
            validate_packed_source_position_alignment(
                perturb_mask=torch.tensor([[0, 0, 1]], dtype=torch.bool),
                position_ids=torch.tensor([[0, 1, 0]]),
                source_ids=torch.tensor([[7, 7, -1]]),
            )
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            validate_packed_source_position_alignment(
                perturb_mask=torch.zeros(1, 2, dtype=torch.bool),
                position_ids=torch.tensor([[0.0, 1.5]]),
                source_ids=torch.tensor([[7, 7]]),
            )
        with self.assertRaisesRegex(ValueError, "adjacent active"):
            validate_packed_source_position_alignment(
                perturb_mask=torch.tensor([[1, 1]], dtype=torch.bool),
                position_ids=torch.tensor([[0, 0]]),
                source_ids=torch.tensor([[7, 8]]),
            )
        with self.assertRaisesRegex(ValueError, "right-padding position"):
            validate_packed_source_position_alignment(
                perturb_mask=torch.tensor([[1, 0]], dtype=torch.bool),
                position_ids=torch.tensor([[0, 9]]),
                source_ids=torch.tensor([[7, -1]]),
            )

    def test_relative_vadv_matches_each_group_user_embedding_norm(self) -> None:
        model = TinyCausalLM()
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        embeddings = model.get_input_embeddings()(ids)
        clean = model(inputs_embeds=embeddings).logits.detach()
        perturb_mask = torch.tensor([[0, 1, 0, 1, 1, 0]], dtype=torch.bool)
        group_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
        labels = torch.tensor([[-100, -100, 2, -100, -100, 5]])
        rho = 0.03

        r_vadv = build_vadv(
            model,
            embeddings,
            clean,
            perturb_mask,
            labels=labels,
            group_ids=group_ids,
            xi=0.1,
            rho=rho,
            generator=torch.Generator().manual_seed(17),
        )
        embedding_norms, embedding_keys = masked_frobenius_norms(
            embeddings,
            perturb_mask,
            group_ids=group_ids,
        )
        perturbation_norms, perturbation_keys = masked_frobenius_norms(
            r_vadv,
            perturb_mask,
            group_ids=group_ids,
        )
        torch.testing.assert_close(embedding_keys, perturbation_keys)
        torch.testing.assert_close(
            perturbation_norms,
            embedding_norms * rho,
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(torch.count_nonzero(r_vadv[~perturb_mask]).item(), 0)

    def test_relative_probe_initializer_matches_question_embedding_scale(self) -> None:
        embeddings = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [2.0, 1.0], [5.0, 0.0]]]
        )
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        groups = torch.tensor([[0, 0, 1, 1]])
        tau = 0.01
        probe = initialize_relative_masked_gaussian(
            embeddings,
            mask,
            tau,
            group_ids=groups,
            generator=torch.Generator().manual_seed(19),
        )
        embedding_norms, embedding_keys = masked_frobenius_norms(
            embeddings, mask, group_ids=groups
        )
        probe_norms, probe_keys = masked_frobenius_norms(
            probe, mask, group_ids=groups
        )
        torch.testing.assert_close(embedding_keys, probe_keys)
        torch.testing.assert_close(
            probe_norms, embedding_norms * tau, rtol=1e-5, atol=1e-7
        )
        self.assertTrue(torch.equal(probe[~mask], torch.zeros_like(probe[~mask])))

    def test_build_vadv_accepts_relative_probe_tau(self) -> None:
        model = TinyCausalLM()
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        embeddings = model.get_input_embeddings()(ids)
        clean = model(inputs_embeds=embeddings).logits.detach()
        mask = torch.tensor([[0, 1, 0, 1, 1, 0]], dtype=torch.bool)
        groups = torch.tensor([[0, 0, 0, 1, 1, 1]])
        labels = torch.tensor([[-100, -100, 2, -100, -100, 5]])
        result = build_vadv(
            model,
            embeddings,
            clean,
            mask,
            labels=labels,
            group_ids=groups,
            xi=0.003,
            xi_mode="relative_question_frobenius",
            rho=0.01,
            generator=torch.Generator().manual_seed(23),
        )
        self.assertFalse(result.requires_grad)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(result[~mask], torch.zeros_like(result[~mask])))

    def test_vadv_rejects_ambiguous_absolute_and_relative_radius(self) -> None:
        model = TinyCausalLM()
        ids = torch.tensor([[0, 1, 2, 3]])
        embeddings = model.get_input_embeddings()(ids)
        clean = model(inputs_embeds=embeddings).logits.detach()
        kwargs = {
            "labels": torch.tensor([[-100, -100, 2, 3]]),
            "xi": 0.1,
        }
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_vadv(
                model,
                embeddings,
                clean,
                torch.tensor([[0, 1, 0, 0]], dtype=torch.bool),
                epsilon=0.1,
                rho=0.01,
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_vadv(
                model,
                embeddings,
                clean,
                torch.tensor([[0, 1, 0, 0]], dtype=torch.bool),
                **kwargs,
            )

    def test_zero_vectors_and_no_user_tokens_stay_finite_zero(self) -> None:
        vectors = torch.zeros(2, 3, 4)
        mask = torch.tensor([[1, 0, 1], [0, 0, 0]], dtype=torch.bool)
        normalized = normalize_masked_per_group(vectors, mask, 2.0)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertEqual(torch.count_nonzero(normalized).item(), 0)

        model = TinyCausalLM()
        ids = torch.tensor([[0, 1, 2, 3]])
        embeddings = model.get_input_embeddings()(ids)
        with torch.no_grad():
            clean = model(inputs_embeds=embeddings).logits
        model.forward_calls = 0
        r = build_vadv(
            model,
            embeddings,
            clean,
            torch.zeros(1, 4, dtype=torch.bool),
            labels=torch.tensor([[-100, -100, 2, 3]]),
            xi=1.0e-3,
            epsilon=0.5,
        )
        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(torch.count_nonzero(r).item(), 0)
        self.assertFalse(r.requires_grad)

    def test_no_assistant_tokens_skips_probe(self) -> None:
        model = TinyCausalLM()
        ids = torch.tensor([[0, 1, 2, 3]])
        embeddings = model.get_input_embeddings()(ids)
        clean = torch.randn(1, 4, 7)
        r = build_vadv(
            model,
            embeddings,
            clean,
            torch.tensor([[0, 1, 1, 0]], dtype=torch.bool),
            labels=torch.full((1, 4), -100),
            xi=1.0e-3,
            epsilon=0.5,
        )
        self.assertEqual(model.forward_calls, 0)
        self.assertEqual(torch.count_nonzero(r).item(), 0)

    def test_power_iteration_is_masked_group_normalized_and_leaves_no_param_grads(
        self,
    ) -> None:
        model = TinyCausalLM()
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        embeddings = model.get_input_embeddings()(ids)
        with torch.no_grad():
            clean = model(inputs_embeds=embeddings).logits

        # Preserve both None and an existing accumulated gradient as evidence
        # that autograd.grad targets d only and does not touch parameter .grad.
        parameters = list(model.parameters())
        parameters[0].grad = torch.full_like(parameters[0], 0.25)
        before = [
            None if parameter.grad is None else parameter.grad.clone()
            for parameter in parameters
        ]

        perturb_mask = torch.tensor([[0, 1, 0, 0, 1, 0]], dtype=torch.bool)
        group_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
        labels = torch.tensor([[-100, -100, 2, -100, -100, 5]])
        r = build_vadv(
            model,
            embeddings,
            clean,
            perturb_mask,
            labels=labels,
            group_ids=group_ids,
            xi=1.0e-3,
            epsilon=0.7,
            generator=torch.Generator().manual_seed(9),
        )

        self.assertFalse(r.requires_grad)
        self.assertIsNone(r.grad_fn)
        self.assertEqual(torch.count_nonzero(r[~perturb_mask]).item(), 0)
        norms, keys = masked_frobenius_norms(r, perturb_mask, group_ids=group_ids)
        self.assertEqual(keys.tolist(), [[0, 0], [0, 1]])
        torch.testing.assert_close(norms, torch.full((2,), 0.7), rtol=1e-5, atol=1e-6)
        for parameter, old_grad in zip(parameters, before):
            if old_grad is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(parameter.grad, old_grad)

    def test_power_iteration_direction_beats_same_norm_gaussian_probe(self) -> None:
        model = TinyCausalLM()
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        embeddings = model.get_input_embeddings()(ids)
        clean = model(inputs_embeds=embeddings).logits.detach()
        perturb_mask = torch.tensor([[0, 1, 0, 0, 1, 0]], dtype=torch.bool)
        group_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
        labels = torch.tensor([[-100, -100, 2, -100, -100, 5]])

        initial = initialize_masked_gaussian(
            embeddings,
            perturb_mask,
            0.1,
            group_ids=group_ids,
            generator=torch.Generator().manual_seed(9),
        )
        r_vadv = build_vadv(
            model,
            embeddings,
            clean,
            perturb_mask,
            labels=labels,
            group_ids=group_ids,
            xi=0.1,
            epsilon=0.1,
            generator=torch.Generator().manual_seed(9),
        )
        initial_kl = assistant_only_kl(
            clean, model(inputs_embeds=embeddings + initial).logits, labels=labels
        )
        adversarial_kl = assistant_only_kl(
            clean, model(inputs_embeds=embeddings + r_vadv).logits, labels=labels
        )
        self.assertGreaterEqual(adversarial_kl.item(), initial_kl.item())

    def test_outer_vat_loss_updates_embedding_block_norm_and_head(self) -> None:
        model = TinyCausalLM()
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        embeddings = model.get_input_embeddings()(ids)
        clean = model(inputs_embeds=embeddings).logits
        labels = torch.tensor([[-100, -100, 2, 3, 4, 5]])
        r = build_vadv(
            model,
            embeddings,
            clean.detach(),
            torch.tensor([[0, 1, 0, 0, 0, 0]], dtype=torch.bool),
            labels=labels,
            xi=1.0e-2,
            epsilon=0.3,
            generator=torch.Generator().manual_seed(21),
        )

        model.zero_grad(set_to_none=True)
        adv_logits = model(inputs_embeds=embeddings + r).logits
        vat_loss = assistant_only_kl(clean, adv_logits, labels=labels)
        vat_loss.backward()

        named = dict(model.named_parameters())
        for name in (
            "embedding.weight",
            "block.weight",
            "norm.weight",
            "lm_head.weight",
        ):
            self.assertIsNotNone(named[name].grad, name)
            self.assertTrue(torch.isfinite(named[name].grad).all(), name)
            self.assertGreater(named[name].grad.norm().item(), 0.0, name)


if __name__ == "__main__":
    unittest.main()
