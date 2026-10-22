from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

import src.axolotl_integration as integration
from src.axolotl_integration import _pad_perturb_mask


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_mask_and_source_padding_match_collator_target(padding_side):
    features = [
        {
            "input_ids": [1, 2, 3],
            "perturb_mask": [0, 1, 0],
            "source_ids": [11, 11, 11],
        },
        {
            "input_ids": [4, 5, 6, 7, 8],
            "perturb_mask": torch.tensor([1, 1, 0, 0, 0]),
            "source_ids": np.array([22, 22, 22, 22, 22]),
        },
    ]
    _pad_perturb_mask(
        features,
        pad_to_multiple_of=8,
        padding_side=padding_side,
    )

    assert len(features[0]["perturb_mask"]) == 8
    assert len(features[0]["source_ids"]) == 8
    assert len(features[1]["perturb_mask"]) == 8
    assert len(features[1]["source_ids"]) == 8
    pad_slice = slice(3, None) if padding_side == "right" else slice(0, 5)
    assert features[0]["perturb_mask"][pad_slice] == [0] * 5
    assert features[0]["source_ids"][pad_slice] == [-1] * 5
    tensor_pad = slice(5, None) if padding_side == "right" else slice(0, 3)
    assert torch.equal(
        features[1]["perturb_mask"][tensor_pad],
        torch.zeros(3, dtype=torch.long),
    )
    assert np.array_equal(
        features[1]["source_ids"][tensor_pad],
        np.full(3, -1),
    )


def test_mask_padding_fails_when_only_part_of_batch_has_mask():
    with pytest.raises(ValueError, match="missing from part"):
        _pad_perturb_mask(
            [
                {"input_ids": [1], "perturb_mask": [1]},
                {"input_ids": [2]},
            ],
            pad_to_multiple_of=8,
            padding_side="right",
        )


def test_mask_padding_fails_on_partial_source_ids_or_unaligned_fields():
    with pytest.raises(ValueError, match="source_ids is missing from part"):
        _pad_perturb_mask(
            [
                {
                    "input_ids": [1],
                    "perturb_mask": [1],
                    "source_ids": [10],
                },
                {"input_ids": [2], "perturb_mask": [1]},
            ],
            pad_to_multiple_of=None,
            padding_side="right",
        )
    with pytest.raises(ValueError, match="perturb_mask is missing from part"):
        _pad_perturb_mask(
            [
                {"input_ids": [1]},
                {"input_ids": [2], "perturb_mask": [1]},
            ],
            pad_to_multiple_of=None,
            padding_side="right",
        )

    with pytest.raises(ValueError, match="length differs"):
        _pad_perturb_mask(
            [{"input_ids": [1, 2], "perturb_mask": [1]}],
            pad_to_multiple_of=None,
            padding_side="right",
        )

    with pytest.raises(ValueError, match="must be binary"):
        _pad_perturb_mask(
            [{"input_ids": [1, 2], "perturb_mask": [1, 2]}],
            pad_to_multiple_of=None,
            padding_side="right",
        )


def test_train_launcher_passes_string_path_to_axolotl(monkeypatch, tmp_path):
    from scripts import train

    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(
        "use_vat: true\n"
        "supervise_assistant_eot: true\n"
        "vat_radius_mode: relative\n"
        "vat_xi: 0.001\n"
        "vat_xi_mode: absolute\n"
        "vat_rho: 0.003\n"
        "phase2_materialized_train_dataset_path: /derived\n"
        "phase2_materialized_train_dataset_tree_sha256: "
        + "a" * 64
        + "\n"
        "phase2_materialized_train_dataset_rows: 50000\n",
        encoding="utf-8",
    )
    observed = {}

    def fake_load_cfg(path):
        observed["path"] = path
        return {}

    monkeypatch.setattr(train, "load_cfg", fake_load_cfg)
    config = train._load_with_vat_fields(config_path)

    assert observed["path"] == str(config_path)
    assert isinstance(observed["path"], str)
    assert config["use_vat"] is True
    assert config["supervise_assistant_eot"] is True
    assert config["vat_radius_mode"] == "relative"
    assert config["vat_xi"] == pytest.approx(0.001)
    assert config["vat_xi_mode"] == "absolute"
    assert config["vat_rho"] == pytest.approx(0.003)
    assert config["phase2_materialized_train_dataset_path"] == "/derived"
    assert config["phase2_materialized_train_dataset_tree_sha256"] == "a" * 64
    assert config["phase2_materialized_train_dataset_rows"] == 50_000


def test_train_launcher_preserves_native_preprocessing_extension(monkeypatch, tmp_path):
    from scripts import train

    config_path = tmp_path / "native.yaml"
    config_path.write_text(
        "native_axolotl_preprocessing: true\n"
        "use_vat: true\n"
        "vat_perturb_mask_source: native_llama3_input_ids\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train, "load_cfg", lambda path: {})
    config = train._load_with_vat_fields(config_path)
    assert config["native_axolotl_preprocessing"] is True
    assert config["vat_perturb_mask_source"] == "native_llama3_input_ids"


def test_train_launcher_preserves_metamorphic_extensions(monkeypatch, tmp_path):
    from scripts import train

    config_path = tmp_path / "metamorphic.yaml"
    config_path.write_text(
        "use_metamorphic_vat: true\n"
        "metamorphic_candidate_db: /candidates.sqlite3\n"
        "metamorphic_sampling_seed: 42\n"
        "metamorphic_soft_worst_calibration: /calibration.json\n"
        "vat_perturbation_mode: metamorphic_text_uniform_sft\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train, "load_cfg", lambda path: {})
    config = train._load_with_vat_fields(config_path)
    assert config["use_metamorphic_vat"] is True
    assert config["metamorphic_candidate_db"] == "/candidates.sqlite3"
    assert config["metamorphic_sampling_seed"] == 42
    assert config["metamorphic_soft_worst_calibration"] == "/calibration.json"
    assert config["vat_perturbation_mode"] == "metamorphic_text_uniform_sft"


def test_install_hook_replaces_index_view_with_bound_materialized_dataset(
    monkeypatch,
):
    from axolotl.utils.data import sft as axolotl_sft

    class LogicalView:
        def __len__(self):
            return 50_000

    logical_view = LogicalView()
    materialized = object()

    def original(*args, **kwargs):
        del args, kwargs
        return logical_view, None, ["prompter"]

    monkeypatch.setattr(axolotl_sft, "load_prepare_datasets", original)
    observed = {}

    def fake_load(path, *, expected_tree_sha256, expected_rows):
        observed.update(
            {
                "path": path,
                "tree": expected_tree_sha256,
                "rows": expected_rows,
            }
        )
        return materialized

    monkeypatch.setattr(integration, "load_materialized_dataset", fake_load)
    integration.install_axolotl_vat_integration(verify_revision=False)
    cfg = SimpleNamespace(
        dataset_shard_num=2,
        dataset_shard_idx=0,
        phase2_materialized_train_dataset_path="/derived",
        phase2_materialized_train_dataset_tree_sha256="b" * 64,
        phase2_materialized_train_dataset_rows=50_000,
    )
    train, evaluation, prompters = axolotl_sft.load_prepare_datasets(
        None, cfg, None
    )
    assert train is materialized
    assert evaluation is None
    assert prompters == ["prompter"]
    assert observed == {
        "path": "/derived",
        "tree": "b" * 64,
        "rows": 50_000,
    }


def test_pilot_saves_exact_requested_endpoint_for_resume():
    from scripts.train import _pilot_save_steps

    assert _pilot_save_steps(10) == 10
    assert _pilot_save_steps(20) == 20
    assert _pilot_save_steps(272, "136") == 136
    with pytest.raises(ValueError, match="positive"):
        _pilot_save_steps(0)
    with pytest.raises(ValueError, match="divide max_steps"):
        _pilot_save_steps(272, "100")
    with pytest.raises(ValueError, match="positive integer"):
        _pilot_save_steps(272, "not-an-integer")
