#!/usr/bin/env python3
"""Thin launcher that installs VAT hooks, then calls Axolotl's own train flow."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from transformers.hf_argparser import HfArgumentParser

from axolotl.cli import load_cfg
from axolotl.cli.train import do_train
from axolotl.common.cli import TrainerCliArgs

from src.axolotl_integration import (
    configure_assistant_eot_supervision,
    install_axolotl_native_trainer_integration,
    install_axolotl_vat_integration,
)
from src.vat_trainer import (
    LOCKED_PILOT_SCHEDULER_TOTAL_STEPS,
    PILOT_SCHEDULER_TOTAL_STEPS_ENV,
    _pilot_scheduler_total_steps,
)


VAT_CONFIG_KEYS = {
    "supervise_assistant_eot",
    "native_axolotl_preprocessing",
    "use_metamorphic_vat",
    "metamorphic_candidate_db",
    "metamorphic_sampling_seed",
    "metamorphic_soft_worst_calibration",
    "use_vat",
    "use_minmax_adversarial_sft",
    "minmax_rho",
    "vat_radius_mode",
    "vat_power_iterations",
    "vat_xi",
    "vat_xi_mode",
    "vat_epsilon",
    "vat_rho",
    "vat_rho_schedule",
    "vat_rho_warmup_fraction",
    "vat_rho_delay_fraction",
    "vat_lambda",
    "vat_lambda_schedule",
    "vat_lambda_warmup_fraction",
    "vat_lambda_delay_fraction",
    "vat_lambda_cycles",
    "vat_gradient_similarity_ema_decay",
    "vat_gradient_telemetry_interval_steps",
    "vat_perturbation_mode",
    "vat_density_mode",
    "vat_density_vocabulary_chunk_size",
    "vat_semantic_margin_manifest",
    "vat_hparams_locked",
    "vat_perturb_mask_source",
    "phase2_materialized_train_dataset_path",
    "phase2_materialized_train_dataset_tree_sha256",
    "phase2_materialized_train_dataset_rows",
}
PILOT_SAVE_STEPS_ENV = "VAT_PILOT_SAVE_STEPS"


def _pop_flag(arguments: list[str], flag: str) -> bool:
    if flag not in arguments:
        return False
    arguments.remove(flag)
    return True


def _pop_option(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    value = arguments[index + 1]
    del arguments[index : index + 2]
    return value


def _load_with_vat_fields(config_path: Path):
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Axolotl v0.4.0 records this value as the string-typed
    # ``axolotl_config_path`` Pydantic field. Passing a Path reaches config
    # validation but is rejected before any training setup can run.
    cfg = load_cfg(str(config_path))
    # The historical Axolotl Pydantic schema ignores unknown extension fields.
    # Reinsert only the explicitly audited VAT namespace after its validation.
    for key in VAT_CONFIG_KEYS:
        if key in raw:
            cfg[key] = raw[key]
    return cfg


def _pilot_save_steps(
    max_steps: int,
    requested_save_steps: str | int | None = None,
) -> int:
    """Return a save cadence that includes the exact requested endpoint.

    Most stages save only their terminal checkpoint. A calibrated joint E1/E2
    stage may instead request ``save_steps=E1`` while running uninterrupted
    through ``E2``. Divisibility guarantees that both declared checkpoints
    are materialized as resumable checkpoint directories.
    """

    if max_steps < 1:
        raise ValueError("pilot max_steps must be positive")
    if requested_save_steps is None:
        return max_steps
    try:
        save_steps = int(requested_save_steps)
    except (TypeError, ValueError) as error:
        raise ValueError("pilot save steps must be a positive integer") from error
    if (
        save_steps < 1
        or save_steps > max_steps
        or max_steps % save_steps != 0
    ):
        raise ValueError(
            "pilot save steps must be positive, no larger than max_steps, "
            "and divide max_steps exactly"
        )
    return save_steps


def _configure_pilot_scheduler_horizon(max_steps: int) -> int:
    """Bind every staged pilot to the uninterrupted main 544-step LR curve."""

    if max_steps < 1:
        raise ValueError("pilot max_steps must be positive")
    raw_value = os.environ.get(
        PILOT_SCHEDULER_TOTAL_STEPS_ENV,
        str(LOCKED_PILOT_SCHEDULER_TOTAL_STEPS),
    )
    try:
        total_steps = _pilot_scheduler_total_steps(raw_value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if total_steps is None:
        raise RuntimeError("locked pilot scheduler horizon unexpectedly absent")
    if max_steps > total_steps:
        raise SystemExit(
            f"pilot max_steps {max_steps} exceeds scheduler horizon {total_steps}"
        )
    os.environ[PILOT_SCHEDULER_TOTAL_STEPS_ENV] = str(total_steps)
    return total_steps


def main() -> None:
    arguments = list(sys.argv[1:])
    pilot = _pop_flag(arguments, "--pilot")
    contract_output = _pop_option(arguments, "--vat-contract-output")
    resume_from_checkpoint = _pop_option(arguments, "--resume-from-checkpoint")
    pilot_grad_output = _pop_option(arguments, "--pilot-grad-output")
    phase2_diagnostic_output = _pop_option(
        arguments, "--phase2-diagnostic-output"
    )
    if not arguments:
        raise SystemExit(
            "usage: accelerate launch scripts/train.py CONFIG [--pilot] "
            "[Axolotl TrainerCliArgs...]"
        )
    config_path = Path(arguments.pop(0)).resolve()
    if not config_path.is_file():
        raise SystemExit(f"config does not exist: {config_path}")

    cfg = _load_with_vat_fields(config_path)
    native_preprocessing = bool(
        getattr(cfg, "native_axolotl_preprocessing", False)
    )
    if native_preprocessing:
        if bool(getattr(cfg, "supervise_assistant_eot", False)):
            raise SystemExit(
                "native Axolotl preprocessing forbids custom EOT supervision"
            )
        if bool(getattr(cfg, "use_vat", False)) and getattr(
            cfg, "vat_perturb_mask_source", None
        ) != "native_llama3_input_ids":
            raise SystemExit(
                "native Axolotl VAT must derive its mask from native input_ids"
            )
        install_axolotl_native_trainer_integration(verify_revision=True)
    else:
        install_axolotl_vat_integration(verify_revision=True)
        configure_assistant_eot_supervision(
            bool(getattr(cfg, "supervise_assistant_eot", False))
        )
    if contract_output:
        contract_path = Path(contract_output).resolve()
        if contract_path.exists() and not resume_from_checkpoint:
            raise SystemExit(
                f"contract already exists; refuse to overwrite fresh run: {contract_path}"
            )
        os.environ["VAT_CONTRACT_OUTPUT"] = str(contract_path)
    if resume_from_checkpoint:
        if not contract_output:
            raise SystemExit(
                "--resume-from-checkpoint requires --vat-contract-output so "
                "cumulative order/token evidence can be restored"
            )
        resolved_checkpoint = Path(resume_from_checkpoint).resolve()
        cfg.resume_from_checkpoint = str(resolved_checkpoint)
        os.environ["VAT_CONTRACT_RESUME"] = "1"
        os.environ["VAT_RESUME_CHECKPOINT"] = str(resolved_checkpoint)
    if pilot_grad_output:
        if not pilot:
            raise SystemExit("--pilot-grad-output is valid only with --pilot")
        grad_path = Path(pilot_grad_output).resolve()
        if grad_path.exists():
            raise SystemExit(f"pilot gradient diagnostic already exists: {grad_path}")
        os.environ["VAT_PILOT_GRAD_OUTPUT"] = str(grad_path)
    if phase2_diagnostic_output:
        if not pilot or not cfg.use_vat:
            raise SystemExit(
                "--phase2-diagnostic-output is valid only for a VAT pilot"
            )
        diagnostic_path = Path(phase2_diagnostic_output).resolve()
        if diagnostic_path.exists():
            raise SystemExit(
                f"Phase 2 diagnostic already exists: {diagnostic_path}"
            )
        os.environ["VAT_PHASE2_DIAGNOSTIC_OUTPUT"] = str(diagnostic_path)
    if cfg.use_vat and not cfg.vat_hparams_locked and not pilot:
        raise SystemExit(
            "VAT main training is locked until pilot evidence sets "
            "vat_hparams_locked: true"
        )

    if pilot:
        cfg.max_steps = int(os.environ.get("VAT_PILOT_MAX_STEPS", "10"))
        _configure_pilot_scheduler_horizon(cfg.max_steps)
        cfg.val_set_size = 0
        configured_shards = int(getattr(cfg, "dataset_shard_num", 100) or 100)
        configured_index = int(getattr(cfg, "dataset_shard_idx", 0) or 0)
        requested_shards = int(
            os.environ.get("VAT_PILOT_SHARDS", str(configured_shards))
        )
        requested_index = int(
            os.environ.get("VAT_PILOT_SHARD_INDEX", str(configured_index))
        )
        if (requested_shards, requested_index) != (
            configured_shards,
            configured_index,
        ):
            raise SystemExit(
                "pilot shard environment differs from the generated YAML; "
                "regenerate it through run_pilots.sh"
            )
        cfg.dataset_shard_num = configured_shards
        cfg.dataset_shard_idx = configured_index
        if os.environ.get("VAT_PILOT_OUTPUT_DIR"):
            cfg.output_dir = str(Path(os.environ["VAT_PILOT_OUTPUT_DIR"]).resolve())
        if "pilots" not in Path(cfg.output_dir).parts:
            raise SystemExit(
                "pilot output_dir must contain a `pilots` path component; set "
                "it in the generated config or VAT_PILOT_OUTPUT_DIR"
            )
        cfg.save_strategy = "steps"
        # run_pilots.sh launches the first segment with max_steps=HALF_STEPS.
        # Saving at that exact endpoint is essential: Trainer's final model
        # save is not a resumable checkpoint and must not leave auto-resume
        # selecting an earlier periodic checkpoint.
        try:
            cfg.save_steps = _pilot_save_steps(
                cfg.max_steps,
                os.environ.get(PILOT_SAVE_STEPS_ENV),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    parser = HfArgumentParser((TrainerCliArgs,))
    parsed_cli_args, remaining = parser.parse_args_into_dataclasses(
        args=arguments,
        return_remaining_strings=True,
    )
    if remaining:
        raise SystemExit(f"unrecognized launcher arguments: {remaining}")
    do_train(cfg, parsed_cli_args)


if __name__ == "__main__":
    main()
