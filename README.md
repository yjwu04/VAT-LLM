# VAT-LLM

VAT-LLM implements question-only virtual adversarial training for instruction-
tuned decoder language models. Perturbations are applied to user-message
embeddings while the supervised objective remains restricted to assistant
tokens.

## Highlights

- Conversation-aware user-token masking with packed-sequence support.
- FP32 masked KL, relative perturbation radii, and configurable VAT schedules.
- Gradient-similarity weighting and metamorphic VAT variants.
- Axolotl integration without replacing its optimizer, scheduler, dataloader,
  distributed training, or checkpointing lifecycle.
- Training configurations and focused unit and integration tests.

## Repository layout

- `src/`: VAT objectives, masking, trainer integration, and artifact helpers.
- `configs/`: SFT and VAT experiment configurations.
- `scripts/train.py`: training entrypoint.
- `tests/`: unit and integration tests.

## Quick start

Prepare the required model, dataset, and an Axolotl checkout at commit
`8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2`, then launch a configuration with
distributed PyTorch. Replace the `/path/to/...` placeholders in the selected
configuration with paths available on your machine:

```bash
PYTHONPATH="$PWD:$PWD/axolotl/src" \
torchrun --standalone --nproc_per_node=4 \
  scripts/train.py configs/native_axolotl_vat_rho_0p03_lambda_warmup.yaml
```

Models, datasets, checkpoints, experiment outputs, and third-party benchmark
fixtures are not included in this repository.
