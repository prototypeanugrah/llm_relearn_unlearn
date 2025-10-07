# LLM Multimodal Unlearning Framework

## Problem Statement
This repository develops machine unlearning workflows for vision-language models so that LLaVA-based checkpoints forget sensitive samples from the MLLMU Bench forget split while retaining core competency on the retain split.

## Installation (uv)
```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

## Quick Start
```bash
# Hydra-driven Accelerate training (gradient ascent baseline)
python train.py --config-name default

# Tweak batch size and epochs for faster experiments
python train.py --config-name default trainer.args.num_train_epochs=3 \
    trainer.args.per_device_train_batch_size=4
```

## Project Structure
```
llm_relearn_unlearn/
├── train.py                # Main Hydra entry point
├── src/                    # Core data, model, and trainer modules
├── configs/                # Hydra configs (model, data, trainer, deepspeed)
├── outputs/                # Generated checkpoints and logs
├── old_scripts/            # Legacy experiments (not part of the workflow)
├── pyproject.toml          # Project metadata and dependencies
└── uv.lock                 # Resolved dependency lockfile
```

## Training Pipeline
- `configs/default.yaml`: Root Hydra config that binds model, trainer, and MLLMU data presets.
- `configs/model/llava-1.5-7b-hf.yaml`: Loads the LLaVA 1.5 7B base model, tokenizer, and PEFT setup.
- `configs/data/unlearn.yaml`: Declares forget/retain dataloaders backed by MLLMU Bench splits.
- `configs/trainer/grad_ascent.yaml`: Accelerate-friendly gradient-ascent settings with warmup handling.
- `configs/trainer/grad_diff.yaml`: DeepSpeed-ready gradient difference recipe pointing at `configs/deepspeed/zero3.json`.
- `configs/deepspeed/zero3.json`: Zero-3 configuration tuned for CPU offloading and automatic batch sizing.

## Unlearning Algorithms
- **Gradient Ascent** (`GradAscent`): Maximizes forget loss while monitoring retain performance with Accelerate.
- **Gradient Difference** (`GradDiff`): Balances forgetting and retention via weighted loss terms and DeepSpeed scaling.

## Dataset
We rely on the MLLMU Bench dataset (`MLLMMU/MLLMU-Bench`) with `forget15` and `retain_85` splits, using question-answer fields for multimodal unlearning experiments.

## License
Distributed under the MIT License.

## Acknowledgments
Built on top of the LLaVA model family and the MLLMU Bench authors' dataset releases; thanks to both communities for open-sourcing their work.
