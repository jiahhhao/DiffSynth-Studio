# Repository Guidelines

## Project Structure & Module Organization

DiffSynth-Studio is a Python package rooted at `diffsynth/`. Core runtime utilities live in `diffsynth/core/`, model definitions in `diffsynth/models/`, diffusion and training abstractions in `diffsynth/diffusion/`, and high-level generation orchestrators in `diffsynth/pipelines/`. Shared model configuration maps are under `diffsynth/configs/`.

Examples are the primary workflows. Use `examples/<model_name>/model_inference/` for standard inference, `examples/<model_name>/model_inference_low_vram/` for offload-heavy inference, and `examples/<model_name>/model_training/` for training scripts and `accelerate` configs. Documentation is maintained in `docs/en/` and `docs/zh/`. Local datasets, downloaded models, and generated outputs commonly appear in `data/`, `models/`, and `results/`.

## Build, Test, and Development Commands

- `pip install -e .`: install the package in editable mode with dependencies from `pyproject.toml`.
- `pip install -r docs/requirements.txt`: install documentation dependencies when editing docs.
- `python examples/flux2/model_inference/FLUX.2-dev.py`: run a representative inference example; choose the script matching the model being changed.
- `accelerate launch examples/<model>/model_training/train.py ...`: run training directly, or prefer the matching `.sh` file under `examples/<model>/model_training/` when one exists.
- `python -m build`: build source and wheel distributions if validating package metadata.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax and existing PEP 8 style: 4-space indentation, `snake_case` functions and variables, `PascalCase` classes, and explicit imports. Keep model and pipeline file names aligned with their family, such as `wan_video.py`, `flux_image.py`, or `qwen_image.py`. Example scripts may preserve upstream model names with capitals, dots, and hyphens.

## Testing Guidelines

There is no global pytest suite in this repository. Validate changes with the smallest relevant example script, preferably a low-cost inference path or validation script under `examples/<model>/model_training/validate_*`. For training changes, run or dry-run the corresponding `accelerate launch` command and document hardware, dtype, and VRAM assumptions. Avoid committing generated files from `results/`, checkpoints, or local datasets.

## Commit & Pull Request Guidelines

Recent history uses short imperative summaries, often with PR numbers, for example `Support ERNIE-Image (#1389)` or `fix version issue of transformers (#1412)`. Keep commits focused and mention the affected model, pipeline, or subsystem.

Pull requests should include a problem statement, implementation summary, validation commands, and any changed model or checkpoint requirements. Link related issues, update both English and Chinese docs for user-facing behavior, and include sample outputs or screenshots for visual changes when practical.

## Agent-Specific Instructions

Respect the architecture boundaries above. When adding inference components, preserve DiffSynth’s native VRAM offload patterns by passing correct `device` and `dtype` dictionaries through model configs. Prefer linking to existing docs in `docs/en/` and `docs/zh/` instead of duplicating long explanations.
