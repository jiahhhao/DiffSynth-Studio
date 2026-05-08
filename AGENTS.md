# DiffSynth-Studio AI Coding Agent Instructions

Welcome to the DiffSynth-Studio workspace. This file contains conventions and guidance to help AI coding agents be productive in this project.

## Architecture Guidelines
- **Core (`diffsynth/core/`)**: Infrastructure components including VRAM management (`diffsynth/core/vram`), data loaders, and device control.
- **Models (`diffsynth/models/`)**: Neural network modules (e.g., DiTs, VAEs, ControlNets).
- **Diffusion (`diffsynth/diffusion/`)**: Diffusion-specific logic, flow matching, losses, and training runners.
- **Pipelines (`diffsynth/pipelines/`)**: High-level orchestrators combining components for generation tasks. Refer to relevant modules or markdown files like `Model.md` (if they exist) for architecture flows.

## Development & Execution
- **Installation**: Uses `pyproject.toml` and requires Python $\ge 3.10.1$. Install dependencies using `pip install -e .`
- **Execution**: The project uses standalone scripts from the `examples/` directory for most workflows.
  - Inference: `python examples/<model_name>/model_inference/<script>.py`
  - Training: Handled via `accelerate launch`. Review `.sh` files in `examples/<model_name>/model_training/` for specific configurations.
- **Testing**: No global unit-test suite exists (e.g., pytest). Validation relies on script-based execution from the `examples/` structures.

## Conventions
- **VRAM Optimizations**: DiffSynth handles layer-by-layer offloads natively. Ensure inference components use the correct dictionaries mapping offload/onload `device` and `dtype`.
- **Checkpoints & Data**: Datasets and checkpoints leverage `modelscope` heavily.
- **Documentation**: Read extensive guides localized in [docs/en/](docs/en/) and [docs/zh/](docs/zh/). Focus on linking to these pages rather than duplicating knowledge.