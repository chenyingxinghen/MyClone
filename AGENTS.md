# Repository Guidelines

## Project Structure & Module Organization
This repository is a compact Python fine-tuning pipeline for chat-style cloning.
`process_data.py` converts WeChat/QQ exports into ShareGPT-format data, using
settings from `config.json`. `train.py` runs QLoRA SFT with transformers, PEFT,
bitsandbytes, TRL, and accelerate-compatible DDP. Generated training data lives
under `dataset/sft/`, especially `sft-my.json` and `dataset_info.json`. Training
outputs default to `output/` locally or `/kaggle/working/output` on Kaggle. Large
archives, raw exports, generated datasets, and model outputs should stay out of
normal commits unless intentionally sanitized and documented.

## Build, Test, and Development Commands
- `pip install pandas`: install the only dependency needed for data processing.
- `python process_data.py`: read chat exports from `config.json` and regenerate
  `dataset/sft/`.
- `pip install -r requirements.txt`: install the full training stack.
- `python train.py --dataset ./dataset/sft`: run single-process training locally.
- `torchrun --nproc_per_node=2 train.py --dataset /kaggle/input/YOUR_DATASET/sft --model Qwen/Qwen3-8B`: recommended Kaggle 2xT4 DDP smoke run.

## Coding Style & Naming Conventions
Use Python 3 with 4-space indentation, UTF-8 files, and `snake_case` for functions,
variables, and CLI flags. Keep constants in `UPPER_SNAKE_CASE`. Prefer
`pathlib.Path` for filesystem paths, `argparse` for script options, and small
dataclasses for structured records. Follow the existing import order: standard
library first, then third-party imports. Add comments only where they explain
non-obvious training, data filtering, or DDP behavior.

## Testing Guidelines
There is no formal test suite yet. Validate data changes by running
`python process_data.py`, then inspect `dataset/sft/sft-my.json` for alternating
`user` and `assistant` turns. Validate training changes with a short single-GPU
or 2xT4 run, low epochs, and evaluation enabled. If adding tests, place them in
`tests/`, name files `test_*.py`, and prefer small fixtures that do not include
private chat logs.

## Commit & Pull Request Guidelines
Recent history uses short, direct Chinese commit summaries, for example
`双卡加速` and `过拟合与数据质量修复`. Keep commits focused and describe the user-visible
change or bug fixed. Pull requests should include a brief summary, commands run,
dataset/model assumptions, and screenshots or loss snippets when training
behavior changes. Link related issues when available and call out any large
artifact or data-format changes.

## Security & Configuration Tips
Treat `config.json`, raw chat exports, generated datasets, and checkpoints as
private by default. Do not add secrets, personal identifiers, or unsanitized chat
content to examples, tests, or pull requests.
