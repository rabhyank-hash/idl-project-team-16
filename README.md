# Multimodal Mind2Web Download + Outcome-Aware SFT

This repo is set up around two entrypoints:

- `download_data_and_model.py`
- `train_qwen3_vl_outcome_sft.py`

The intended workflow is:

1. install dependencies
2. download the Multimodal Mind2Web dataset and Qwen3-VL-8B-Thinking
3. run supervised fine-tuning with QLoRA on the outcome-aware SFT targets

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

If you are on PSC and want to avoid home-directory quota issues, redirect caches before installing or training:

```bash
mkdir -p /ocean/projects/cis260137p/mramnath/.cache/{huggingface,pip,wandb,torch,tmp}
export HF_HOME=/ocean/projects/cis260137p/mramnath/.cache/huggingface
export HF_DATASETS_CACHE=/ocean/projects/cis260137p/mramnath/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/ocean/projects/cis260137p/mramnath/.cache/huggingface/transformers
export HUGGINGFACE_HUB_CACHE=/ocean/projects/cis260137p/mramnath/.cache/huggingface/hub
export TORCH_HOME=/ocean/projects/cis260137p/mramnath/.cache/torch
export WANDB_DIR=/ocean/projects/cis260137p/mramnath/.cache/wandb
export PIP_CACHE_DIR=/ocean/projects/cis260137p/mramnath/.cache/pip
export TMPDIR=/ocean/projects/cis260137p/mramnath/.cache/tmp
```

## 1. Download data and model

The download script pulls:

- `osunlp/Multimodal-Mind2Web` into `data/`
- `Qwen/Qwen3-VL-8B-Thinking` into `assets/`

Basic usage:

```bash
python download_data_and_model.py
```

Useful variants:

```bash
python download_data_and_model.py --dataset-only
python download_data_and_model.py --model-only
python download_data_and_model.py --splits train
python download_data_and_model.py --force-redownload
```

Default locations:

- dataset cache: `data/multimodal_mind2web/`
- model snapshot: `assets/Qwen3-VL-8B-Thinking/`
- download manifest: `data/mind2web_download_manifest.json`

## 2. Run outcome-aware SFT

The SFT script fine-tunes `Qwen/Qwen3-VL-8B-Thinking` with:

- 4-bit quantization via bitsandbytes
- LoRA
- default LoRA rank `2`
- default training length `3` epochs
- assistant-only causal LM cross-entropy loss
- W&B logging and checkpoint reporting by default

Basic usage:

```bash
python train_qwen3_vl_outcome_sft.py
```

Recommended PSC example:

```bash
python train_qwen3_vl_outcome_sft.py \
  --data_cache_dir data/multimodal_mind2web \
  --pruned_html_path data/pruned_html/train.jsonl \
  --outcome_path sft_data/sft_data/outcome_aware_sft.jsonl \
  --output_dir outputs/qwen3_vl_outcome_sft
```

Useful hyperparameters:

```bash
python train_qwen3_vl_outcome_sft.py \
  --lora_rank 4 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4
```

### What the script trains on

Training uses the intersection of:

- the Mind2Web `train` split
- `sft_data/sft_data/outcome_aware_sft.jsonl`

Matching is done by `dataset_index`.

Not every train example has an outcome-aware target, so the script automatically keeps only the common samples. At runtime it writes a `dataset_report.json` containing:

- train split size
- number of outcome-aware rows
- number of matched rows
- whether all train samples were covered
- the first missing `dataset_index` values
- output target length statistics

### Prompt and target format

The model is trained to produce exactly:

```text
Answer: <index>
Outcome: {"transition_type":"...","changed_region":"...","change_magnitude":"...","confidence":"..."}
```

Inputs include:

- screenshot
- pruned HTML
- task
- candidate actions

If `data/pruned_html/train.jsonl` is missing, the script falls back to pruning `cleaned_html` on the fly.

### Weights & Biases

W&B is enabled by default. Authenticate first:

```bash
wandb login
```

Or:

```bash
export WANDB_API_KEY=your_token_here
```

Then run training normally. To disable W&B:

```bash
python train_qwen3_vl_outcome_sft.py --disable_wandb
```

### Training outputs

Each run writes to a timestamped subdirectory under the chosen `--output_dir`, including:

- model checkpoints
- adapter weights
- `dataset_report.json`
- `run_config.json`
- processor files

## File summary

- [download_data_and_model.py](/ocean/projects/cis260137p/mramnath/idl-project-team-16/download_data_and_model.py)
- [train_qwen3_vl_outcome_sft.py](/ocean/projects/cis260137p/mramnath/idl-project-team-16/train_qwen3_vl_outcome_sft.py)
- [requirements.txt](/ocean/projects/cis260137p/mramnath/idl-project-team-16/requirements.txt)
