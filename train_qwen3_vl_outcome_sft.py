"""
QLoRA SFT for Qwen3-VL-8B-Thinking on outcome-aware Multimodal Mind2Web.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from outcome_aware_sft_dataset import (
    OutcomeAwareMind2WebSFTDataset,
    OutcomeAwareSFTCollator,
    analyze_outcome_targets,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_DATASET_ID = "osunlp/Multimodal-Mind2Web"
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_output_dir = repo_root / "outputs" / "qwen3_vl_outcome_sft"

    parser = argparse.ArgumentParser(
        description="Run QLoRA-based supervised fine-tuning for outcome-aware Mind2Web."
    )
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--data_cache_dir",
        type=Path,
        default=repo_root / "data" / "multimodal_mind2web",
    )
    parser.add_argument(
        "--pruned_html_path",
        type=Path,
        default=repo_root / "data" / "pruned_html" / "train.jsonl",
        help="Preprocessed pruned HTML JSONL. If missing, HTML is pruned on the fly.",
    )
    parser.add_argument(
        "--outcome_path",
        type=Path,
        default=repo_root / "sft_data" / "sft_data" / "outcome_aware_sft.jsonl",
    )
    parser.add_argument("--output_dir", type=Path, default=default_output_dir)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--wandb_project", default="mind2web-outcome-aware-sft")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--max_html_chars", type=int, default=8_000)
    parser.add_argument("--max_seq_length", type=int, default=4_096)
    parser.add_argument("--suggested_max_new_tokens", type=int, default=96)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_wandb", action="store_true")
    return parser.parse_args()


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"qwen3-vl-outcome-aware-r{args.lora_rank}-e{args.num_train_epochs:g}-{timestamp}"


def _resolve_lora_targets(model) -> list[str]:
    available_module_names = {name for name, _ in model.named_modules()}
    resolved = [
        target
        for target in DEFAULT_LORA_TARGET_MODULES
        if any(name.endswith(target) for name in available_module_names)
    ]
    if not resolved:
        raise ValueError("Could not find any matching LoRA target modules in the loaded model.")
    return resolved


def _load_model_and_processor(args: argparse.Namespace):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank} if torch.cuda.is_available() else None

    model_load_kwargs = {
        "quantization_config": quant_config,
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            attn_implementation="flash_attention_2",
            **model_load_kwargs,
        )
    except Exception:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            attn_implementation="sdpa",
            **model_load_kwargs,
        )

    processor = AutoProcessor.from_pretrained(args.model_id)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_resolve_lora_targets(model),
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    return model, processor, lora_config


def main() -> None:
    args = parse_args()
    run_name = _resolve_run_name(args)
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.disable_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")
        os.environ.setdefault("WANDB_WATCH", "false")
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)

    target_stats = analyze_outcome_targets(args.outcome_path)
    dataset = OutcomeAwareMind2WebSFTDataset(
        split=args.split,
        dataset_id=args.dataset_id,
        data_cache_dir=args.data_cache_dir,
        pruned_html_path=args.pruned_html_path if args.pruned_html_path.exists() else None,
        outcome_path=args.outcome_path,
        max_html_chars=args.max_html_chars,
    )

    dataset_report = {
        **dataset.dataset_report,
        "output_target_stats": target_stats,
        "configured_max_seq_length": args.max_seq_length,
        "configured_suggested_max_new_tokens": args.suggested_max_new_tokens,
        "loss_function": "assistant_only_causal_lm_cross_entropy",
    }
    _json_dump(output_dir / "dataset_report.json", dataset_report)

    model, processor, lora_config = _load_model_and_processor(args)
    collator = OutcomeAwareSFTCollator(processor=processor, max_length=args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[] if args.disable_wandb else ["wandb"],
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
    )

    _json_dump(
        output_dir / "run_config.json",
        {
            "model_id": args.model_id,
            "dataset_id": args.dataset_id,
            "split": args.split,
            "pruned_html_path": str(args.pruned_html_path),
            "outcome_path": str(args.outcome_path),
            "run_name": run_name,
            "output_dir": str(output_dir),
            "lora_config": {
                "r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "lora_dropout": lora_config.lora_dropout,
                "target_modules": list(lora_config.target_modules),
            },
            "training": {
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "max_seq_length": args.max_seq_length,
                "suggested_max_new_tokens": args.suggested_max_new_tokens,
                "loss_function": "assistant_only_causal_lm_cross_entropy",
            },
        },
    )

    trainer.train()
    trainer.save_model()
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
