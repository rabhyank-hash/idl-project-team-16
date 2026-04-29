import argparse
import importlib
import train_qwen3_fixed as train_module
DEFAULT_CONFIG=train_module.DEFAULT_CONFIG
def _config_to_dict(config):
    if isinstance(config,dict):
        return dict(config)
    raw={}
    for name in dir(config):
        if name.startswith("_"):
            continue
        try:
            value=getattr(config,name)
        except AttributeError:
            continue
        if callable(value):
            continue
        raw[name]=value
    return raw
def run_training(config):
    importlib.reload(train_module)
    train_module.train(_config_to_dict(config))
def parse_args():
    p=argparse.ArgumentParser(description="Train Qwen3-VL on next-action prediction from HTML and screenshots.")
    p.add_argument("--model-id",default=DEFAULT_CONFIG["model_id"])
    p.add_argument("--hf-token",default=DEFAULT_CONFIG["hf_token"])
    p.add_argument("--hf-cache-dir",default=DEFAULT_CONFIG["hf_cache_dir"])
    p.add_argument("--image-root",default=DEFAULT_CONFIG["image_root"])
    p.add_argument("--jsonl-path",default=DEFAULT_CONFIG["jsonl_path"])
    p.add_argument("--output-dir",default=DEFAULT_CONFIG["output_dir"])
    p.add_argument("--max-seq-length",type=int,default=DEFAULT_CONFIG["max_seq_length"])
    p.add_argument("--max-length",type=int,dest="max_seq_length",help="Alias for --max-seq-length")
    p.add_argument("--epochs",type=float,default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--lr",type=float,default=DEFAULT_CONFIG["lr"])
    p.add_argument("--lr-scheduler-type",default=DEFAULT_CONFIG["lr_scheduler_type"])
    p.add_argument("--train-bs",type=int,default=DEFAULT_CONFIG["train_bs"])
    p.add_argument("--eval-bs",type=int,default=DEFAULT_CONFIG["eval_bs"])
    p.add_argument("--grad-accum",type=int,default=DEFAULT_CONFIG["grad_accum"])
    p.add_argument("--warmup-ratio",type=float,default=DEFAULT_CONFIG["warmup_ratio"])
    p.add_argument("--weight-decay",type=float,default=DEFAULT_CONFIG["weight_decay"])
    p.add_argument("--seed",type=int,default=DEFAULT_CONFIG["seed"])
    p.add_argument("--log-steps",type=int,default=DEFAULT_CONFIG["log_steps"])
    p.add_argument("--save-steps",type=int,default=DEFAULT_CONFIG["save_steps"])
    p.add_argument("--use-bf16",action="store_true",default=DEFAULT_CONFIG["use_bf16"])
    p.add_argument("--no-bf16",action="store_false",dest="use_bf16")
    p.add_argument("--grad-checkpointing",action="store_true",default=DEFAULT_CONFIG["grad_checkpointing"])
    p.add_argument("--no-grad-checkpointing",action="store_false",dest="grad_checkpointing")
    p.add_argument("--use-lora",action="store_true",default=DEFAULT_CONFIG["use_lora"])
    p.add_argument("--no-lora",action="store_false",dest="use_lora")
    p.add_argument("--lora-r",type=int,default=DEFAULT_CONFIG["lora_r"])
    p.add_argument("--lora-alpha",type=int,default=DEFAULT_CONFIG["lora_alpha"])
    p.add_argument("--lora-dropout",type=float,default=DEFAULT_CONFIG["lora_dropout"])
    p.add_argument("--lora-targets",nargs="*",default=list(DEFAULT_CONFIG["lora_targets"]))
    p.add_argument("--use-flash-attn",action="store_true",default=DEFAULT_CONFIG["use_flash_attn"])
    p.add_argument("--no-flash-attn",action="store_false",dest="use_flash_attn")
    p.add_argument("--use-quantization",action="store_true",default=DEFAULT_CONFIG["use_quantization"])
    p.add_argument("--no-quantization",action="store_false",dest="use_quantization")
    return p.parse_args()
def main():
    args=parse_args()
    cfg={
        "model_id": args.model_id,
        "hf_token": args.hf_token,
        "hf_cache_dir": args.hf_cache_dir,
        "image_root": args.image_root,
        "jsonl_path": args.jsonl_path,
        "output_dir": args.output_dir,
        "max_seq_length": args.max_seq_length if args.max_seq_length is not None else DEFAULT_CONFIG["max_seq_length"],
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_scheduler_type": args.lr_scheduler_type,
        "train_bs": args.train_bs,
        "eval_bs": args.eval_bs,
        "grad_accum": args.grad_accum,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "log_steps": args.log_steps,
        "save_steps": args.save_steps,
        "use_bf16": args.use_bf16,
        "grad_checkpointing": args.grad_checkpointing,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_targets": tuple(args.lora_targets),
        "use_flash_attn": args.use_flash_attn,
        "use_quantization": args.use_quantization,
    }
    train_module.train(cfg)
if __name__ == "__main__":
    main()
