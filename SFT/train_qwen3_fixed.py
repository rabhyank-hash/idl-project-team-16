import os
import gc
from pathlib import Path
import torch
from PIL import Image
from peft import LoraConfig,get_peft_model
from transformers import AutoProcessor,Trainer,TrainerCallback,TrainingArguments,Qwen3VLForConditionalGeneration
from dataloading_qwen3_fixed import load_records
LOCAL_ROOT=os.environ.get("LOCAL")
DEFAULT_IMAGE_ROOT=str(Path(LOCAL_ROOT) / "project_data") if LOCAL_ROOT else "./data/images"
DEFAULT_CONFIG={
    "model_id": "Qwen/Qwen3-VL-8B-Thinking",
    "hf_token": None,
    "hf_cache_dir": "./.hf_cache",
    "use_flash_attn": True,
    "use_quantization": False,
    "image_root": DEFAULT_IMAGE_ROOT,
    "jsonl_path": "./vanilla_matched_6445.jsonl",
    "output_dir": "./outputs/qwen3-vl-8b-thinking-sft",
    "max_seq_length": 2048,
    "epochs": 3,
    "lr": 1e-4,
    "lr_scheduler_type": "cosine",
    "train_bs": 1,
    "eval_bs": 1,
    "grad_accum": 8,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "seed": 42,
    "log_steps": 10,
    "save_steps": 200,
    "use_bf16": True,
    "grad_checkpointing": True,
    "use_lora": True,
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_targets": ("q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"),
    "dataloader_num_workers": 0,
    "cache_clear_steps": 50,
}
def build_messages(system_prompt: str,prompt_text: str,target_text: str | None=None):
    user_msg={
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text","text": prompt_text},
        ],
    }
    msgs=[]
    if system_prompt:
        msgs.append({"role": "system","content": [{"type": "text","text": system_prompt}]})
    msgs.append(user_msg)
    if target_text is None:
        return msgs
    msgs.append({"role": "assistant","content": [{"type": "text","text": target_text.strip()}]})
    return msgs
class VLMActionCollator:
    def __init__(self,processor,max_seq_length: int=4608):
        self.processor=processor
        self.max_seq_length=max_seq_length
    def __call__(self,features):
        imgs=[]
        full_texts=[]
        prompt_texts=[]
        for ex in features:
            img_path=ex["image"]
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Screenshot not found: {img_path}")
            img=Image.open(img_path).convert("RGB")
            imgs.append(img)
            raw_prompt=ex["prompt_text"]
            markers=["Candidate actions","Candidates","Action candidates","Valid actions"]
            cut=-1
            for m in markers:
                idx=raw_prompt.lower().find(m.lower())
                if idx != -1:
                    cut=idx
                    break
            if cut != -1:
                head=raw_prompt[:2500]
                candidates=raw_prompt[cut:]
                if len(candidates) > 3500:
                    candidates=candidates[:3500]
                prompt_text=head + "\n...\n" + candidates
            else:
                prompt_text=raw_prompt[:2500] + "\n...\n" + raw_prompt[-3500:]
            full_msgs=build_messages(ex.get("system_prompt",""),prompt_text,ex["target"])
            prompt_msgs=build_messages(ex.get("system_prompt",""),prompt_text,None)
            full_texts.append(
                self.processor.apply_chat_template(
                    full_msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        full_batch=self.processor(
            text=full_texts,
            images=imgs,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        prompt_batch=self.processor(
            text=prompt_texts,
            images=imgs,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        labels=full_batch["input_ids"].clone()
        for i in range(labels.size(0)):
            prompt_len=int(prompt_batch["attention_mask"][i].sum().item())
            labels[i,:prompt_len]=-100
            labels[i,full_batch["attention_mask"][i] == 0]=-100
        full_batch["labels"]=labels
        return full_batch
class CUDACacheCleanupCallback(TrainerCallback):
    def __init__(self,every_n_steps: int):
        self.every_n_steps=max(0,int(every_n_steps))
    def on_step_end(self,args,state,control,**kwargs):
        if self.every_n_steps <= 0:
            return control
        if torch.cuda.is_available() and state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            gc.collect()
            torch.cuda.empty_cache()
        return control
def build_model_and_processor(config: dict):
    print("Loading model...")
    attn_impl="flash_attention_2" if config["use_flash_attn"] else "sdpa"
    token=config.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    cache_dir=config.get("hf_cache_dir")
    if config.get("use_quantization",False):
        raise ValueError("Quantization is disabled for this script. Use bf16 training.")
    dtype=torch.bfloat16 if config["use_bf16"] and torch.cuda.is_available() else torch.float16
    model_id=config["model_id"]
    print(f"Loading model_id={model_id}")
    try:
        try:
            model=Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                trust_remote_code=True,
                token=token,
            )
            print(f"Model loaded with attn_implementation={attn_impl}")
        except Exception as exc:
            if attn_impl != "sdpa":
                print(f"Failed with {attn_impl}: {exc}")
                print("Falling back to sdpa...")
                model=Qwen3VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    attn_implementation="sdpa",
                    low_cpu_mem_usage=True,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    token=token,
                )
                print("Model loaded with attn_implementation=sdpa")
            else:
                raise
        processor=AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=token,
            cache_dir=cache_dir,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 28 * 28,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load model_id={model_id}. If gated,set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or login first."
        ) from exc
    if config["grad_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache=False
        if hasattr(model,"enable_input_require_grads"):
            model.enable_input_require_grads()
    if config["use_lora"]:
        peft_cfg=LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            target_modules=list(config["lora_targets"]),
            task_type="CAUSAL_LM",
        )
        model=get_peft_model(model,peft_cfg)
        model.print_trainable_parameters()
    return model,processor
def train(config: dict):
    config={**DEFAULT_CONFIG,**config}
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
    hf_cache_dir=os.path.abspath(config["hf_cache_dir"])
    os.makedirs(hf_cache_dir,exist_ok=True)
    os.environ["HF_HOME"]=hf_cache_dir
    os.environ["HF_HUB_CACHE"]=os.path.join(hf_cache_dir,"hub")
    os.environ["TRANSFORMERS_CACHE"]=os.path.join(hf_cache_dir,"transformers")
    os.makedirs(config["output_dir"],exist_ok=True)
    dataset=load_records(config["jsonl_path"],config["image_root"])
    print(dataset)
    print("Columns:",dataset.column_names)
    split=dataset.train_test_split(test_size=0.02,seed=config["seed"])
    train_ds=split["train"]
    eval_ds=split["test"]
    print(f"Train size: {len(train_ds)}")
    print(f"Eval size:  {len(eval_ds)}")
    print(f"Total loaded: {len(dataset)}")
    model,processor=build_model_and_processor(config)
    collator=VLMActionCollator(processor=processor,max_seq_length=config["max_seq_length"])
    args=TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        learning_rate=config["lr"],
        lr_scheduler_type=config["lr_scheduler_type"],
        per_device_train_batch_size=config["train_bs"],
        per_device_eval_batch_size=config["eval_bs"],
        gradient_accumulation_steps=config["grad_accum"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        logging_steps=config["log_steps"],
        save_steps=config["save_steps"],
        eval_strategy="steps",
        eval_steps=config["save_steps"],
        save_total_limit=2,
        bf16=config["use_bf16"] and torch.cuda.is_available(),
        fp16=(not config["use_bf16"]) and torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=config.get("dataloader_num_workers",0),
        gradient_checkpointing=config["grad_checkpointing"],
        seed=config["seed"],
        data_seed=config["seed"],
        ddp_find_unused_parameters=False,
    )
    cbs=[]
    if config.get("cache_clear_steps",0):
        cbs.append(CUDACacheCleanupCallback(config["cache_clear_steps"]))
    trainer=Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=cbs,
    )
    trainer.train()
    trainer.save_model(config["output_dir"])
    processor.save_pretrained(config["output_dir"])
    print("Saved to",config["output_dir"])
