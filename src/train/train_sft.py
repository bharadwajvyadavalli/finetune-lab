"""SFT training — handles both QLoRA and LoRA(bf16) via config.

Usage:
    python -m src.train.train_sft --config configs/qlora_sft.yaml
    python -m src.train.train_sft --config configs/lora_sft.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_quant_config(cfg: dict) -> BitsAndBytesConfig | None:
    q = cfg.get("quantization") or {}
    if not q.get("enabled"):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=getattr(torch, q.get("bnb_4bit_compute_dtype", "bfloat16")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    base_model_id = cfg["base_model"]
    dataset_dir = Path(cfg["dataset_dir"])
    output_dir = Path(cfg["output_dir"])
    hub_repo = cfg.get("hub_repo") or None

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        # Mistral has no pad token by default — reuse EOS.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # critical for causal LM training

    # ---- Base model (optionally 4-bit) ----
    quant_config = build_quant_config(cfg)
    # For multi-GPU: place model on correct device per process
    device_map = {"": PartialState().local_process_index}
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if cfg["training"].get("bf16") else None,
        device_map=device_map,
    )
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # required for gradient checkpointing
    model.config.pretraining_tp = 1

    # ---- LoRA ----
    lora_cfg_dict = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg_dict["r"],
        lora_alpha=lora_cfg_dict["alpha"],
        lora_dropout=lora_cfg_dict.get("dropout", 0.05),
        target_modules=lora_cfg_dict["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- Data ----
    dataset_file = cfg.get("dataset_file", "train.jsonl")
    data_files = {"train": str(dataset_dir / dataset_file)}
    train_ds = load_dataset("json", data_files=data_files)["train"]
    print(f"Training on {len(train_ds)} examples")

    # ---- Training args (TRL 0.9.x uses TrainingArguments) ----
    t = cfg["training"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        bf16=t.get("bf16", True),
        logging_steps=t.get("logging_steps", 25),
        save_steps=t.get("save_steps", 500),
        save_total_limit=t.get("save_total_limit", 2),
        seed=t.get("seed", 42),
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=bool(hub_repo),
        hub_model_id=hub_repo,
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        peft_config=peft_config,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=t.get("max_seq_length", 1024),
        packing=False,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {output_dir}")
    trainer.save_model(str(output_dir))

    if hub_repo:
        print(f"Pushing adapter to hub: {hub_repo}")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
