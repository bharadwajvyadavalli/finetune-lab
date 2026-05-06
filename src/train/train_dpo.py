"""DPO training, starting from the Phase 1 SFT adapter.

Usage:
    python -m src.train.train_dpo --config configs/dpo.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer


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
    sft_adapter = cfg["sft_adapter"]
    if not sft_adapter:
        raise ValueError("dpo.yaml: 'sft_adapter' must point to your Phase 1 adapter (local or hub)")
    dataset_dir = Path(cfg["dpo_dataset_dir"])
    output_dir = Path(cfg["output_dir"])
    hub_repo = cfg.get("hub_repo") or None

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Base + Phase 1 LoRA as starting policy ----
    quant_config = build_quant_config(cfg)
    # For multi-GPU: place model on correct device per process
    device_map = {"": PartialState().local_process_index}
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if cfg["training"].get("bf16") else None,
        device_map=device_map,
    )
    if quant_config is not None:
        base = prepare_model_for_kbit_training(base)

    model = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True)
    model.config.use_cache = False

    # New LoRA config — DPO will train fresh adapter weights on top of the policy.
    # (When `peft_config` is set, DPOTrainer re-wraps the model with it.)
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
    train_ds = load_dataset(
        "json", data_files={"train": str(dataset_dir / "train.jsonl")}
    )["train"]
    print(f"DPO training on {len(train_ds)} pairs")

    # ---- Training args (TRL 0.9.x DPOConfig) ----
    t = cfg["training"]
    d = cfg["dpo"]
    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.05),
        bf16=t.get("bf16", True),
        logging_steps=t.get("logging_steps", 10),
        save_steps=t.get("save_steps", 200),
        save_total_limit=t.get("save_total_limit", 1),
        seed=t.get("seed", 42),
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=bool(hub_repo),
        hub_model_id=hub_repo,
        remove_unused_columns=False,
        beta=d.get("beta", 0.1),
        loss_type=d.get("loss_type", "sigmoid"),
        max_length=t.get("max_length", 512),
        max_prompt_length=t.get("max_prompt_length", 384),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT trick: ref = base policy with adapter disabled
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving DPO adapter to {output_dir}")
    trainer.save_model(str(output_dir))

    if hub_repo:
        print(f"Pushing DPO adapter to hub: {hub_repo}")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
