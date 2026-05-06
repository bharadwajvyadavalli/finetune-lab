"""Eval harness — runs base / SFT / LoRA / DPO over the held-out set and
reports SQL exact-match accuracy plus a side-by-side comparison table.

Usage:
    python -m src.eval.eval_sql \\
        --base_model mistralai/Mistral-7B-v0.1 \\
        --eval_file data/spider_processed/holdout.jsonl \\
        --variants base=NONE qlora=outputs/qlora_sft lora=outputs/lora_sft dpo=outputs/dpo_lora \\
        --out_csv eval_results.csv

Each variant arg is `name=path_or_hub_repo`. Use `NONE` for the base model with no adapter.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.prompts import format_prompt, extract_sql


def normalize_sql(sql: str) -> str:
    s = sql.strip().rstrip(";").lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_variant(base_model_id: str, adapter: str | None):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter and adapter != "NONE":
        model = PeftModel.from_pretrained(base, adapter)
    else:
        model = base
    model.eval()

    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def generate_sql(model, tok, schema: str, question: str) -> str:
    prompt = format_prompt(schema, question)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return extract_sql(tok.decode(out[0], skip_special_tokens=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--variants", nargs="+", required=True,
                        help="name=path entries, e.g. base=NONE qlora=outputs/qlora_sft")
    parser.add_argument("--out_csv", default="eval_results.csv")
    args = parser.parse_args()

    eval_rows = [json.loads(l) for l in open(args.eval_file)]
    print(f"Loaded {len(eval_rows)} eval rows")

    variants = []
    for v in args.variants:
        name, _, path = v.partition("=")
        variants.append((name, path or "NONE"))

    # results[i] = {variant_name: predicted_sql}
    results: list[dict] = [{"question": r["question"], "gold": r["sql"]} for r in eval_rows]
    accuracies: dict[str, float] = {}

    for name, adapter in variants:
        print(f"\n=== Variant: {name}  ({adapter}) ===")
        model, tok = load_variant(args.base_model, adapter)

        correct = 0
        for i, row in enumerate(eval_rows):
            pred = generate_sql(model, tok, row["schema"], row["question"])
            results[i][name] = pred
            if normalize_sql(pred) == normalize_sql(row["sql"]):
                correct += 1
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(eval_rows)}] running acc: {correct/(i+1):.3f}")

        acc = correct / len(eval_rows)
        accuracies[name] = acc
        print(f"  {name} exact-match accuracy: {acc:.3f}")

        # Free VRAM before loading the next variant.
        del model
        torch.cuda.empty_cache()

    # ---- Summary ----
    print("\n=== Final accuracies ===")
    for name, acc in accuracies.items():
        print(f"  {name:10s}  {acc:.3f}")

    # ---- CSV output ----
    fieldnames = ["question", "gold"] + [name for name, _ in variants]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)
    print(f"\nWrote per-row predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
