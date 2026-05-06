"""Build DPO (chosen, rejected) pairs from the Phase 1 SFT adapter.

Strategy: run the Phase 1 model over Spider train rows. For each row where the
model's SQL doesn't match the gold SQL (under loose normalization), save a pair:
    chosen   = gold SQL
    rejected = model's incorrect SQL

This gives us task-aligned preferences for free — much better than generic
chat preferences for our use case.

Usage:
    python -m src.dpo_synth.build_pairs \\
        --base_model mistralai/Mistral-7B-v0.1 \\
        --sft_adapter outputs/qlora_sft \\
        --train_file data/spider_processed/train.jsonl \\
        --out_file data/dpo_pairs/train.jsonl \\
        --max_pairs 5000
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.prompts import format_prompt, extract_sql


def normalize_sql(sql: str) -> str:
    """Loose normalization for comparing SQL strings.
    Not a real SQL parser — good enough to skip trivially-equivalent answers."""
    s = sql.strip().rstrip(";").lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_model(base_model_id: str, sft_adapter: str):
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
    model = PeftModel.from_pretrained(base, sft_adapter)
    model.eval()

    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def generate_sql(model, tok, schema: str, question: str, max_new_tokens: int = 200) -> str:
    prompt = format_prompt(schema, question)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # deterministic for repeatable pair generation
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    return extract_sql(full)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--sft_adapter", required=True,
                        help="Phase 1 adapter — local path or HF hub repo")
    parser.add_argument("--train_file", default="data/spider_processed/train.jsonl")
    parser.add_argument("--out_file", default="data/dpo_pairs/train.jsonl")
    parser.add_argument("--max_pairs", type=int, default=5000)
    parser.add_argument("--max_inputs", type=int, default=8000,
                        help="Cap on rows to attempt before stopping (for time control).")
    args = parser.parse_args()

    print(f"Loading {args.base_model} + adapter {args.sft_adapter}")
    model, tok = load_model(args.base_model, args.sft_adapter)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs_written = 0
    skipped_match = 0

    # Read all lines up to max_inputs
    with open(args.train_file) as fin:
        lines = [line for i, line in zip(range(args.max_inputs), fin)]

    with out_path.open("w") as fout:
        pbar = tqdm(lines, desc="Generating pairs")
        for line in pbar:
            if pairs_written >= args.max_pairs:
                break
            row = json.loads(line)
            schema, question, gold = row["schema"], row["question"], row["sql"]

            try:
                generated = generate_sql(model, tok, schema, question)
            except Exception as e:
                pbar.write(f"generation failed: {e}")
                continue

            if normalize_sql(generated) == normalize_sql(gold):
                skipped_match += 1
                pbar.set_postfix(pairs=pairs_written, matched=skipped_match)
                continue
            if not generated.strip():
                continue

            pair = {
                "prompt": format_prompt(schema, question),
                "chosen": gold,
                "rejected": generated,
            }
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
            pairs_written += 1
            pbar.set_postfix(pairs=pairs_written, matched=skipped_match)

    print(f"\nDone. written={pairs_written} matched={skipped_match}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
