"""Gradio app for HuggingFace Spaces.

Loads Mistral-7B + a chosen LoRA adapter on demand and serves a Text-to-SQL UI.
The model dropdown is the centerpiece: visitors can A/B base / QLoRA / LoRA / DPO
on the same prompt and see what each technique does.

Layout note: this file lives in `app/` so it can be deployed to a HF Space as-is.
The companion `app/requirements.txt` specifies the inference deps.
"""
from __future__ import annotations

import os

import gradio as gr
import spaces
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL = "mistralai/Mistral-7B-v0.1"

ADAPTERS: dict[str, str | None] = {
    "Base (no fine-tuning)": None,
    "QLoRA SFT": "bharadwajvyadavalli/mistral-sql-qlora",
    "LoRA SFT (bf16)": "bharadwajvyadavalli/mistral-sql-lora",
    "DPO (on top of QLoRA)": "bharadwajvyadavalli/mistral-sql-dpo",
}

SAMPLE_SCHEMA = """CREATE TABLE customer (
  id INT PRIMARY KEY,
  name TEXT,
  signup_date DATE,
  country TEXT
);

CREATE TABLE orders (
  id INT PRIMARY KEY,
  customer_id INT,
  amount DECIMAL,
  created_at TIMESTAMP
);"""

SAMPLE_QUESTIONS = [
    "How many customers signed up from Germany?",
    "What are the top 5 customers by total order amount?",
    "Show monthly order revenue for 2025.",
]

# ---------------------------------------------------------------------------
# Prompt formatting (mirrors src/prompts.py — kept inline so the Space is
# self-contained and doesn't need to import the rest of the repo).
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are a SQL expert. Given a database schema and a natural-language "
    "question, write a SQL query that answers the question. "
    "Return only the SQL query, no explanation."
)


def format_prompt(schema: str, question: str) -> str:
    return (
        f"### Instruction:\n{SYSTEM_INSTRUCTION}\n\n"
        f"### Schema:\n{schema.strip()}\n\n"
        f"### Question:\n{question.strip()}\n\n"
        f"### SQL:\n"
    )


def extract_sql(generated: str) -> str:
    marker = "### SQL:\n"
    if marker in generated:
        generated = generated.split(marker, 1)[1]
    for stop in ("\n###", "\n\n"):
        if stop in generated:
            generated = generated.split(stop, 1)[0]
            break
    return generated.strip()


# ---------------------------------------------------------------------------
# Lazy model loading — keep one variant resident at a time to fit in CPU/small
# GPU. Switching variants reloads.
# ---------------------------------------------------------------------------
_state: dict = {"variant": None, "model": None, "tokenizer": None}


def _load_variant(variant_name: str):
    adapter = ADAPTERS.get(variant_name)
    print(f"Loading variant '{variant_name}' (adapter={adapter or 'NONE'})")

    # Use 4-bit quantization for memory efficiency
    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        # CPU fallback (slow but works)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    model = PeftModel.from_pretrained(base, adapter) if adapter else base
    model.eval()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _ensure_loaded(variant_name: str):
    if _state["variant"] != variant_name:
        # Clear previous to free memory.
        _state["model"] = None
        _state["tokenizer"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, tok = _load_variant(variant_name)
        _state.update({"variant": variant_name, "model": model, "tokenizer": tok})


@spaces.GPU
def run_inference(variant_name: str, schema: str, question: str) -> str:
    if not schema.strip() or not question.strip():
        return "Please fill in both schema and question."
    _ensure_loaded(variant_name)
    model, tok = _state["model"], _state["tokenizer"]

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


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Mistral Text-to-SQL — Fine-tuning Sprint") as demo:
    gr.Markdown(
        "# Mistral-7B Text-to-SQL — Fine-tuning Sprint\n"
        "Compare the same base model fine-tuned three different ways "
        "(QLoRA SFT, LoRA SFT bf16, DPO) on the Spider dataset. "
        "Pick a variant, paste a schema, ask a question."
    )
    with gr.Row():
        with gr.Column():
            variant = gr.Dropdown(
                choices=[k for k, v in ADAPTERS.items() if v is not None or k.startswith("Base")],
                value="Base (no fine-tuning)",
                label="Model variant",
            )
            schema_box = gr.Textbox(
                label="Schema (CREATE TABLE statements)",
                value=SAMPLE_SCHEMA,
                lines=10,
            )
            question_box = gr.Textbox(
                label="Question",
                value=SAMPLE_QUESTIONS[0],
                lines=2,
            )
            with gr.Row():
                for q in SAMPLE_QUESTIONS:
                    gr.Button(q, size="sm").click(lambda v=q: v, outputs=question_box)
            submit = gr.Button("Generate SQL", variant="primary")
        with gr.Column():
            sql_out = gr.Code(label="Generated SQL", language="sql")

    submit.click(
        run_inference,
        inputs=[variant, schema_box, question_box],
        outputs=sql_out,
    )


if __name__ == "__main__":
    demo.launch()
