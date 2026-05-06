"""Prompt templates — the single source of truth.

Every script that touches the model (SFT training, DPO training, DPO synthesis,
eval, Gradio app) imports from this module. If train/inference prompts diverge,
the model produces garbage silently — this module exists to prevent that.
"""

SYSTEM_INSTRUCTION = (
    "You are a SQL expert. Given a database schema and a natural-language "
    "question, write a SQL query that answers the question. "
    "Return only the SQL query, no explanation."
)


def format_prompt(schema: str, question: str) -> str:
    """The prompt seen by the model at inference time.

    Generation continues from the trailing `### SQL:\\n`.
    """
    return (
        f"### Instruction:\n{SYSTEM_INSTRUCTION}\n\n"
        f"### Schema:\n{schema.strip()}\n\n"
        f"### Question:\n{question.strip()}\n\n"
        f"### SQL:\n"
    )


def format_full(schema: str, question: str, sql: str) -> str:
    """The full string seen at training time: prompt + target SQL + EOS."""
    # Note: the trainer appends EOS via the tokenizer; we just append it textually
    # here as a marker. SFTTrainer + DataCollator handle EOS tokenization.
    return format_prompt(schema, question) + sql.strip()


def extract_sql(generated_text: str) -> str:
    """Pull just the SQL out of model output. Strips anything after a newline
    boundary if the model continues past the SQL (which can happen pre-DPO)."""
    # If the prompt template was echoed back, take only the part after `### SQL:`.
    marker = "### SQL:\n"
    if marker in generated_text:
        generated_text = generated_text.split(marker, 1)[1]
    # Stop at the first `###` heading or double newline (model rambling).
    for stop in ("\n###", "\n\n"):
        if stop in generated_text:
            generated_text = generated_text.split(stop, 1)[0]
            break
    return generated_text.strip()
