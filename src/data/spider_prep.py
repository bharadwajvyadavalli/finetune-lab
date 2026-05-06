"""Spider data prep — load, format, and split.

Spider's HF representation has shifted across versions, so this script tries the
common column shapes and falls back with a clear error. The smoke test (Task 10)
runs this script first and lets you eyeball formatted examples before any GPU spend.

Output layout under --out_dir:
    train.jsonl       # main training set (Spider train minus holdout)
    holdout.jsonl     # held-out rows for eval (sampled from train)
    dev.jsonl         # Spider's dev split (used for final eval)

Each row in train.jsonl / dev.jsonl / holdout.jsonl has the shape:
    {"schema": "<schema text>", "question": "...", "sql": "...", "text": "<formatted full prompt+SQL>"}
The "text" field is what SFTTrainer consumes directly.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

from src.prompts import format_full


SCHEMA_KEYS = ("schema", "context", "db_schema", "create_table_statements")
QUESTION_KEYS = ("question", "input", "natural_language_query")
SQL_KEYS = ("query", "sql", "answer", "output")


def _first_present(row: dict, keys: Iterable[str]) -> str | None:
    for k in keys:
        if k in row and row[k]:
            v = row[k]
            return v if isinstance(v, str) else str(v)
    return None


def _build_schema_from_db_metadata(row: dict) -> str | None:
    """Spider's canonical version stores schema as a separate `db_id` -> tables.json
    mapping. Some HF mirrors flatten this into the row as `tables` / `column_names`.
    This is a best-effort reconstructor; if it can't find what it needs, returns None
    and the caller falls back / errors with a clear message."""
    if "db_id" in row and "tables" in row and "column_names" in row:
        # Flattened variant — reconstruct CREATE TABLE-ish text.
        tables = row["tables"]  # list of table names
        columns = row["column_names"]  # list of [table_idx, col_name]
        col_types = row.get("column_types", ["text"] * len(columns))
        lines = []
        for t_idx, t_name in enumerate(tables):
            cols = [
                f"  {c_name} {col_types[i] if i < len(col_types) else 'text'}"
                for i, (ti, c_name) in enumerate(columns)
                if ti == t_idx and c_name != "*"
            ]
            if cols:
                lines.append(f"CREATE TABLE {t_name} (\n" + ",\n".join(cols) + "\n);")
        return "\n\n".join(lines) if lines else None
    return None


def _row_to_record(row: dict) -> dict | None:
    schema = _first_present(row, SCHEMA_KEYS) or _build_schema_from_db_metadata(row)
    question = _first_present(row, QUESTION_KEYS)
    sql = _first_present(row, SQL_KEYS)
    if not (schema and question and sql):
        return None
    text = format_full(schema, question, sql)
    return {"schema": schema, "question": question, "sql": sql, "text": text}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="b-mc2/sql-create-context",
                        help="HF dataset id. Default has schema inlined in each row "
                             "(78k rows). Pass 'xlangai/spider' to use raw Spider, "
                             "but you'll need to reconstruct schemas from tables.json.")
    parser.add_argument("--out_dir", default="data/spider_processed")
    parser.add_argument("--holdout_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="Print 3 formatted examples and exit (no files written).")
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    ds = load_dataset(args.dataset)

    # Some datasets only have a 'train' split.
    splits = {s: ds[s] for s in ds.keys()}
    print(f"Splits found: {list(splits.keys())}")

    # Convert each split.
    converted: dict[str, list[dict]] = {}
    for name, split in splits.items():
        rows = []
        skipped = 0
        for row in split:
            rec = _row_to_record(row)
            if rec is None:
                skipped += 1
                continue
            rows.append(rec)
        converted[name] = rows
        print(f"  {name}: kept {len(rows)}, skipped {skipped}")

    if not any(converted.values()):
        raise RuntimeError(
            f"No rows could be formatted from {args.dataset}. "
            f"Inspect a row's keys and update SCHEMA_KEYS/QUESTION_KEYS/SQL_KEYS, "
            f"or pass --dataset b-mc2/sql-create-context (already in instruction shape)."
        )

    if args.smoke:
        sample_split = next(iter(converted.values()))
        for i, rec in enumerate(sample_split[:3]):
            print(f"\n----- Example {i+1} -----")
            print(rec["text"])
        return

    out_dir = Path(args.out_dir)

    train = converted.get("train") or converted[next(iter(converted))]
    rng = random.Random(args.seed)
    rng.shuffle(train)
    holdout = train[: args.holdout_size]
    train_main = train[args.holdout_size :]
    _write_jsonl(out_dir / "train.jsonl", train_main)
    _write_jsonl(out_dir / "holdout.jsonl", holdout)
    print(f"Wrote {len(train_main)} train rows + {len(holdout)} holdout rows")

    if "validation" in converted:
        _write_jsonl(out_dir / "dev.jsonl", converted["validation"])
        print(f"Wrote {len(converted['validation'])} dev rows")
    elif "dev" in converted:
        _write_jsonl(out_dir / "dev.jsonl", converted["dev"])
        print(f"Wrote {len(converted['dev'])} dev rows")


if __name__ == "__main__":
    main()
