"""Run a small independent behavioral screen on the fixed benchmark.

This is intentionally separate from the Gemma sparse-circuit analysis. It is
an inexpensive cross-model behavioral check, not a replacement for the
preregistered mechanistic replication.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def model_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--benchmark", type=Path, default=Path("publication_package/benchmark_v1/benchmark_v1.csv"))
    parser.add_argument("--out", type=Path, default=Path("behavioral_screen"))
    args = parser.parse_args()

    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.benchmark.open(encoding="utf-8", newline="")))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()

    output = []
    started = time.time()
    for index, row in enumerate(rows):
        prompt = model_prompt(tokenizer, row["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=4,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        gold = str(row["gold"])
        output.append({
            **row,
            "model": args.model,
            "generated": continuation,
            "exact_first_answer": int(continuation == gold or continuation.startswith(gold)),
        })
        if (index + 1) % 25 == 0:
            print(f"completed {index + 1}/{len(rows)}")

    with (args.out / "behavioral_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    direct = [r for r in output if r["kind"] == "direct"]
    summary = {
        "model": args.model,
        "rows": len(output),
        "direct_rows": len(direct),
        "direct_accuracy": sum(r["exact_first_answer"] for r in direct) / len(direct),
        "accuracy_by_family": {},
        "accuracy_by_form": {},
        "runtime_seconds": time.time() - started,
        "interpretation": "Independent behavioral screen only; no mechanistic or module claim.",
    }
    for key in ["family", "form", "split"]:
        groups = {}
        for row in direct:
            groups.setdefault(row[key], []).append(row["exact_first_answer"])
        summary[f"accuracy_by_{key}"] = {name: sum(values) / len(values) for name, values in groups.items()}
    (args.out / "behavioral_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
