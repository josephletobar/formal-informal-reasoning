from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import run_qwen3_mechanistic_batch as base


def write_union(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    out = base.OUT
    snapshot_root = base.HF_HOME / "hub" / "models--Qwen--Qwen3-8B" / "snapshots"
    snapshot = max((p for p in snapshot_root.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    rows = base.make_rows(tokenizer, base.BEHAVIOR_N)
    num_layers = int(model.config.num_hidden_layers)
    component_layers = [max(0, num_layers - 3), max(0, num_layers - 2)]
    component_rows = base.component_ablation(model, tokenizer, rows, component_layers)
    write_union(out / "component_ablation.csv", component_rows)
    metadata = {"model": base.MODEL_NAME, "component_rows": len(component_rows), "component_layers": component_layers, "runtime_seconds": round(time.time() - started, 2), "status": "completed"}
    (out / "component_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"component continuation complete ({len(component_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
