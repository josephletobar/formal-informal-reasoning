from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import run_gemma9_mechanistic_batch as base


def continuation_id(tokenizer, prompt, answer):
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    full_ids = tokenizer(prompt + answer, add_special_tokens=True).input_ids
    if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        # This fallback handles tokenizers whose boundary merge changes the prefix.
        ids = tokenizer(answer, add_special_tokens=False).input_ids
        return int(ids[0])
    return int(full_ids[len(prompt_ids)])


def foil_answer(tokenizer, prompt, gold):
    numeric = False
    try:
        int(gold)
        numeric = True
    except ValueError:
        pass
    candidates = [str(i) for i in range(200)] if numeric else ["false", "true", "P", "Q", "unknown", "blue", "green", "granted"]
    gold_id = continuation_id(tokenizer, prompt, gold)
    for candidate in candidates:
        if candidate != gold and continuation_id(tokenizer, prompt, candidate) != gold_id:
            return candidate
    raise ValueError((prompt, gold))


def behavior(model, tokenizer, rows):
    records = []
    for kind in ("direct", "explicit", "applied", "implicit", "surface_control", "answer_control", "unrelated"):
        batch = []
        for row in rows:
            if kind == "surface_control":
                gold = row["surface_gold"]
            elif kind == "unrelated":
                gold = row["unrelated_gold"]
            else:
                gold = row["gold"]
            prompt = row[f"{kind}_prompt"]
            foil = foil_answer(tokenizer, prompt, gold)
            batch.append({"family": row["family"], "example": row["example"], "kind": kind, "prompt": prompt,
                          "gold": gold, "foil": foil, "gold_id": continuation_id(tokenizer, prompt, gold),
                          "foil_id": continuation_id(tokenizer, prompt, foil)})
        records.extend(base.batched_behavior(model, tokenizer, batch))
        print(f"corrected behavior {kind} complete ({len(batch)} rows)", flush=True)
    return records


def corrected_causal(model, tokenizer, rows, layer_ids):
    out = []
    selected = [r for r in rows if r["example"] < base.CAUSAL_N]
    for target in selected:
        target_prompt = target["applied_prompt"]
        target_gold = target["gold"]
        target_foil = foil_answer(tokenizer, target_prompt, target_gold)
        target_gold_id = continuation_id(tokenizer, target_prompt, target_gold)
        target_foil_id = continuation_id(tokenizer, target_prompt, target_foil)
        target_states, target_logits = base.layer_capture(model, tokenizer, target_prompt, layer_ids)
        target_margin = float((target_logits[target_gold_id] - target_logits[target_foil_id]).item())
        sources = [("same_computation", target["direct_prompt"], target["gold"]),
                   ("surface_wrong", target["surface_control_prompt"], target["surface_gold"]),
                   ("unrelated", target["unrelated_prompt"], target["unrelated_gold"])]
        for source_kind, source_prompt, source_gold in sources:
            source_foil = foil_answer(tokenizer, source_prompt, source_gold)
            source_gold_id = continuation_id(tokenizer, source_prompt, source_gold)
            source_foil_id = continuation_id(tokenizer, source_prompt, source_foil)
            source_states, source_logits = base.layer_capture(model, tokenizer, source_prompt, layer_ids)
            source_margin = float((source_logits[source_gold_id] - source_logits[source_foil_id]).item())
            for layer in layer_ids:
                patched = base.patch_layer(model, tokenizer, target_prompt, layer, source_states[layer])
                patched_margin = float((patched[target_gold_id] - patched[target_foil_id]).item())
                out.append({"family": target["family"], "example": target["example"], "source_kind": source_kind,
                    "layer": layer, "target_margin": target_margin, "source_margin": source_margin,
                    "patched_margin": patched_margin, "patch_effect": patched_margin - target_margin,
                    "normalized_rescue": (patched_margin - target_margin) / (abs(source_margin - target_margin) + 1e-6)})
    return out


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
    cache_name = "models--" + base.MODEL_NAME.replace("/", "--")
    snapshot_root = base.HF_HOME / "hub" / cache_name / "snapshots"
    snapshot = max((p for p in snapshot_root.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    rows = base.make_rows(tokenizer, base.BEHAVIOR_N)
    behavior_rows = behavior(model, tokenizer, rows)
    write_union(out / "behavior_matrix_corrected.csv", behavior_rows)
    num_layers = int(model.config.num_hidden_layers)
    layer_ids = sorted(set([0, num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers - 1]))
    causal_rows = corrected_causal(model, tokenizer, rows, layer_ids)
    write_union(out / "residual_patching_corrected.csv", causal_rows)
    # Reuse the component hook machinery with prompt-conditioned answer IDs.
    for row in rows:
        prompt = row["applied_prompt"]
        row["gold_id"] = continuation_id(tokenizer, prompt, row["gold"])
        row["foil_id"] = continuation_id(tokenizer, prompt, foil_answer(tokenizer, prompt, row["gold"]))
    component_layers = [max(0, num_layers - 3), max(0, num_layers - 2)]
    component_rows = base.component_ablation(model, tokenizer, rows, component_layers)
    write_union(out / "component_ablation_corrected.csv", component_rows)
    meta = {"model": base.MODEL_NAME, "objective": "prompt-conditioned first continuation token", "behavior_rows": len(behavior_rows), "causal_rows": len(causal_rows), "component_rows": len(component_rows), "runtime_seconds": round(time.time() - started, 2), "status": "completed"}
    (out / "corrected_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"corrected objective complete behavior={len(behavior_rows)} causal={len(causal_rows)} component={len(component_rows)}", flush=True)


if __name__ == "__main__":
    main()
