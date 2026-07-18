from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("QWEN3_MECH_OUT", "results/research_v13_qwen3_mechanistic"))
HF_HOME = Path(os.environ.get("HF_HOME", "huggingface"))
MODEL_NAME = os.environ.get("MECH_MODEL", "Qwen/Qwen3-8B")
BEHAVIOR_N = 100
REP_N = 24
CAUSAL_N = 6
BATCH = int(os.environ.get("MECH_BATCH", "4"))


def first_id(tokenizer, answer: str) -> int:
    ids = tokenizer(answer, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(answer)
    return int(ids[0])


def foil_id(tokenizer, answer: str) -> int:
    gold = first_id(tokenizer, answer)
    candidates = [str(i) for i in range(0, 200)] + ["true", "false", "blue", "green"]
    for candidate in candidates:
        fid = first_id(tokenizer, candidate)
        if fid != gold:
            return fid
    raise ValueError(answer)


def family_specs():
    specs = {}
    specs["addition"] = {
        "make": lambda i: (i % 50, (i * 7 + 3) % 50),
        "answer": lambda a, b: str(a + b),
        "direct": lambda a, b: f"calc: {a}+{b}=",
        "explicit": lambda a, b: f"Add {a} and {b}. Answer:",
        "applied": lambda a, b: f"A basket has {a} apples and receives {b} more. Total apples:",
        "implicit": lambda a, b: f"There are {a} red tiles and {b} blue tiles. Total tiles:",
        "surface_control": lambda a, b: f"calc: {a}*{b}=",
        "surface_answer": lambda a, b: str(a * b),
        "answer_control": lambda a, b: f"calc: {a + b - 1}+1=",
        "unrelated": lambda a, b: f"A rectangle has side lengths {a} and {b}. Area:",
        "unrelated_answer": lambda a, b: str(a * b),
    }
    specs["subtraction"] = {
        "make": lambda i: (50 + (i % 40), 1 + ((i * 7 + 2) % 40)),
        "answer": lambda a, b: str(a - b),
        "direct": lambda a, b: f"calc: {a}-{b}=",
        "explicit": lambda a, b: f"Subtract {b} from {a}. Answer:",
        "applied": lambda a, b: f"A box has {a} items and {b} are removed. Items left:",
        "implicit": lambda a, b: f"There are {a} birds; {b} fly away. Birds remaining:",
        "surface_control": lambda a, b: f"calc: {a}+{b}=",
        "surface_answer": lambda a, b: str(a + b),
        "answer_control": lambda a, b: f"calc: {a - b}+0=",
        "unrelated": lambda a, b: f"A rectangle has side lengths {a} and {b}. Area:",
        "unrelated_answer": lambda a, b: str(a * b),
    }
    specs["multiplication"] = {
        "make": lambda i: (2 + (i % 8), 2 + ((i * 3 + 1) % 8)),
        "answer": lambda a, b: str(a * b),
        "direct": lambda a, b: f"calc: {a}*{b}=",
        "explicit": lambda a, b: f"Multiply {a} by {b}. Answer:",
        "applied": lambda a, b: f"There are {a} rows with {b} seats each. Total seats:",
        "implicit": lambda a, b: f"Make {a} groups, placing {b} coins in each. Coins total:",
        "surface_control": lambda a, b: f"calc: {a}+{b}=",
        "surface_answer": lambda a, b: str(a + b),
        "answer_control": lambda a, b: f"calc: {a * b - 1}+1=",
        "unrelated": lambda a, b: f"A rectangle has side lengths {a} and {b}. Area:",
        "unrelated_answer": lambda a, b: str(a * b),
    }
    specs["modular"] = {
        "make": lambda i: (i % 20, (i * 5 + 1) % 20),
        "answer": lambda a, b: str((a + b) % 10),
        "direct": lambda a, b: f"calc: ({a}+{b}) mod 10 =",
        "explicit": lambda a, b: f"Add {a} and {b}, then wrap the result around modulo 10. Answer:",
        "applied": lambda a, b: f"A clock counter is at {a} and advances {b} steps on a 10-position dial. Final position:",
        "implicit": lambda a, b: f"On a dial labeled 0 through 9, start at {a} and move {b} spaces. Position:",
        "surface_control": lambda a, b: f"calc: {a}+{b}=",
        "surface_answer": lambda a, b: str(a + b),
        "answer_control": lambda a, b: f"calc: ({a + b}) mod 10 =",
        "unrelated": lambda a, b: f"A rectangle has side lengths {a} and {b}. Area:",
        "unrelated_answer": lambda a, b: str(a * b),
    }
    specs["pythagorean"] = {
        "triples": [(3, 4, 5), (5, 12, 13), (6, 8, 10), (8, 15, 17), (9, 12, 15), (12, 16, 20), (7, 24, 25), (20, 21, 29)],
        "make": lambda i: (i % 8, 0),
        "answer": lambda a, b: str(specs["pythagorean"]["triples"][a][2]),
        "direct": lambda a, b: f"calc: sqrt({specs['pythagorean']['triples'][a][0]}^2+{specs['pythagorean']['triples'][a][1]}^2)=",
        "explicit": lambda a, b: f"A right triangle has legs {specs['pythagorean']['triples'][a][0]} and {specs['pythagorean']['triples'][a][1]}. Hypotenuse:",
        "applied": lambda a, b: f"A robot moves {specs['pythagorean']['triples'][a][0]} east and {specs['pythagorean']['triples'][a][1]} north. Straight-line distance:",
        "implicit": lambda a, b: f"Points (0,0) and ({specs['pythagorean']['triples'][a][0]},{specs['pythagorean']['triples'][a][1]}). Distance:",
        "surface_control": lambda a, b: f"calc: {specs['pythagorean']['triples'][a][0]}+{specs['pythagorean']['triples'][a][1]}=",
        "surface_answer": lambda a, b: str(specs["pythagorean"]["triples"][a][0] + specs["pythagorean"]["triples"][a][1]),
        "answer_control": lambda a, b: f"calc: {specs['pythagorean']['triples'][a][2]}+0=",
        "unrelated": lambda a, b: f"A rectangle has side lengths {specs['pythagorean']['triples'][a][0]} and {specs['pythagorean']['triples'][a][1]}. Area:",
        "unrelated_answer": lambda a, b: str(specs["pythagorean"]["triples"][a][0] * specs["pythagorean"]["triples"][a][1]),
    }
    specs["implication"] = {
        "make": lambda i: (i % 4, (i // 4) % 4),
        "answer": lambda a, b: "blue",
        "direct": lambda a, b: "Rule: If P then Q. P. Therefore:",
        "explicit": lambda a, b: "If a badge is valid, access is granted. The badge is valid. Access:",
        "applied": lambda a, b: "If the sensor is active, the alarm sounds. The sensor is active. Alarm:",
        "implicit": lambda a, b: "Rule table: active -> on. Input: active. Output:",
        "surface_control": lambda a, b: "Rule: If P then Q. Q. Therefore:",
        "surface_answer": lambda a, b: "P",
        "answer_control": lambda a, b: "Rule: If P then Q. P and R. Therefore:",
        "unrelated": lambda a, b: "Rule: If birds have wings then they can fly. A stone is present. Therefore:",
        "unrelated_answer": lambda a, b: "unknown",
    }
    return specs


def make_rows(tokenizer, n):
    rows = []
    for family, spec in family_specs().items():
        for i in range(n):
            a, b = spec["make"](i)
            answer = spec["answer"](a, b)
            row = {"family": family, "example": i, "a": a, "b": b, "gold": answer}
            for kind in ("direct", "explicit", "applied", "implicit", "surface_control", "answer_control", "unrelated"):
                row[f"{kind}_prompt"] = spec[kind](a, b)
            row["surface_gold"] = spec["surface_answer"](a, b)
            row["answer_gold"] = answer
            row["unrelated_gold"] = spec["unrelated_answer"](a, b)
            row["gold_id"] = first_id(tokenizer, answer)
            row["foil_id"] = foil_id(tokenizer, answer)
            rows.append(row)
    return rows


def get_device(model):
    return next(model.parameters()).device


def batched_behavior(model, tokenizer, records):
    import torch
    out = []
    for start in range(0, len(records), BATCH):
        chunk = records[start:start + BATCH]
        enc = tokenizer([x["prompt"] for x in chunk], return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(get_device(model)) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc, use_cache=False).logits
        positions = enc["attention_mask"].sum(dim=1) - 1
        for j, item in enumerate(chunk):
            vec = logits[j, positions[j]].float()
            gold = item["gold_id"]
            foil = item["foil_id"]
            item = dict(item)
            item["pred_id"] = int(vec.argmax().item())
            item["correct"] = int(item["pred_id"] == gold)
            item["margin"] = float((vec[gold] - vec[foil]).item())
            out.append(item)
    return out


def hidden_batch(model, tokenizer, prompts, layer_ids):
    import torch
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(get_device(model)) for k, v in enc.items()}
    with torch.inference_mode():
        result = model(**enc, output_hidden_states=True, use_cache=False)
    positions = enc["attention_mask"].sum(dim=1) - 1
    vectors = []
    for layer in layer_ids:
        h = result.hidden_states[layer]
        vectors.append(torch.stack([h[j, positions[j]].float().cpu() for j in range(len(prompts))]))
    del result, enc
    return vectors


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def run_representation(model, tokenizer, rows, layer_ids):
    rep = []
    selected = [r for r in rows if r["example"] < REP_N]
    for family in family_specs():
        for r in [x for x in selected if x["family"] == family]:
            kinds = ["direct", "explicit", "applied", "implicit", "surface_control", "answer_control", "unrelated"]
            prompts = [r[f"{k}_prompt"] for k in kinds]
            hs = hidden_batch(model, tokenizer, prompts, layer_ids)
            for li, layer in enumerate(layer_ids):
                arr = {k: hs[li][j].numpy() for j, k in enumerate(kinds)}
                rep.append({"family": family, "example": r["example"], "layer": layer,
                    "same_explicit": cosine(arr["direct"], arr["explicit"]),
                    "same_applied": cosine(arr["direct"], arr["applied"]),
                    "same_implicit": cosine(arr["direct"], arr["implicit"]),
                    "surface_control": cosine(arr["direct"], arr["surface_control"]),
                    "answer_control": cosine(arr["direct"], arr["answer_control"]),
                    "unrelated": cosine(arr["direct"], arr["unrelated"])})
    return rep


def layer_capture(model, tokenizer, prompt, layer_ids):
    import torch
    layers = model.model.layers
    saved = {}
    handles = []
    def make_hook(layer):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            saved[layer] = value[0, -1, :].detach().float().clone()
        return hook
    for layer in layer_ids:
        handles.append(layers[layer].register_forward_hook(make_hook(layer)))
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(get_device(model)) for k, v in enc.items()}
    with torch.inference_mode():
        logits = model(**enc, use_cache=False).logits[0, -1].float().detach().cpu()
    for handle in handles:
        handle.remove()
    return saved, logits


def patch_layer(model, tokenizer, prompt, layer, source_state):
    import torch
    module = model.model.layers[layer]
    def hook(mod, inputs, output):
        if isinstance(output, tuple):
            value = output[0].clone()
            value[:, -1, :] = source_state.to(value.device, dtype=value.dtype)
            return (value, *output[1:])
        value = output.clone()
        value[:, -1, :] = source_state.to(value.device, dtype=value.dtype)
        return value
    handle = module.register_forward_hook(hook)
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(get_device(model)) for k, v in enc.items()}
    with torch.inference_mode():
        logits = model(**enc, use_cache=False).logits[0, -1].float().detach().cpu()
    handle.remove()
    return logits


def run_residual_causal(model, tokenizer, rows, layer_ids):
    out = []
    selected = [r for r in rows if r["example"] < CAUSAL_N]
    specs = family_specs()
    for target in [r for r in selected if r["family"] in specs]:
        sources = [("same_computation", target["direct_prompt"], target["gold"]),
                   ("surface_wrong", target["surface_control_prompt"], target["surface_gold"]),
                   ("unrelated", target["unrelated_prompt"], target["unrelated_gold"])]
        target_prompt = target["applied_prompt"]
        target_base = first_id(tokenizer, target["gold"])
        target_foil = foil_id(tokenizer, target["gold"])
        _, target_logits = layer_capture(model, tokenizer, target_prompt, layer_ids)
        target_margin = float((target_logits[target_base] - target_logits[target_foil]).item())
        for source_kind, source_prompt, source_gold in sources:
            source_states, source_logits = layer_capture(model, tokenizer, source_prompt, layer_ids)
            source_id = first_id(tokenizer, source_gold)
            source_foil = foil_id(tokenizer, source_gold)
            source_margin = float((source_logits[source_id] - source_logits[source_foil]).item())
            for layer in layer_ids:
                patched = patch_layer(model, tokenizer, target_prompt, layer, source_states[layer])
                patched_margin = float((patched[target_base] - patched[target_foil]).item())
                out.append({"family": target["family"], "example": target["example"], "source_kind": source_kind,
                    "layer": layer, "target_margin": target_margin, "source_margin": source_margin,
                    "patched_margin": patched_margin, "patch_effect": patched_margin - target_margin,
                    "normalized_rescue": (patched_margin - target_margin) / (abs(source_margin - target_margin) + 1e-6)})
    return out


def component_ablation(model, tokenizer, rows, layer_ids):
    import torch
    out = []
    selected = [r for r in rows if r["example"] < CAUSAL_N and r["family"] in ("addition", "subtraction", "multiplication")]
    for row in selected:
        prompt = row["applied_prompt"]
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(get_device(model)) for k, v in enc.items()}
        with torch.inference_mode():
            base = model(**enc, use_cache=False).logits[0, -1].float().detach().cpu()
        gold, foil = row["gold_id"], row["foil_id"]
        base_margin = float((base[gold] - base[foil]).item())
        for layer in layer_ids:
            mlp = model.model.layers[layer].mlp
            def mlp_hook(mod, inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                value = value.clone()
                value[:, -1, :] = 0
                return value
            h = mlp.register_forward_hook(mlp_hook)
            with torch.inference_mode():
                ablated = model(**enc, use_cache=False).logits[0, -1].float().detach().cpu()
            h.remove()
            out.append({"family": row["family"], "example": row["example"], "component": "whole_mlp_output",
                        "layer": layer, "base_margin": base_margin,
                        "ablated_margin": float((ablated[gold] - ablated[foil]).item()),
                        "effect": float(((ablated[gold] - ablated[foil]) - (base[gold] - base[foil])).item())})
            o_proj = model.model.layers[layer].self_attn.o_proj
            n_heads = int(getattr(model.config, "num_attention_heads", 32))
            head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // n_heads))
            for head in [0, n_heads // 2, n_heads - 2, n_heads - 1]:
                def pre_hook(mod, inputs, head=head):
                    value = inputs[0].clone()
                    value[:, -1, head * head_dim:(head + 1) * head_dim] = 0
                    return (value,)
                hp = o_proj.register_forward_pre_hook(pre_hook)
                with torch.inference_mode():
                    ablated_head = model(**enc, use_cache=False).logits[0, -1].float().detach().cpu()
                hp.remove()
                out.append({"family": row["family"], "example": row["example"], "component": "attention_head_ov_input",
                            "layer": layer, "head": head, "base_margin": base_margin,
                            "ablated_margin": float((ablated_head[gold] - ablated_head[foil]).item()),
                            "effect": float(((ablated_head[gold] - ablated_head[foil]) - (base[gold] - base[foil])).item())})
    return out


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cache_name = "models--" + MODEL_NAME.replace("/", "--")
    snapshot_root = HF_HOME / "hub" / cache_name / "snapshots"
    snapshot = max((p for p in snapshot_root.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    rows = make_rows(tokenizer, BEHAVIOR_N)
    behavior_records = []
    for kind in ("direct", "explicit", "applied", "implicit", "surface_control", "answer_control", "unrelated"):
        batch = []
        for r in rows:
            gold_key = "gold"
            if kind == "surface_control":
                gold_key = "surface_gold"
            elif kind == "unrelated":
                gold_key = "unrelated_gold"
            batch.append({"family": r["family"], "example": r["example"], "kind": kind, "prompt": r[f"{kind}_prompt"],
                          "gold": r[gold_key], "gold_id": first_id(tokenizer, r[gold_key]), "foil_id": foil_id(tokenizer, r[gold_key])})
        behavior_records.extend(batched_behavior(model, tokenizer, batch))
        print(f"behavior {kind} complete ({len(batch)} rows)", flush=True)
    write_csv(OUT / "behavior_matrix.csv", behavior_records)
    num_layers = int(model.config.num_hidden_layers)
    layer_ids = sorted(set([0, num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers - 1]))
    rep_rows = run_representation(model, tokenizer, rows, layer_ids)
    write_csv(OUT / "representation_similarity.csv", rep_rows)
    print(f"representation complete ({len(rep_rows)} rows)", flush=True)
    causal_rows = run_residual_causal(model, tokenizer, rows, layer_ids)
    write_csv(OUT / "residual_patching.csv", causal_rows)
    print(f"residual patching complete ({len(causal_rows)} rows)", flush=True)
    component_layers = [max(0, num_layers - 3), max(0, num_layers - 2)]
    component_rows = component_ablation(model, tokenizer, rows, component_layers)
    write_csv(OUT / "component_ablation.csv", component_rows)
    print(f"component ablation complete ({len(component_rows)} rows)", flush=True)
    by_kind = {}
    for r in behavior_records:
        by_kind.setdefault((r["family"], r["kind"]), []).append(r)
    lines = ["# Qwen3-8B Mechanistic Extension", "", f"Model: `{MODEL_NAME}`", f"Behavior rows: `{len(behavior_records)}`", f"Representation rows: `{len(rep_rows)}`", f"Residual patch rows: `{len(causal_rows)}`", f"Component rows: `{len(component_rows)}`", "", "## Behavioral accuracy", "", "| Family | Direct | Explicit | Applied | Implicit | Surface control | Same-answer control | Unrelated |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for family in family_specs():
        vals = []
        for kind in ("direct", "explicit", "applied", "implicit", "surface_control", "answer_control", "unrelated"):
            group = by_kind[(family, kind)]
            vals.append(f"{np.mean([x['correct'] for x in group]):.3f}")
        lines.append("| " + family + " | " + " | ".join(vals) + " |")
    lines += ["", "## Analysis boundary", "", "Residual patching here replaces the target's final-token post-block residual with the source's corresponding post-block residual at selected layers. Component ablation zeros the final-position whole MLP output or one attention head's input to the output projection. These are causal component probes, not sparse-transcoder attribution graphs.", "", f"Selected layers: `{layer_ids}`; component layers: `{component_layers}`; runtime seconds: `{time.time() - started:.1f}`.", ""]
    (OUT / "qwen3_mechanistic_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "run_metadata.json").write_text(json.dumps({"model": MODEL_NAME, "behavior_n_per_family": BEHAVIOR_N, "representation_n_per_family": REP_N, "causal_n_per_family": CAUSAL_N, "layers": layer_ids, "component_layers": component_layers, "runtime_seconds": round(time.time() - started, 2)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
