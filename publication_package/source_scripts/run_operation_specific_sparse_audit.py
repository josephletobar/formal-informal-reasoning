from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_anthropic_addition_reproduction_v2 as base


OUT = Path(os.environ.get("OP_SPARSE_OUT", "results/research_v15_operation_specific"))
TOP_K = 32
MAX_CANDIDATES = 12


def specs():
    discovery = {
        "addition": [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (12, 3), (13, 4), (23, 5), (34, 6)],
        "subtraction": [(3, 1), (4, 2), (5, 3), (6, 4), (7, 5), (8, 6), (9, 7), (10, 8), (15, 3), (16, 4), (25, 5), (36, 6)],
        "multiplication": [(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 3), (3, 4), (3, 5), (4, 2), (4, 3), (4, 4), (5, 2)],
    }
    confirmation = {
        "addition": [(2, 7), (3, 8), (4, 9), (5, 12), (6, 13), (7, 14), (8, 15), (9, 16), (11, 8), (21, 7), (32, 9), (43, 8)],
        "subtraction": [(9, 2), (10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (15, 8), (16, 9), (19, 8), (28, 7), (39, 9), (48, 8)],
        "multiplication": [(2, 6), (2, 7), (2, 8), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6), (5, 3), (5, 4), (6, 3), (7, 2)],
    }
    return discovery, confirmation


def make_prompt(family, form, a, b):
    if family == "addition":
        if form == "explicit": return f"calc: {a}+{b}="
        if form == "applied": return f"A shelf has {a} books and receives {b} more. How many books now?"
        return f"Record: start={a}; added={b}; total=?"
    if family == "subtraction":
        if form == "explicit": return f"calc: {a}-{b}="
        if form == "applied": return f"A shelf has {a} books and {b} are removed. How many remain?"
        return f"Record: start={a}; removed={b}; remaining=?"
    if form == "explicit": return f"calc: {a}*{b}="
    if form == "applied": return f"There are {a} rows with {b} seats each. Total seats?"
    return f"Record: groups={a}; items_each={b}; total=?"


def answer(family, a, b):
    return {"addition": a + b, "subtraction": a - b, "multiplication": a * b}[family]


def continuation_id(tokenizer, prompt, answer_text):
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    full_ids = tokenizer(prompt + answer_text, add_special_tokens=True).input_ids
    if full_ids[:len(prompt_ids)] == prompt_ids and len(full_ids) > len(prompt_ids):
        return int(full_ids[len(prompt_ids)])
    return int(tokenizer(answer_text, add_special_tokens=False).input_ids[0])


def foil_text(tokenizer, prompt, gold):
    gold_id = continuation_id(tokenizer, prompt, str(gold))
    for delta in [-1, 1, -10, 10, -2, 2, -20, 20]:
        candidate = gold + delta
        if candidate >= 0 and continuation_id(tokenizer, prompt, str(candidate)) != gold_id:
            return str(candidate)
    return str(gold + 37)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_observations(model, tokenizer, rows):
    import torch
    observations = []
    for index, row in enumerate(rows, 1):
        with torch.inference_mode():
            logits, acts = model.get_activations(row["prompt"], sparse=False)
        position = int(acts.shape[1] - 1)
        for layer in range(int(acts.shape[0])):
            values = acts[layer, position].float()
            top_values, top_indices = torch.topk(torch.relu(values), k=TOP_K)
            for rank, (value, feature) in enumerate(zip(top_values.tolist(), top_indices.tolist()), 1):
                if value <= 0:
                    continue
                observations.append({"split": row["split"], "family": row["family"], "form": row["form"], "example": row["example"], "a": row["a"], "b": row["b"], "prompt": row["prompt"], "layer": layer, "feature": int(feature), "feature_id": f"L{layer}_F{int(feature)}", "rank": rank, "activation": float(value), "position": position})
        del logits, acts
        base.cleanup()
        if index % 12 == 0:
            print(f"observations {index}/{len(rows)}", flush=True)
    return observations


def select_candidates(observations):
    groups = defaultdict(list)
    for row in observations:
        groups[(int(row["layer"]), int(row["feature"]))].append(row)
    candidates = []
    for (layer, feature), rows in groups.items():
        discovery = [r for r in rows if r["split"] == "discovery"]
        add = [r for r in discovery if r["family"] == "addition"]
        controls = [r for r in discovery if r["family"] in {"subtraction", "multiplication"}]
        add_examples = {int(r["example"]) for r in add}
        add_forms = {r["form"] for r in add}
        ctrl_values = [float(r["activation"]) for r in controls]
        add_values = [float(r["activation"]) for r in add]
        if len(add_examples) < 3 or len(add_forms) < 2 or not add_values:
            continue
        add_mean = float(np.mean(add_values))
        ctrl_mean = float(np.mean(ctrl_values)) if ctrl_values else 0.0
        contrast = add_mean - ctrl_mean
        recurrence = len(add_examples) / 36.0
        score = contrast * math.sqrt(recurrence) * math.log1p(len(add_examples))
        candidates.append({"layer": layer, "feature": feature, "feature_id": f"L{layer}_F{feature}", "addition_examples": len(add_examples), "addition_forms": len(add_forms), "addition_mean": add_mean, "control_mean": ctrl_mean, "addition_minus_control": contrast, "score": score})
    candidates.sort(key=lambda r: r["score"], reverse=True)
    return candidates[:MAX_CANDIDATES]


def margin(logits, gold, foil):
    if logits.ndim == 3: logits = logits[0, -1]
    elif logits.ndim == 2: logits = logits[-1]
    return float((logits[gold].float() - logits[foil].float()).item())


def run_confirmation(model, tokenizer, candidates, rows):
    import torch
    result = []
    for i, row in enumerate(rows, 1):
        prompt = row["prompt"]
        gold = continuation_id(tokenizer, prompt, str(row["total"]))
        foil = continuation_id(tokenizer, prompt, foil_text(tokenizer, prompt, row["total"]))
        with torch.inference_mode():
            base_logits, acts = model.get_activations(prompt, sparse=False)
        position = int(acts.shape[1] - 1)
        base_margin = margin(base_logits, gold, foil)
        for candidate in candidates:
            layer, feature = int(candidate["layer"]), int(candidate["feature"])
            current = float(acts[layer, position, feature].float().item())
            try:
                with torch.inference_mode():
                    changed, _ = model.feature_intervention(prompt, [(layer, position, feature, 0.0)], sparse=True, return_activations=False)
                changed_margin = margin(changed, gold, foil)
                result.append({**row, **candidate, "gold_id": gold, "foil_id": foil, "baseline_margin": base_margin, "target_activation": current, "ablated_margin": changed_margin, "necessity_damage": base_margin - changed_margin, "status": "ok"})
                del changed
            except Exception as exc:
                result.append({**row, **candidate, "gold_id": gold, "foil_id": foil, "baseline_margin": base_margin, "target_activation": current, "status": "error", "error": repr(exc)})
        del base_logits, acts
        base.cleanup()
        if i % 12 == 0:
            print(f"confirmation {i}/{len(rows)}", flush=True)
    return result


def ci(values):
    if len(values) < 2: return float(np.mean(values)) if values else 0.0, 0.0, 0.0
    mean = float(np.mean(values)); half = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    discovery_pairs, confirmation_pairs = specs()
    all_rows = []
    for split, pair_map in (("discovery", discovery_pairs), ("confirmation", confirmation_pairs)):
        for family, pairs in pair_map.items():
            for example, (a, b) in enumerate(pairs):
                for form in ("explicit", "applied", "implicit"):
                    prompt = make_prompt(family, form, a, b)
                    all_rows.append({"split": split, "family": family, "form": form, "example": example, "a": a, "b": b, "total": answer(family, a, b), "prompt": prompt})
    model, tokenizer = base.load_replacement_model()
    observations = extract_observations(model, tokenizer, all_rows)
    candidates = select_candidates(observations)
    discovery_rows = [r for r in observations if r["split"] == "discovery"]
    confirmation_rows = [r for r in all_rows if r["split"] == "confirmation"]
    causal = run_confirmation(model, tokenizer, candidates, confirmation_rows)
    write_csv(OUT / "operation_feature_observations.csv", observations)
    write_csv(OUT / "operation_specific_candidates.csv", candidates)
    write_csv(OUT / "heldout_feature_necessity.csv", causal)
    summary = []
    for family in ("addition", "subtraction", "multiplication"):
        for form in ("explicit", "applied", "implicit"):
            vals = [float(r["necessity_damage"]) for r in causal if r.get("status") == "ok" and r["family"] == family and r["form"] == form]
            m, lo, hi = ci(vals)
            summary.append({"family": family, "form": form, "n": len(vals), "mean_necessity_damage": m, "ci_low": lo, "ci_high": hi})
    write_csv(OUT / "heldout_summary.csv", summary)
    (OUT / "run_metadata.json").write_text(json.dumps({"model": "google/gemma-2-2b-it", "discovery_prompts": 108, "confirmation_prompts": 108, "observation_rows": len(observations), "candidate_count": len(candidates), "causal_rows": len(causal), "runtime_seconds": round(time.time() - started, 2), "selection": "addition discovery recurrence and addition-minus-subtraction/multiplication activation contrast"}, indent=2), encoding="utf-8")
    lines = ["# Operation-Specific Sparse Feature Audit", "", "This run selects sparse candidates using addition-minus-control activation contrast on discovery prompts, then tests them on held-out addition, subtraction, and multiplication prompts.", "", "## Candidates", "", f"Frozen candidates: `{len(candidates)}`", "", "| Feature | Layer | Addition forms | Addition examples | Addition mean | Control mean | Contrast |", "|---|---:|---:|---:|---:|---:|---:|"]
    for c in candidates:
        lines.append(f"| `{c['feature_id']}` | {c['layer']} | {c['addition_forms']} | {c['addition_examples']} | {c['addition_mean']:.4f} | {c['control_mean']:.4f} | {c['addition_minus_control']:.4f} |")
    lines += ["", "## Held-out necessity", "", "| Family | Form | n | Mean damage | 95% CI |", "|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['family']} | {row['form']} | {row['n']} | {row['mean_necessity_damage']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] |")
    lines += ["", "Positive necessity damage means ablating the feature reduced the gold-minus-foil answer margin. Candidate selection was performed on discovery rows only; summary rows use held-out prompts.", ""]
    (OUT / "operation_specific_sparse_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
