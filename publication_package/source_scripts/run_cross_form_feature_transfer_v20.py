from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from run_graph_edge_poc import load_model, margin, token_id, write_csv

OUT = Path(os.environ.get("TRANSFER_OUT", "results/research_v20_cross_form_transfer"))
MODEL_NAME = "google/gemma-2-2b-it"
# Keep addition, subtraction, and multiplication answers single-token.
PAIRS = [(1, 2), (1, 3), (2, 2), (2, 3)]


def prompt_for(form: str, a: int, b: int) -> tuple[str, int]:
    if form == "explicit":
        return f"calc: {a}+{b}=", a + b
    if form == "applied":
        return f"A shelf has {a} books and receives {b} more. How many books now?", a + b
    if form == "implicit":
        return f"Record: start={a}; added={b}; total=?", a + b
    if form == "subtraction":
        # Keep the control answer nonnegative and single-token for the
        # existing gold-minus-foil objective.
        return f"calc: {b}-{a}=", b - a
    if form == "multiplication":
        return f"calc: {a}*{b}=", a * b
    raise ValueError(form)


def ci(values):
    values = np.asarray(values, dtype=float)
    h = 1.96 * values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(values.mean() - h), float(values.mean() + h)


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    candidate_path = Path(os.environ.get("CANDIDATES", "results/research_v17_recurrent_graph_causal/frozen_candidates.json"))
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))[:6]
    (OUT / "candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    model = load_model()
    source_forms = ["explicit"]
    target_forms = ["explicit", "applied", "implicit", "subtraction", "multiplication"]
    rows = []
    for a, b in PAIRS:
        source_prompt, _ = prompt_for("explicit", a, b)
        source_tokens = model.ensure_tokenized(source_prompt)
        source_position = int(source_tokens.shape[-1] - 1)
        with torch.inference_mode():
            _, source_acts = model.get_activations(source_prompt, sparse=False)
        for target_form in target_forms:
            target_prompt, total = prompt_for(target_form, a, b)
            target_tokens = model.ensure_tokenized(target_prompt)
            target_position = int(target_tokens.shape[-1] - 1)
            gold = token_id(model, str(total))
            foil_text = str(total + 1) if total < 9 else str(total - 1)
            foil = token_id(model, foil_text)
            with torch.inference_mode():
                target_logits, target_acts = model.get_activations(target_prompt, sparse=False)
            baseline = margin(model, target_logits, gold, foil)
            for cand in candidates:
                layer, feature = int(cand["layer"]), int(cand["feature"])
                source_value = float(source_acts[layer, source_position, feature].item())
                target_value = float(target_acts[layer, target_position, feature].item())
                for mode, replacement in (("zero", 0.0), ("source_transfer", source_value)):
                    try:
                        with torch.inference_mode():
                            changed, _ = model.feature_intervention(target_prompt, [(layer, target_position, feature, replacement)], sparse=True, return_activations=False)
                        changed_margin = margin(model, changed, gold, foil)
                        rows.append({"source_form": "explicit", "target_form": target_form, "a": a, "b": b,
                                     "source_prompt": source_prompt, "target_prompt": target_prompt, "total": total,
                                     "feature_id": cand["feature_id"], "layer": layer, "feature": feature,
                                     "source_activation": source_value, "target_activation": target_value,
                                     "baseline_margin": baseline, "changed_margin": changed_margin,
                                     "damage": baseline - changed_margin, "mode": mode, "status": "ok"})
                        del changed
                    except Exception as exc:
                        rows.append({"source_form": "explicit", "target_form": target_form, "a": a, "b": b,
                                     "source_prompt": source_prompt, "target_prompt": target_prompt, "total": total,
                                     "feature_id": cand["feature_id"], "layer": layer, "feature": feature,
                                     "source_activation": source_value, "target_activation": target_value,
                                     "baseline_margin": baseline, "mode": mode, "status": "error", "error": repr(exc)})
            del target_logits, target_acts
            torch.cuda.empty_cache()
        del source_acts
        torch.cuda.empty_cache()
    write_csv(OUT / "cross_form_feature_transfer.csv", rows)
    summaries = []
    for target_form in target_forms:
        for mode in ("zero", "source_transfer"):
            vals = [float(r["damage"]) for r in rows if r.get("status") == "ok" and r["target_form"] == target_form and r["mode"] == mode]
            mean, lo, hi = ci(vals)
            summaries.append({"target_form": target_form, "mode": mode, "n": len(vals), "mean_damage": mean, "ci_low": lo, "ci_high": hi})
    report = ["# Cross-Form Addition Feature Transfer", "", f"Model: `{MODEL_NAME}`", "", "Six discovery-selected recurring features were read from an explicit addition source and tested on same-number explicit, applied, implicit, subtraction, and multiplication targets. Zero is a necessity control; source transfer replaces the target feature value with the source activation.", "", "| Target form | Mode | N | Mean damage | 95% CI |", "|---|---|---:|---:|---:|"]
    for s in summaries:
        report.append(f"| {s['target_form']} | {s['mode']} | {s['n']} | {s['mean_damage']:.4f} | [{s['ci_low']:.4f}, {s['ci_high']:.4f}] |")
    report.extend(["", "Positive damage means the intervention reduced the target gold-minus-foil margin. A source-transfer effect that is specific to addition forms, and stronger than zeroing and operation controls, would support operation-linked transfer. This is a small causal screen, not a complete circuit claim.", ""])
    (OUT / "cross_form_feature_transfer_report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "cross_form_feature_transfer_summary.json").write_text(json.dumps({"model": MODEL_NAME, "rows": len(rows), "summaries": summaries, "runtime_seconds": round(time.time() - started, 2)}, indent=2), encoding="utf-8")
    print(f"cross-form transfer complete rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
