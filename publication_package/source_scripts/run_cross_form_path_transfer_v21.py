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

OUT = Path(os.environ.get("PATH_TRANSFER_OUT", "results/research_v21_cross_form_path_transfer"))
PATH_FILE = Path(os.environ.get("PATH_FEATURES", "results/research_v18_full_path_component/multilayer_path_features.csv"))
PAIRS = [(1, 2), (1, 3), (2, 2), (2, 3)]


def prompt_for(form, a, b):
    if form == "explicit": return f"calc: {a}+{b}=", a + b
    if form == "applied": return f"A shelf has {a} books and receives {b} more. How many books now?", a + b
    if form == "implicit": return f"Record: start={a}; added={b}; total=?", a + b
    if form == "subtraction": return f"calc: {b}-{a}=", b - a
    if form == "multiplication": return f"calc: {a}*{b}=", a * b
    raise ValueError(form)


def ci(values):
    values = np.asarray(values, dtype=float)
    h = 1.96 * values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(values.mean() - h), float(values.mean() + h)


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    path_rows = list(csv.DictReader(PATH_FILE.open(encoding="utf-8")))
    paths = {}
    for row in path_rows:
        if row["status"] == "ok" and int(row["path_rank"]) == 1:
            paths.setdefault(row["graph_id"], []).append(row)
    source_ids = [f"addition_explicit_{i}" for i in range(4)]
    frozen = [{"source_graph": graph_id, "feature_count": len(paths[graph_id]), "features": paths[graph_id]} for graph_id in source_ids]
    (OUT / "frozen_paths.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    model = load_model()
    rows = []
    for i, (a, b) in enumerate(PAIRS):
        source_prompt, _ = prompt_for("explicit", a, b)
        source_position = int(model.ensure_tokenized(source_prompt).shape[-1] - 1)
        specs = [(int(x["layer"]), int(x["feature"])) for x in paths[source_ids[i]]]
        with torch.inference_mode():
            _, source_acts = model.get_activations(source_prompt, sparse=False)
        source_values = [float(source_acts[layer, source_position, feature].item()) for layer, feature in specs]
        for target_form in ["explicit", "applied", "implicit", "subtraction", "multiplication"]:
            target_prompt, total = prompt_for(target_form, a, b)
            target_position = int(model.ensure_tokenized(target_prompt).shape[-1] - 1)
            gold = token_id(model, str(total))
            foil = token_id(model, str(total + 1) if total < 9 else str(total - 1))
            with torch.inference_mode():
                target_logits, _ = model.get_activations(target_prompt, sparse=False)
            baseline = margin(model, target_logits, gold, foil)
            zero_specs = [(layer, target_position, feature, 0.0) for layer, feature in specs]
            transfer_specs = [(layer, target_position, feature, value) for (layer, feature), value in zip(specs, source_values)]
            for mode, interventions in (("zero_path", zero_specs), ("source_path_transfer", transfer_specs)):
                try:
                    with torch.inference_mode():
                        changed, _ = model.feature_intervention(target_prompt, interventions, sparse=True, return_activations=False)
                    changed_margin = margin(model, changed, gold, foil)
                    rows.append({"source_graph": source_ids[i], "source_prompt": source_prompt, "target_form": target_form,
                                 "target_prompt": target_prompt, "a": a, "b": b, "total": total, "path_rank": 1,
                                 "feature_count": len(specs), "baseline_margin": baseline, "changed_margin": changed_margin,
                                 "damage": baseline - changed_margin, "mode": mode, "status": "ok"})
                    del changed
                except Exception as exc:
                    rows.append({"source_graph": source_ids[i], "source_prompt": source_prompt, "target_form": target_form,
                                 "target_prompt": target_prompt, "a": a, "b": b, "total": total, "path_rank": 1,
                                 "feature_count": len(specs), "baseline_margin": baseline, "mode": mode,
                                 "status": "error", "error": repr(exc)})
            del target_logits
            torch.cuda.empty_cache()
        del source_acts
        torch.cuda.empty_cache()
    write_csv(OUT / "cross_form_path_transfer.csv", rows)
    summaries = []
    for target_form in ["explicit", "applied", "implicit", "subtraction", "multiplication"]:
        for mode in ["zero_path", "source_path_transfer"]:
            vals = [float(r["damage"]) for r in rows if r.get("status") == "ok" and r["target_form"] == target_form and r["mode"] == mode]
            mean, lo, hi = ci(vals)
            summaries.append({"target_form": target_form, "mode": mode, "n": len(vals), "mean_damage": mean, "ci_low": lo, "ci_high": hi})
    (OUT / "cross_form_path_transfer_summary.json").write_text(json.dumps({"rows": len(rows), "summaries": summaries, "runtime_seconds": round(time.time() - started, 2)}, indent=2), encoding="utf-8")
    (OUT / "cross_form_path_transfer_report.md").write_text("\n".join([
        "# Cross-Form Multilayer Path Transfer", "", "This run transfers the rank-1 seven-feature path from explicit addition into same-operation and wrong-operation targets. Positive damage means lower gold-minus-foil margin.", "",
        "| Target | Mode | N | Mean damage | 95% CI |", "|---|---|---:|---:|---:|",
        *[f"| {s['target_form']} | {s['mode']} | {s['n']} | {s['mean_damage']:.4f} | [{s['ci_low']:.4f}, {s['ci_high']:.4f}] |" for s in summaries], "",
        "This is a direct path-level sufficiency/transfer screen. Because the source path is selected from explicit addition graphs, positive operation-specific transfer would be stronger evidence than graph recurrence alone; it still would require larger held-out panels and necessity controls.", ""
    ]), encoding="utf-8")
    print(f"cross-form path transfer complete rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
