from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_operation_specific_sparse_audit as audit
import run_anthropic_addition_reproduction_v2 as base

OUT = Path(os.environ.get("GRAPH_CAUSAL_OUT", "results/research_v17_recurrent_graph_causal"))
GRAPH_DIR = Path(os.environ.get("GRAPH_PANEL_DIR", "results/research_v16_abc_graph_panel"))
SEED = 1701


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ci(values):
    values = [float(x) for x in values]
    mean = float(np.mean(values)) if values else 0.0
    half = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, mean - half, mean + half


def main():
    import torch

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    parents = read_csv(GRAPH_DIR / "graph_target_parents.csv")
    addition = [r for r in parents if r["family"] == "addition"]
    discovery = [r for r in addition if int(r["example"]) < 4]
    confirmation = [r for r in addition if int(r["example"]) >= 4]

    # Freeze candidates using only the first half of addition graphs. Require recurrence
    # across at least two ABC forms, then test only on the second half.
    by_feature = defaultdict(list)
    for row in discovery:
        by_feature[(int(row["layer"]), int(row["feature"]))].append(row)
    ranked = []
    for key, rows in by_feature.items():
        graph_ids = {r["graph_id"] for r in rows}
        forms = {r["form"] for r in rows}
        edge_sum = sum(abs(float(r["edge_to_target"])) for r in rows)
        if len(graph_ids) >= 3 and len(forms) >= 2:
            ranked.append((len(graph_ids), len(forms), edge_sum, key, rows))
    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    selected = ranked[:12]
    frozen = [{"layer": layer, "feature": feature, "feature_id": f"L{layer}_F{feature}",
               "discovery_graph_count": graph_count, "discovery_form_count": form_count,
               "discovery_edge_abs_sum": edge_sum}
              for graph_count, form_count, edge_sum, (layer, feature), _ in selected]
    (OUT / "frozen_candidates.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    selected_keys = {(x["layer"], x["feature"]) for x in frozen}
    test_rows = [r for r in confirmation if (int(r["layer"]), int(r["feature"])) in selected_keys]
    # One matched random feature at each exact layer/position creates a local control.
    random.seed(SEED)
    model, tokenizer = base.load_replacement_model()
    rows = []
    feature_count = 16384
    for idx, graph_row in enumerate(test_rows):
        prompt = graph_row["prompt"]
        total = int(graph_row["total"])
        gold = audit.continuation_id(tokenizer, prompt, str(total))
        foil = audit.continuation_id(tokenizer, prompt, audit.foil_text(tokenizer, prompt, total))
        layer = int(graph_row["layer"])
        position = int(graph_row["position"])
        feature = int(graph_row["feature"])
        with torch.inference_mode():
            base_logits, acts = model.get_activations(prompt, sparse=False)
        baseline = audit.margin(base_logits, gold, foil)
        del base_logits
        # A random feature must be active at this exact layer and position. An
        # arbitrary feature is usually zero already, making it an invalid null
        # control for a feature intervention.
        active_indices = torch.nonzero(acts[layer, position].abs() > 1e-5, as_tuple=False).flatten().tolist()
        candidates = [int(x) for x in active_indices if int(x) != feature and (layer, int(x)) not in selected_keys]
        if not candidates:
            candidates = [int(x) for x in active_indices if int(x) != feature]
        if not candidates:
            rows.append({"family": "addition", "form": graph_row["form"], "example": graph_row["example"],
                         "prompt": prompt, "graph_id": graph_row["graph_id"], "layer": layer,
                         "position": position, "feature": feature, "feature_id": f"L{layer}_F{feature}",
                         "source_feature": feature, "kind": "random_control", "mode": "unavailable",
                         "status": "error", "error": "no active matched-layer control feature"})
            del acts
            base.cleanup()
            continue
        random_feature = candidates[random.randrange(len(candidates))]
        for label, intervention_feature in (("recurrent", feature), ("random_control", random_feature)):
            current = float(acts[layer, position, intervention_feature].float().item())
            for mode, replacement in (("zero", 0.0), ("negative_original", -current)):
                try:
                    with torch.inference_mode():
                        changed, _ = model.feature_intervention(prompt, [(layer, position, intervention_feature, replacement)], sparse=True, return_activations=False)
                    changed_margin = audit.margin(changed, gold, foil)
                    rows.append({"family": "addition", "form": graph_row["form"], "example": graph_row["example"],
                                 "prompt": prompt, "graph_id": graph_row["graph_id"], "layer": layer,
                                 "position": position, "feature": intervention_feature,
                                 "feature_id": f"L{layer}_F{intervention_feature}", "source_feature": feature,
                                 "kind": label, "edge_to_target": graph_row["edge_to_target"] if label == "recurrent" else "",
                                 "mode": mode, "current_activation": current, "baseline_margin": baseline,
                                 "changed_margin": changed_margin, "damage": baseline - changed_margin, "status": "ok"})
                    del changed
                except Exception as exc:
                    rows.append({"family": "addition", "form": graph_row["form"], "example": graph_row["example"],
                                 "prompt": prompt, "graph_id": graph_row["graph_id"], "layer": layer,
                                 "position": position, "feature": intervention_feature,
                                 "feature_id": f"L{layer}_F{intervention_feature}", "source_feature": feature,
                                 "kind": label, "edge_to_target": graph_row["edge_to_target"] if label == "recurrent" else "",
                                 "mode": mode, "current_activation": current, "baseline_margin": baseline,
                                 "status": "error", "error": repr(exc)})
        del acts
        base.cleanup()
        if (idx + 1) % 10 == 0:
            print(f"tested {idx + 1}/{len(test_rows)} graph-parent locations", flush=True)

    write_csv(OUT / "recurrent_graph_causal.csv", rows)
    summaries = []
    for kind in ("recurrent", "random_control"):
        for mode in ("zero", "negative_original"):
            vals = [float(r["damage"]) for r in rows if r.get("status") == "ok" and r["kind"] == kind and r["mode"] == mode]
            mean, lo, hi = ci(vals)
            summaries.append({"kind": kind, "mode": mode, "n": len(vals), "mean_damage": mean, "ci_low": lo, "ci_high": hi})
    (OUT / "recurrent_graph_causal_summary.json").write_text(json.dumps({"rows": len(rows), "frozen_candidates": frozen, "summaries": summaries, "runtime_seconds": round(time.time() - started, 2)}, indent=2), encoding="utf-8")
    print(f"v17 recurrent graph causal complete rows={len(rows)} runtime={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
