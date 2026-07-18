from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_operation_specific_sparse_audit as audit
import run_anthropic_addition_reproduction_v2 as base

OUT = Path(os.environ.get("ABC_GRAPH_OUT", "results/research_v16_abc_graph_panel"))


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def graph_specs():
    pairs = [(1, 2), (2, 3), (3, 1), (4, 2), (2, 2), (1, 4), (3, 3), (4, 1)]
    rows = []
    for example, (a, b) in enumerate(pairs):
        for form in ("explicit", "applied", "implicit"):
            rows.append({"family": "addition", "form": form, "example": example, "a": a, "b": b, "total": a + b, "prompt": audit.make_prompt("addition", form, a, b)})
    controls = [("subtraction", "explicit", 3, 1), ("subtraction", "applied", 4, 2), ("multiplication", "explicit", 2, 3), ("multiplication", "applied", 2, 2)]
    for example, (family, form, a, b) in enumerate(controls):
        rows.append({"family": family, "form": form, "example": example, "a": a, "b": b, "total": audit.answer(family, a, b), "prompt": audit.make_prompt(family, form, a, b)})
    return rows


def main():
    import torch
    from circuit_tracer.attribution.attribute import attribute

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    model, tokenizer = base.load_replacement_model()
    specs = graph_specs()
    summaries = []
    parents = []
    for index, spec in enumerate(specs, 1):
        prompt = spec["prompt"]
        target_id = audit.continuation_id(tokenizer, prompt, str(spec["total"]))
        target = torch.tensor([target_id], device=model.cfg.device)
        slug = f"{spec['family']}_{spec['form']}_{spec['example']}"
        path = OUT / f"graph_{slug}.pt"
        try:
            graph = attribute(prompt, model, attribution_targets=target, max_n_logits=8, desired_logit_prob=0.90, batch_size=1, max_feature_nodes=384, offload="cpu", verbose=False, update_interval=8)
            graph.to_pt(str(path))
            data = torch.load(path, map_location="cpu", weights_only=False)
            active = data["active_features"].detach().cpu().numpy()
            selected = data["selected_features"].detach().cpu().numpy().reshape(-1)
            adjacency = data["adjacency_matrix"].detach().float().cpu().numpy()
            n_logits = len(data["logit_targets"])
            scores = adjacency[adjacency.shape[0] - n_logits, :len(selected)]
            order = np.argsort(np.abs(scores))[::-1][:80]
            for rank, node_index in enumerate(order, 1):
                active_index = int(selected[int(node_index)])
                layer, position, feature = [int(x) for x in active[active_index]]
                parents.append({**spec, "graph_id": slug, "rank": rank, "layer": layer, "position": position, "feature": feature, "feature_id": f"L{layer}_F{feature}", "edge_to_target": float(scores[int(node_index)]), "abs_edge": float(abs(scores[int(node_index)]))})
            summaries.append({**spec, "graph_id": slug, "graph_path": str(path), "status": "ok", "active_features": int(len(active)), "selected_features": int(len(selected)), "adjacency_shape": json.dumps(list(adjacency.shape))})
            del graph, data
        except Exception as exc:
            summaries.append({**spec, "graph_id": slug, "status": "error", "error": repr(exc)})
        base.cleanup()
        print(f"graph {index}/{len(specs)}", flush=True)
    write_csv(OUT / "graph_summaries.csv", summaries)
    write_csv(OUT / "graph_target_parents.csv", parents)
    ok = [r for r in summaries if r["status"] == "ok"]
    graph_sets = {r["graph_id"]: {(int(p["layer"]), int(p["feature"])) for p in parents if p["graph_id"] == r["graph_id"]} for r in ok}
    recurrence = defaultdict(set)
    for graph_id, features in graph_sets.items():
        for feature in features: recurrence[feature].add(graph_id)
    recurring_rows = []
    for (layer, feature), graph_ids in sorted(recurrence.items(), key=lambda item: (-len(item[1]), item[0])):
        forms = sorted({next(r["form"] for r in ok if r["graph_id"] == gid) for gid in graph_ids})
        families = sorted({next(r["family"] for r in ok if r["graph_id"] == gid) for gid in graph_ids})
        recurring_rows.append({"layer": layer, "feature": feature, "feature_id": f"L{layer}_F{feature}", "graph_count": len(graph_ids), "form_count": len(forms), "forms": ",".join(forms), "families": ",".join(families)})
    write_csv(OUT / "recurring_graph_features.csv", recurring_rows)
    addition_ids = [r["graph_id"] for r in ok if r["family"] == "addition"]
    pairwise = []
    for i, left in enumerate(addition_ids):
        for right in addition_ids[i + 1:]:
            a, b = graph_sets[left], graph_sets[right]
            union = len(a | b)
            pairwise.append({"left": left, "right": right, "same_form": int(next(r["form"] for r in ok if r["graph_id"] == left) == next(r["form"] for r in ok if r["graph_id"] == right)), "jaccard": len(a & b) / union if union else 0.0})
    write_csv(OUT / "graph_pairwise_jaccard.csv", pairwise)
    (OUT / "graph_panel_metadata.json").write_text(json.dumps({"graphs_requested": len(specs), "graphs_ok": len(ok), "parent_rows": len(parents), "recurring_features": len(recurring_rows), "runtime_seconds": round(time.time() - started, 2), "status": "completed"}, indent=2), encoding="utf-8")
    print(f"ABC graph panel complete graphs={len(ok)}/{len(specs)} recurring={len(recurring_rows)}", flush=True)


if __name__ == "__main__":
    main()
