from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_operation_specific_sparse_audit as audit
import run_anthropic_addition_reproduction_v2 as base


OUT = Path(os.environ.get("OP_SPARSE_OUT", "results/research_v15_operation_specific"))


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


def main():
    import torch
    from circuit_tracer.attribution.attribute import attribute

    started = time.time()
    candidates = read_csv(OUT / "operation_specific_candidates.csv")
    observations = read_csv(OUT / "operation_feature_observations.csv")
    selected_keys = {(int(r["layer"]), int(r["feature"])) for r in candidates}
    by_layer = defaultdict(set)
    for row in observations:
        if row["split"] == "discovery":
            by_layer[int(row["layer"])].add(int(row["feature"]))
    rng = random.Random(17)
    random_controls = []
    for candidate in candidates:
        layer = int(candidate["layer"])
        pool = sorted(by_layer[layer] - {feature for lyr, feature in selected_keys if lyr == layer})
        if not pool:
            pool = sorted(by_layer[layer])
        feature = rng.choice(pool)
        values = [float(r["activation"]) for r in observations if r["split"] == "discovery" and int(r["layer"]) == layer and int(r["feature"]) == feature]
        random_controls.append({"layer": layer, "feature": feature, "feature_id": f"L{layer}_F{feature}", "reference_value": float(np.mean(values)) if values else 0.0})
    discovery, confirmation = audit.specs()
    confirmation_rows = []
    for family, pairs in confirmation.items():
        for example, (a, b) in enumerate(pairs):
            for form in ("explicit", "applied", "implicit"):
                confirmation_rows.append({"split": "confirmation", "family": family, "form": form, "example": example, "a": a, "b": b, "total": audit.answer(family, a, b), "prompt": audit.make_prompt(family, form, a, b)})
    model, tokenizer = base.load_replacement_model()
    rows = []
    all_features = [("selected", r, float(r["addition_mean"])) for r in candidates] + [("random_control", r, float(r["reference_value"])) for r in random_controls]
    for i, spec in enumerate(confirmation_rows, 1):
        prompt = spec["prompt"]
        gold = audit.continuation_id(tokenizer, prompt, str(spec["total"]))
        foil_text = audit.foil_text(tokenizer, prompt, spec["total"])
        foil = audit.continuation_id(tokenizer, prompt, foil_text)
        with torch.inference_mode():
            base_logits, acts = model.get_activations(prompt, sparse=False)
        position = int(acts.shape[1] - 1)
        base_margin = audit.margin(base_logits, gold, foil)
        for source, candidate, reference in all_features:
            layer, feature = int(candidate["layer"]), int(candidate["feature"])
            current = float(acts[layer, position, feature].float().item())
            for mode, replacement in (("zero", 0.0), ("reference", reference)):
                try:
                    with torch.inference_mode():
                        changed, _ = model.feature_intervention(prompt, [(layer, position, feature, replacement)], sparse=True, return_activations=False)
                    changed_margin = audit.margin(changed, gold, foil)
                    rows.append({**spec, "source": source, "feature_id": candidate["feature_id"], "layer": layer, "feature": feature, "mode": mode, "reference_value": reference, "target_activation": current, "baseline_margin": base_margin, "changed_margin": changed_margin, "effect": changed_margin - base_margin, "status": "ok"})
                    del changed
                except Exception as exc:
                    rows.append({**spec, "source": source, "feature_id": candidate["feature_id"], "layer": layer, "feature": feature, "mode": mode, "reference_value": reference, "target_activation": current, "baseline_margin": base_margin, "status": "error", "error": repr(exc)})
        del base_logits, acts
        base.cleanup()
        if i % 12 == 0:
            print(f"sufficiency {i}/{len(confirmation_rows)}", flush=True)
    write_csv(OUT / "heldout_feature_sufficiency.csv", rows)
    graph_rows = []
    graph_specs = [r for r in confirmation_rows if r["family"] == "addition" and r["form"] in {"explicit", "applied", "implicit"} and r["example"] < 2]
    for index, spec in enumerate(graph_specs, 1):
        prompt = spec["prompt"]
        target_id = audit.continuation_id(tokenizer, prompt, str(spec["total"]))
        target = torch.tensor([target_id], device=model.cfg.device)
        try:
            graph = attribute(prompt, model, attribution_targets=target, max_n_logits=8, desired_logit_prob=0.90, batch_size=1, max_feature_nodes=256, offload="cpu", verbose=False, update_interval=8)
            path = OUT / f"graph_{spec['family']}_{spec['form']}_{spec['example']}.pt"
            graph.to_pt(str(path))
            data = torch.load(path, map_location="cpu", weights_only=False)
            active = data["active_features"].detach().cpu().numpy()
            selected = data["selected_features"].detach().cpu().numpy().reshape(-1)
            adjacency = data["adjacency_matrix"].detach().float().cpu().numpy()
            n_logits = len(data["logit_targets"])
            scores = adjacency[adjacency.shape[0] - n_logits, :len(selected)]
            candidate_set = {(int(r["layer"]), int(r["feature"])) for r in candidates}
            hits = []
            for node_index, active_index in enumerate(selected):
                layer, position, feature = [int(x) for x in active[int(active_index)]]
                if (layer, feature) in candidate_set:
                    hits.append({"layer": layer, "position": position, "feature": feature, "feature_id": f"L{layer}_F{feature}", "edge_to_target": float(scores[node_index])})
            graph_rows.append({**spec, "graph_path": str(path), "active_features": int(len(active)), "selected_features": int(len(selected)), "candidate_hits": len(hits), "candidate_hit_details": json.dumps(hits), "status": "ok"})
            del graph, data
        except Exception as exc:
            graph_rows.append({**spec, "candidate_hits": 0, "status": "error", "error": repr(exc)})
        base.cleanup()
        print(f"graph {index}/{len(graph_specs)}", flush=True)
    write_csv(OUT / "heldout_graph_candidate_hits.csv", graph_rows)
    (OUT / "sufficiency_graph_metadata.json").write_text(json.dumps({"sufficiency_rows": len(rows), "graph_rows": len(graph_rows), "runtime_seconds": round(time.time() - started, 2), "status": "completed"}, indent=2), encoding="utf-8")
    print(f"sufficiency and graphs complete rows={len(rows)} graphs={len(graph_rows)}", flush=True)


if __name__ == "__main__":
    main()
