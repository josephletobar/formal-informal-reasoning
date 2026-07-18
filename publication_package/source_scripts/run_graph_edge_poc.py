from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np


MODEL_NAME = "google/gemma-2-2b-it"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def final_logits(logits):
    if logits.ndim == 3:
        return logits[0, -1]
    if logits.ndim == 2:
        return logits[-1]
    return logits


def token_id(model, text: str) -> int:
    ids = model.tokenizer(text, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if len(ids) != 1:
        raise ValueError(f"Expected one token for {text!r}, got {ids!r}")
    return int(ids[0])


def load_model():
    import torch
    from transformers import AutoTokenizer
    from circuit_tracer.replacement_model import ReplacementModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=False)
    model = ReplacementModel.from_pretrained(
        MODEL_NAME,
        "gemma",
        backend="transformerlens",
        device=torch.device("cuda"),
        dtype=torch.float16,
        lazy_encoder=False,
        lazy_decoder=True,
        tokenizer=tokenizer,
    )
    model.eval()
    return model


def margin(model, logits, gold_id: int, foil_id: int) -> float:
    vector = final_logits(logits).float()
    return float((vector[gold_id] - vector[foil_id]).item())


def node_name(active, selected, node_index: int, n_selected: int, n_error: int, n_pos: int, n_logits: int) -> str:
    if node_index < n_selected:
        active_index = int(selected[node_index])
        layer, position, feature = [int(value) for value in active[active_index]]
        return f"feature:L{layer}:P{position}:F{feature}"
    error_start = n_selected
    token_start = error_start + n_error
    logit_start = token_start + n_pos
    if node_index < token_start:
        relative = node_index - error_start
        layer = relative // n_pos
        position = relative % n_pos
        return f"error:L{layer}:P{position}"
    if node_index < logit_start:
        return f"token:P{node_index - token_start}"
    return f"logit:{node_index - logit_start}"


def graph_parents(graph_path: Path, top_k: int) -> tuple[dict, list[dict]]:
    import torch

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    active = data["active_features"].detach().cpu().numpy()
    selected = data["selected_features"].detach().cpu().numpy().reshape(-1)
    adjacency = data["adjacency_matrix"].detach().float().cpu().numpy()
    cfg = data["cfg"]
    n_pos = int(data["input_tokens"].shape[0])
    n_selected = len(selected)
    n_error = int(cfg.n_layers) * n_pos
    n_logits = len(data["logit_targets"])
    logit_start = adjacency.shape[0] - n_logits
    source_scores = adjacency[logit_start].copy()
    order = np.argsort(np.abs(source_scores[:n_selected]))[::-1]
    rows = []
    for rank, node_index in enumerate(order[:top_k], 1):
        active_index = int(selected[int(node_index)])
        layer, position, feature = [int(value) for value in active[active_index]]
        rows.append({
            "rank": rank,
            "graph_node": int(node_index),
            "active_feature_index": active_index,
            "layer": layer,
            "position": position,
            "feature": feature,
            "feature_id": f"L{layer}_P{position}_F{feature}",
            "edge_to_logit": float(source_scores[int(node_index)]),
            "abs_edge_to_logit": float(abs(source_scores[int(node_index)])),
            "node_name": node_name(active, selected, int(node_index), n_selected, n_error, n_pos, n_logits),
        })
    metadata = {
        "input_string": data["input_string"],
        "active_features": int(len(active)),
        "selected_features": int(len(selected)),
        "adjacency_shape": list(adjacency.shape),
        "logit_target": str(data["logit_targets"][0]),
        "n_pos": n_pos,
        "n_error_nodes": n_error,
        "n_logit_nodes": n_logits,
    }
    return metadata, rows


def run(model, results: Path, graph_dir: Path, top_k: int) -> None:
    import torch

    started = time.time()
    graph_specs = [
        ("calc_3_plus_4", graph_dir / "graph_calc_3_plus_4.pt", "calc: 3+4=", 7),
        ("calc_8_plus_7", graph_dir / "graph_calc_8_plus_7.pt", "calc: 8+7=1", 5),
    ]
    edge_rows = []
    causal_rows = []
    for graph_id, graph_path, prompt, gold_digit in graph_specs:
        metadata, parents = graph_parents(graph_path, top_k)
        metadata["graph_id"] = graph_id
        metadata["prompt"] = prompt
        metadata["gold_digit"] = gold_digit
        write_json(results / f"{graph_id}_metadata.json", metadata)
        write_csv(results / f"{graph_id}_top_logit_parents.csv", parents)
        edge_rows.extend({"graph_id": graph_id, **row} for row in parents)
        tokens = model.ensure_tokenized(prompt)
        position = int(tokens.shape[0] - 1)
        gold = token_id(model, str(gold_digit))
        foil = token_id(model, str(gold_digit + 1 if gold_digit < 9 else gold_digit - 1))
        with torch.inference_mode():
            base_logits, _ = model.get_activations(prompt, sparse=False)
        base_margin = margin(model, base_logits, gold, foil)
        for parent in parents:
            layer = int(parent["layer"])
            feature = int(parent["feature"])
            feature_position = int(parent["position"])
            try:
                with torch.inference_mode():
                    intervened_logits, _ = model.feature_intervention(
                        prompt,
                        [(layer, feature_position, feature, 0.0)],
                        sparse=True,
                        return_activations=False,
                    )
                changed_margin = margin(model, intervened_logits, gold, foil)
                causal_rows.append({
                    "graph_id": graph_id,
                    "prompt": prompt,
                    "target_gold": gold_digit,
                    "layer": layer,
                    "position": feature_position,
                    "feature": feature,
                    "feature_id": parent["feature_id"],
                    "edge_to_logit": parent["edge_to_logit"],
                    "baseline_margin": base_margin,
                    "intervened_margin": changed_margin,
                    "margin_change": changed_margin - base_margin,
                    "gold_damage": base_margin - changed_margin,
                    "status": "ok",
                })
            except Exception as exc:
                causal_rows.append({
                    "graph_id": graph_id,
                    "prompt": prompt,
                    "layer": layer,
                    "position": feature_position,
                    "feature": feature,
                    "feature_id": parent["feature_id"],
                    "edge_to_logit": parent["edge_to_logit"],
                    "status": "error",
                    "error": repr(exc),
                })
        del base_logits
        torch.cuda.empty_cache()

    write_csv(results / "graph_edge_summary.csv", edge_rows)
    write_csv(results / "graph_edge_causal_tests.csv", causal_rows)
    ok = [row for row in causal_rows if row.get("status") == "ok"]
    report = f"""# Addition Attribution-Graph Edge Proof of Concept

## What this reproduces

This is a graph-level circuit test. The existing Circuit Tracer attribution graphs explicitly store feature nodes, error nodes, input-token nodes, and answer-logit nodes. For each graph, this run selected the strongest feature nodes directly upstream of the answer-logit node and then zeroed those exact feature nodes at their original layer and token position.

This is much closer to an Anthropic-style circuit test than a whole-layer representation comparison. The graph itself proposes the path; the intervention tests whether its proposed source feature changes the answer.

## Graphs

- `calc: 3+4=` -> next-token answer `7`
- `calc: 8+7=1` -> final answer digit `5` for the multi-token answer `15`
- Top feature parents tested per graph: {top_k}
- Successful graph-node interventions: {len(ok)}/{len(causal_rows)}

## Results

The raw graph edges are in `graph_edge_summary.csv`. The causal tests are in `graph_edge_causal_tests.csv`.

For each row:

- `edge_to_logit` is the graph's signed direct attribution from the feature to the answer-logit node.
- `gold_damage` is the drop in the correct-answer-minus-foil margin after zeroing the feature.
- Positive `gold_damage` means the feature was helping the answer under this intervention.

## Interpretation

The important reproduction criterion is convergence between the graph and intervention: features that the attribution graph places directly on the answer path should tend to produce measurable output changes when removed. This small run tests that criterion on two addition examples.

It is still not a complete reusable-circuit result because there are only two graphs and no held-out controls yet. The next step is to repeat graph construction across held-out additions, intersect recurring graph parents, and test those recurring nodes on new prompts and wrong-operation controls.
"""
    (results / "graph_edge_poc_report.md").write_text(report, encoding="utf-8")
    write_json(results / "graph_edge_poc_summary.json", {
        "model": MODEL_NAME,
        "graphs": len(graph_specs),
        "top_k_per_graph": top_k,
        "causal_rows": len(causal_rows),
        "successful_causal_rows": len(ok),
        "runtime_seconds": time.time() - started,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="addition_graph_edge_poc")
    parser.add_argument("--graph-dir", default="discovery_smoke_v2")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    model = load_model()
    run(model, results, Path(args.graph_dir), args.top_k)


if __name__ == "__main__":
    main()
