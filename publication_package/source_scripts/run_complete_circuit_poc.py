from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from run_graph_edge_poc import load_model, margin, token_id, write_csv, write_json


MODEL_NAME = "google/gemma-2-2b-it"


def prompt_total(prompt: str) -> int:
    expression = prompt.split(":", 1)[1].strip().rstrip("=")
    left, right = expression.split("+")
    return int(left) + int(right)


def load_graph(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def feature_node_name(graph: dict, node: int, n_selected: int) -> str:
    if node >= n_selected:
        n_pos = int(graph["input_tokens"].numel())
        n_error = int(graph["cfg"].n_layers) * n_pos
        token_start = n_selected + n_error
        logit_start = int(graph["adjacency_matrix"].shape[0]) - len(graph["logit_targets"])
        if node < token_start:
            relative = node - n_selected
            return f"error:L{relative // n_pos}:P{relative % n_pos}"
        if node < logit_start:
            return f"token:P{node - token_start}"
        return f"logit:{node - logit_start}"
    active_index = int(graph["selected_features"][node])
    layer, pos, feature = [int(value) for value in graph["active_features"][active_index]]
    return f"L{layer}:P{pos}:F{feature}"


def graph_top_paths(graph: dict, beam_width: int = 128, top_k: int = 8, max_depth: int = 8) -> list[dict]:
    """Find high-weight source-to-logit paths by backward beam search.

    The serialized graph stores rows as target nodes and columns as source
    nodes. The node ordering is selected features, error nodes, input tokens,
    then logit nodes.
    """
    adjacency = graph["adjacency_matrix"].detach().float().cpu().numpy()
    n_selected = int(graph["selected_features"].numel())
    n_pos = int(graph["input_tokens"].numel())
    n_error = int(graph["cfg"].n_layers) * n_pos
    n_logits = len(graph["logit_targets"])
    token_start = n_selected + n_error
    logit_start = adjacency.shape[0] - n_logits
    beam = [(1.0, logit_start, [logit_start], [])]
    completed = []
    for _ in range(max_depth):
        next_beam = []
        for score, current, nodes_back, weights_back in beam:
            if token_start <= current < logit_start:
                completed.append((score, nodes_back, weights_back))
                continue
            incoming = adjacency[current].copy()
            incoming[np.asarray(nodes_back, dtype=int)] = 0.0
            candidates = np.argsort(np.abs(incoming))[::-1]
            added = 0
            for source in candidates:
                weight = float(incoming[source])
                if abs(weight) < 1e-6:
                    break
                next_score = score * abs(weight)
                next_beam.append((next_score, int(source), nodes_back + [int(source)], weights_back + [weight]))
                added += 1
                if added >= 12:
                    break
        next_beam.sort(key=lambda item: item[0], reverse=True)
        beam = next_beam[:beam_width]
        if not beam:
            break
    completed.extend((score, nodes, weights) for score, _, nodes, weights in beam if token_start <= _ < logit_start)
    paths = []
    seen = set()
    for score, nodes_back, weights_back in sorted(completed, key=lambda item: item[0], reverse=True):
        nodes = list(reversed(nodes_back))
        weights = list(reversed(weights_back))
        feature_nodes = [node for node in nodes if node < n_selected]
        key = tuple(nodes)
        if not feature_nodes or key in seen:
            continue
        seen.add(key)
        feature_specs = []
        for node in feature_nodes:
            active_index = int(graph["selected_features"][node])
            layer, pos, feature = [int(value) for value in graph["active_features"][active_index]]
            feature_specs.append({"node": node, "layer": layer, "position": pos, "feature": feature, "feature_id": f"L{layer}_P{pos}_F{feature}"})
        paths.append({
            "path_rank": len(paths) + 1,
            "path_score_abs_product": float(score),
            "nodes": nodes,
            "node_names": [feature_node_name(graph, node, n_selected) for node in nodes],
            "edge_weights_source_to_target": weights,
            "feature_specs": feature_specs,
            "feature_count": len(feature_specs),
        })
        if len(paths) >= top_k:
            break
    return paths


def select_heads(source_dir: Path, count: int = 4) -> list[dict]:
    rows = list(csv.DictReader((source_dir / "component_ablations.csv").open(encoding="utf-8")))
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok" or not row.get("prompt", "").startswith("calc:") or row.get("component_type") != "attention_head":
            continue
        groups[(int(row["layer"]), int(row["head"]))].append(float(row["effect"]))
    candidates = []
    for (layer, head), effects in groups.items():
        negative = sum(value < 0 for value in effects)
        if negative >= 4:
            candidates.append({"layer": layer, "head": head, "mean_ablation_effect": float(np.mean(effects)), "negative_count": negative, "prompt_count": len(effects)})
    candidates.sort(key=lambda row: row["mean_ablation_effect"])
    return candidates[:count]


def pattern_source_hook(head: int, query_position: int, source_position: int):
    def hook(value, hook=None):
        out = value.clone()
        out[:, head, query_position, source_position] = 0
        return out
    return hook


def run_feature_paths(model, graph_manifest: dict, results: Path, top_paths: int, top_features: int) -> tuple[list[dict], list[dict]]:
    path_rows = []
    path_feature_rows = []
    for graph_row in graph_manifest["graphs"]:
        if graph_row.get("status") != "ok":
            continue
        graph = load_graph(Path(graph_row["path"]))
        paths = graph_top_paths(graph, top_k=top_paths)
        prompt = graph_row["prompt"]
        total = prompt_total(prompt)
        gold_text = str(total)
        gold = token_id(model, gold_text)
        foil_total = total + 1 if total < 9 else total - 1
        foil = token_id(model, str(foil_total))
        with torch.inference_mode():
            base_logits, acts = model.get_activations(prompt, sparse=False)
        base_margin = margin(model, base_logits, gold, foil)
        for path in paths:
            specs = path["feature_specs"]
            zero_interventions = [(spec["layer"], spec["position"], spec["feature"], 0.0) for spec in specs]
            boost_interventions = [(spec["layer"], spec["position"], spec["feature"], float(acts[spec["layer"], spec["position"], spec["feature"]].item()) * 2.0) for spec in specs]
            try:
                with torch.inference_mode():
                    zero_logits, _ = model.feature_intervention(prompt, zero_interventions, sparse=True, return_activations=False)
                    boost_logits, _ = model.feature_intervention(prompt, boost_interventions, sparse=True, return_activations=False)
                zero_margin = margin(model, zero_logits, gold, foil)
                boost_margin = margin(model, boost_logits, gold, foil)
                path_rows.append({
                    "graph_id": graph_row["graph_id"],
                    "prompt": prompt,
                    "total": total,
                    "path_rank": path["path_rank"],
                    "feature_count": len(specs),
                    "path_score_abs_product": path["path_score_abs_product"],
                    "path_nodes": json.dumps(path["node_names"], separators=(",", ":")),
                    "base_margin": base_margin,
                    "zero_path_margin": zero_margin,
                    "boost_path_margin": boost_margin,
                    "path_gold_damage": base_margin - zero_margin,
                    "path_boost_gain": boost_margin - base_margin,
                    "status": "ok",
                })
                for spec in specs:
                    path_feature_rows.append({"graph_id": graph_row["graph_id"], "prompt": prompt, "path_rank": path["path_rank"], **spec, "status": "ok"})
            except Exception as exc:
                path_rows.append({"graph_id": graph_row["graph_id"], "prompt": prompt, "total": total, "path_rank": path["path_rank"], "path_nodes": json.dumps(path["node_names"], separators=(",", ":")), "status": "error", "error": repr(exc)})
        del graph, base_logits, acts
        torch.cuda.empty_cache()
    write_csv(results / "multilayer_path_tests.csv", path_rows)
    write_csv(results / "multilayer_path_features.csv", path_feature_rows)
    return path_rows, path_feature_rows


def run_attention(model, source_dir: Path, results: Path, prompts: list[tuple[str, int]]) -> tuple[list[dict], list[dict]]:
    heads = select_heads(source_dir)
    write_csv(results / "selected_attention_heads.csv", heads)
    decomposition_rows = []
    causal_rows = []
    layers = sorted({int(row["layer"]) for row in heads})
    for prompt, total in prompts:
        tokens = model.ensure_tokenized(prompt)
        query_position = int(tokens.shape[0] - 1)
        names = []
        for layer in layers:
            names.extend([f"blocks.{layer}.attn.hook_q", f"blocks.{layer}.attn.hook_k", f"blocks.{layer}.attn.hook_v", f"blocks.{layer}.attn.hook_attn_scores", f"blocks.{layer}.attn.hook_pattern", f"blocks.{layer}.attn.hook_result"])
        with torch.inference_mode():
            logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in set(names), return_type="logits")
        gold = token_id(model, str(total))
        foil_total = total + 1 if total < 9 else total - 1
        foil = token_id(model, str(foil_total))
        base_margin = margin(model, logits, gold, foil)
        delta_unembed = (model.W_U[:, gold] - model.W_U[:, foil]).float().detach()
        for head_row in heads:
            layer = int(head_row["layer"]); head = int(head_row["head"])
            q = cache.cache_dict[f"blocks.{layer}.attn.hook_q"][0].float()
            k = cache.cache_dict[f"blocks.{layer}.attn.hook_k"][0].float()
            v = cache.cache_dict[f"blocks.{layer}.attn.hook_v"][0].float()
            scores = cache.cache_dict[f"blocks.{layer}.attn.hook_attn_scores"][0].float()
            pattern = cache.cache_dict[f"blocks.{layer}.attn.hook_pattern"][0].float()
            result = cache.cache_dict[f"blocks.{layer}.attn.hook_result"][0].float()
            kv_group = head // max(1, int(model.cfg.n_heads) // int(k.shape[1]))
            attention = pattern[head, query_position]
            source_order = torch.argsort(attention, descending=True).tolist()[:4]
            for source_position in source_order:
                source_position = int(source_position)
                value_to_residual = v[source_position, kv_group] @ model.W_O[layer, head].float()
                source_contribution = float(torch.dot(value_to_residual * attention[source_position], delta_unembed).item())
                decomposition_rows.append({
                    "prompt": prompt, "total": total, "layer": layer, "head": head,
                    "query_position": query_position, "source_position": source_position,
                    "qk_score": float(scores[head, query_position, source_position].item()),
                    "attention_weight": float(attention[source_position].item()),
                    "source_ov_logit_projection": source_contribution,
                    "head_result_norm": float(result[query_position, head].norm().item()),
                    "query_norm": float(q[query_position, head].norm().item()),
                    "key_norm": float(k[source_position, kv_group].norm().item()),
                    "status": "ok",
                })
                hook_name = f"blocks.{layer}.attn.hook_pattern"
                try:
                    with torch.inference_mode():
                        changed_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, pattern_source_hook(head, query_position, source_position))], return_type="logits")
                    changed_margin = margin(model, changed_logits, gold, foil)
                    causal_rows.append({
                        "prompt": prompt, "total": total, "layer": layer, "head": head,
                        "query_position": query_position, "source_position": source_position,
                        "attention_weight": float(attention[source_position].item()),
                        "baseline_margin": base_margin, "source_ablated_margin": changed_margin,
                        "source_gold_damage": base_margin - changed_margin, "status": "ok",
                    })
                except Exception as exc:
                    causal_rows.append({"prompt": prompt, "total": total, "layer": layer, "head": head, "query_position": query_position, "source_position": source_position, "status": "error", "error": repr(exc)})
        del cache, logits
        torch.cuda.empty_cache()
    write_csv(results / "attention_qk_ov_decomposition.csv", decomposition_rows)
    write_csv(results / "attention_source_position_causal_tests.csv", causal_rows)
    return decomposition_rows, causal_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="complete_circuit_poc")
    parser.add_argument("--panel-dir", default="addition_graph_panel")
    parser.add_argument("--source-dir", default="component_analysis_full")
    parser.add_argument("--top-paths", type=int, default=6)
    args = parser.parse_args()
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    panel_manifest = json.loads((Path(args.panel_dir) / "graph_panel_manifest.json").read_text(encoding="utf-8"))
    prompts = [(row["prompt"], prompt_total(row["prompt"])) for row in panel_manifest["graphs"] if row.get("status") == "ok"]
    started = time.time()
    model = load_model()
    model.set_use_attn_result(True)
    path_rows, path_feature_rows = run_feature_paths(model, panel_manifest, results, args.top_paths, 8)
    decomposition_rows, attention_causal_rows = run_attention(model, Path(args.source_dir), results, prompts)
    path_ok = [row for row in path_rows if row.get("status") == "ok"]
    attn_ok = [row for row in attention_causal_rows if row.get("status") == "ok"]
    path_damage = [float(row["path_gold_damage"]) for row in path_ok]
    path_gain = [float(row["path_boost_gain"]) for row in path_ok]
    attn_damage = [float(row["source_gold_damage"]) for row in attn_ok]
    report = f"""# Complete Circuit POC: Paths, Attention Decomposition, Necessity, Sufficiency

## Scope

This run addresses the three gaps in the earlier graph result:

1. It extracts complete multi-node paths through the attribution graph rather than only listing direct parents of the answer logit.
2. It decomposes selected attention heads by query-key score, attention weight, value-to-residual OV contribution, and source-position ablation.
3. It tests feature-path necessity by zeroing paths and a sufficiency-like dose response by setting graph features to twice their clean activation.

## Completed tests

- Multi-layer path rows: {len(path_ok)}/{len(path_rows)} successful
- Path feature rows: {len(path_feature_rows)}
- Attention QK/OV decomposition rows: {len(decomposition_rows)}
- Attention source-position causal rows: {len(attn_ok)}/{len(attention_causal_rows)} successful
- Model: `{MODEL_NAME}`

## Multi-layer pathways

The path table is `multilayer_path_tests.csv`. Each path is a sequence of graph nodes from an input-token side of the graph through one or more sparse feature/error nodes to the answer-logit node.

Across successful paths:

- Mean path gold-answer damage after zeroing: {float(np.mean(path_damage)) if path_damage else None:.4f}
- Mean path margin change after doubling feature activations: {float(np.mean(path_gain)) if path_gain else None:.4f}

The path intervention is stronger than a one-node test because all feature nodes on a graph path are intervened on together. The path feature table gives the exact layer, token position, and feature index for every node.

## Attention decomposition

The attention tables separate four pieces of an attention computation:

- QK score: how strongly the query and source key match.
- Attention weight: the normalized routing weight.
- OV projection: the source value after the head's output projection, projected onto the correct-answer-versus-foil direction.
- Source-position causal damage: the output change when that one source position is removed from the selected head's attention pattern.

The raw decomposition is in `attention_qk_ov_decomposition.csv`. The source-position causal table is in `attention_source_position_causal_tests.csv`.

Mean source-position gold damage across successful tests: {float(np.mean(attn_damage)) if attn_damage else None:.4f}.

## Necessity and sufficiency

Necessity is tested by zeroing a graph feature or an entire multi-layer path. A positive gold-damage value means the feature/path was helping the correct answer.

Sufficiency is tested as a dose response: the same graph feature values are set to twice their clean activation, and the answer margin is measured. This is not yet a full transplantation proof; it asks whether increasing the graph-identified signal moves the answer in the graph-predicted direction.

The full paired results are in `multilayer_path_tests.csv`.

## Scientific status

This is now a substantially closer circuit reproduction: it has explicit graph paths, attention source decomposition, component-level interventions, and paired necessity/dose-response tests. The remaining limitation is that sufficiency is still tested by activation scaling within the same prompt rather than by transplanting a complete path into a held-out control. That is the next decisive experiment.
"""
    (results / "complete_circuit_poc_report.md").write_text(report, encoding="utf-8")
    write_json(results / "complete_circuit_poc_summary.json", {
        "model": MODEL_NAME,
        "path_rows": len(path_rows), "successful_path_rows": len(path_ok),
        "attention_decomposition_rows": len(decomposition_rows),
        "attention_causal_rows": len(attention_causal_rows), "successful_attention_causal_rows": len(attn_ok),
        "mean_path_gold_damage": float(np.mean(path_damage)) if path_damage else None,
        "mean_path_boost_gain": float(np.mean(path_gain)) if path_gain else None,
        "mean_attention_source_gold_damage": float(np.mean(attn_damage)) if attn_damage else None,
        "runtime_seconds": time.time() - started,
    })


if __name__ == "__main__":
    main()
