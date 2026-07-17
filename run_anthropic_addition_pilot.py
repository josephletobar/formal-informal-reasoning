from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


MODEL_NAME = "google/gemma-2-2b-it"
PAIRS = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 5), (2, 6), (3, 2), (4, 5)]
FORMS = ("A_explicit", "B_applied", "C_implicit")
TOP_K = 16


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def set_state(results: Path, stage: str, **extra) -> None:
    write_json(results / "state.json", {"stage": stage, "timestamp": time.time(), **extra})


def token_id(model, text: str) -> int:
    ids = model.tokenizer(text, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if len(ids) != 1:
        raise ValueError(f"Expected a one-token answer for {text!r}, got {ids!r}")
    return int(ids[0])


def prompt_for(a: int, b: int, form: str) -> str:
    if form == "A_explicit":
        return f"calc: {a}+{b}="
    if form == "B_applied":
        return f"A shelf has {a} books and receives {b} more. How many books now?"
    if form == "C_implicit":
        return f"Record: start={a}; added={b}; total=?"
    raise ValueError(form)


def final_position_logits(logits):
    """Normalize model logits to the vocabulary vector at the final sequence position."""
    if logits.ndim == 3:
        return logits[0, -1]
    if logits.ndim == 2:
        return logits[-1]
    if logits.ndim == 1:
        return logits
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


def extract(model, prompt: str):
    import torch

    with torch.inference_mode():
        logits, acts = model.get_activations(prompt, sparse=False)
        final_acts = acts[:, -1, :].float().cpu().numpy()
        final_logits = final_position_logits(logits).float().cpu().numpy()
    top_idx = np.empty((final_acts.shape[0], TOP_K), dtype=np.int32)
    top_val = np.empty((final_acts.shape[0], TOP_K), dtype=np.float32)
    for layer, row in enumerate(final_acts):
        take = min(TOP_K, row.shape[0])
        idx = np.argpartition(row, -take)[-take:]
        idx = idx[np.argsort(row[idx])[::-1]]
        top_idx[layer, :take] = idx
        top_val[layer, :take] = row[idx]
        if take < TOP_K:
            top_idx[layer, take:] = -1
            top_val[layer, take:] = 0
    del logits, acts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return top_idx, top_val, final_logits, final_acts


def feature_set(top_idx: np.ndarray, top_val: np.ndarray) -> set[tuple[int, int]]:
    return {
        (int(layer), int(feature))
        for layer, row in enumerate(top_idx)
        for feature, value in zip(row, top_val[layer])
        if int(feature) >= 0 and float(value) > 0
    }


def margin(model, logits: np.ndarray, total: int) -> tuple[float, int, int]:
    gold = token_id(model, str(total))
    foil_total = total + 1 if total < 9 else total - 1
    foil = token_id(model, str(foil_total))
    if gold == foil:
        foil = token_id(model, str(total - 1 if total > 1 else total + 2))
    return float(logits[gold] - logits[foil]), gold, foil


def load_model():
    import torch
    from transformers import AutoTokenizer
    from circuit_tracer.replacement_model import ReplacementModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot")
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


def run_graph(model, results: Path, prompt: str, gold: int) -> dict:
    import torch
    from circuit_tracer.attribution import attribute

    try:
        target = torch.tensor([token_id(model, str(gold))], device=model.cfg.device)
        graph = attribute(
            prompt,
            model,
            attribution_targets=target,
            max_n_logits=4,
            desired_logit_prob=0.90,
            batch_size=1,
            max_feature_nodes=256,
            offload="cpu",
            verbose=False,
        )
        graph_path = results / "addition_graph_single.pt"
        graph.to_pt(str(graph_path))
        summary = {
            "status": "ok",
            "prompt": prompt,
            "gold": gold,
            "graph_path": str(graph_path),
            "active_features": int(graph.active_features.shape[0]),
            "selected_features": int(graph.selected_features.shape[0]),
            "adjacency_shape": list(graph.adjacency_matrix.shape),
        }
        del graph
        torch.cuda.empty_cache()
        return summary
    except Exception as exc:
        return {"status": "error", "prompt": prompt, "gold": gold, "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    started = time.time()

    write_json(results / "manifest.json", {
        "model": MODEL_NAME,
        "pairs": PAIRS,
        "forms": list(FORMS),
        "top_k": TOP_K,
        "analysis": "small Anthropic-style addition circuit reproduction",
    })
    set_state(results, "loading_model")

    model = load_model()
    set_state(results, "model_loaded")
    direct: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    behavior_rows: list[dict] = []
    counts: Counter[tuple[int, int]] = Counter()
    sums: defaultdict[tuple[int, int], float] = defaultdict(float)

    for i, (a, b) in enumerate(PAIRS):
        prompt = prompt_for(a, b, "A_explicit")
        idx, values, logits, dense = extract(model, prompt)
        keys = feature_set(idx, values)
        score, gold_id, foil_id = margin(model, logits, a + b)
        direct[(a, b)] = (idx, values, logits, dense)
        for layer, feature in keys:
            counts[(layer, feature)] += 1
            hit = np.flatnonzero(idx[layer] == feature)
            if hit.size:
                sums[(layer, feature)] += float(values[layer, hit[0]])
        behavior_rows.append({
            "a": a, "b": b, "form": "A_explicit", "prompt": prompt,
            "correct_vs_foil": int(score >= 0),
            "gold_minus_foil": score,
            "gold_token_id": gold_id, "foil_token_id": foil_id,
            "top_feature_count": len(keys),
        })
        set_state(results, "direct", completed=i + 1, total=len(PAIRS))

    candidate_keys = [key for key, support in counts.most_common() if support >= 2][:8]
    candidates = [{
        "layer": layer,
        "feature": feature,
        "feature_id": f"L{layer}_F{feature}",
        "support": counts[(layer, feature)],
        "support_rate": counts[(layer, feature)] / len(PAIRS),
        "mean_activation": sums[(layer, feature)] / counts[(layer, feature)],
    } for layer, feature in candidate_keys]
    write_csv(results / "feature_candidates.csv", candidates)
    write_json(results / "frozen_candidates.json", candidates)
    write_csv(results / "behavior.csv", behavior_rows)

    representation_rows: list[dict] = []
    dense_by_form: dict[str, list[np.ndarray]] = {form: [] for form in FORMS}
    for i, (a, b) in enumerate(PAIRS):
        akeys = feature_set(direct[(a, b)][0], direct[(a, b)][1])
        dense_by_form["A_explicit"].append(direct[(a, b)][3])
        for form in ("B_applied", "C_implicit"):
            prompt = prompt_for(a, b, form)
            idx, values, logits, dense = extract(model, prompt)
            keys = feature_set(idx, values)
            dense_by_form[form].append(dense)
            score, gold_id, foil_id = margin(model, logits, a + b)
            a_dense = direct[(a, b)][3]
            a_flat = a_dense.reshape(-1).astype(np.float32)
            b_flat = dense.reshape(-1).astype(np.float32)
            dense_cosine = float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8))
            union = len(akeys | keys)
            representation_rows.append({
                "a": a, "b": b, "form": form, "prompt": prompt,
                "correct_vs_foil": int(score >= 0),
                "gold_minus_foil": score,
                "same_pair_jaccard": len(akeys & keys) / union if union else 0.0,
                "dense_activation_cosine": dense_cosine,
                "same_pair_overlap": len(akeys & keys),
                "gold_token_id": gold_id, "foil_token_id": foil_id,
            })
        set_state(results, "abc_representation", completed=i + 1, total=len(PAIRS))
    write_csv(results / "representation.csv", representation_rows)

    # Compare the low-dimensional subspaces spanned by dense feature
    # activations across the eight matched prompts. This is a representation
    # test, not a causal result.
    def subspace_affinity(left: list[np.ndarray], right: list[np.ndarray]) -> float:
        left_matrix = np.stack([item.reshape(-1).astype(np.float32) for item in left])
        right_matrix = np.stack([item.reshape(-1).astype(np.float32) for item in right])
        left_matrix -= left_matrix.mean(axis=0, keepdims=True)
        right_matrix -= right_matrix.mean(axis=0, keepdims=True)
        rank = max(1, min(4, left_matrix.shape[0] - 1, right_matrix.shape[0] - 1))
        left_basis = np.linalg.svd(left_matrix, full_matrices=False)[2][:rank].T
        right_basis = np.linalg.svd(right_matrix, full_matrices=False)[2][:rank].T
        singular_values = np.linalg.svd(left_basis.T @ right_basis, compute_uv=False)
        return float(np.mean(singular_values))

    subspace_rows = [
        {"comparison": "A_vs_B", "subspace_affinity": subspace_affinity(dense_by_form["A_explicit"], dense_by_form["B_applied"])},
        {"comparison": "A_vs_C", "subspace_affinity": subspace_affinity(dense_by_form["A_explicit"], dense_by_form["C_implicit"])},
    ]
    write_csv(results / "subspace_similarity.csv", subspace_rows)
    write_csv(results / "activation_similarity.csv", representation_rows)

    causal_rows: list[dict] = []
    rng = random.Random(7)
    for a, b in PAIRS[:4]:
        prompt = prompt_for(a, b, "A_explicit")
        idx, values, logits, _ = direct[(a, b)]
        base_score, _, _ = margin(model, logits, a + b)
        tokens = model.ensure_tokenized(prompt)
        position = int(tokens.shape[-1] - 1)
        active = [key for key in candidate_keys if key in feature_set(idx, values)]
        for layer, feature in active[:4]:
            random_feature = rng.randrange(int(model.transcoders.d_transcoder))
            for mode, intervention_feature in (("candidate_zero", feature), ("random_zero", random_feature)):
                try:
                    new_logits, _ = model.feature_intervention(
                        prompt,
                        [(layer, position, intervention_feature, 0.0)],
                        sparse=True,
                        return_activations=False,
                    )
                    new_score, _, _ = margin(model, final_position_logits(new_logits).float().cpu().numpy(), a + b)
                    causal_rows.append({
                        "a": a, "b": b, "prompt": prompt,
                        "layer": layer, "candidate_feature": feature,
                        "intervention_feature": intervention_feature,
                        "mode": mode, "base_gold_minus_foil": base_score,
                        "patched_gold_minus_foil": new_score,
                        "effect": new_score - base_score, "status": "ok",
                    })
                    del new_logits
                except Exception as exc:
                    causal_rows.append({
                        "a": a, "b": b, "prompt": prompt,
                        "layer": layer, "candidate_feature": feature,
                        "intervention_feature": intervention_feature,
                        "mode": mode, "status": "error", "error": repr(exc),
                    })
    write_csv(results / "causal.csv", causal_rows)
    set_state(results, "causal_complete", rows=len(causal_rows))

    graph_summary = run_graph(model, results, prompt_for(3, 4, "A_explicit"), 7)
    write_json(results / "graph_summary.json", graph_summary)
    set_state(results, "graph_complete", status=graph_summary.get("status"))

    def mean(rows, field):
        values = [float(row[field]) for row in rows if row.get("status", "ok") == "ok" and field in row]
        return float(np.mean(values)) if values else None

    candidate_effects = [row for row in causal_rows if row.get("mode") == "candidate_zero"]
    random_effects = [row for row in causal_rows if row.get("mode") == "random_zero"]
    report = f"""# Small Anthropic-Style Addition Circuit Reproduction

## Scope

Gemma 2 2B was analyzed with the Circuit Tracer replacement model and Gemma transcoder set. The run used {len(PAIRS)} direct addition prompts, matched applied and implicit forms, sparse feature recurrence, candidate-versus-random suppression, and one attribution graph.

This is a reproduction of the analysis pattern, not Claude or Anthropic's exact weights.

## Results

- Direct behavioral accuracy versus a nearby foil: {mean(behavior_rows, "correct_vs_foil")}
- ABC behavioral accuracy: {mean(behavior_rows + representation_rows, "correct_vs_foil")}
- Same-pair A-to-B/C sparse-feature Jaccard: {mean(representation_rows, "same_pair_jaccard")}
- Same-pair dense-feature activation cosine: {mean(representation_rows, "dense_activation_cosine")}
- A-vs-B/C subspace affinity: {subspace_rows}
- Candidate suppression mean effect: {mean(candidate_effects, "effect")}
- Random suppression mean effect: {mean(random_effects, "effect")}
- Causal rows: {len(causal_rows)}
- Attribution graph status: **{graph_summary.get("status")}**

## Interpretation

Evidence for a reusable computation requires behavioral reliability, above-control cross-context feature recurrence, and candidate suppression stronger than random suppression. Because this is a small pilot, any positive result is preliminary and does not establish a discrete reasoning module.

## Artifacts

- `behavior.csv`
- `feature_candidates.csv`
- `frozen_candidates.json`
- `representation.csv`
- `activation_similarity.csv`
- `subspace_similarity.csv`
- `causal.csv`
- `graph_summary.json`
- `addition_graph_single.pt` when graph construction succeeds
"""
    (results / "report.md").write_text(report, encoding="utf-8")
    write_json(results / "run_metadata.json", {
        "model": MODEL_NAME,
        "runtime_seconds": time.time() - started,
        "candidate_count": len(candidates),
        "causal_rows": len(causal_rows),
        "graph_status": graph_summary.get("status"),
    })
    set_state(results, "complete", runtime_seconds=time.time() - started)


if __name__ == "__main__":
    main()
