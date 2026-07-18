from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_graph_edge_poc import load_model, margin, token_id, write_csv, write_json


PAIRS = [(1, 2, 2, 1), (2, 3, 3, 2), (3, 4, 4, 3), (4, 5, 5, 4)]


def select_features(recurrence_path: Path, minimum_count: int = 3, max_features: int = 6) -> list[dict]:
    rows = list(csv.DictReader(recurrence_path.open(encoding="utf-8")))
    rows = [row for row in rows if int(row["graph_count"]) >= minimum_count]
    rows.sort(key=lambda row: (-int(row["graph_count"]), int(row["layer"]), int(row["feature"])))
    return rows[:max_features]


def prompt(a: int, b: int) -> str:
    return f"calc: {a}+{b}="


def run(model, results: Path, recurrence_path: Path) -> None:
    started = time.time()
    features = select_features(recurrence_path)
    write_csv(results / "selected_transfer_features.csv", features)
    rows = []
    path_rows = []
    for source_a, source_b, target_a, target_b in PAIRS:
        source_prompt = prompt(source_a, source_b)
        target_prompt = prompt(target_a, target_b)
        total = source_a + source_b
        source_tokens = model.ensure_tokenized(source_prompt)
        target_tokens = model.ensure_tokenized(target_prompt)
        source_position = int(source_tokens.shape[-1] - 1)
        target_position = int(target_tokens.shape[-1] - 1)
        with torch.inference_mode():
            source_logits, source_acts = model.get_activations(source_prompt, sparse=False)
            target_logits, target_acts = model.get_activations(target_prompt, sparse=False)
        gold = token_id(model, str(total))
        foil_total = total + 1 if total < 9 else total - 1
        foil = token_id(model, str(foil_total))
        base = margin(model, target_logits, gold, foil)
        specs = [(int(row["layer"]), target_position, int(row["feature"])) for row in features]
        zero_values = [(layer, pos, feature, 0.0) for layer, pos, feature in specs]
        transferred_values = [(layer, pos, feature, float(source_acts[layer, source_position, feature].item())) for layer, pos, feature in specs]
        with torch.inference_mode():
            zero_logits, _ = model.feature_intervention(target_prompt, zero_values, sparse=True, return_activations=False)
            transferred_logits, _ = model.feature_intervention(target_prompt, transferred_values, sparse=True, return_activations=False)
        zero_margin = margin(model, zero_logits, gold, foil)
        transferred_margin = margin(model, transferred_logits, gold, foil)
        corruption = base - zero_margin
        path_rows.append({
            "source_prompt": source_prompt, "target_prompt": target_prompt, "total": total,
            "feature_count": len(specs), "baseline_margin": base, "zero_chain_margin": zero_margin,
            "transferred_chain_margin": transferred_margin, "chain_necessity_damage": corruption,
            "chain_transfer_gain": transferred_margin - zero_margin,
            "chain_rescue_fraction": (transferred_margin - zero_margin) / corruption if abs(corruption) > 1e-6 else 0.0,
            "status": "ok",
        })
        for row, (layer, pos, feature) in zip(features, specs):
            source_value = float(source_acts[layer, source_position, feature].item())
            target_value = float(target_acts[layer, target_position, feature].item())
            with torch.inference_mode():
                zero_one_logits, _ = model.feature_intervention(target_prompt, [(layer, pos, feature, 0.0)], sparse=True, return_activations=False)
                transfer_one_logits, _ = model.feature_intervention(target_prompt, [(layer, pos, feature, source_value)], sparse=True, return_activations=False)
            zero_one = margin(model, zero_one_logits, gold, foil)
            transfer_one = margin(model, transfer_one_logits, gold, foil)
            damage = base - zero_one
            rows.append({
                "source_prompt": source_prompt, "target_prompt": target_prompt, "total": total,
                "feature_id": row["feature_id"], "layer": layer, "feature": feature,
                "source_activation": source_value, "target_activation": target_value,
                "baseline_margin": base, "zero_margin": zero_one, "transferred_margin": transfer_one,
                "necessity_damage": damage, "transfer_gain": transfer_one - zero_one,
                "rescue_fraction": (transfer_one - zero_one) / damage if abs(damage) > 1e-6 else 0.0,
                "status": "ok",
            })
        del source_logits, source_acts, target_logits, target_acts
        torch.cuda.empty_cache()
    write_csv(results / "same_answer_feature_transfer.csv", rows)
    write_csv(results / "same_answer_chain_transfer.csv", path_rows)
    ok = rows
    path_ok = path_rows
    report = f"""# Same-Answer Circuit Necessity and Sufficiency POC

## Protocol

This run tests whether recurring graph features can be transferred between commutative addition prompts with the same answer:

- `1+2` -> `2+1`
- `2+3` -> `3+2`
- `3+4` -> `4+3`
- `4+5` -> `5+4`

For each target, recurring graph features are tested in three states:

1. Clean target activation.
2. Feature set to zero, testing necessity.
3. Feature replaced by the source prompt's activation, testing transfer/sufficiency.

The same is repeated with the recurring features patched as a small multilayer chain.

## Results

- Individual feature tests: {len(ok)}
- Multilayer chain tests: {len(path_ok)}
- Mean individual necessity damage: {float(np.mean([float(row['necessity_damage']) for row in ok])) if ok else None:.4f}
- Mean individual transfer rescue fraction: {float(np.mean([float(row['rescue_fraction']) for row in ok])) if ok else None:.4f}
- Mean chain necessity damage: {float(np.mean([float(row['chain_necessity_damage']) for row in path_ok])) if path_ok else None:.4f}
- Mean chain transfer rescue fraction: {float(np.mean([float(row['chain_rescue_fraction']) for row in path_ok])) if path_ok else None:.4f}

Raw individual tests are in `same_answer_feature_transfer.csv`. Raw chain tests are in `same_answer_chain_transfer.csv`.

## Interpretation

This is a stronger sufficiency test than simply doubling an activation. It asks whether a feature or small feature chain selected from one addition example can be placed into a different-looking but answer-equivalent addition example after corruption.

Positive rescue means the source activation moved the corrupted target back toward its clean margin. The test is still small and uses commutative arithmetic rather than unrelated controls, but it directly evaluates transfer of a graph-identified computation.
"""
    (results / "same_answer_transfer_report.md").write_text(report, encoding="utf-8")
    write_json(results / "same_answer_transfer_summary.json", {"features": features, "individual_rows": len(rows), "chain_rows": len(path_rows), "mean_necessity_damage": float(np.mean([float(row["necessity_damage"]) for row in rows])) if rows else None, "mean_rescue_fraction": float(np.mean([float(row["rescue_fraction"]) for row in rows])) if rows else None, "mean_chain_necessity_damage": float(np.mean([float(row["chain_necessity_damage"]) for row in path_rows])) if path_rows else None, "mean_chain_rescue_fraction": float(np.mean([float(row["chain_rescue_fraction"]) for row in path_rows])) if path_rows else None, "runtime_seconds": time.time() - started})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="same_answer_transfer")
    parser.add_argument("--recurrence", default="addition_graph_panel/graph_panel_recurrence.csv")
    args = parser.parse_args()
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    run(load_model(), results, Path(args.recurrence))


if __name__ == "__main__":
    main()
