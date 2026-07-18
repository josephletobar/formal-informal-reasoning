"""Recompute headline manuscript values from the retained raw CSV files."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "publication_package" / "data"


def mean(path: str, column: str, **filters: object) -> tuple[int, float]:
    data = pd.read_csv(DATA / Path(path).name)
    for key, value in filters.items():
        data = data[data[key] == value]
    values = pd.to_numeric(data[column], errors="raise")
    return len(values), float(values.mean())


# The graph table contains 80 retained parents per graph. Recompute the two
# overlap groups used in the manuscript rather than trusting a summary file.
parents = pd.read_csv(DATA / "graph_target_parents.csv")
addition = parents[parents["family"] == "addition"]
sets = {
    graph_id: set(group["feature_id"])
    for graph_id, group in addition.groupby("graph_id")
}
meta = addition.groupby("graph_id")["form"].first().to_dict()
same = []
different = []
for left, right in combinations(sorted(sets), 2):
    score = len(sets[left] & sets[right]) / len(sets[left] | sets[right])
    if meta[left] == meta[right]:
        same.append(score)
    else:
        different.append(score)
assert len(same) == 84, len(same)
assert len(different) == 192, len(different)
assert abs(np.mean(same) - 0.6104398438213321) < 1e-10
assert abs(np.mean(different) - 0.10668925041353705) < 1e-10

checks = [
    ("path_zero", "multilayer_path_tests.csv", "path_gold_damage", {"status": "ok"}, 64, 0.6602783203125),
    ("source_position", "attention_source_position_causal_tests.csv", "source_gold_damage", {"status": "ok"}, 128, 0.20648193359375),
    ("heldout_recurrent_zero", "recurrent_graph_causal.csv", "damage", {"status": "ok", "kind": "recurrent", "mode": "zero"}, 141, -0.012219070542788675),
    ("heldout_active_zero", "recurrent_graph_causal.csv", "damage", {"status": "ok", "kind": "random_control", "mode": "zero"}, 141, 0.009215283901133436),
    ("same_answer_chain_zero", "same_answer_chain_transfer.csv", "chain_necessity_damage", {"status": "ok"}, 4, -0.541015625),
    ("same_answer_chain_transfer", "same_answer_chain_transfer.csv", "chain_transfer_gain", {"status": "ok"}, 4, -0.53515625),
    ("cross_form_applied", "cross_form_feature_transfer.csv", "damage", {"status": "ok", "target_form": "applied", "mode": "source_transfer"}, 24, -0.0075893402099609375),
    ("cross_form_implicit", "cross_form_feature_transfer.csv", "damage", {"status": "ok", "target_form": "implicit", "mode": "source_transfer"}, 24, -0.005289713541666667),
    ("cross_form_subtraction", "cross_form_feature_transfer.csv", "damage", {"status": "ok", "target_form": "subtraction", "mode": "source_transfer"}, 24, 0.0026041666666666665),
    ("cross_form_multiplication", "cross_form_feature_transfer.csv", "damage", {"status": "ok", "target_form": "multiplication", "mode": "source_transfer"}, 24, -0.0029296875),
]

for label, path, column, filters, expected_n, expected_mean in checks:
    n, observed = mean(path, column, **filters)
    assert n == expected_n, (label, n, expected_n)
    assert abs(observed - expected_mean) < 1e-10, (label, observed, expected_mean)

print("headline claim audit passed")
print(f"graph_overlap_same_form n={len(same)} mean={np.mean(same):.12f}")
print(f"graph_overlap_different_form n={len(different)} mean={np.mean(different):.12f}")
for label, _, _, _, n, expected in checks:
    print(f"{label} n={n} mean={expected:.12f}")
