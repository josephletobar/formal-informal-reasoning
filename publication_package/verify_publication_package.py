from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
PKG = ROOT / "publication_package"
DATA = PKG / "data"


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    mean = float(x.mean())
    h = 1.96 * float(x.std(ddof=1)) / math.sqrt(len(x)) if len(x) > 1 else 0.0
    return mean, mean - h, mean + h


def close(actual, expected, tol=1e-6):
    if not np.isclose(actual, expected, atol=tol, rtol=0):
        raise AssertionError(f"value mismatch: actual={actual} expected={expected}")


def main():
    required = [
        PKG / "NMI_ARTICLE_DRAFT.md",
        PKG / "REPRODUCIBILITY_MANIFEST.md",
        PKG / "SUBMISSION_CHECKLIST.md",
        PKG / "environment-lock.txt",
        PKG / "tables/metric_ledger.csv",
    ]
    required += list((PKG / "figures").glob("fig*.png"))
    if any(not p.exists() for p in required):
        raise AssertionError(f"missing package artifacts: {[str(p) for p in required if not p.exists()]}")

    graph = pd.read_csv(DATA / "graph_summaries.csv")
    if len(graph) != 28 or not (graph["status"] == "ok").all():
        raise AssertionError("graph panel is not 28/28 successful")
    j = pd.read_csv(DATA / "graph_pairwise_jaccard.csv")
    same = j.loc[j["same_form"] == 1, "jaccard"]
    different = j.loc[j["same_form"] == 0, "jaccard"]
    close(float(same.mean()), 0.6104398438213321)
    close(float(different.mean()), 0.10668925041353705)

    heldout = pd.read_csv(DATA / "recurrent_graph_causal.csv")
    if len(heldout) != 564 or not (heldout["status"] == "ok").all():
        raise AssertionError("held-out causal panel is not 564/564 successful")
    recurrent = heldout.loc[(heldout["kind"] == "recurrent") & (heldout["mode"] == "zero"), "damage"]
    control = heldout.loc[(heldout["kind"] == "random_control") & (heldout["mode"] == "zero"), "damage"]
    close(float(recurrent.mean()), -0.012219070542788675)
    close(float(control.mean()), 0.009215283901133436)

    paths = pd.read_csv(DATA / "multilayer_path_tests.csv")
    source = pd.read_csv(DATA / "attention_source_position_causal_tests.csv")
    if len(paths) != 64 or not (paths["status"] == "ok").all():
        raise AssertionError("path panel is not 64/64 successful")
    if len(source) != 128 or not (source["status"] == "ok").all():
        raise AssertionError("attention source panel is not 128/128 successful")
    close(float(paths["path_gold_damage"].mean()), 0.6602783203125)
    close(float(source["source_gold_damage"].mean()), 0.20648193359375)

    transfer = pd.read_csv(DATA / "cross_form_feature_transfer.csv")
    if len(transfer) != 240 or not (transfer["status"] == "ok").all():
        raise AssertionError("cross-form transfer panel is not 240/240 successful")
    applied = transfer[(transfer["target_form"] == "applied") & (transfer["mode"] == "source_transfer")]["damage"]
    implicit = transfer[(transfer["target_form"] == "implicit") & (transfer["mode"] == "source_transfer")]["damage"]
    close(float(applied.mean()), -0.0075893402099609375)
    close(float(implicit.mean()), -0.005289713541666667)

    draft = (PKG / "NMI_ARTICLE_DRAFT.md").read_text(encoding="utf-8")
    abstract = draft[draft.index("## Abstract"):draft.index("## Introduction")]
    main = draft[draft.index("## Introduction"):draft.index("## Methods")]
    if len(abstract.split()) > 150:
        raise AssertionError(f"abstract exceeds 150 words: {len(abstract.split())}")
    if len(main.split()) > 3500:
        raise AssertionError(f"main text exceeds 3500 words: {len(main.split())}")

    print("publication package verification passed")
    print(f"graphs={len(graph)} heldout_rows={len(heldout)} path_rows={len(paths)} source_rows={len(source)} transfer_rows={len(transfer)}")
    print(f"abstract_words={len(abstract.split())} main_text_words={len(main.split())}")


if __name__ == "__main__":
    main()
