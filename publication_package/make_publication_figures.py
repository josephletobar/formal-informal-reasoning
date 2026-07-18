from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
DATA = ROOT / "publication_package" / "data"
OUT = Path(__file__).parent
FIG = OUT / "figures"
TABLE = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TABLE.mkdir(parents=True, exist_ok=True)


def mean_ci(values):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean()) if len(x) else 0.0
    h = 1.96 * float(x.std(ddof=1)) / math.sqrt(len(x)) if len(x) > 1 else 0.0
    return mean, mean - h, mean + h


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# Figure 1: graph recurrence versus matched form controls.
panel = DATA
j = pd.read_csv(panel / "graph_pairwise_jaccard.csv")
same = j.loc[j["same_form"] == 1, "jaccard"].to_numpy()
different = j.loc[j["same_form"] == 0, "jaccard"].to_numpy()
rows = []
for label, x in [("same form", same), ("different form", different)]:
    m, lo, hi = mean_ci(x)
    rows.append({"comparison": label, "n": len(x), "mean_jaccard": m, "ci_low": lo, "ci_high": hi})
pd.DataFrame(rows).to_csv(TABLE / "graph_overlap_summary.csv", index=False)
fig, ax = plt.subplots(figsize=(4.3, 3.1))
means = [r["mean_jaccard"] for r in rows]
errors = [[r["mean_jaccard"] - r["ci_low"] for r in rows], [r["ci_high"] - r["mean_jaccard"] for r in rows]]
ax.bar([0, 1], means, yerr=errors, color=["#2f6f9f", "#b8c7d1"], capsize=4, width=0.58)
ax.set_xticks([0, 1], ["same prompt\nform", "different\nprompt form"])
ax.set_ylabel("Top-parent Jaccard overlap")
ax.set_ylim(0, 0.75)
ax.set_title("Attribution graphs recur within matched forms")
save(fig, "fig1_graph_recurrence")

# Figure 2: full circuit proof-of-concept and held-out causal check.
v18 = DATA
v17 = DATA
path = pd.read_csv(v18 / "multilayer_path_tests.csv")
attn = pd.read_csv(v18 / "attention_source_position_causal_tests.csv")
causal = pd.read_csv(v17 / "recurrent_graph_causal.csv")
entries = []
for label, x in [("path zero", path["path_gold_damage"]), ("source-position\nremoval", attn["source_gold_damage"]),
                 ("recurrent feature\nzero", causal.loc[causal["kind"] == "recurrent", "damage"]),
                 ("active control\nzero", causal.loc[causal["kind"] == "random_control", "damage"])]:
    m, lo, hi = mean_ci(x)
    entries.append({"condition": label, "n": len(x), "mean_damage": m, "ci_low": lo, "ci_high": hi})
pd.DataFrame(entries).to_csv(TABLE / "causal_summary.csv", index=False)
fig, ax = plt.subplots(figsize=(5.8, 3.3))
means = [r["mean_damage"] for r in entries]
errs = [[r["mean_damage"] - r["ci_low"] for r in entries], [r["ci_high"] - r["mean_damage"] for r in entries]]
ax.axhline(0, color="#555", lw=0.8)
ax.bar(range(4), means, yerr=errs, capsize=3, color=["#2f6f9f", "#4f8f72", "#bf7c52", "#b8c7d1"], width=0.62)
ax.set_xticks(range(4), [r["condition"] for r in entries])
ax.set_ylabel("Gold-minus-foil damage")
ax.set_title("Path influence is positive; held-out feature necessity is not")
save(fig, "fig2_causal_audit")

# Figure 3: larger-model addition residual transfer.
model_rows = []
for label, filename in [("Qwen3-8B", "qwen3_residual_patching_corrected.csv"), ("Gemma2-9B", "gemma9_residual_patching_corrected.csv")]:
    d = pd.read_csv(DATA / filename)
    d = d[d["family"].str.lower() == "addition"]
    for kind, g in d.groupby("source_kind"):
        m, lo, hi = mean_ci(g["patch_effect"])
        model_rows.append({"model": label, "source_kind": kind, "n": len(g), "mean_effect": m, "ci_low": lo, "ci_high": hi})
models = pd.DataFrame(model_rows)
models.to_csv(TABLE / "larger_model_addition_transfer.csv", index=False)
order = ["same_computation", "surface_wrong_computation", "unrelated"]
fig, ax = plt.subplots(figsize=(6.3, 3.4))
x = np.arange(len(order)); width = 0.35
for i, model in enumerate(["Qwen3-8B", "Gemma2-9B"]):
    sub = models[models["model"] == model].set_index("source_kind").reindex(order)
    means = sub["mean_effect"].to_numpy()
    errs = np.vstack([means - sub["ci_low"].to_numpy(), sub["ci_high"].to_numpy() - means])
    ax.bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=3, label=model)
ax.axhline(0, color="#555", lw=0.8)
ax.set_xticks(x, ["same\ncomputation", "surface / wrong\ncomputation", "unrelated"])
ax.set_ylabel("Residual patch effect")
ax.set_title("Larger models: addition source transfer")
ax.legend(frameon=False)
save(fig, "fig3_larger_model_transfer")

# Figure 4: Gemma 9B layer distribution for addition residual patching.
g = pd.read_csv(DATA / "gemma9_residual_patching_corrected.csv")
g = g[(g["family"].str.lower() == "addition") & (g["source_kind"] == "same_computation")]
layer_rows = []
for layer, x in g.groupby("layer"):
    m, lo, hi = mean_ci(x["patch_effect"])
    layer_rows.append({"layer": int(layer), "n": len(x), "mean_effect": m, "ci_low": lo, "ci_high": hi})
layer_rows = sorted(layer_rows, key=lambda r: r["layer"])
pd.DataFrame(layer_rows).to_csv(TABLE / "gemma9_addition_layer_effects.csv", index=False)
fig, ax = plt.subplots(figsize=(5.5, 3.2))
xx = np.arange(len(layer_rows)); mm = np.array([r["mean_effect"] for r in layer_rows])
ee = np.vstack([[r["mean_effect"] - r["ci_low"] for r in layer_rows], [r["ci_high"] - r["mean_effect"] for r in layer_rows]])
ax.axhline(0, color="#555", lw=0.8)
ax.errorbar(xx, mm, yerr=ee, fmt="o-", color="#2f6f9f", capsize=3)
ax.set_xticks(xx, [str(r["layer"]) for r in layer_rows])
ax.set_xlabel("Selected transformer layer")
ax.set_ylabel("Same-computation patch effect")
ax.set_title("Gemma 2 9B addition transfer is distributed")
save(fig, "fig4_gemma9_layer_effect")

print("publication figures and tables complete")
