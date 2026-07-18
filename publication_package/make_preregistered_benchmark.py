from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).parent / "benchmark_v1"
OUT.mkdir(parents=True, exist_ok=True)

# All answers are small nonnegative integers so the strict next-token objective
# can be calibrated without negative or multi-token answer edge cases.
PAIRS = [(1, 2), (1, 3), (2, 2), (2, 3), (1, 4), (1, 5), (1, 6), (1, 7)]


def render(family: str, form: str, a: int, b: int) -> tuple[str, int]:
    if family == "addition":
        answer = a + b
        if form == "explicit": return f"calc: {a}+{b}=", answer
        if form == "applied": return f"A shelf has {a} books and receives {b} more. How many books now?", answer
        if form == "implicit": return f"Record: start={a}; added={b}; total=?", answer
    if family == "subtraction":
        answer = b - a
        if form == "explicit": return f"calc: {b}-{a}=", answer
        if form == "applied": return f"A shelf has {b} books and sells {a}. How many books remain?", answer
        if form == "implicit": return f"Record: start={b}; removed={a}; remaining=?", answer
    if family == "multiplication":
        answer = a * b
        if form == "explicit": return f"calc: {a}*{b}=", answer
        if form == "applied": return f"There are {a} boxes with {b} items each. How many items total?", answer
        if form == "implicit": return f"Record: groups={a}; each={b}; total=?", answer
    raise ValueError((family, form))


def control_rows(family: str, form: str, a: int, b: int, gold: int):
    same_a = max(0, gold - 1)
    if form == "explicit":
        same_answer_prompt = f"calc: {same_a}+1="
    elif form == "applied":
        same_answer_prompt = f"A shelf has {same_a} books and receives 1 more. How many books now?"
    else:
        same_answer_prompt = f"Record: start={same_a}; added=1; total=?"
    if family == "addition":
        wrong_prompt, wrong = render("subtraction", form, a, b)
        same_answer = gold
        near_prompt, near = render("addition", form, a, b + 1)
    elif family == "subtraction":
        wrong_prompt, wrong = render("addition", form, a, b)
        same_answer = gold
        near_prompt, near = render("subtraction", form, a, b + 1)
    else:
        wrong_prompt, wrong = render("addition", form, a, b)
        same_answer = gold
        near_prompt, near = render("multiplication", form, a, b + 1)
    return [
        ("surface_wrong_computation", wrong_prompt, wrong),
        ("near_miss", near_prompt, near),
        ("same_answer_control", same_answer_prompt, same_answer),
        ("unrelated", "Unrelated record: status=complete; output=?", 0),
    ]


rows = []
for family in ("addition", "subtraction", "multiplication"):
    for example, (a, b) in enumerate(PAIRS):
        split = "discovery" if example < 4 else "confirmation"
        for form in ("explicit", "applied", "implicit"):
            prompt, gold = render(family, form, a, b)
            rows.append({"id": f"{family}_{form}_{example}", "family": family, "form": form, "split": split, "kind": "direct", "a": a, "b": b, "prompt": prompt, "gold": gold})
            for control, control_prompt, control_gold in control_rows(family, form, a, b, gold):
                rows.append({"id": f"{family}_{form}_{example}_{control}", "family": family, "form": form, "split": split, "kind": control, "a": a, "b": b, "prompt": control_prompt, "gold": control_gold})

fields = list(rows[0])
with (OUT / "benchmark_v1.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
with (OUT / "benchmark_v1.jsonl").open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
(OUT / "benchmark_manifest.json").write_text(json.dumps({"families": ["addition", "subtraction", "multiplication"], "forms": ["explicit", "applied", "implicit"], "pairs": PAIRS, "rows": len(rows), "discovery_direct_rows": 36, "confirmation_direct_rows": 36, "tokenization_policy": "exclude any item whose gold answer is not one token before causal analysis"}, indent=2), encoding="utf-8")
print(f"wrote {len(rows)} deterministic benchmark rows")
