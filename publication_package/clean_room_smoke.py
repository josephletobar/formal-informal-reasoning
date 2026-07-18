"""Verify package structure and one serialized graph without running inference."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "publication_package"

benchmark = pd.read_csv(PACKAGE / "benchmark_v1" / "benchmark_v1.csv")
assert len(benchmark) == 360, len(benchmark)
assert benchmark["gold"].between(0, 9).all()
assert (benchmark["kind"] == "direct").sum() == 72

manifest = json.loads((PACKAGE / "benchmark_v1" / "benchmark_manifest.json").read_text(encoding="utf-8"))
assert manifest["rows"] == 360
assert manifest["discovery_direct_rows"] + manifest["confirmation_direct_rows"] == 72

data = PACKAGE / "data"
assert (data / "graph_summaries.csv").exists()
assert len(pd.read_csv(data / "graph_summaries.csv")) == 28
assert len(pd.read_csv(data / "graph_target_parents.csv")) == 2240
assert len(pd.read_csv(data / "recurrent_graph_causal.csv")) == 564
print("serialized graph files are intentionally omitted from the lightweight public release")
print("package smoke passed: benchmark_rows=360 graph_summaries=28 graph_parent_rows=2240 heldout_rows=564")
