"""Remove private runtime paths from copied CSV metadata."""

from __future__ import annotations

import csv
from pathlib import Path


path = Path(__file__).resolve().parent / "data" / "graph_summaries.csv"
rows = []
with path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames or []
    for row in reader:
        if row.get("graph_path"):
            row["graph_path"] = "local_graph_archive/" + Path(row["graph_path"]).name
        rows.append(row)

with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"sanitized {path}")
