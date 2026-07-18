"""Create SHA256 hashes for the local publication package's key inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "tables" / "artifact_hashes.json"

FILES = [
    Path("publication_package/NMI_ARTICLE_DRAFT.md"),
    Path("publication_package/PREREGISTERED_REPLICATION_PROTOCOL.md"),
    Path("publication_package/benchmark_v1/benchmark_v1.csv"),
    Path("publication_package/benchmark_v1/benchmark_v1.jsonl"),
    Path("publication_package/benchmark_v1/benchmark_manifest.json"),
    Path("publication_package/tables/metric_ledger.csv"),
    Path("publication_package/data/graph_target_parents.csv"),
    Path("publication_package/data/recurring_graph_features.csv"),
    Path("publication_package/data/recurrent_graph_causal.csv"),
    Path("publication_package/data/multilayer_path_tests.csv"),
    Path("publication_package/data/attention_source_position_causal_tests.csv"),
    Path("publication_package/data/same_answer_chain_transfer.csv"),
    Path("publication_package/data/cross_form_feature_transfer.csv"),
]

FILES.extend(Path("publication_package/source_scripts").glob("*.py"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


records = []
for relative in FILES:
    path = ROOT / relative
    records.append({
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    })

OUT.write_text(json.dumps({"algorithm": "sha256", "files": records}, indent=2) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(records)} files)")
