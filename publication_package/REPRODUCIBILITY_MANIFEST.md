# Reproducibility Manifest

## Current status

This package is the lightweight public release of the executed experiments used in the article draft. It includes repository-relative derived data, the deterministic benchmark, principal driver scripts, an observed environment lock, SHA256 hashes and package-level checks. It intentionally does not redistribute model weights, gated credentials or large serialized graph objects.

## Primary evidence

| Evidence | Raw artifact | Report |
|---|---|---|
| ABC attribution graph panel | `data/graph_target_parents.csv`, `data/graph_summaries.csv`, `data/graph_pairwise_jaccard.csv` | `NMI_ARTICLE_DRAFT.md` and metric ledger |
| Held-out recurrent-feature causality | `data/recurrent_graph_causal.csv` | `NMI_ARTICLE_DRAFT.md` and metric ledger |
| Multilayer paths and source heads | `data/multilayer_path_tests.csv`, `data/attention_qk_ov_decomposition.csv`, `data/attention_source_position_causal_tests.csv` | `NMI_ARTICLE_DRAFT.md` and metric ledger |
| Same-answer transfer | `data/same_answer_feature_transfer.csv`, `data/same_answer_chain_transfer.csv` | `NMI_ARTICLE_DRAFT.md` and metric ledger |
| Cross-form feature transfer | `data/cross_form_feature_transfer.csv` | `NMI_ARTICLE_DRAFT.md` and metric ledger |
| Qwen3 8B larger-model screen | `data/qwen3_residual_patching_corrected.csv` | `NMI_ARTICLE_DRAFT.md` and metric ledger |
| Gemma 2 9B larger-model screen | `data/gemma9_residual_patching_corrected.csv` | `NMI_ARTICLE_DRAFT.md` and metric ledger |

## Figure generation

Run from the project root:

```powershell
python publication_package\make_publication_figures.py
```

The script reads the raw CSVs above and writes PNG/SVG figures and CSV summary tables under `publication_package/figures` and `publication_package/tables`.

## Primary run commands

The primary Gemma 2B runs were executed in the original remote workspace with:

```bash
export HF_HOME="${HF_HOME:-huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
python run_full_abc_attribution_graphs.py
python run_operation_specific_sufficiency_graphs.py
python run_recurrent_graph_causal_v17c.py
python source_scripts/run_complete_circuit_poc.py --panel-dir research_v16_abc_graph_panel --source-dir component_analysis_full --results research_v18_full_path_component --top-paths 8
python source_scripts/run_same_answer_path_transfer.py --results research_v19_same_answer_transfer --recurrence research_v16_abc_graph_panel/recurring_graph_features.csv
python run_cross_form_feature_transfer_v20c.py
```

The raw outputs from those runs are copied into `data/` for this public release. The lightweight release does not include model weights, gated credentials or serialized `.pt` graph objects; the local archive retains the serialized graphs separately.

## Model and transcoder requirements

- Primary model: `google/gemma-2-2b-it`.
- Primary sparse representation: Gemma Scope replacement-model/transcoder snapshots for layers 0-25.
- Larger residual screens: `Qwen/Qwen3-8B` and `google/gemma-2-9b-it`.
- Precision: FP16 for reported larger-model screens; no quantization.
- Hardware used: RunPod NVIDIA L4, 23,034 MiB VRAM.
- The model cache was kept outside the project code and was not copied into this package.

## Verification checks

The following checks were completed before writing the draft:

- 28/28 attribution graphs succeeded.
- 564/564 corrected held-out causal rows succeeded.
- 64/64 multilayer path rows succeeded.
- 128/128 attention source-position rows succeeded.
- 240/240 cross-form feature-transfer rows succeeded.
- Figure generation completed and all four figures were visually inspected.
- Headline-claim audit recomputed the manuscript's graph-overlap, path, source-position, held-out and transfer means from raw CSVs.
- Principal experiment driver scripts are included under `source_scripts/` and included in the artifact hash manifest.
- The Word manuscript passes structural auditing. Visual PDF rendering was attempted but could not run because `soffice` is not installed and the renderer's Windows scratch directory is permission-blocked.

Run the package checks from the project root:

```powershell
python publication_package\clean_room_smoke.py
python publication_package\audit_claims.py
python publication_package\verify_publication_package.py
```

## Current release blockers

1. Convert the observed `environment-lock.txt` into a tested install file and pin every dependency, including plotting and transformer-lens packages.
2. Add SHA256 hashes for the exact model and transcoder snapshots; the current hash manifest covers the local tabular and graph-analysis inputs.
3. Publish the full serialized graph objects and exact model/transcoder snapshot hashes if a byte-for-byte graph rerun is required.
4. Run the smoke test in a genuinely fresh environment and add a one-command end-to-end rerun for one inference/intervention.
5. Add an explicit code/data license to the public repository.
6. Run a larger preregistered held-out replication before making a strong modularity claim.
