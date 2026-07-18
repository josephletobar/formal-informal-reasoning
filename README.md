# Anthropic-Style Addition Circuit Reproduction

This repository contains a small, reproducible addition-circuit pilot for Gemma 2 2B using the Decoder Research Circuit Tracer tooling.

The experiment is intentionally bounded:

- one validated attribution graph for a simple addition prompt;
- eight direct addition prompts;
- matched explicit, applied, and implicit ABC prompts;
- sparse feature recurrence and cross-context Jaccard overlap;
- a small candidate-versus-random feature suppression test;
- CSV artifacts and one Markdown report.

This is an Anthropic-style reproduction of the analysis pattern, not a reproduction of Claude's weights or Anthropic's exact internal pipeline.

## RunPod

Use a CUDA PyTorch pod with at least 16 GB VRAM and 32 GB system RAM. An L4 with 24 GB VRAM and 62 GB RAM is a suitable low-cost choice.

In the pod:

```bash
git clone https://github.com/josephletobar/informa--formal-reasoning.git
cd informa--formal-reasoning
export HF_TOKEN='your_token_here'
bash setup_runpod.sh
```

Do not commit or write the Hugging Face token into the repository. Results are written to `results/`.

## Audited publication package

The `publication_package/` directory contains the current Nature Machine Intelligence article draft, Word initial-submission file, figures, benchmark, repository-relative derived data, experiment-driver copies, metric ledger and reproducibility checks. Run the lightweight public-release checks from the repository root:

```bash
python publication_package/audit_claims.py
python publication_package/clean_room_smoke.py
python publication_package/verify_publication_package.py
```

The manuscript reports both positive graph/path effects and the null held-out recurrent-feature necessity result. It does not claim that a discrete reasoning module has been established. Model weights, gated credentials and large serialized graph objects are intentionally not committed.

The publication-package audits also run automatically through GitHub Actions on changes to the package.
