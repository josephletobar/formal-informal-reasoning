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
