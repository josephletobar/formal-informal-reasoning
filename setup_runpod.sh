#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

CT_ROOT="${CT_ROOT:-/workspace/circuit-tracer}"
if [[ ! -d "$CT_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/decoderesearch/circuit-tracer.git "$CT_ROOT"
fi
python -m pip install -e "$CT_ROOT"

python run_anthropic_addition_pilot.py --results results
