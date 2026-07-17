#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

VENV_ROOT="${VENV_ROOT:-/workspace/venv}"
if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV_ROOT"
fi
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

CT_ROOT="${CT_ROOT:-/workspace/circuit-tracer}"
if [[ ! -d "$CT_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/decoderesearch/circuit-tracer.git "$CT_ROOT"
fi
python -m pip install -e "$CT_ROOT"
python -m py_compile run_anthropic_addition_pilot.py
python run_anthropic_addition_pilot.py --results results
