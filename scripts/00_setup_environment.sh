#!/bin/bash
# Setup the project environment (macOS / Apple Silicon).
#
# Creates a local virtualenv and installs minimal dependencies.
# RL training uses MLX (Apple's framework); inference uses Ollama models.
# vLLM/CUDA-based stacks from the paper's cluster setup do not run on macOS.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Creating virtual environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -q -r requirements.txt

echo ""
echo "Checking Ollama..."
if command -v ollama >/dev/null 2>&1; then
    echo "✓ ollama found"
else
    echo "✗ ollama not found — install from https://ollama.ai"
fi

echo ""
echo "Checking Prover9 (optional, needed for proof-level verification)..."
if command -v prover9 >/dev/null 2>&1; then
    echo "✓ prover9 found"
else
    echo "✗ prover9 not found — see scripts/02_setup_prover9.sh"
fi

echo ""
echo "✓ Setup complete. Activate with: source .venv/bin/activate"
