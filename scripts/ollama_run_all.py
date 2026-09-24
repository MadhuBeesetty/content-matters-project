#!/usr/bin/env python3
"""
Run the full local evaluation pipeline:
  1. generate synthetic data (if missing)
  2. baseline evaluation (answer-level, vs gold labels)
  3. prompt-level treatment comparison (inference only — no training)
  4. write docs/RESULTS.md
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"


def run_script(script_name: str, description: str, args: list = None) -> bool:
    print()
    print("=" * 80)
    print(f"STEP: {description}")
    print("=" * 80)
    print()

    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)] + (args or [])
    result = subprocess.run(cmd)

    print()
    print(f"{'✓' if result.returncode == 0 else '✗'} {description}")
    return result.returncode == 0


def main() -> bool:
    print()
    print("CONTENT MATTERS PROJECT - LOCAL EVALUATION PIPELINE")
    print("(inference-only; no model training happens in this pipeline)")
    print()

    result = subprocess.run(["which", "ollama"], capture_output=True)
    if result.returncode != 0:
        print("✗ Ollama not found. Install from https://ollama.ai")
        return False
    print("✓ Ollama detected")

    steps = []
    if not (DATA_DIR / "test.json").exists():
        steps.append(("generate_synthetic_data.py", "0. Generate synthetic dataset", []))
    steps += [
        ("ollama_baseline.py", "1. Baseline evaluation", ["50"]),
        ("ollama_treatment_eval.py", "2. Treatment comparison (prompt-level)", ["50"]),
        ("generate_report.py", "3. Write docs/RESULTS.md", []),
    ]

    ok = sum(run_script(s, d, a) for s, d, a in steps)
    print()
    print(f"PIPELINE COMPLETE: {ok}/{len(steps)} steps succeeded")
    print("Results: data/results/  |  Report: docs/RESULTS.md")
    return ok == len(steps)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
