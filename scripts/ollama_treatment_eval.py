#!/usr/bin/env python3
"""
Prompt-level feedback treatment comparison (NO training).

IMPORTANT: This script does not train anything. Ollama models are
inference-only — no gradients, no weight updates. What this measures is how
different *prompt-time* feedback framings (the paper's treatment contents)
affect a frozen local model's answer accuracy against gold labels.

This is a cheap inference-time probe. The paper's actual claim concerns RL
post-training with these treatments; testing that requires real training
(see README roadmap).

Treatments (contents from the paper, Table 1):
  control  - no feedback
  econst   - generic constraints ("focus on logical structure...")
  cdiag    - structured diagnostics checklist

Usage:
    python3 scripts/ollama_treatment_eval.py [num_examples] [model]
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
DEFAULT_MODEL = "phi3.5"
TIMEOUT = 60

TREATMENTS = {
    "control": "",
    "econst": "Focus on logical structure and valid inference rules.\n\n",
    "cdiag": (
        "Diagnostic feedback:\n"
        "- Verify each inference step is valid\n"
        "- Check that all axioms are applied correctly\n"
        "- Ensure the proof is complete\n\n"
    ),
}


def load_split(split: str) -> List[Dict]:
    with open(DATA_DIR / f"{split}.json") as f:
        return json.load(f)


def format_prompt(example: Dict, treatment: str) -> str:
    base = f"""Given the following logical premises, determine whether the conclusion is True, False, or Uncertain.

Premises:
{example['context']}

Conclusion: {example['theorem']}

Reason briefly, then end your answer with exactly one line:
ANSWER: True / False / Uncertain"""
    return TREATMENTS[treatment] + base


def query_ollama(prompt: str, model: str) -> Tuple[str, bool]:
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        return (result.stdout.strip(), True) if result.returncode == 0 else (result.stderr, False)
    except subprocess.TimeoutExpired:
        return "Timeout", False
    except Exception as e:
        return str(e), False


def parse_answer(response: str) -> Optional[str]:
    match = re.findall(r"ANSWER:\s*(True|False|Uncertain)", response, re.IGNORECASE)
    return match[-1].capitalize() if match else None


def evaluate_treatment(treatment: str, examples: List[Dict], model: str) -> Dict:
    print(f"\n--- Treatment: {treatment} ({len(examples)} problems) ---")
    correct = wrong = unparsed = 0
    start = time.time()

    for i, example in enumerate(examples):
        response, ok = query_ollama(format_prompt(example, treatment), model)
        if not ok:
            unparsed += 1
            continue
        predicted = parse_answer(response)
        if predicted is None:
            unparsed += 1
        elif predicted == example["label"]:
            correct += 1
        else:
            wrong += 1

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(examples)} correct={correct}")

    scored = correct + wrong + unparsed
    accuracy = correct / scored if scored else 0.0
    print(f"  => {treatment}: {accuracy*100:.1f}% ({correct}/{scored}), "
          f"wrong={wrong}, unparsed={unparsed}, {time.time()-start:.0f}s")

    return {
        "treatment": treatment,
        "model": model,
        "metric": "answer-level accuracy vs gold label (prompt-level treatment, no training)",
        "total": len(examples),
        "correct": correct,
        "wrong": wrong,
        "unparsed": unparsed,
        "accuracy": accuracy,
    }


def main():
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    print("=" * 80)
    print("TREATMENT COMPARISON (inference-time prompting, NOT RL training)")
    print("=" * 80)
    print(f"Model: {model} | Problems per treatment: {num}")
    print("Random-guess floor: ~33%")

    examples = load_split("test")[:num]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for treatment in TREATMENTS:
        results[treatment] = evaluate_treatment(treatment, examples, model)
        with open(RESULTS_DIR / f"ollama_treatment_{treatment}.json", "w") as f:
            json.dump(results[treatment], f, indent=2)

    print()
    print("SUMMARY")
    print("-" * 40)
    for t, r in results.items():
        print(f"  {t:<10} {r['accuracy']*100:5.1f}%  ({r['correct']}/{r['total']})")

    with open(RESULTS_DIR / "ollama_treatments_all.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
