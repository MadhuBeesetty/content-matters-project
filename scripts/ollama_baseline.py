#!/usr/bin/env python3
"""
Baseline evaluation with a local Ollama model.

Scores answers against the dataset's gold labels (True / False / Uncertain).
This is answer-level evaluation: the model is asked for a final verdict and
we compare it to the label. It does NOT check formal proofs via Prover9.

Usage:
    python3 scripts/ollama_baseline.py [num_examples] [model]

Example:
    python3 scripts/ollama_baseline.py 50 phi3.5
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
TIMEOUT = 60  # seconds per problem


def load_split(split: str) -> List[Dict]:
    with open(DATA_DIR / f"{split}.json") as f:
        return json.load(f)


def format_problem(example: Dict) -> str:
    return f"""Given the following logical premises, determine whether the conclusion is True, False, or Uncertain.

Premises:
{example['context']}

Conclusion: {example['theorem']}

Reason briefly, then end your answer with exactly one line:
ANSWER: True
or
ANSWER: False
or
ANSWER: Uncertain"""


def query_ollama(prompt: str, model: str) -> Tuple[str, bool]:
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        if result.returncode == 0:
            return result.stdout.strip(), True
        return result.stderr, False
    except subprocess.TimeoutExpired:
        return "Timeout", False
    except Exception as e:
        return str(e), False


def parse_answer(response: str) -> Optional[str]:
    """Extract the final True/False/Uncertain verdict, or None if unparsable."""
    match = re.findall(r"ANSWER:\s*(True|False|Uncertain)", response, re.IGNORECASE)
    if match:
        return match[-1].capitalize()
    return None


def run_baseline(num_examples: Optional[int] = None, model: str = DEFAULT_MODEL) -> Dict:
    print("=" * 80)
    print(f"BASELINE EVALUATION - {model} (answer-level, scored against gold labels)")
    print("=" * 80)
    print()

    test_data = load_split("test")
    if num_examples:
        test_data = test_data[:num_examples]
    total = len(test_data)

    print(f"Dataset: {DATA_DIR} (test split)")
    print(f"Model: {model}")
    print(f"Problems: {total}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    correct, wrong, unparsed, failures = 0, 0, 0, 0
    records = []
    start = time.time()

    for i, example in enumerate(test_data):
        response, ok = query_ollama(format_problem(example), model)

        if not ok:
            failures += 1
            records.append({"id": example["id"], "label": example["label"],
                            "predicted": None, "correct": False, "error": response[:200]})
        else:
            predicted = parse_answer(response)
            if predicted is None:
                unparsed += 1
            elif predicted == example["label"]:
                correct += 1
            else:
                wrong += 1
            records.append({"id": example["id"], "label": example["label"],
                            "predicted": predicted, "correct": predicted == example["label"],
                            "response": response[:300]})

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  {i+1:4d}/{total} | correct {correct} | wrong {wrong} | "
                  f"unparsed {unparsed} | {rate:.2f} prob/s | ETA {(total-i-1)/rate if rate else 0:.0f}s")

    elapsed = time.time() - start
    scored = correct + wrong + unparsed
    accuracy = correct / scored if scored else 0.0

    print()
    print("RESULTS")
    print("-" * 80)
    print(f"Correct:  {correct}/{scored} = {accuracy*100:.1f}%")
    print(f"Wrong:    {wrong}")
    print(f"Unparsed: {unparsed}  (no ANSWER line - format failures)")
    print(f"Errors:   {failures}  (model call failed)")
    print(f"Time: {elapsed:.0f}s ({elapsed/total:.1f}s/problem)")
    print()
    print("Note: random guessing scores ~33% on this 3-way task.")

    summary = {
        "model": model,
        "dataset": str(DATA_DIR),
        "metric": "answer-level accuracy vs gold label",
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "unparsed": unparsed,
        "errors": failures,
        "accuracy": accuracy,
        "elapsed_seconds": round(elapsed, 1),
        "records": records,
    }

    out = RESULTS_DIR / f"ollama_{model.replace(':', '_')}_baseline.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")
    return summary


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    m = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    run_baseline(num_examples=n, model=m)
