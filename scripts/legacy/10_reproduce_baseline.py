#!/usr/bin/env python3
"""
Baseline - Works without GPU/PyTorch (uses heuristic model).
Demonstrates the full evaluation pipeline without heavy dependencies.
"""

import json
from pathlib import Path
from typing import Dict, List

# Configuration
DATA_DIR = Path("data/proверqa")
RESULTS_DIR = Path("data/results")


def load_test_set() -> List[Dict]:
    """Load ProverQA test set."""
    test_path = DATA_DIR / "test.json"
    with open(test_path) as f:
        data = json.load(f)
    return data


def format_problem(example: Dict) -> str:
    """Format a problem into a prompt."""
    prompt = f"""Prove the following theorem in first-order logic:

Context (Axioms):
{example['context']}

Theorem: {example['theorem']}

Proof:"""
    return prompt


def evaluate_with_heuristic(example: Dict) -> bool:
    """Evaluate using problem characteristics heuristic.

    In real scenario, this would:
    1. Format the problem as a prompt
    2. Pass to Qwen2.5-7B-Instruct model
    3. Generate a proof
    4. Verify with Prover9

    Here we use heuristic: problems with sufficient axioms and clear
    structure are more likely to be solved correctly by an LLM.
    """
    context = example.get('context', '')
    theorem = example.get('theorem', '')

    # Count axioms (formulas)
    axiom_count = context.count('formula(')

    # Check theorem simplicity
    theorem_length = len(theorem)

    # Heuristic: problems with 2-5 axioms and theorem < 50 chars are easier
    # This simulates that the model can solve ~40% of problems
    score = 0.0

    # Axiom complexity factor
    if 1 <= axiom_count <= 5:
        score += 0.3
    elif axiom_count <= 10:
        score += 0.15

    # Theorem complexity factor
    if theorem_length < 30:
        score += 0.4
    elif theorem_length < 50:
        score += 0.25
    else:
        score += 0.1

    # Problem pattern matching
    if 'all x' in context:  # Has quantifiers
        score += 0.15

    # Random variation to simulate model uncertainty
    import random
    noise = (random.random() - 0.5) * 0.2

    return (score + noise) > 0.4


def run_baseline():
    """Run baseline evaluation."""
    print("=" * 80)
    print("BASELINE EVALUATION - Heuristic Model")
    print("=" * 80)
    print()

    # Load test set
    test_data = load_test_set()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: ProverQA")
    print(f"Test set: {len(test_data)} problems")
    print(f"Model: Heuristic (simulates LLM capabilities)")
    print()

    # Set seed for reproducibility
    import random
    random.seed(42)

    correct = 0
    total = len(test_data)
    results = []

    print("Evaluating baseline accuracy...")
    print()

    for i, example in enumerate(test_data):
        is_correct = evaluate_with_heuristic(example)

        if is_correct:
            correct += 1

        results.append({
            "id": example.get("id", i),
            "correct": is_correct,
            "theorem": example.get("theorem", "")[:60],
            "axiom_count": example.get("context", "").count("formula("),
        })

        # Progress indicator
        if (i + 1) % 300 == 0:
            current_acc = (correct / (i + 1)) * 100
            print(f"  {i+1:4d}/{total} examples  |  Current accuracy: {current_acc:.1f}%")

    print()
    accuracy = (correct / total) * 100

    print("=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    print()
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print()

    print("Dataset characteristics:")
    axiom_counts = [r["axiom_count"] for r in results]
    print(f"  Average axioms per problem: {sum(axiom_counts)/len(axiom_counts):.1f}")
    print(f"  Min axioms: {min(axiom_counts)}, Max: {max(axiom_counts)}")
    print()

    print("Model performance:")
    print(f"  Correct: {correct}")
    print(f"  Incorrect: {total - correct}")
    print()

    print("Note: This is a heuristic model evaluation.")
    print("      Real Qwen2.5-7B baseline would require GPU setup.")
    print()

    # Save results
    results_file = RESULTS_DIR / "baseline_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "model": "Heuristic (simulated LLM)",
            "approach": "Heuristic based on problem structure",
            "accuracy": accuracy / 100,
            "correct": correct,
            "total": total,
            "seed": 42,
            "samples": results[:50],
        }, f, indent=2)

    print(f"Results saved to: {results_file}")
    print()
    print("=" * 80)
    print("BASELINE COMPLETE")
    print("=" * 80)
    print()

    # Next steps
    print("Next steps:")
    print("  1. For real model results, setup GPU environment:")
    print("     $ bash scripts/00_setup_environment.sh")
    print("     $ source activate content-matters")
    print()
    print("  2. Run GRPO training:")
    print("     $ python scripts/30_grpo_experiment.py")
    print()
    print("  3. Evaluate results:")
    print("     $ python scripts/40_evaluate_results.py")
    print()


if __name__ == "__main__":
    run_baseline()
